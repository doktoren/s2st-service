"""Combined Azure HTTP translation and WebSocket VAD service."""

from __future__ import annotations

import asyncio
import base64
import os
import queue
import threading

import azure.cognitiveservices.speech as speechsdk
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .common import (
    AudioChunkMessage,
    AudioFormat,
    AudioMessage,
    AudioUtils,
    CloseMessage,
    Codec,
    EndOfAudioMessage,
    ErrorMessage,
    IncomingMessage,
    ReadyMessage,
    SetupMessage,
)

# Mapping from target language code to Azure voice name.
VOICE_MAP: dict[str, str] = {"en": "en-US-JennyNeural"}


def _env(name: str) -> str:
    """Return required environment variable or raise ``ValueError``."""
    val = os.getenv(name)
    if not val:
        msg = f"Missing required environment variable: {name}"
        raise ValueError(msg)
    return val


def _make_config(
    source_language: str, target_language: str
) -> speechsdk.translation.SpeechTranslationConfig:
    """Build a translation config for Azure Speech."""
    key = _env("SPEECH_KEY")
    region = os.getenv("SPEECH_REGION", "northeurope")
    cfg = speechsdk.translation.SpeechTranslationConfig(subscription=key, region=region)
    cfg.speech_recognition_language = source_language
    cfg.add_target_language(target_language)
    voice = VOICE_MAP.get(target_language)
    if voice is None:
        msg = f"Unsupported target language: {target_language}"
        raise ValueError(msg)
    cfg.voice_name = voice
    return cfg


def _stream_format() -> speechsdk.audio.AudioStreamFormat:
    """Return the default PCM16 mono stream format."""
    return speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)


def _azure_s2st(
    pcm_bytes: bytes, cfg: speechsdk.translation.SpeechTranslationConfig
) -> bytes:
    """Translate PCM16 audio using Azure Speech."""
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=_stream_format())
    audio_in = speechsdk.audio.AudioConfig(stream=push_stream)
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=cfg, audio_config=audio_in
    )

    audio_chunks: list[bytes] = []
    done = threading.Event()

    def _on_synth(_s: object, e: speechsdk.SpeechSynthesisEventArgs) -> None:
        """Collect synthesized audio chunks and signal completion."""
        reason = getattr(e.result, "reason", None)
        if reason == speechsdk.ResultReason.SynthesizingAudio:
            chunk = getattr(e.result, "audio", None)
            if isinstance(chunk, (bytes, bytearray)):
                audio_chunks.append(bytes(chunk))
        elif reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            done.set()

    recognizer.synthesizing.connect(_on_synth)
    recognizer.start_continuous_recognition_async().get()
    push_stream.write(pcm_bytes)
    push_stream.close()
    done.wait()
    recognizer.stop_continuous_recognition_async().get()
    return b"".join(audio_chunks)


app = FastAPI(title="Azure S2ST")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TranslateRequest(BaseModel):
    """Request payload for ``/translate``."""

    audio_b64: str = Field(..., description="Base64 encoded audio in the given format.")
    audio_format: AudioFormat
    source_language: str
    target_language: str


class TranslateResponse(BaseModel):
    """Translated audio payload."""

    audio_b64: str
    duration_ms: int


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    """Translate an audio segment using Azure Speech."""
    fmt = req.audio_format
    if fmt.codec is not Codec.PCM16 or fmt.sample_rate != 16000:
        raise HTTPException(status_code=400, detail="only 16 kHz PCM16 supported")
    pcm_bytes = AudioUtils.b64_to_bytes(req.audio_b64)
    cfg = _make_config(req.source_language, req.target_language)
    out_pcm = await asyncio.to_thread(_azure_s2st, pcm_bytes, cfg)
    duration_ms = len(out_pcm) * 1000 // (2 * 16000)
    return TranslateResponse(audio_b64=AudioUtils.bytes_to_b64(out_pcm), duration_ms=duration_ms)


@app.get("/")
async def index() -> dict[str, str]:
    """Return service status."""
    return {"status": "ok"}


@app.websocket("/ws")
async def ws_handler(ws: WebSocket) -> None:  # noqa: C901, PLR0915
    """WebSocket endpoint forwarding audio to Azure Speech."""
    await ws.accept()
    try:
        raw = await ws.receive_text()
        setup = SetupMessage.model_validate_json(raw)
    except Exception as exc:
        await ws.send_text(ErrorMessage(type="error", code="bad_setup", message=str(exc)).model_dump_json())
        await ws.close()
        return

    frame_q: queue.Queue[bytes | None] = queue.Queue()
    loop = asyncio.get_event_loop()
    utterance_seq = 0
    seq = 0

    async def send_ready() -> None:
        """Send the initial ready message to the client."""
        await ws.send_text(ReadyMessage(type="ready", session_id=str(id(ws))).model_dump_json())

    async def send_audio(chunk: bytes) -> None:
        """Send a translated audio chunk to the client."""
        nonlocal seq, utterance_seq
        msg = AudioChunkMessage(
            type="audio_chunk",
            utterance_id=f"utt-{utterance_seq}",
            seq=seq,
            audio_b64=base64.b64encode(chunk).decode("ascii"),
            duration_ms=len(chunk) * 1000 // (2 * 16000),
        )
        seq += 1
        await ws.send_text(msg.model_dump_json())

    async def send_end(tgt_ms: int) -> None:
        """Send the end-of-audio marker for one utterance."""
        nonlocal utterance_seq, seq
        await ws.send_text(
            EndOfAudioMessage(
                type="end_of_audio",
                utterance_id=f"utt-{utterance_seq}",
                latency_ms=0,
                src_duration_ms=0,
                tgt_duration_ms=tgt_ms,
            ).model_dump_json()
        )
        utterance_seq += 1
        seq = 0

    def worker() -> None:
        """Background thread feeding audio to Azure and relaying results."""
        cfg = _make_config(setup.source_language, setup.target_language)
        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=_stream_format())
        audio_in = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.translation.TranslationRecognizer(translation_config=cfg, audio_config=audio_in)

        current_audio = bytearray()

        def _on_synth(_s: object, e: speechsdk.SpeechSynthesisEventArgs) -> None:
            """Handle synthesized audio callbacks from Azure Speech."""
            reason = getattr(e.result, "reason", None)
            if reason == speechsdk.ResultReason.SynthesizingAudio:
                chunk = getattr(e.result, "audio", None)
                if isinstance(chunk, (bytes, bytearray)):
                    data = bytes(chunk)
                    current_audio.extend(data)
                    asyncio.run_coroutine_threadsafe(send_audio(data), loop)
            elif reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                tgt_ms = len(current_audio) * 1000 // (2 * 16000)
                current_audio.clear()
                asyncio.run_coroutine_threadsafe(send_end(tgt_ms), loop)

        recognizer.synthesizing.connect(_on_synth)

        recognizer.start_continuous_recognition_async().get()
        asyncio.run_coroutine_threadsafe(send_ready(), loop)

        while True:
            frame = frame_q.get()
            if frame is None:
                break
            push_stream.write(frame)
        push_stream.close()
        recognizer.stop_continuous_recognition_async().get()

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        while True:
            msg = await ws.receive()
            if (text := msg.get("text")) is not None:
                incoming = IncomingMessage.model_validate_json(text).root
                if isinstance(incoming, AudioMessage):
                    frame_q.put(base64.b64decode(incoming.audio_b64))
                elif isinstance(incoming, CloseMessage):
                    frame_q.put(None)
                    break
                else:
                    await ws.send_text(
                        ErrorMessage(
                            type="error", code="bad_setup", message="setup already completed"
                        ).model_dump_json()
                    )
            elif msg.get("bytes") is not None:
                await ws.send_text(
                    ErrorMessage(
                        type="error", code="invalid_audio", message="binary frames not supported"
                    ).model_dump_json()
                )
            else:
                break
    except WebSocketDisconnect:
        pass
    finally:
        frame_q.put(None)
        thread.join()
