"""Combined Azure HTTP translation and WebSocket VAD service."""

from __future__ import annotations

import asyncio
import base64
import logging
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Mapping from target language code to Azure voice name.
VOICE_MAP: dict[str, str] = {"en": "en-US-JennyNeural"}
TARGET_MAP: dict[str, str] = {"en": "en-US", "da": "da-DK"}


def _env(name: str) -> str:
    """Return required environment variable or raise ``ValueError``."""
    val = os.getenv(name)
    if not val:
        msg = f"Missing required environment variable: {name}"
        raise ValueError(msg)
    return val


def _stream_format() -> speechsdk.audio.AudioStreamFormat:
    """Return the default PCM16 mono stream format."""
    return speechsdk.audio.AudioStreamFormat(samples_per_second=16000, bits_per_sample=16, channels=1)


def _azure_s2st(pcm_bytes: bytes, source_language: str, target_language: str) -> bytes:
    """Translate PCM16 audio using Azure Speech."""
    logger.info(
        f"_azure_s2st(pcm_bytes of len {len(pcm_bytes)}, "
        f"source_language={source_language}, target_language={target_language}): "
    )
    config = speechsdk.translation.SpeechTranslationConfig(
        subscription=_env("SPEECH_KEY"), region=os.getenv("SPEECH_REGION", "northeurope")
    )
    config.speech_recognition_language = TARGET_MAP[source_language]
    config.add_target_language(target_language)
    config.voice_name = VOICE_MAP[target_language]
    # config.speech_synthesis_voice_name = VOICE_MAP[target_language]  # No, this is for Text-To-Speech
    config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm)

    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=_stream_format())
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=config,
        audio_config=speechsdk.audio.AudioConfig(stream=push_stream),
    )
    audio_chunks: list[bytes] = []
    done = threading.Event()

    def _on_synthesizing(event_args: speechsdk.translation.TranslationSynthesisEventArgs) -> None:
        """Collect synthesized audio chunks and signal completion."""
        match event_args.result.reason:
            case speechsdk.ResultReason.SynthesizingAudio:
                audio_chunks.append(event_args.result.audio)
            case speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("Synthesizing audio completed")
            case _:
                logger.info(f"Unknown event: {event_args}")

    def _on_recognizing(event_args: speechsdk.translation.TranslationRecognitionEventArgs) -> None:
        """Collect synthesized audio chunks and signal completion."""
        logger.info(f"_on_recognizing({event_args}) called")

    def _set_done(event_args: object) -> None:
        logger.info(f"_set_done({event_args}) called")
        done.set()

    recognizer.synthesizing.connect(_on_synthesizing)
    recognizer.recognizing.connect(_on_recognizing)
    recognizer.session_stopped.connect(_set_done)
    recognizer.canceled.connect(_set_done)
    try:
        recognizer.start_continuous_recognition()
        logger.info(
            f"Writing {len(pcm_bytes)}: {base64.b64encode(pcm_bytes[:30])!r}...{base64.b64encode(pcm_bytes[-30:])!r}"
        )
        push_stream.write(pcm_bytes)
        push_stream.close()
        done.wait()
    finally:
        recognizer.stop_continuous_recognition()
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
def translate(req: TranslateRequest) -> TranslateResponse:
    """Translate an audio segment using Azure Speech."""
    # logger.info(req.model_dump())
    fmt = req.audio_format
    if fmt.codec is not Codec.PCM16 or fmt.sample_rate != 16000:
        raise HTTPException(status_code=400, detail="only 16 kHz PCM16 supported")
    pcm_bytes = AudioUtils.b64_to_bytes(req.audio_b64)
    out_pcm = _azure_s2st(pcm_bytes, req.source_language, req.target_language)
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
        count = 0
        config = speechsdk.translation.SpeechTranslationConfig(
            subscription=_env("SPEECH_KEY"), region=os.getenv("SPEECH_REGION", "northeurope")
        )
        config.speech_recognition_language = TARGET_MAP[setup.source_language]
        config.add_target_language(setup.target_language)
        config.voice_name = VOICE_MAP[setup.target_language]
        # config.speech_synthesis_voice_name = VOICE_MAP[target_language]  # No, this is for Text-To-Speech
        config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm)

        push_stream = speechsdk.audio.PushAudioInputStream(stream_format=_stream_format())
        audio_in = speechsdk.audio.AudioConfig(stream=push_stream)
        recognizer = speechsdk.translation.TranslationRecognizer(translation_config=config, audio_config=audio_in)

        def _on_synthesizing(event_args: speechsdk.translation.TranslationSynthesisEventArgs) -> None:
            """Collect synthesized audio chunks and signal completion."""
            match event_args.result.reason:
                case speechsdk.ResultReason.SynthesizingAudio:
                    asyncio.run_coroutine_threadsafe(send_audio(event_args.result.audio), loop)
                case speechsdk.ResultReason.SynthesizingAudioCompleted:
                    logger.info("Synthesizing audio completed")
                case _:
                    logger.info(f"Unknown event: {event_args}")

        def _on_recognizing(event_args: speechsdk.translation.TranslationRecognitionEventArgs) -> None:
            """Collect synthesized audio chunks and signal completion."""
            logger.info(f"_on_recognizing({event_args}) called")

        def _set_done(event_args: object) -> None:
            logger.info(f"_set_done({event_args}) called")
            frame_q.put(None)

        recognizer.synthesizing.connect(_on_synthesizing)
        recognizer.recognizing.connect(_on_recognizing)
        recognizer.session_stopped.connect(_set_done)
        recognizer.canceled.connect(_set_done)

        recognizer.start_continuous_recognition_async().get()
        asyncio.run_coroutine_threadsafe(send_ready(), loop)

        while True:
            frame = frame_q.get()
            if frame is None:
                break

            count += 1
            if count % 50 == 0:
                logger.info("Pushing audio frame")
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
