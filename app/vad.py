"""WebSocket service for VAD and forwarding to the translation API."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from contextlib import suppress
from dataclasses import dataclass

import audioop
import httpx
import torch
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .common import (
    AudioChunkMessage,
    AudioFormat,
    AudioMessage,
    CloseMessage,
    Codec,
    EndOfAudioMessage,
    ErrorMessage,
    IncomingMessage,
    ReadyMessage,
    SetupMessage,
)

logger = logging.getLogger("seamless.ws")
logging.basicConfig(level=logging.INFO)

TRANSLATE_URL = os.environ.get("TRANSLATE_URL", "http://localhost:8001/translate")

silero_model: torch.nn.Module | None = None


def _silero_is_speech(pcm16_a: bytes, pcm16_b: bytes) -> bool:
    """Return True if Silero VAD detects speech in the frame."""
    global silero_model
    if silero_model is None:
        silero_model, _ = torch.hub.load(  # type: ignore[no-untyped-call]
            "snakers4/silero-vad", "silero_vad", trust_repo=True
        )
    audio_tensor = torch.frombuffer(pcm16_a + pcm16_b, dtype=torch.int16).to(torch.float32) / 32768.0
    with torch.no_grad():
        prob: float = float(silero_model(audio_tensor, 16000).item())
    return prob > 0.5


# ---------------------------
# Session handling
# ---------------------------

SPEECH_STREAK_COUNT_THRESHOLD = 3
SILENCE_NUM_FRAMES = 30


@dataclass
class Audio:
    pcm16_16k: bytes
    webrtc_is_speech: bool
    # silero_is_speech is set to True if this frame joined with the previous or next
    # frame passes silero speech detection
    silero_is_speech: bool = False


@dataclass
class SessionState:
    vad: webrtcvad.Vad
    vad_aggr: int
    utterance_seq: int
    audio: list[Audio]
    speech_detected: bool


def make_vad(aggr: int) -> webrtcvad.Vad:
    """Create a configured WebRTC VAD instance."""
    vad = webrtcvad.Vad()
    vad.set_mode(aggr)
    return vad


# ---------------------------
# FastAPI app
# ---------------------------


app = FastAPI(title="Seamless S2ST WebSocket")


@app.get("/")
async def index() -> HTMLResponse:
    """Serve a minimal page to verify the server is up."""
    return HTMLResponse("<pre>Seamless S2ST WebSocket service is running.</pre>")


def _decode_and_resample(msg: AudioMessage, fmt: AudioFormat) -> bytes:
    """Decode wire audio to 16 kHz mono PCM16 bytes."""
    data = base64.b64decode(msg.audio_b64)
    if fmt.codec is Codec.G711_ULAW:
        pcm16 = audioop.ulaw2lin(data, 2)
        return audioop.ratecv(pcm16, 2, 1, 8000, 16000, None)[0]
    if fmt.sample_rate != 16000:
        return audioop.ratecv(data, 2, 1, fmt.sample_rate, 16000, None)[0]
    return data


async def _handle_audio_msg(msg: AudioMessage, setup: SetupMessage, state: SessionState) -> None:
    """Decode incoming frame, update VAD state and buffer speech."""
    if msg.duration_ms != 20:
        error_msg = "duration_ms must be 20 for every frame"
        raise ValueError(error_msg)

    pcm16k = _decode_and_resample(msg, setup.audio_format)

    audio = Audio(
        pcm16_16k=pcm16k,
        # 330 is ~0.01 in int16 units. This check may be better to omit
        webrtc_is_speech=state.vad.is_speech(pcm16k, 16000),  # if audioop.rms(pcm16k, 2) >= 330 else False,
    )
    state.audio.append(audio)
    if len(state.audio) >= 2 and state.audio[-2].webrtc_is_speech and state.audio[-1].webrtc_is_speech:
        silero_is_speech = _silero_is_speech(state.audio[-2].pcm16_16k, state.audio[-1].pcm16_16k)
        state.audio[-2].silero_is_speech |= silero_is_speech
        state.audio[-1].silero_is_speech |= silero_is_speech
    if (
        not state.speech_detected
        and len(state.audio) >= SPEECH_STREAK_COUNT_THRESHOLD
        and all(audio.silero_is_speech for audio in state.audio[-SPEECH_STREAK_COUNT_THRESHOLD:])
    ):
        state.speech_detected = True


async def _maybe_infer_and_emit(ws: WebSocket, setup: SetupMessage, state: SessionState) -> None:
    """Send buffered audio to the translation service and emit the result."""
    try:
        if not state.speech_detected:
            return

        utterance_id = f"utt-{state.utterance_seq}"
        state.utterance_seq += 1

        pcm_bytes = b"".join(audio.pcm16_16k for audio in state.audio)
        min_samples = 16000 // 10  # 0.1 second
        if len(pcm_bytes) // 2 < min_samples:
            logger.info("Dropping short utterance: samples=%d", len(pcm_bytes) // 2)
            return

        src_duration_ms = len(pcm_bytes) * 1000 // (2 * 16000)
        start = asyncio.get_event_loop().time()
        payload = {
            "audio_b64": base64.b64encode(pcm_bytes).decode("ascii"),
            "audio_format": {"codec": "pcm16", "sample_rate": 16000, "channels": 1},
            "target_language": setup.target_language,
        }
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(TRANSLATE_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:  # pragma: no cover - network failure
            logger.exception("Translation failed")
            await ws.send_text(ErrorMessage(type="error", code="server_error", message=str(e)).model_dump_json())
            return
        latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)

        tgt_duration_ms = int(data["duration_ms"])
        chunk = AudioChunkMessage(
            type="audio_chunk",
            utterance_id=utterance_id,
            seq=0,
            audio_b64=data["audio_b64"],
            duration_ms=tgt_duration_ms,
        )
        await ws.send_text(chunk.model_dump_json())
        await ws.send_text(
            EndOfAudioMessage(
                type="end_of_audio",
                utterance_id=utterance_id,
                latency_ms=latency_ms,
                src_duration_ms=src_duration_ms,
                tgt_duration_ms=tgt_duration_ms,
            ).model_dump_json()
        )

    finally:
        state.audio.clear()
        state.speech_detected = False


@app.websocket("/ws")
async def ws_handler(ws: WebSocket) -> None:  # noqa: C901
    """WebSocket endpoint implementing the agreed protocol."""
    await ws.accept()

    try:
        raw = await ws.receive_text()
        setup = SetupMessage.model_validate_json(raw)
    except Exception as exc:
        await ws.send_text(ErrorMessage(type="error", code="bad_setup", message=str(exc)).model_dump_json())
        with suppress(Exception):
            await ws.close()
        return

    vad_aggr = 1
    state = SessionState(
        vad=make_vad(vad_aggr),
        vad_aggr=vad_aggr,
        audio=[],
        utterance_seq=0,
        speech_detected=False,
    )

    await ws.send_text(ReadyMessage(type="ready", session_id=id(ws).__str__()).model_dump_json())

    try:
        while True:
            msg = await ws.receive()
            if (text := msg.get("text")) is not None:
                incoming = IncomingMessage.model_validate_json(text).root
                if isinstance(incoming, CloseMessage):
                    await _maybe_infer_and_emit(ws, setup, state)
                    break
                if isinstance(incoming, SetupMessage):
                    await ws.send_text(
                        ErrorMessage(
                            type="error",
                            code="bad_setup",
                            message="setup already completed",
                        ).model_dump_json()
                    )
                elif isinstance(incoming, AudioMessage):
                    await _handle_audio_msg(incoming, setup, state)
                    webrtc = "".join("T" if audio.webrtc_is_speech else "F" for audio in state.audio)
                    silero = "".join("T" if audio.silero_is_speech else "F" for audio in state.audio)
                    logger.info(f"Now state is len={len(state.audio)}\nWebRTC: {webrtc}\nSilero: {silero}")
                    if (
                        state.speech_detected
                        and len(state.audio) > SILENCE_NUM_FRAMES
                        and not any(audio.silero_is_speech for audio in state.audio[-SILENCE_NUM_FRAMES:])
                    ):
                        await _maybe_infer_and_emit(ws, setup, state)

            elif "bytes" in msg and msg["bytes"] is not None:
                await ws.send_text(
                    ErrorMessage(
                        type="error", code="invalid_audio", message="binary frames not supported"
                    ).model_dump_json()
                )
            else:
                logger.warning(f"Received invalid message - aborting: {msg}")
                break
    except WebSocketDisconnect:
        state.audio.clear()
        state.speech_detected = False
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.exception("Server error")
        await ws.send_text(ErrorMessage(type="error", code="server_error", message=str(exc)).model_dump_json())
