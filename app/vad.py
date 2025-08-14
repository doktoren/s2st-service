"""WebSocket service for VAD and forwarding to the translation API."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import audioop
import httpx
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, RootModel

logger = logging.getLogger("seamless.ws")
logging.basicConfig(level=logging.INFO)

TRANSLATE_URL = os.environ.get("TRANSLATE_URL", "http://localhost:8001/translate")


# ---------------------------
# Protocol models
# ---------------------------


class Codec(str, Enum):
    """Supported audio codecs for the client/server wire format."""

    G711_ULAW = "g711_ulaw"
    PCM16 = "pcm16"


class AudioFormat(BaseModel):
    """Audio format negotiated with the client."""

    codec: Codec
    sample_rate: Literal[8000, 16000, 24000]
    channels: Literal[1]


class SetupMessage(BaseModel):
    """Initial setup message sent by the client."""

    type: Literal["setup"] = "setup"
    target_language: str = Field(..., description="Target language code as supported by Seamless.")
    audio_format: AudioFormat
    chunking: dict[str, Any]


class AudioMessage(BaseModel):
    """Audio frame payload."""

    type: Literal["audio"] = "audio"
    seq: int
    audio_b64: str
    duration_ms: int


class CloseMessage(BaseModel):
    """Client intends to close the session."""

    type: Literal["close"] = "close"


IncomingMessage = RootModel[SetupMessage | AudioMessage | CloseMessage]


class ReadyMessage(BaseModel):
    """Server response after setup negotiation."""

    type: Literal["ready"] = "ready"
    session_id: str


class AudioChunkMessage(BaseModel):
    """Chunk of synthesized translation audio."""

    type: Literal["audio_chunk"] = "audio_chunk"
    utterance_id: str
    seq: int
    audio_b64: str
    duration_ms: int


class EndOfAudioMessage(BaseModel):
    """Marks completion of one utterance translation."""

    type: Literal["end_of_audio"] = "end_of_audio"
    utterance_id: str
    latency_ms: int
    src_duration_ms: int
    tgt_duration_ms: int


class ErrorMessage(BaseModel):
    """Error payload."""

    type: Literal["error"] = "error"
    code: Literal["bad_setup", "invalid_audio", "over_limit", "server_error"]
    message: str


# ---------------------------
# Session handling
# ---------------------------


@dataclass
class SessionState:
    vad: webrtcvad.Vad
    vad_aggr: int
    buffer_pcm16_16k: list[bytes]
    utterance_seq: int
    has_speech: bool
    silence_ms: int


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
    is_speech = state.vad.is_speech(pcm16k, 16000)
    if is_speech:
        rms = audioop.rms(pcm16k, 2)
        if rms < 330:  # ~0.01 in int16 units
            is_speech = False
    if is_speech:
        state.has_speech = True
        state.silence_ms = 0
    elif state.has_speech:
        state.silence_ms += 20
    if state.has_speech:
        state.buffer_pcm16_16k.append(pcm16k)


async def _maybe_infer_and_emit(ws: WebSocket, setup: SetupMessage, state: SessionState) -> None:
    """Send buffered audio to the translation service and emit the result."""
    try:
        if not state.buffer_pcm16_16k or not state.has_speech:
            return

        utterance_id = f"utt-{state.utterance_seq}"
        state.utterance_seq += 1

        pcm_bytes = b"".join(state.buffer_pcm16_16k)
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
        state.buffer_pcm16_16k.clear()
        state.has_speech = False
        state.silence_ms = 0
        state.vad = make_vad(state.vad_aggr)


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

    vad_aggr = int(setup.chunking.get("vad_aggressiveness", 1))
    state = SessionState(
        vad=make_vad(vad_aggr),
        vad_aggr=vad_aggr,
        buffer_pcm16_16k=[],
        utterance_seq=0,
        has_speech=False,
        silence_ms=0,
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
                    logger.info(
                        f"Now state is len={len(state.buffer_pcm16_16k)}, "
                        f"has_speech={state.has_speech}, silence_ms={state.silence_ms}"
                    )
                    if state.has_speech and state.silence_ms >= 400:
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
        state.buffer_pcm16_16k.clear()
        state.has_speech = False
        state.silence_ms = 0
    except Exception as exc:  # pragma: no cover - unexpected failure
        logger.exception("Server error")
        await ws.send_text(ErrorMessage(type="error", code="server_error", message=str(exc)).model_dump_json())
