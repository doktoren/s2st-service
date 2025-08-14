"""WebSocket service for VAD and forwarding to the translation API."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from typing import Any, Literal, TypedDict

import httpx
import numpy as np
import orjson
import torch
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, RootModel

from .common import AudioFormat, AudioUtils, Codec

logger = logging.getLogger("seamless.ws")
logging.basicConfig(level=logging.INFO)

SILENCE_RMS_THRESHOLD = 0.01
TRANSLATE_URL = os.environ.get("TRANSLATE_URL", "http://localhost:8001/translate")
# ---------------------------
# Protocol models
# ---------------------------


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
    negotiated: dict[str, Any]


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


class ErrorMessage(BaseModel):
    """Error payload."""

    type: Literal["error"] = "error"
    code: Literal["bad_setup", "invalid_audio", "over_limit", "server_error"]
    message: str

# ---------------------------
# Session handling
# ---------------------------


class SessionState(TypedDict):
    audio_format: AudioFormat
    vad: webrtcvad.Vad
    vad_frame_ms: int
    vad_aggr: int
    buffer_pcm16_16k: list[torch.Tensor]
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


async def _handle_audio_msg(msg: AudioMessage, setup: SetupMessage, state: SessionState) -> None:
    """Decode incoming frame, update VAD state and buffer speech."""
    if msg.duration_ms != 20:
        error_msg = "server_vad requires duration_ms == 20 for every frame"
        raise ValueError(error_msg)

    fmt = setup.audio_format
    data = AudioUtils.b64_to_bytes(msg.audio_b64)
    if fmt.codec is Codec.G711_ULAW:
        pcm16 = AudioUtils.ulaw_bytes_to_pcm16(data)
        tens = torch.from_numpy(pcm16.astype(np.float32) / 32768.0)
        wave_16k = AudioUtils.resample(tens, 8000, 16000)
    else:
        tens = AudioUtils.pcm16_bytes_to_tensor_mono(data)
        wave_16k = tens if fmt.sample_rate == 16000 else AudioUtils.resample(tens, fmt.sample_rate, 16000)
    pcm_bytes_16k = AudioUtils.tensor_to_pcm16_bytes_mono(wave_16k)
    is_speech = state["vad"].is_speech(pcm_bytes_16k, 16000)
    if is_speech:
        rms = float(torch.sqrt(torch.mean(wave_16k * wave_16k)).item())
        if rms < SILENCE_RMS_THRESHOLD:
            is_speech = False
    if is_speech:
        state["has_speech"] = True
        state["silence_ms"] = 0
    else:
        state["silence_ms"] += state["vad_frame_ms"]
    if state["has_speech"]:
        state["buffer_pcm16_16k"].append(wave_16k)


def _vad_segment_closed(state: SessionState) -> bool:
    """Return ``True`` when speech is followed by >=400 ms of silence."""
    return state["has_speech"] and state["silence_ms"] >= 400


async def _maybe_infer_and_emit(ws: WebSocket, setup: SetupMessage, state: SessionState) -> None:
    """Send buffered audio to the translation service and emit the result."""
    if not state["buffer_pcm16_16k"] or not state["has_speech"]:
        state["buffer_pcm16_16k"].clear()
        state["has_speech"] = False
        state["silence_ms"] = 0
        return

    utterance_id = f"utt-{state['utterance_seq']}"
    state["utterance_seq"] += 1

    wave_16k = torch.cat(state["buffer_pcm16_16k"])
    state["buffer_pcm16_16k"].clear()

    min_samples = int(0.1 * 16000)
    if wave_16k.numel() < min_samples:
        logger.info("Dropping short utterance: samples=%d", wave_16k.numel())
        state["has_speech"] = False
        state["silence_ms"] = 0
        state["vad"] = make_vad(state["vad_aggr"])
        return

    start = asyncio.get_event_loop().time()
    pcm_bytes = AudioUtils.tensor_to_pcm16_bytes_mono(wave_16k)
    payload = {
        "audio_b64": AudioUtils.bytes_to_b64(pcm_bytes),
        "audio_format": setup.audio_format.model_dump(),
        "target_language": setup.target_language,
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(TRANSLATE_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.exception("Translation failed")
        await ws.send_text(ErrorMessage(type="error", code="server_error", message=str(e)).model_dump_json())
        return
    latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)

    chunk = AudioChunkMessage(
        type="audio_chunk",
        utterance_id=utterance_id,
        seq=0,
        audio_b64=data["audio_b64"],
        duration_ms=int(data["duration_ms"]),
    )
    await ws.send_text(chunk.model_dump_json())
    await ws.send_text(
        EndOfAudioMessage(type="end_of_audio", utterance_id=utterance_id, latency_ms=latency_ms).model_dump_json()
    )

    state["has_speech"] = False
    state["silence_ms"] = 0
    state["vad"] = make_vad(state["vad_aggr"])


@app.websocket("/ws")
async def ws_handler(ws: WebSocket) -> None:  # noqa: C901, PLR0912, PLR0915
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

    if setup.chunking.get("strategy") != "server_vad":
        await ws.send_text(
            ErrorMessage(
                type="error",
                code="bad_setup",
                message="chunking.strategy must be 'server_vad'",
            ).model_dump_json()
        )
        with suppress(Exception):
            await ws.close()
        return

    vad_aggr = int(setup.chunking.get("vad_aggressiveness", 1))
    state: SessionState = {
        "audio_format": setup.audio_format,
        "vad": make_vad(vad_aggr),
        "vad_frame_ms": 20,
        "vad_aggr": vad_aggr,
        "buffer_pcm16_16k": [],
        "utterance_seq": 0,
        "has_speech": False,
        "silence_ms": 0,
    }

    chunking: dict[str, Any] = {
        "strategy": "server_vad",
        "vad": {"aggressiveness": state["vad_aggr"], "frame_ms": state["vad_frame_ms"]},
    }
    negotiated = {
        "audio_format": setup.audio_format.model_dump(),
        "model_audio": {"sample_rate": 16000},
        "chunking": chunking,
    }
    await ws.send_text(ReadyMessage(type="ready", session_id=id(ws).__str__(), negotiated=negotiated).model_dump_json())

    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"] is not None:
                try:
                    incoming = IncomingMessage.model_validate_json(msg["text"]).root
                except Exception:
                    data = orjson.loads(msg["text"].encode("utf-8"))
                    if data.get("type") == "close":
                        await _maybe_infer_and_emit(ws, setup, state)
                        break
                    await ws.send_text(
                        ErrorMessage(type="error", code="invalid_audio", message="invalid message").model_dump_json()
                    )
                    continue

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
                    if _vad_segment_closed(state):
                        await _maybe_infer_and_emit(ws, setup, state)

            elif "bytes" in msg and msg["bytes"] is not None:
                await ws.send_text(
                    ErrorMessage(
                        type="error", code="invalid_audio", message="binary frames not supported"
                    ).model_dump_json()
                )
            else:
                break
    except WebSocketDisconnect:
        state["buffer_pcm16_16k"].clear()
        state["has_speech"] = False
        state["silence_ms"] = 0
    except Exception as exc:
        logger.exception("Server error")
        await ws.send_text(ErrorMessage(type="error", code="server_error", message=str(exc)).model_dump_json())
