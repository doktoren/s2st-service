"""
FastAPI WebSocket S2ST service using SeamlessM4T-V2-Large.

- Single mode: s2st.
- Client chooses input/output audio format (must match), one of:
  * g711_ulaw mono @ 8000 Hz
  * pcm16 mono @ 16000 Hz
  * pcm16 mono @ 24000 Hz
- Two chunking strategies:
  * server_vad: client sends 20 ms frames; server segments with VAD
  * client_chunked: client sends frames then an explicit end_of_segment; each segment is translated individually

This module targets Python 3.13 and strict typing/mypy.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, TypedDict

import g711
import numpy as np
import orjson
import torch
import torchaudio
import webrtcvad
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, RootModel
from transformers import AutoProcessor, SeamlessM4Tv2Model

logger = logging.getLogger("seamless.ws")
logging.basicConfig(level=logging.INFO)

# ---------------------------
# Protocol models
# ---------------------------


class Codec(str, Enum):
    """Supported audio codecs for the client/server wire format."""

    G711_ULAW = "g711_ulaw"
    PCM16 = "pcm16"


class ChunkingStrategy(str, Enum):
    """Chunking strategies supported by the server."""

    SERVER_VAD = "server_vad"
    CLIENT_CHUNKED = "client_chunked"


class AudioFormat(BaseModel):
    """
    Audio format negotiated with the client.

    The same format is used for input and output.
    """

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


class EndOfSegmentMessage(BaseModel):
    """Marks the end of a client-chunked segment."""

    type: Literal["end_of_segment"] = "end_of_segment"


class CloseMessage(BaseModel):
    """Client intends to close the session."""

    type: Literal["close"] = "close"


IncomingMessage = RootModel[SetupMessage | AudioMessage | EndOfSegmentMessage | CloseMessage]


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
# Audio utilities
# ---------------------------


class AudioUtils:
    """Helpers for codec (de)serialisation and resampling."""

    @staticmethod
    def b64_to_bytes(data_b64: str) -> bytes:
        """Decode a base64 string to raw bytes."""
        return base64.b64decode(data_b64)

    @staticmethod
    def bytes_to_b64(buf: bytes) -> str:
        """Encode raw bytes to base64 string without newlines."""
        return base64.b64encode(buf).decode("ascii")

    @staticmethod
    def ulaw_bytes_to_pcm16(bytes_in: bytes) -> np.ndarray:
        """Decode μ-law bytes to PCM16 int16 numpy array (mono)."""
        ulaw = np.frombuffer(bytes_in, dtype=np.uint8)
        pcm16 = g711.ulaw.decode(ulaw)
        return pcm16.astype(np.int16)  # type: ignore[no-any-return]

    @staticmethod
    def pcm16_to_ulaw_bytes(pcm16: np.ndarray) -> bytes:
        ulaw = g711.ulaw.encode(pcm16.astype(np.int16))
        return bytes(ulaw.tolist())

    @staticmethod
    def pcm16_bytes_to_tensor_mono(bytes_in: bytes) -> torch.Tensor:
        """Convert PCM16 bytes (mono) to torch float32 tensor in [-1, 1] shape (T,)."""
        pcm = np.frombuffer(bytes_in, dtype=np.int16)
        return torch.from_numpy(pcm.astype(np.float32) / 32768.0)

    @staticmethod
    def tensor_to_pcm16_bytes_mono(wave: torch.Tensor) -> bytes:
        """Convert torch float tensor in [-1, 1] to PCM16 little-endian bytes (mono)."""
        clamped = torch.clamp(wave, -1.0, 1.0)
        pcm16 = (clamped * 32767.0).round().to(dtype=torch.int16).cpu().numpy()
        return pcm16.tobytes()

    @staticmethod
    def resample(wave: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
        """Resample mono waveform using torchaudio."""
        if src_sr == dst_sr:
            return wave
        resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr)
        # torchaudio expects shape (channels, time)
        return resampler(wave.view(1, -1)).view(-1)  # type: ignore[no-any-return]


# ---------------------------
# Seamless engine
# ---------------------------


@dataclass
class SeamlessConfig:
    model_id: str = "facebook/seamless-m4t-v2-large"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # ROCm is crash-prone with fp16 here; prefer float32 for stability.
    dtype: torch.dtype = torch.float32


class SeamlessEngine:
    """Thin wrapper around SeamlessM4T-V2 for S2ST."""

    def __init__(self, cfg: SeamlessConfig) -> None:
        self.cfg = cfg
        logger.info("Loading model %s on %s", cfg.model_id, cfg.device)
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)  # type: ignore[no-untyped-call]
        self.model = SeamlessM4Tv2Model.from_pretrained(cfg.model_id, torch_dtype=cfg.dtype)
        self.model.to(cfg.device)
        self.model.eval()

    @torch.inference_mode()
    def s2st(
        self,
        audio_16k: torch.Tensor,
        target_language: str,
    ) -> torch.Tensor:
        """
        Speech→Speech translation.

        Args:
            audio_16k: mono waveform tensor at 16 kHz in [-1, 1].
            target_language: language code supported by Seamless.

        Returns:
            torch.Tensor: mono 16 kHz waveform in [-1, 1].

        """
        # Prepare processor inputs; expects list of waveforms
        inputs = self.processor(audios=[audio_16k.cpu().numpy()], sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

        # Ask explicitly for speech; some TF versions default to text
        generated = self.model.generate(
            **inputs, tgt_lang=target_language, generate_speech=True, speech_use_cache=False
        )

        if isinstance(generated, torch.Tensor):
            # (batch, time)
            out = generated[0].to("cpu")
        elif isinstance(generated, dict):
            # Common keys for speech
            for k in ("waveform", "audio_values", "audio"):
                if k in generated:
                    t = generated[k]
                    if isinstance(t, torch.Tensor):
                        out = (t[0] if t.ndim > 1 else t).to("cpu")
                    else:
                        arr = t[0] if hasattr(t, "__getitem__") else t
                        out = torch.as_tensor(arr, dtype=torch.float32)
                    break
            else:
                msg = f"Unexpected generate() dict keys: {list(generated.keys())}"
                raise RuntimeError(msg)
        elif isinstance(generated, (list, tuple)):
            g0 = generated[0]
            out = torch.as_tensor(g0, dtype=torch.float32)
        else:
            out = torch.as_tensor(generated, dtype=torch.float32)

        return torch.clamp(out.to(torch.float32), -1.0, 1.0)


# ---------------------------
# Session handling
# ---------------------------


class SessionState(TypedDict):
    audio_format: AudioFormat
    strategy: ChunkingStrategy
    vad: webrtcvad.Vad | None
    vad_frame_ms: int
    vad_aggr: int
    buffer_pcm16_16k: list[torch.Tensor]
    utterance_seq: int


def make_vad(aggr: int) -> webrtcvad.Vad:
    """Create a configured WebRTC VAD instance."""
    vad = webrtcvad.Vad()
    vad.set_mode(aggr)
    return vad


# ---------------------------
# FastAPI app
# ---------------------------

app = FastAPI(title="Seamless S2ST WebSocket")
engine = SeamlessEngine(SeamlessConfig())


@app.get("/")
async def index() -> HTMLResponse:
    """Serve a minimal page to verify the server is up."""
    return HTMLResponse("<pre>Seamless S2ST WebSocket service is running.</pre>")


@app.websocket("/ws")
async def ws_handler(ws: WebSocket) -> None:  # noqa: C901, PLR0912, PLR0915
    """WebSocket endpoint implementing the agreed protocol."""
    await ws.accept()

    # 1) Expect setup
    try:
        raw = await ws.receive_text()
        setup = SetupMessage.model_validate_json(raw)
    except Exception as exc:
        await ws.send_text(ErrorMessage(type="error", code="bad_setup", message=str(exc)).model_dump_json())
        # Client may have already initiated close; suppress double-close errors
        with suppress(Exception):
            await ws.close()
        return

    # Validate chunking
    strategy = _parse_strategy(setup.chunking)
    if strategy is None:
        await ws.send_text(
            ErrorMessage(
                type="error",
                code="bad_setup",
                message="chunking.strategy must be 'server_vad' or 'client_chunked'",
            ).model_dump_json()
        )
        # Client may have already initiated close; suppress double-close errors
        with suppress(Exception):
            await ws.close()
        return

    # Prepare session state
    vad_aggr = int(setup.chunking.get("vad_aggressiveness", 1))
    state: SessionState = {
        "audio_format": setup.audio_format,
        "strategy": strategy,
        "vad": make_vad(vad_aggr) if strategy is ChunkingStrategy.SERVER_VAD else None,
        "vad_frame_ms": 20,
        "vad_aggr": vad_aggr,
        "buffer_pcm16_16k": [],
        "utterance_seq": 0,
    }

    # Send ready (negotiate)
    chunking: dict[str, Any]
    if strategy is ChunkingStrategy.SERVER_VAD:
        chunking = {
            "strategy": "server_vad",
            "vad": {"aggressiveness": state["vad_aggr"], "frame_ms": state["vad_frame_ms"]},
        }
    else:
        chunking = {"strategy": "client_chunked", "max_segment_ms": 60000}
    negotiated = {
        "audio_format": setup.audio_format.model_dump(),
        "model_audio": {"sample_rate": 16000},
        "chunking": chunking,
    }
    await ws.send_text(ReadyMessage(type="ready", session_id=id(ws).__str__(), negotiated=negotiated).model_dump_json())

    # 2) Stream loop
    try:
        while True:
            msg = await ws.receive()
            if "text" in msg and msg["text"] is not None:
                try:
                    incoming = IncomingMessage.model_validate_json(msg["text"]).root
                except Exception:
                    # Some clients may send control messages as plain JSON
                    data = orjson.loads(msg["text"].encode("utf-8"))
                    if data.get("type") == "end_of_segment":
                        await _maybe_infer_and_emit(ws, setup, state)
                        continue
                    if data.get("type") == "close":
                        break
                    await ws.send_text(
                        ErrorMessage(type="error", code="invalid_audio", message="invalid message").model_dump_json()
                    )
                    continue

                if isinstance(incoming, EndOfSegmentMessage):
                    await _maybe_infer_and_emit(ws, setup, state)
                elif isinstance(incoming, CloseMessage):
                    # Drop any buffered audio before closing
                    state["buffer_pcm16_16k"].clear()
                    break
                elif isinstance(incoming, SetupMessage):
                    await ws.send_text(
                        ErrorMessage(
                            type="error",
                            code="bad_setup",
                            message="setup already completed",
                        ).model_dump_json()
                    )
                elif isinstance(incoming, AudioMessage):
                    await _handle_audio_msg(incoming, setup, state)
                    if state["strategy"] is ChunkingStrategy.SERVER_VAD and _vad_segment_closed(state):
                        # In VAD mode, every 20ms frame may end an utterance
                        await _maybe_infer_and_emit(ws, setup, state)

            elif "bytes" in msg and msg["bytes"] is not None:
                # Not used: we only accept base64-in-JSON
                await ws.send_text(
                    ErrorMessage(
                        type="error", code="invalid_audio", message="binary frames not supported"
                    ).model_dump_json()
                )
            else:
                # Client closed
                break
    except WebSocketDisconnect:
        # Drop any leftovers if the client vanishes mid-utterance
        state["buffer_pcm16_16k"].clear()
    except Exception as exc:
        logger.exception("Server error")
        await ws.send_text(ErrorMessage(type="error", code="server_error", message=str(exc)).model_dump_json())
    finally:
        # Always clear per-connection buffers
        state["buffer_pcm16_16k"].clear()
        # Client may have already initiated close; suppress double-close errors
        with suppress(Exception):
            await ws.close()


# ---------------------------
# Handlers
# ---------------------------


def _parse_strategy(chunking: dict[str, Any]) -> ChunkingStrategy | None:
    """Extract chunking strategy from the setup message."""
    try:
        strategy = chunking.get("strategy")
        if strategy == ChunkingStrategy.SERVER_VAD.value:
            return ChunkingStrategy.SERVER_VAD
        if strategy == ChunkingStrategy.CLIENT_CHUNKED.value:
            return ChunkingStrategy.CLIENT_CHUNKED
    except Exception as e:
        print(f"Caught and ignored exception {e}")
    return None


async def _handle_audio_msg(msg: AudioMessage, setup: SetupMessage, state: SessionState) -> None:
    """Decode incoming frame, convert to 16 kHz PCM float, append to session buffer."""
    # Validate duration in server_vad
    if state["strategy"] is ChunkingStrategy.SERVER_VAD and msg.duration_ms != 20:
        error_msg = "server_vad requires duration_ms == 20 for every frame"
        raise ValueError(error_msg)

    fmt = setup.audio_format
    data = AudioUtils.b64_to_bytes(msg.audio_b64)

    # Decode to PCM16 mono tensor and resample to 16 kHz
    if fmt.codec is Codec.G711_ULAW:
        pcm16 = AudioUtils.ulaw_bytes_to_pcm16(data)
        tens = torch.from_numpy(pcm16.astype(np.float32) / 32768.0)
        wave_16k = AudioUtils.resample(tens, 8000, 16000)
    else:
        tens = AudioUtils.pcm16_bytes_to_tensor_mono(data)
        wave_16k = tens if fmt.sample_rate == 16000 else AudioUtils.resample(tens, fmt.sample_rate, 16000)

    state["buffer_pcm16_16k"].append(wave_16k)


def _vad_segment_closed(state: SessionState) -> bool:
    """Return ``True`` when the buffered audio ends with 400 ms of silence."""
    vad = state["vad"]
    if vad is None:
        return False
    frame_ms = state["vad_frame_ms"]
    buf = torch.cat(state["buffer_pcm16_16k"]) if state["buffer_pcm16_16k"] else None
    if buf is None or buf.numel() == 0:
        return False

    # Analyze up to the last 800 ms for trailing silence
    tail_ms = 800
    samples_tail = int(16000 * tail_ms / 1000)
    tail = buf[-samples_tail:] if buf.numel() >= samples_tail else buf

    # Slice into frames and require consecutive non-speech totaling 400+ ms
    frame_len = int(16000 * frame_ms / 1000)
    silent_run = 0
    needed_ms = 400
    for i in range(0, tail.numel() - frame_len + 1, frame_len):
        frame = tail[i : i + frame_len]
        pcm_bytes = AudioUtils.tensor_to_pcm16_bytes_mono(frame)
        is_speech = vad.is_speech(pcm_bytes, 16000)
        if is_speech:
            silent_run = 0
        else:
            silent_run += frame_ms
            if silent_run >= needed_ms:
                return True
    return False


async def _maybe_infer_and_emit(ws: WebSocket, setup: SetupMessage, state: SessionState) -> None:
    """
    Run S2ST on buffered audio and emit the translated waveform.

    The buffer is cleared and the VAD instance is re-created after each
    successful inference to avoid cross-utterance artifacts.
    """
    if not state["buffer_pcm16_16k"]:
        return

    utterance_id = f"utt-{state['utterance_seq']}"
    state["utterance_seq"] += 1

    # Concatenate buffered frames and clear
    wave_16k = torch.cat(state["buffer_pcm16_16k"])  # shape (T,)
    state["buffer_pcm16_16k"].clear()

    # Enforce max length for client_chunked
    if state["strategy"] is ChunkingStrategy.CLIENT_CHUNKED:
        max_samples = 16000 * 60  # 60000 ms at 16k
        if wave_16k.numel() > max_samples:
            await ws.send_text(
                ErrorMessage(
                    type="error",
                    code="over_limit",
                    message="segment exceeds 60000 ms",
                ).model_dump_json()
            )
            return

    # Inference
    start = asyncio.get_event_loop().time()
    try:
        logger.info("S2ST start: samples=%d strategy=%s", wave_16k.numel(), state["strategy"].value)
        if bool(int(os.environ.get("BYPASS_ENGINE", "0"))):
            # Loopback for debugging
            out_16k = torch.clamp(wave_16k, -1.0, 1.0).to(torch.float32)
        else:
            out_16k = engine.s2st(wave_16k.to(engine.cfg.device), setup.target_language)
    except Exception as e:
        logger.exception("Inference failed")
        await ws.send_text(ErrorMessage(type="error", code="server_error", message=str(e)).model_dump_json())
        return
    latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)
    logger.info("S2ST done: latency_ms=%d out_len=%d", latency_ms, int(out_16k.numel()))

    # Resample/encode to negotiated format
    fmt = setup.audio_format
    if fmt.codec is Codec.G711_ULAW:
        out_wave = out_16k if fmt.sample_rate == 16000 else AudioUtils.resample(out_16k, 16000, 8000)
        pcm16_bytes = AudioUtils.tensor_to_pcm16_bytes_mono(out_wave)
        pcm16 = np.frombuffer(pcm16_bytes, dtype=np.int16)
        wire_bytes = AudioUtils.pcm16_to_ulaw_bytes(pcm16)
        sr = 8000
    else:
        dst_sr = fmt.sample_rate
        out_wave = out_16k if dst_sr == 16000 else AudioUtils.resample(out_16k, 16000, dst_sr)
        wire_bytes = AudioUtils.tensor_to_pcm16_bytes_mono(out_wave)
        sr = dst_sr

    # Stream as one chunk (you can chunk if desired)
    chunk = AudioChunkMessage(
        type="audio_chunk",
        utterance_id=utterance_id,
        seq=0,
        audio_b64=AudioUtils.bytes_to_b64(wire_bytes),
        duration_ms=int(out_wave.numel() * 1000 / sr),
    )
    await ws.send_text(chunk.model_dump_json())

    await ws.send_text(
        EndOfAudioMessage(type="end_of_audio", utterance_id=utterance_id, latency_ms=latency_ms).model_dump_json()
    )

    # Reset VAD state so that previous utterances do not influence the
    # next segment.  WebRTC VAD keeps an internal state machine and may
    # mis-classify initial frames if not re-instantiated.
    if state["vad"] is not None:
        state["vad"] = make_vad(state["vad_aggr"])
