"""HTTP API for speech translation."""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoProcessor, SeamlessM4Tv2Model

from .common import AudioFormat, AudioUtils, Codec

logger = logging.getLogger("seamless.http")
logging.basicConfig(level=logging.INFO)


@dataclass
class SeamlessConfig:
    """Configuration for the Seamless translation engine."""

    model_id: str = "facebook/seamless-m4t-v2-large"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
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
    def s2st(self, audio_16k: torch.Tensor, target_language: str) -> torch.Tensor:
        """
        Speech-to-speech translation.

        Args:
            audio_16k: Mono waveform tensor at 16 kHz in ``[-1, 1]``.
            target_language: Target language code supported by Seamless.

        Returns:
            Translated waveform at 16 kHz in ``[-1, 1]``.

        """
        inputs = self.processor(audios=[audio_16k.cpu().numpy()], sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}
        generated = self.model.generate(
            **inputs,
            tgt_lang=target_language,
            generate_speech=True,
            speech_use_cache=False,
        )
        if isinstance(generated, torch.Tensor):
            out = generated[0].to("cpu")
        elif isinstance(generated, dict):
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
            out = torch.as_tensor(generated[0], dtype=torch.float32)
        else:
            out = torch.as_tensor(generated, dtype=torch.float32)
        return torch.clamp(out.to(torch.float32), -1.0, 1.0)


engine = SeamlessEngine(SeamlessConfig())
app = FastAPI(title="Seamless S2ST HTTP")
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
    target_language: str


class TranslateResponse(BaseModel):
    """Translated audio payload."""

    audio_b64: str
    duration_ms: int


@app.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    """Translate an audio segment using Seamless."""
    fmt = req.audio_format
    data = AudioUtils.b64_to_bytes(req.audio_b64)
    if fmt.codec is Codec.G711_ULAW:
        pcm16 = AudioUtils.ulaw_bytes_to_pcm16(data)
        tens = torch.from_numpy(pcm16.astype(np.float32) / 32768.0)
        wave_16k = AudioUtils.resample(tens, 8000, 16000)
    else:
        tens = AudioUtils.pcm16_bytes_to_tensor_mono(data)
        wave_16k = tens if fmt.sample_rate == 16000 else AudioUtils.resample(tens, fmt.sample_rate, 16000)

    start = asyncio.get_event_loop().time()
    if bool(int(os.environ.get("BYPASS_ENGINE", "0"))):
        out_16k = torch.clamp(wave_16k, -1.0, 1.0).to(torch.float32)
    else:
        out_16k = engine.s2st(wave_16k.to(engine.cfg.device), req.target_language)
    latency_ms = int((asyncio.get_event_loop().time() - start) * 1000)
    logger.info("S2ST done: latency_ms=%d out_len=%d", latency_ms, int(out_16k.numel()))

    if fmt.codec is Codec.G711_ULAW:
        out_wave = out_16k if fmt.sample_rate == 16000 else AudioUtils.resample(out_16k, 16000, 8000)
        pcm16_bytes = AudioUtils.tensor_to_pcm16_bytes_mono(out_wave)
        pcm16_arr = np.frombuffer(pcm16_bytes, dtype=np.int16)
        wire_bytes = AudioUtils.pcm16_to_ulaw_bytes(pcm16_arr)
        sr = 8000
    else:
        dst_sr = fmt.sample_rate
        out_wave = out_16k if dst_sr == 16000 else AudioUtils.resample(out_16k, 16000, dst_sr)
        wire_bytes = AudioUtils.tensor_to_pcm16_bytes_mono(out_wave)
        sr = dst_sr

    return TranslateResponse(
        audio_b64=AudioUtils.bytes_to_b64(wire_bytes),
        duration_ms=int(out_wave.numel() * 1000 / sr),
    )


@app.get("/")
async def index() -> dict[str, str]:
    """Return service status."""
    return {"status": "ok"}
