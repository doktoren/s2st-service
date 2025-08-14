"""HTTP API for speech translation."""

from __future__ import annotations

import asyncio
import logging
import os

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech

from .common import AudioFormat, AudioUtils, Codec

logger = logging.getLogger("seamless.http")
logging.basicConfig(level=logging.INFO)

logger.info(f"bf16 supported?: {torch.cuda.is_bf16_supported()}")


class SeamlessConfig:
    """Configuration for the Seamless translation engine."""

    model_id: str = "facebook/seamless-m4t-v2-large"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # ROCm uses the CUDA API
    dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class SeamlessEngine:
    """Thin wrapper around SeamlessM4T-V2 for S2ST."""

    def __init__(self, cfg: SeamlessConfig) -> None:
        self.cfg = cfg
        logger.info("Loading model %s on %s", cfg.model_id, cfg.device)
        self.processor = AutoProcessor.from_pretrained(cfg.model_id)  # type: ignore[no-untyped-call]
        self.model = (
            SeamlessM4Tv2ForSpeechToSpeech.from_pretrained(
                cfg.model_id,
                torch_dtype=torch.float16,  # cfg.dtype,
                attn_implementation="eager",  # sidestep SDPA dispatch entirely
            )
            .to(cfg.device)
            .eval()
        )
        logger.info("Compile after move+eval for ROCm speedups - BEGIN")
        try:
            self.model = torch.compile(self.model, mode="max-autotune")  # PyTorch ≥2.2
            logger.info("torch.compile enabled")
        except Exception as e:
            logger.info("torch.compile not enabled: %s", e)
        logger.info("Compile after move+eval for ROCm speedups - END")

    @torch.inference_mode()
    def s2st(self, audio_16k: torch.Tensor, target_language: str) -> torch.Tensor:
        """
        Speech-to-speech translation.

        audio_16k: Mono waveform tensor at 16 kHz in [-1, 1], CPU.
        """
        logger.info("Pin CPU tensor for faster host→device transfer")
        audio_np = audio_16k.cpu().numpy() if audio_16k.is_cuda else audio_16k.contiguous().pin_memory().numpy()

        logger.info("Processor runs on CPU")
        inputs = self.processor(audios=[audio_np], sampling_rate=16000, return_tensors="pt")

        logger.info("Move features to GPU")
        inputs = {k: v.to(self.cfg.device, non_blocking=True) for k, v in inputs.items()}

        logger.info("Inference in mixed precision")
        with torch.autocast(device_type="cuda", dtype=torch.float16), sdpa_kernel(SDPBackend.MATH):
            # With torch version 2.8.0 this call does a hard kill of the program
            generated = self.model.generate(
                **inputs,
                tgt_lang=target_language,
                text_num_beams=1,
                use_cache=False,
                return_dict_in_generate=False,
            )

        logger.info("Handle different output formats")
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

        return torch.clamp(out, -1.0, 1.0)


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

    loop = asyncio.get_event_loop()
    t0 = loop.time()
    if bool(int(os.environ.get("BYPASS_ENGINE", "0"))):
        out_16k = torch.clamp(wave_16k, -1.0, 1.0).to(torch.float32)
    else:
        out_16k = engine.s2st(wave_16k, req.target_language)

    latency_ms = int((loop.time() - t0) * 1000)
    logger.info("end2end_ms=%d out_samples=%d", latency_ms, int(out_16k.numel()))

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
