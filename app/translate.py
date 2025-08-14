"""HTTP API for speech translation."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from torch.nn.attention import sdpa_kernel
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToSpeech

from .common import AudioFormat, AudioUtils, Codec

# Keep MIOpen warnings down while allowing workspace for best perf.
os.environ.setdefault("MIOPEN_LOG_LEVEL", "1")  # 0=silent, 1=warn+err
os.environ.setdefault("MIOPEN_DEBUG_DISABLE_WORKSPACE", "0")  # keep workspace enabled
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
# Additional perf hints for this platform (RDNA3/ROCm).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MIOPEN_FIND_MODE", "1")
os.environ.setdefault("MIOPEN_FIND_ENFORCE", "2")

logger = logging.getLogger("seamless.http")
logging.basicConfig(level=logging.INFO)

# Prefer fast SDPA kernels when available (no-op if unsupported).
with contextlib.suppress(Exception):
    sdpa_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True)
    logger.info("Fast SDPA attention requested")

# Cap CPU threads used by preprocessing/tokenization a bit.
with contextlib.suppress(Exception):
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    torch.set_num_interop_threads(max(1, (os.cpu_count() or 4) // 4))

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
                torch_dtype=cfg.dtype,  # use bf16 on RDNA3+, else fp16
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
        # Warm up once to populate MIOpen/rocBLAS/attention caches.
        try:
            logger.info("Warmup generate() - BEGIN")
            sr = 16000
            warm = torch.zeros(sr // 2, dtype=torch.float32)
            _ = self.s2st(warm, target_language="eng")
            logger.info("Warmup generate() - END")
        except Exception as e:
            logger.info("Warmup skipped: %s", e)

    @torch.inference_mode()
    def s2st(self, audio_16k: torch.Tensor, target_language: str) -> torch.Tensor:  # noqa: C901, PLR0912
        """
        Speech-to-speech translation.
        audio_16k: Mono waveform tensor at 16 kHz in [-1, 1], CPU.
        """
        logger.info("Pin CPU tensor for faster host→device transfer")
        audio_tensor = audio_16k.contiguous()
        if not audio_tensor.is_cuda:
            with contextlib.suppress(Exception):
                audio_tensor = audio_tensor.pin_memory()
        audio_np = audio_tensor.cpu().numpy()

        logger.info("Processor runs on CPU")
        inputs = self.processor(audios=[audio_np], sampling_rate=16000, return_tensors="pt")

        logger.info("Move features to GPU")
        dev = self.cfg.device
        dt = self.cfg.dtype
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    inputs[k] = v.to(dev, dtype=dt, non_blocking=True)
                else:
                    inputs[k] = v.to(dev, non_blocking=True)

        logger.info("Inference in mixed precision")
        # Re-enable optimized attention paths (no forced MATH backend).
        # With torch 2.8.0 this call previously crashed; on 2.7.0 it is stable.
        with torch.autocast(device_type="cuda", dtype=dt):
            generated = self.model.generate(
                **inputs,
                tgt_lang=target_language,
                speaker_id=5,
                text_num_beams=1,
                speech_num_beams=1,
                use_cache=True,  # allow kv-cache for throughput
                return_dict_in_generate=False,
            )

        logger.info("Handle different output formats")
        if isinstance(generated, torch.Tensor):
            out = generated[0].to("cpu", dtype=torch.float32)
        elif isinstance(generated, dict):
            for k in ("waveform", "audio_values", "audio"):
                if k in generated:
                    t = generated[k]
                    if isinstance(t, torch.Tensor):
                        out = (t[0] if t.ndim > 1 else t).to("cpu", dtype=torch.float32)
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
