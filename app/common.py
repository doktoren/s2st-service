"""
Shared utilities and models for audio processing.

This module defines audio formats and helpers for encoding,
decoding and resampling audio data used by both the WebSocket and
HTTP translation services.
"""
from __future__ import annotations

import base64
from enum import Enum
from typing import Literal

import g711
import numpy as np
import torch
import torchaudio
from pydantic import BaseModel


class Codec(str, Enum):
    """Supported audio codecs for the client/server wire format."""

    G711_ULAW = "g711_ulaw"
    PCM16 = "pcm16"


class AudioFormat(BaseModel):
    """
    Audio format negotiated with the client.

    The same format is used for input and output.
    """

    codec: Codec
    sample_rate: Literal[8000, 16000, 24000]
    channels: Literal[1]


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
        """Decode μ-law bytes to a PCM16 ``numpy`` array (mono)."""
        ulaw = np.frombuffer(bytes_in, dtype=np.uint8)
        pcm16 = g711.ulaw.decode(ulaw)
        return pcm16.astype(np.int16)

    @staticmethod
    def pcm16_to_ulaw_bytes(pcm16: np.ndarray) -> bytes:
        """Encode a PCM16 array to μ-law bytes."""
        ulaw = g711.ulaw.encode(pcm16.astype(np.int16))
        return bytes(ulaw.tolist())

    @staticmethod
    def pcm16_bytes_to_tensor_mono(bytes_in: bytes) -> torch.Tensor:
        """Convert PCM16 bytes (mono) to a float tensor in ``[-1, 1]``."""
        pcm = np.frombuffer(bytes_in, dtype=np.int16)
        return torch.from_numpy(pcm.astype(np.float32) / 32768.0)

    @staticmethod
    def tensor_to_pcm16_bytes_mono(wave: torch.Tensor) -> bytes:
        """Convert a float tensor in ``[-1, 1]`` to PCM16 bytes (mono)."""
        clamped = torch.clamp(wave, -1.0, 1.0)
        pcm16 = (clamped * 32767.0).round().to(dtype=torch.int16).cpu().numpy()
        return bytes(pcm16.tobytes())

    @staticmethod
    def resample(wave: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
        """Resample a mono waveform using ``torchaudio``."""
        if src_sr == dst_sr:
            return wave
        resampler = torchaudio.transforms.Resample(orig_freq=src_sr, new_freq=dst_sr)
        return resampler(wave.view(1, -1)).view(-1)
