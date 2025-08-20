"""
Shared utilities and models for audio processing and protocol handling.

This module defines audio formats, helpers for encoding, decoding and
resampling audio data, and the WebSocket protocol message models used by
both the VAD and translation services.
"""

from __future__ import annotations

import base64
from enum import Enum
from typing import Literal, cast

import g711
import numpy as np
import torch
import torchaudio
from pydantic import BaseModel, Field, RootModel


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
        return np.asarray(g711.ulaw.decode(ulaw), dtype=np.int16)

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
        return cast("torch.Tensor", resampler(wave.view(1, -1))).view(-1)


# ---------------------------
# Protocol models
# ---------------------------


class SetupMessage(BaseModel):
    """Initial setup message sent by the client."""

    type: Literal["setup"] = "setup"
    target_language: str = Field(..., description="Target language code as supported by the service.")
    audio_format: AudioFormat


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


# ----------------

LANGUAGE_MAP = {
    # Nordics
    "da": "dan",
    "sv": "swe",
    "nb": "nob",
    "nn": "nno",
    "no": "nob",
    "is": "isl",
    "fi": "fin",
    # Big ones
    "en": "eng",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "pt": "por",
    "it": "ita",
    "nl": "nld",
    "pl": "pol",
    "cs": "ces",
    "sk": "slk",
    "sl": "slv",
    "hr": "hrv",
    "sr": "srp",
    "ro": "ron",
    "bg": "bul",
    "ru": "rus",
    "uk": "ukr",
    "tr": "tur",
    "el": "ell",
    "hu": "hun",
    # Balkans/Baltics
    "bs": "bos",
    "sq": "als",  # note: model uses 'bos'; 'als' (Tosk Albanian) may be unsupported for speech
    "lv": "lvs",
    "lt": "lit",
    "et": "est",
    # Asian
    "zh": "cmn",
    "zh-cn": "cmn",
    "zh-hans": "cmn",
    "zh-hant": "cmn_Hant",
    "zh-tw": "cmn_Hant",
    "yue": "yue",
    "ja": "jpn",
    "ko": "kor",
    "hi": "hin",
    "id": "ind",
    "ms": "zlm",
    "th": "tha",
    "vi": "vie",
    # Semitic
    "he": "heb",
    "ar": "arb",
    # Others seen in the model list
    "fa": "pes",
    "ur": "urd",
    "az": "azj",
    "kk": "kaz",
    "ky": "kir",
    "uz": "uzn",
}
