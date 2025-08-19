"""
Speech-to-speech translation using Azure Speech.

SPEECH_KEY=<secret> uv run python test/azure_test_s2st.py

Usage patterns:
- call `translate_once_s2st(...)` for one-shot mic → translated TTS to default speaker.
- integrate `stream_s2st(...)` in a backend to push PCM frames and yield translated text and/or TTS audio chunks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import azure.cognitiveservices.speech as speechsdk  # type: ignore[import-untyped]

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

# ----------------------------- Data models -----------------------------


@dataclass(frozen=True)
class SpeechAuth:
    """Authentication for Azure Speech."""

    key: str
    region: str


@dataclass(frozen=True)
class S2STConfig:
    """Configuration for speech-to-speech translation."""

    src_lang: str  # e.g. "da-DK"
    tgt_lang: str  # e.g. "en"
    tgt_voice: str  # e.g. "en-US-JennyNeural"
    sample_rate_hz: int = 16_000
    bits_per_sample: Literal[16] = 16
    channels: Literal[1] = 1


# ----------------------------- Utilities -----------------------------


def _env(name: str, default: str | None = None) -> str:
    """Return required environment variable or raise ValueError."""
    val = os.getenv(name)
    if not val:
        if default is None:
            msg = f"Missing required environment variable: {name}"
            raise ValueError(msg)
        return default
    return val


def _make_translation_config(auth: SpeechAuth, cfg: S2STConfig) -> speechsdk.translation.SpeechTranslationConfig:
    """Build and return a SpeechTranslationConfig."""
    tr = speechsdk.translation.SpeechTranslationConfig(
        subscription=auth.key,
        region=auth.region,
    )
    tr.speech_recognition_language = cfg.src_lang
    tr.add_target_language(cfg.tgt_lang)
    tr.voice_name = cfg.tgt_voice
    return tr


def _default_stream_format(cfg: S2STConfig) -> speechsdk.audio.AudioStreamFormat:
    """Return AudioStreamFormat for PCM mono."""
    return speechsdk.audio.AudioStreamFormat(
        samples_per_second=cfg.sample_rate_hz,
        bits_per_sample=cfg.bits_per_sample,
        channels=cfg.channels,
    )


# ----------------------------- One-shot S2ST (simple) -----------------------------


def translate_once_s2st(
    auth: SpeechAuth,
    cfg: S2STConfig,
    *,
    use_default_microphone: bool = True,
    use_default_speaker: bool = True,
) -> str | None:
    """
    Perform one-shot S2ST:
    - Capture a single utterance from mic (or raise if mic not available).
    - Translate to target language.
    - Synthesize target speech to default speaker.
    Returns the translated text, or None if no match.
    """
    tr_cfg = _make_translation_config(auth, cfg)
    audio_in = speechsdk.audio.AudioConfig(use_default_microphone=use_default_microphone)
    auto = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=[cfg.src_lang])
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=tr_cfg,
        auto_detect_source_language_config=auto,
        audio_config=audio_in,
    )

    result = recognizer.recognize_once_async().get()
    assert result is not None
    if result.reason == speechsdk.ResultReason.TranslatedSpeech:
        translated_text: str | None = result.translations.get(cfg.tgt_lang)
        if translated_text:
            tts_cfg = speechsdk.SpeechConfig(subscription=auth.key, region=auth.region)
            tts_cfg.speech_synthesis_voice_name = cfg.tgt_voice
            out_cfg = speechsdk.audio.AudioOutputConfig(use_default_speaker=use_default_speaker)
            tts = speechsdk.SpeechSynthesizer(speech_config=tts_cfg, audio_config=out_cfg)
            _ = tts.speak_text_async(translated_text).get()
            return translated_text
        return None
    if result.reason == speechsdk.ResultReason.NoMatch:
        return None
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        msg = f"S2ST canceled: {details.reason}; {details.error_details}"
        raise RuntimeError(msg)
    return None


# ----------------------------- Streaming S2ST (backend-friendly) -----------------------------


@dataclass(frozen=True)
class StreamEvent:
    """
    Streaming event from the recognizer.

    kind:
      - "partial": partial transcript/translation (low latency, not final)
      - "final": final translation for a segment
      - "audio": synthesized audio chunk (bytes, PCM or SDK-managed)
    """

    kind: Literal["partial", "final", "audio"]
    text: str | None = None
    audio: bytes | None = None


def stream_s2st(  # noqa: C901
    auth: SpeechAuth,
    cfg: S2STConfig,
    pcm_frames: Iterable[bytes],
    *,
    on_ready: Callable[[], None] | None = None,
) -> Iterator[StreamEvent]:
    """
    Stream PCM16 mono frames (16000 Hz default) → yield translation and synthesized audio.

    - Provide ~100-200 ms per frame for good latency.
    - This yields:
        StreamEvent(kind="partial", text=...)
        StreamEvent(kind="final", text=...)
        StreamEvent(kind="audio", audio=<bytes>)
    - Caller is responsible for playback of audio bytes if desired.
    """
    tr_cfg = _make_translation_config(auth, cfg)
    push_stream = speechsdk.audio.PushAudioInputStream(stream_format=_default_stream_format(cfg))
    audio_in = speechsdk.audio.AudioConfig(stream=push_stream)
    recognizer = speechsdk.translation.TranslationRecognizer(
        translation_config=tr_cfg,
        audio_config=audio_in,
    )

    # Wire up events
    partial_buffer: list[StreamEvent] = []

    def _on_recognizing(_s: object, e: object) -> None:
        text = getattr(e.result, "translations", {}).get(cfg.tgt_lang)  # type: ignore[attr-defined]
        if text:
            partial_buffer.append(StreamEvent(kind="partial", text=text))

    def _on_recognized(_s: object, e: object) -> None:
        text = getattr(e.result, "translations", {}).get(cfg.tgt_lang)  # type: ignore[attr-defined]
        if text:
            partial_buffer.append(StreamEvent(kind="final", text=text))

    def _on_synth(_s: object, e: object) -> None:
        if getattr(e.result, "reason", None) == speechsdk.ResultReason.SynthesizingAudio:  # type: ignore[attr-defined]
            chunk = getattr(e.result, "audio", None)  # type: ignore[attr-defined]
            if isinstance(chunk, (bytes, bytearray)):
                partial_buffer.append(StreamEvent(kind="audio", audio=bytes(chunk)))

    recognizer.recognizing.connect(_on_recognizing)
    recognizer.recognized.connect(_on_recognized)
    recognizer.synthesizing.connect(_on_synth)

    # Start recognition
    recognizer.start_continuous_recognition_async().get()
    if on_ready:
        on_ready()

    try:
        for frame in pcm_frames:
            # Push caller-provided PCM frames
            push_stream.write(frame)
            # Yield any accumulated events without blocking the producer
            while partial_buffer:
                yield partial_buffer.pop(0)
        # Signal end of stream
        push_stream.close()
    finally:
        recognizer.stop_continuous_recognition_async().get()


# ----------------------------- Convenience factory -----------------------------


def from_env(
    *,
    src_lang: str,
    tgt_lang: str,
    tgt_voice: str,
    sample_rate_hz: int = 16_000,
) -> tuple[SpeechAuth, S2STConfig]:
    """
    Build (auth, config) from environment:
      SPEECH_KEY
      SPEECH_REGION
    """
    auth = SpeechAuth(key=_env("SPEECH_KEY"), region=_env("SPEECH_REGION", default="northeurope"))
    cfg = S2STConfig(
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        tgt_voice=tgt_voice,
        sample_rate_hz=sample_rate_hz,
    )
    return auth, cfg


# ----------------------------- Example main (manual test) -----------------------------

if __name__ == "__main__":
    # One-shot mic → TTS example (manual sanity check; do not run in CI)
    AUTH, CFG = from_env(src_lang="da-DK", tgt_lang="en", tgt_voice="en-US-JennyNeural")
    print("Speak a short sentence...")
    text = translate_once_s2st(AUTH, CFG)
    print(f"Translated: {text!r}")
