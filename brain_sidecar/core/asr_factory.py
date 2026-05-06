from __future__ import annotations

from brain_sidecar.config import Settings
from brain_sidecar.core.asr import ASR_BACKEND_FASTER_WHISPER, ASR_BACKEND_NEMOTRON_STREAMING
from brain_sidecar.core.nemotron_streaming import NemotronStreamingTranscriber
from brain_sidecar.core.transcription import FasterWhisperTranscriber


def create_asr_backend(settings: Settings):
    if settings.asr_backend == ASR_BACKEND_NEMOTRON_STREAMING:
        return NemotronStreamingTranscriber(settings)
    if settings.asr_backend == ASR_BACKEND_FASTER_WHISPER:
        return FasterWhisperTranscriber(settings)
    raise RuntimeError(f"Unsupported ASR backend: {settings.asr_backend}")

