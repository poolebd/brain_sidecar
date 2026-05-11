from __future__ import annotations

from brain_sidecar.config import Settings
from brain_sidecar.core.asr import ASR_BACKEND_FASTER_WHISPER
from brain_sidecar.core.transcription import FasterWhisperTranscriber


def create_asr_backend(settings: Settings):
    if settings.asr_backend == ASR_BACKEND_FASTER_WHISPER:
        return FasterWhisperTranscriber(settings)
    raise RuntimeError("Only the Faster-Whisper ASR backend is supported.")
