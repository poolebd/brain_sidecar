from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


ASR_BACKEND_FASTER_WHISPER = "faster_whisper"
ASR_BACKENDS = {ASR_BACKEND_FASTER_WHISPER}


@dataclass(frozen=True)
class TranscribedSpan:
    start_s: float
    end_s: float
    text: str
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


@dataclass(frozen=True)
class TranscriptionResult:
    model: str
    language: str | None
    spans: list[TranscribedSpan]
    audio_rms: float | None = None


@runtime_checkable
class BatchAsrBackend(Protocol):
    backend_name: str
    model_size: str | None
    last_error: str | None

    async def load(self) -> None:
        ...

    async def transcribe_pcm16(
        self,
        pcm: bytes,
        start_offset_s: float,
        *,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        ...

def validate_asr_backend(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in ASR_BACKENDS:
        raise ValueError("Unsupported BRAIN_SIDECAR_ASR_BACKEND={!r}; only faster_whisper is supported.".format(value))
    return normalized
