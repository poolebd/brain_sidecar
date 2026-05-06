from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable


ASR_BACKEND_FASTER_WHISPER = "faster_whisper"
ASR_BACKEND_NEMOTRON_STREAMING = "nemotron_streaming"
ASR_BACKENDS = {ASR_BACKEND_FASTER_WHISPER, ASR_BACKEND_NEMOTRON_STREAMING}
NEMOTRON_CHUNK_MS_VALUES = {80, 160, 560, 1120}
NEMOTRON_ATT_CONTEXT_BY_CHUNK_MS = {
    80: [70, 0],
    160: [70, 1],
    560: [70, 6],
    1120: [70, 13],
}
NEMOTRON_DTYPES = {"float32"}


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


@dataclass(frozen=True)
class StreamingAsrEvent:
    kind: Literal["partial", "final"]
    text: str
    start_s: float
    end_s: float
    model: str
    confidence: float | None = None
    tokens: list[str] = field(default_factory=list)
    words: list[dict[str, object]] = field(default_factory=list)
    source: str = "streaming_asr"


@dataclass(frozen=True)
class StreamingPcmChunk:
    pcm: bytes
    start_offset_s: float
    end_offset_s: float
    final: bool = False


@runtime_checkable
class BatchAsrBackend(Protocol):
    backend_name: str
    streaming_supported: bool
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


@runtime_checkable
class StreamingAsrBackend(Protocol):
    backend_name: str
    streaming_supported: bool
    model_size: str | None
    last_error: str | None

    async def load(self) -> None:
        ...

    async def open_stream(
        self,
        session_id: str,
        start_offset_s: float = 0.0,
    ) -> "StreamingAsrSession":
        ...


@runtime_checkable
class StreamingAsrSession(Protocol):
    async def accept_pcm16(self, pcm: bytes, start_offset_s: float) -> list[StreamingAsrEvent]:
        ...

    async def flush(self, final_offset_s: float | None = None) -> list[StreamingAsrEvent]:
        ...

    async def close(self) -> None:
        ...


def validate_asr_backend(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in ASR_BACKENDS:
        allowed = ", ".join(sorted(ASR_BACKENDS))
        raise ValueError(f"Unsupported BRAIN_SIDECAR_ASR_BACKEND={value!r}; expected one of: {allowed}.")
    return normalized


def validate_nemotron_chunk_ms(value: int) -> int:
    chunk_ms = int(value)
    if chunk_ms not in NEMOTRON_CHUNK_MS_VALUES:
        allowed = ", ".join(str(item) for item in sorted(NEMOTRON_CHUNK_MS_VALUES))
        raise ValueError(f"Unsupported BRAIN_SIDECAR_NEMOTRON_CHUNK_MS={value!r}; expected one of: {allowed}.")
    return chunk_ms


def validate_nemotron_dtype(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in NEMOTRON_DTYPES:
        allowed = ", ".join(sorted(NEMOTRON_DTYPES))
        raise ValueError(f"Unsupported BRAIN_SIDECAR_NEMOTRON_DTYPE={value!r}; expected one of: {allowed}.")
    return normalized


def validate_nemotron_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized != "cuda":
        raise ValueError("BRAIN_SIDECAR_NEMOTRON_DEVICE must be 'cuda'; CPU streaming ASR is not supported.")
    return normalized
