from __future__ import annotations

import asyncio
import hashlib
import re
import threading
from dataclasses import dataclass

from brain_sidecar.config import Settings
from brain_sidecar.core.gpu import cuda_out_of_memory, prepare_asr_gpu


@dataclass(frozen=True)
class TranscribedSpan:
    start_s: float
    end_s: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    model: str
    language: str | None
    spans: list[TranscribedSpan]


class FasterWhisperTranscriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_size: str | None = None
        self._model = None
        self._load_lock = threading.Lock()

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            prepare_asr_gpu(self.settings)
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:
                raise RuntimeError(f"faster-whisper import failed: {exc}") from exc

            errors: list[str] = []
            for attempt in range(2):
                attempt_errors = self._try_load_models(WhisperModel)
                if self._model is not None:
                    return
                errors.extend(attempt_errors)
                if attempt == 0 and any(cuda_out_of_memory(error) for error in attempt_errors):
                    try:
                        prepare_asr_gpu(self.settings, force_unload=True)
                    except RuntimeError as exc:
                        errors.append(f"GPU cleanup after CUDA OOM: {exc}")
                        break
                    continue
                break

            joined = "; ".join(errors)
            raise RuntimeError(
                "Could not load a CUDA Faster-Whisper model. "
                f"Tried primary and fallback, unloading GPU-resident Ollama models when needed: {joined}"
            )

    def _try_load_models(self, whisper_model_class) -> list[str]:
        errors: list[str] = []
        for model_size in [self.settings.asr_primary_model, self.settings.asr_fallback_model]:
            try:
                self._model = whisper_model_class(
                    model_size,
                    device="cuda",
                    compute_type=self.settings.asr_compute_type,
                )
                self.model_size = model_size
                return []
            except Exception as exc:
                errors.append(f"{model_size}: {exc}")
        return errors

    async def transcribe_pcm16(
        self,
        pcm: bytes,
        start_offset_s: float,
        *,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        if self._model is None:
            await self.load()
        return await asyncio.to_thread(self._transcribe_sync, pcm, start_offset_s, initial_prompt)

    def _transcribe_sync(
        self,
        pcm: bytes,
        start_offset_s: float,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        import numpy as np

        if self._model is None or self.model_size is None:
            raise RuntimeError("Transcriber model is not loaded.")

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        segments, info = self._model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": self.settings.asr_vad_min_silence_ms},
            condition_on_previous_text=self.settings.asr_condition_on_previous_text,
            beam_size=self.settings.asr_beam_size,
            temperature=0.0,
            initial_prompt=initial_prompt if initial_prompt is not None else self.settings.asr_initial_prompt,
        )
        spans = [
            TranscribedSpan(
                start_s=start_offset_s + float(segment.start),
                end_s=start_offset_s + float(segment.end),
                text=clean_transcript_text(segment.text),
            )
            for segment in segments
            if is_signal_text(segment.text, self.settings.min_segment_chars)
        ]
        return TranscriptionResult(
            model=self.model_size,
            language=getattr(info, "language", None),
            spans=spans,
        )


def transcript_fingerprint(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_transcript_text(text: str) -> str:
    return " ".join(text.split()).strip()


def is_signal_text(text: str, min_chars: int) -> bool:
    cleaned = clean_transcript_text(text)
    if len(cleaned) < min_chars:
        return False
    alnum_count = sum(char.isalnum() for char in cleaned)
    if alnum_count < min_chars:
        return False
    if re.fullmatch(r"[\W_]+", cleaned):
        return False
    return True
