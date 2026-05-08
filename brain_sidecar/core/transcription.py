from __future__ import annotations

import asyncio
import hashlib
import re
import threading

import numpy as np

from brain_sidecar.config import Settings
from brain_sidecar.core.asr import ASR_BACKEND_FASTER_WHISPER, TranscribedSpan, TranscriptionResult
from brain_sidecar.core.gpu import cuda_out_of_memory, prepare_asr_gpu


class FasterWhisperTranscriber:
    backend_name = ASR_BACKEND_FASTER_WHISPER

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_size: str | None = None
        self.last_error: str | None = None
        self._model = None
        self._load_lock = threading.Lock()

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            if self._uses_cuda():
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
                if self._uses_cuda() and attempt == 0 and any(cuda_out_of_memory(error) for error in attempt_errors):
                    try:
                        prepare_asr_gpu(self.settings, force_unload=True)
                    except RuntimeError as exc:
                        errors.append(f"GPU cleanup after CUDA OOM: {exc}")
                        break
                    continue
                break

            joined = "; ".join(errors)
            self.last_error = joined
            raise RuntimeError(
                f"Could not load a {self._device()} Faster-Whisper model. "
                f"Tried primary and fallback models: {joined}"
            )

    def _try_load_models(self, whisper_model_class) -> list[str]:
        errors: list[str] = []
        for model_size in [self.settings.asr_primary_model, self.settings.asr_fallback_model]:
            try:
                self._model = whisper_model_class(
                    model_size,
                    device=self._device(),
                    compute_type=self._compute_type(),
                )
                self.model_size = model_size
                self.last_error = None
                return []
            except Exception as exc:
                self.last_error = str(exc)
                errors.append(f"{model_size}: {exc}")
        return errors

    def _device(self) -> str:
        return getattr(self.settings, "asr_device", "cuda")

    def _uses_cuda(self) -> bool:
        return self._device() == "cuda"

    def _compute_type(self) -> str:
        compute_type = self.settings.asr_compute_type
        if self._device() == "cpu" and compute_type in {"float16", "int8_float16"}:
            return "int8"
        return compute_type

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
        if self._model is None or self.model_size is None:
            raise RuntimeError("Transcriber model is not loaded.")

        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        rms = audio_rms(audio)
        if rms < self.settings.asr_min_audio_rms:
            return TranscriptionResult(model=self.model_size, language=None, spans=[], audio_rms=rms)

        segments, info = self._model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": self.settings.asr_vad_min_silence_ms},
            condition_on_previous_text=self.settings.asr_condition_on_previous_text,
            beam_size=self.settings.asr_beam_size,
            temperature=0.0,
            no_speech_threshold=self.settings.asr_no_speech_threshold,
            log_prob_threshold=self.settings.asr_log_prob_threshold,
            compression_ratio_threshold=self.settings.asr_compression_ratio_threshold,
            initial_prompt=initial_prompt if initial_prompt is not None else self.settings.asr_initial_prompt,
        )
        spans = [
            TranscribedSpan(
                start_s=start_offset_s + float(segment.start),
                end_s=start_offset_s + float(segment.end),
                text=clean_transcript_text(segment.text),
                avg_logprob=_segment_float(segment, "avg_logprob"),
                compression_ratio=_segment_float(segment, "compression_ratio"),
                no_speech_prob=_segment_float(segment, "no_speech_prob"),
            )
            for segment in segments
            if is_signal_segment(
                segment,
                min_chars=self.settings.min_segment_chars,
                no_speech_threshold=self.settings.asr_no_speech_threshold,
                log_prob_threshold=self.settings.asr_log_prob_threshold,
                compression_ratio_threshold=self.settings.asr_compression_ratio_threshold,
            )
        ]
        return TranscriptionResult(
            model=self.model_size,
            language=getattr(info, "language", None),
            spans=spans,
            audio_rms=rms,
        )


def transcript_fingerprint(text: str) -> str:
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_transcript_text(text: str) -> str:
    return " ".join(text.split()).strip()


def audio_rms(audio) -> float:
    if len(audio) == 0:
        return 0.0
    magnitudes = np.abs(np.asarray(audio, dtype=np.float32))
    if magnitudes.size >= 20:
        magnitudes = np.clip(magnitudes, 0.0, float(np.quantile(magnitudes, 0.95)))
    return float(np.sqrt(np.mean(magnitudes * magnitudes)))


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


def is_signal_segment(
    segment,
    *,
    min_chars: int,
    no_speech_threshold: float,
    log_prob_threshold: float,
    compression_ratio_threshold: float,
) -> bool:
    if not is_signal_text(getattr(segment, "text", ""), min_chars):
        return False
    no_speech_prob = _segment_float(segment, "no_speech_prob")
    if no_speech_prob is not None and no_speech_prob > no_speech_threshold:
        return False
    avg_logprob = _segment_float(segment, "avg_logprob")
    if avg_logprob is not None and avg_logprob < log_prob_threshold:
        return False
    compression_ratio = _segment_float(segment, "compression_ratio")
    if compression_ratio is not None and compression_ratio > compression_ratio_threshold:
        return False
    return True


def _segment_float(segment, name: str) -> float | None:
    value = getattr(segment, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
