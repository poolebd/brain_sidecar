from __future__ import annotations

import asyncio
import inspect
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from brain_sidecar.config import Settings
from brain_sidecar.core.asr import (
    ASR_BACKEND_NEMOTRON_STREAMING,
    NEMOTRON_ATT_CONTEXT_BY_CHUNK_MS,
    StreamingAsrEvent,
    StreamingPcmChunk,
    validate_nemotron_chunk_ms,
)
from brain_sidecar.core.gpu import prepare_asr_gpu
from brain_sidecar.core.transcription import clean_transcript_text


NEMOTRON_INSTALL_HELP = (
    "Nemotron streaming ASR requires NVIDIA NeMo ASR. Install the optional local stack with system packages "
    "`libsndfile1` and `ffmpeg`, then `pip install -e '.[nemotron]'` and "
    "`pip install 'git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[asr]'`."
)
DEFAULT_STREAMING_MIN_FINAL_WORDS = 10
DEFAULT_STREAMING_MIN_FINAL_SECONDS = 2.8
STREAMING_MIN_SHORT_FINAL_WORDS = 6
STREAMING_MIN_PARTIAL_WORDS = 2


@dataclass(frozen=True)
class StableTextChunk:
    text: str
    start_s: float
    end_s: float


class StreamingPcmChunker:
    """Reframe arbitrary PCM16 capture chunks into model-supported streaming chunks."""

    def __init__(self, *, sample_rate: int, chunk_ms: int) -> None:
        self.sample_rate = int(sample_rate)
        self.chunk_ms = validate_nemotron_chunk_ms(chunk_ms)
        self.bytes_per_second = self.sample_rate * 2
        self.chunk_bytes = int(self.bytes_per_second * (self.chunk_ms / 1000.0))
        self._buffer = bytearray()
        self._buffer_start_s: float | None = None

    def accept(self, pcm: bytes, start_offset_s: float) -> list[StreamingPcmChunk]:
        if not pcm:
            return []
        if self._buffer_start_s is None:
            self._buffer_start_s = start_offset_s
        self._buffer.extend(pcm)
        return self._drain(final=False)

    def flush(self) -> list[StreamingPcmChunk]:
        return self._drain(final=True)

    def _drain(self, *, final: bool) -> list[StreamingPcmChunk]:
        chunks: list[StreamingPcmChunk] = []
        while len(self._buffer) >= self.chunk_bytes or (final and self._buffer):
            size = self.chunk_bytes if len(self._buffer) >= self.chunk_bytes else len(self._buffer)
            size -= size % 2
            if size <= 0:
                self._buffer.clear()
                break
            start_s = self._buffer_start_s or 0.0
            end_s = start_s + (size / self.bytes_per_second)
            chunk = bytes(self._buffer[:size])
            del self._buffer[:size]
            chunks.append(StreamingPcmChunk(chunk, start_s, end_s, final=final and not self._buffer))
            self._buffer_start_s = end_s if self._buffer else None
            if final and size < self.chunk_bytes:
                break
        return chunks


class StablePrefixFinalizer:
    """Turn cumulative streaming hypotheses into non-duplicated final text."""

    def __init__(
        self,
        *,
        stable_chunks: int,
        partials_enabled: bool,
        min_final_words: int = DEFAULT_STREAMING_MIN_FINAL_WORDS,
        min_final_seconds: float = DEFAULT_STREAMING_MIN_FINAL_SECONDS,
    ) -> None:
        self.stable_chunks = max(1, int(stable_chunks))
        self.partials_enabled = partials_enabled
        self.min_final_words = max(4, int(min_final_words))
        self.min_final_seconds = max(1.0, float(min_final_seconds))
        self.min_short_final_words = min(self.min_final_words, STREAMING_MIN_SHORT_FINAL_WORDS)
        self._last_text = ""
        self._committed_words = 0
        self._stream_start_s: float | None = None
        self._committed_until_s: float | None = None
        self._pending_final_words: list[str] = []
        self._pending_final_start_s: float | None = None
        self._pending_final_end_s: float | None = None

    def accept_text(
        self,
        text: str,
        *,
        start_s: float,
        end_s: float,
        model: str,
    ) -> list[StreamingAsrEvent]:
        cleaned = clean_transcript_text(text)
        if not cleaned:
            return []
        if self._stream_start_s is None:
            self._stream_start_s = start_s
        if self._committed_until_s is None:
            self._committed_until_s = self._stream_start_s
        events: list[StreamingAsrEvent] = []
        words = _words(cleaned)
        stable_words = _common_prefix_words(self._last_text, cleaned)
        holdback_words = max(0, self.stable_chunks - 1)
        commit_to = min(len(stable_words), max(0, len(words) - holdback_words))
        if commit_to > self._committed_words:
            final_words = stable_words[self._committed_words : commit_to]
            if final_words:
                if self._pending_final_start_s is None:
                    self._pending_final_start_s = self._uncommitted_start_s(start_s)
                final_start_s = self._pending_final_start_s
                final_end_s = self._estimate_commit_end_s(
                    start_s=final_start_s,
                    end_s=end_s,
                    total_words=len(words),
                    commit_to=commit_to,
                )
                self._pending_final_words.extend(final_words)
                self._pending_final_end_s = final_end_s
                self._committed_words = commit_to
                final_event = self._maybe_emit_pending_final(model=model)
                if final_event is not None:
                    events.append(final_event)
        if self.partials_enabled and cleaned != self._last_text:
            partial_text = self._preview_text(cleaned)
            if self._should_emit_partial(partial_text):
                partial_start_s = self._pending_final_start_s
                if partial_start_s is None:
                    partial_start_s = self._uncommitted_start_s(start_s)
                events.append(
                    StreamingAsrEvent(
                        kind="partial",
                        text=partial_text,
                        start_s=partial_start_s,
                        end_s=max(partial_start_s, end_s),
                        model=model,
                    )
                )
        self._last_text = cleaned
        return [event for event in events if event.text.strip()]

    def flush(self, *, final_offset_s: float | None, model: str) -> list[StreamingAsrEvent]:
        words = _words(self._last_text)
        final_words = [*self._pending_final_words, *words[self._committed_words :]]
        final_text = " ".join(final_words).strip()
        if not final_text:
            return []
        self._committed_words = len(words)
        start_s = self._pending_final_start_s if self._pending_final_start_s is not None else self._uncommitted_start_s(0.0)
        end_s = final_offset_s if final_offset_s is not None else start_s
        self._committed_until_s = max(start_s, end_s)
        self._clear_pending_final()
        return [
            StreamingAsrEvent(
                kind="final",
                text=final_text,
                start_s=start_s,
                end_s=max(start_s, end_s),
                model=model,
            )
        ]

    def _preview_text(self, text: str) -> str:
        words = _words(text)
        if self._committed_words <= 0 and not self._pending_final_words:
            return text
        return " ".join([*self._pending_final_words, *words[self._committed_words :]]).strip()

    def _uncommitted_start_s(self, fallback: float) -> float:
        if self._committed_until_s is not None:
            return self._committed_until_s
        return self._stream_start_s if self._stream_start_s is not None else fallback

    def _estimate_commit_end_s(self, *, start_s: float, end_s: float, total_words: int, commit_to: int) -> float:
        uncommitted_words = max(1, total_words - self._committed_words)
        committed_delta = max(1, commit_to - self._committed_words)
        ratio = min(1.0, committed_delta / uncommitted_words)
        return max(start_s, start_s + ((end_s - start_s) * ratio))

    def _maybe_emit_pending_final(self, *, model: str) -> StreamingAsrEvent | None:
        if not self._pending_final_words or not self._pending_final_ready():
            return None
        text = " ".join(self._pending_final_words).strip()
        start_s = self._pending_final_start_s if self._pending_final_start_s is not None else self._uncommitted_start_s(0.0)
        end_s = self._pending_final_end_s if self._pending_final_end_s is not None else start_s
        self._committed_until_s = max(start_s, end_s)
        self._clear_pending_final()
        return StreamingAsrEvent(
            kind="final",
            text=text,
            start_s=start_s,
            end_s=max(start_s, end_s),
            model=model,
        )

    def _pending_final_ready(self) -> bool:
        word_count = len(self._pending_final_words)
        if word_count >= self.min_final_words:
            return True
        if word_count < self.min_short_final_words:
            return False
        text = " ".join(self._pending_final_words).strip()
        if text.endswith((".", "?", "!")):
            return True
        if self._pending_final_start_s is None or self._pending_final_end_s is None:
            return False
        return self._pending_final_end_s - self._pending_final_start_s >= self.min_final_seconds

    def _should_emit_partial(self, text: str) -> bool:
        words = _words(text)
        if len(words) >= STREAMING_MIN_PARTIAL_WORDS:
            return True
        return bool(text and text.endswith((".", "?", "!")))

    def _clear_pending_final(self) -> None:
        self._pending_final_words = []
        self._pending_final_start_s = None
        self._pending_final_end_s = None


class NemotronStreamingTranscriber:
    backend_name = ASR_BACKEND_NEMOTRON_STREAMING
    streaming_supported = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_size: str | None = None
        self.last_error: str | None = None
        self._model = None
        self._torch = None
        self._streaming_buffer_cls = None
        self._load_lock = threading.Lock()

    async def load(self) -> None:
        await asyncio.to_thread(self._load_sync)

    def _load_sync(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            prepare_asr_gpu(self.settings)
            if self.settings.nemotron_device != "cuda":
                raise RuntimeError("Nemotron streaming ASR is CUDA-only in Brain Sidecar.")
            try:
                import torch
                import nemo.collections.asr as nemo_asr
                from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
            except Exception as exc:
                self.last_error = f"Could not import NVIDIA NeMo ASR: {exc}. {NEMOTRON_INSTALL_HELP}"
                raise RuntimeError(self.last_error) from exc

            if not torch.cuda.is_available():
                self.last_error = "PyTorch CUDA is unavailable; Nemotron streaming ASR cannot run."
                raise RuntimeError(self.last_error)

            previous_token = os.environ.get("HF_TOKEN")
            if self.settings.nemotron_hf_token:
                os.environ["HF_TOKEN"] = self.settings.nemotron_hf_token
            previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
            if self.settings.nemotron_local_files_only:
                os.environ["HF_HUB_OFFLINE"] = "1"
            try:
                kwargs = {"model_name": self.settings.nemotron_model_id}
                from_pretrained = nemo_asr.models.ASRModel.from_pretrained
                if "map_location" in inspect.signature(from_pretrained).parameters:
                    kwargs["map_location"] = self.settings.nemotron_device
                model = from_pretrained(**kwargs)
                if hasattr(model.encoder, "set_default_att_context_size"):
                    model.encoder.set_default_att_context_size(
                        NEMOTRON_ATT_CONTEXT_BY_CHUNK_MS[self.settings.nemotron_chunk_ms]
                    )
                model = model.to(device=self.settings.nemotron_device, dtype=torch.float32)
                model.eval()
                self._model = model
                self._torch = torch
                self._streaming_buffer_cls = CacheAwareStreamingAudioBuffer
                self.model_size = self.settings.nemotron_model_id
                self.last_error = None
            except Exception as exc:
                self.last_error = f"Could not load Nemotron streaming ASR model: {exc}"
                raise RuntimeError(self.last_error) from exc
            finally:
                if self.settings.nemotron_hf_token:
                    if previous_token is None:
                        os.environ.pop("HF_TOKEN", None)
                    else:
                        os.environ["HF_TOKEN"] = previous_token
                if self.settings.nemotron_local_files_only:
                    if previous_hf_offline is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = previous_hf_offline

    async def open_stream(self, session_id: str, start_offset_s: float = 0.0) -> "NemotronStreamingSession":
        if self._model is None:
            await self.load()
        assert self._model is not None
        assert self._torch is not None
        return NemotronStreamingSession(
            model=self._model,
            torch_module=self._torch,
            streaming_buffer_cls=self._streaming_buffer_cls,
            settings=self.settings,
            model_id=self.model_size or self.settings.nemotron_model_id,
            start_offset_s=start_offset_s,
        )


class NemotronStreamingSession:
    def __init__(
        self,
        *,
        model: Any,
        torch_module: Any,
        streaming_buffer_cls: Any,
        settings: Settings,
        model_id: str,
        start_offset_s: float,
    ) -> None:
        self.model = model
        self.torch = torch_module
        self.streaming_buffer = streaming_buffer_cls(
            model=self.model,
            online_normalization=False,
            pad_and_drop_preencoded=False,
        )
        self.settings = settings
        self.model_id = model_id
        self.finalizer = StablePrefixFinalizer(
            stable_chunks=settings.streaming_stable_final_chunks,
            partials_enabled=settings.streaming_partials_enabled,
            min_final_words=settings.streaming_min_final_words,
            min_final_seconds=settings.streaming_min_final_seconds,
        )
        self.start_offset_s = start_offset_s
        self._step_num = 0
        self._previous_hypotheses = None
        self._previous_pred_out = None
        self._pred_out_stream = None
        self._closed = False
        self._raw_audio = np.zeros(0, dtype=np.float32)
        self._cache_last_channel, self._cache_last_time, self._cache_last_channel_len = (
            self.model.encoder.get_initial_cache_state(batch_size=1)
        )

    async def accept_pcm16(self, pcm: bytes, start_offset_s: float) -> list[StreamingAsrEvent]:
        if self._closed or not pcm:
            return []
        return await asyncio.to_thread(self._accept_sync, pcm, start_offset_s)

    def _accept_sync(self, pcm: bytes, start_offset_s: float) -> list[StreamingAsrEvent]:
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size == 0:
            return []
        started_at = time.monotonic()
        end_s = start_offset_s + (samples.size / self.settings.audio_sample_rate)
        events: list[StreamingAsrEvent] = []
        with self.torch.inference_mode():
            self._raw_audio = np.concatenate((self._raw_audio, samples))
            processed_signal, processed_signal_length = self.streaming_buffer.preprocess_audio(self._raw_audio)
            self.streaming_buffer.buffer = processed_signal
            self.streaming_buffer.streams_length = processed_signal_length.reshape(1).to(processed_signal.device)
            events.extend(self._drain_model_chunks(start_s=start_offset_s, end_s=end_s, final=False))
        elapsed_s = max(0.001, time.monotonic() - started_at)
        for event in events:
            object.__setattr__(event, "source", f"nemotron_rt:{elapsed_s:.3f}s")
        return events

    def _drain_model_chunks(self, *, start_s: float, end_s: float, final: bool) -> list[StreamingAsrEvent]:
        events: list[StreamingAsrEvent] = []
        while self._has_ready_streaming_chunk(final=final):
            try:
                chunk_audio, chunk_lengths = next(iter(self.streaming_buffer))
            except StopIteration:
                break
            chunk_audio = chunk_audio.to(self.torch.float32)
            (
                self._pred_out_stream,
                transcribed_texts,
                self._cache_last_channel,
                self._cache_last_time,
                self._cache_last_channel_len,
                self._previous_hypotheses,
            ) = self.model.conformer_stream_step(
                processed_signal=chunk_audio,
                processed_signal_length=chunk_lengths,
                cache_last_channel=self._cache_last_channel,
                cache_last_time=self._cache_last_time,
                cache_last_channel_len=self._cache_last_channel_len,
                keep_all_outputs=self.streaming_buffer.is_buffer_empty(),
                previous_hypotheses=self._previous_hypotheses,
                previous_pred_out=self._previous_pred_out,
                drop_extra_pre_encoded=self._drop_extra_pre_encoded(),
                return_transcription=True,
            )
            self._step_num += 1
            self._previous_pred_out = self._pred_out_stream
            text = _extract_text(transcribed_texts)
            events.extend(self.finalizer.accept_text(text, start_s=start_s, end_s=end_s, model=self.model_id))
        return events

    def _has_ready_streaming_chunk(self, *, final: bool) -> bool:
        buffer = getattr(self.streaming_buffer, "buffer", None)
        lengths = getattr(self.streaming_buffer, "streams_length", None)
        if buffer is None or lengths is None:
            return False
        stream_len = int(lengths[0].item())
        buffer_idx = int(self.streaming_buffer.buffer_idx)
        if final:
            return buffer_idx < stream_len
        return buffer_idx + self._next_streaming_chunk_size() <= stream_len

    def _next_streaming_chunk_size(self) -> int:
        chunk_size = self.streaming_buffer.streaming_cfg.chunk_size
        if isinstance(chunk_size, list):
            if self.streaming_buffer.buffer_idx == 0:
                return int(chunk_size[1] if self.streaming_buffer.pad_and_drop_preencoded else chunk_size[0])
            return int(chunk_size[1])
        return int(chunk_size)

    def _drop_extra_pre_encoded(self) -> int | None:
        if self._step_num == 0:
            return 0
        streaming_cfg = getattr(self.model.encoder, "streaming_cfg", None)
        return getattr(streaming_cfg, "drop_extra_pre_encoded", None)

    async def flush(self, final_offset_s: float | None = None) -> list[StreamingAsrEvent]:
        if self._closed:
            return []
        return await asyncio.to_thread(self._flush_sync, final_offset_s)

    def _flush_sync(self, final_offset_s: float | None) -> list[StreamingAsrEvent]:
        end_s = final_offset_s if final_offset_s is not None else self.start_offset_s
        events = self._drain_model_chunks(start_s=self.start_offset_s, end_s=end_s, final=True)
        events.extend(self.finalizer.flush(final_offset_s=final_offset_s, model=self.model_id))
        for event in events:
            object.__setattr__(event, "source", "nemotron_flush")
        return events

    async def close(self) -> None:
        self._closed = True
        self._raw_audio = np.zeros(0, dtype=np.float32)


def _extract_text(value: Any) -> str:
    if isinstance(value, str):
        return clean_transcript_text(value)
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        text = getattr(first, "text", first)
        return clean_transcript_text(str(text))
    text = getattr(value, "text", value)
    return clean_transcript_text(str(text or ""))


def _words(text: str) -> list[str]:
    return [word for word in clean_transcript_text(text).split(" ") if word]


def _common_prefix_words(left: str, right: str) -> list[str]:
    result: list[str] = []
    for left_word, right_word in zip(_words(left), _words(right), strict=False):
        if left_word != right_word:
            break
        result.append(right_word)
    return result
