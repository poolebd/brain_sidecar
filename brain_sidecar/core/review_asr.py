from __future__ import annotations

import asyncio
import math
import re
import shutil
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from brain_sidecar.config import Settings
from brain_sidecar.core.asr import TranscribedSpan, TranscriptionResult
from brain_sidecar.core.gpu import cuda_out_of_memory, prepare_asr_gpu
from brain_sidecar.core.transcription import clean_transcript_text


ProgressCallback = Callable[[str, int], Awaitable[None]]


@dataclass(frozen=True)
class AudioChunk:
    path: Path
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class ReviewAsrBackend(Protocol):
    backend_name: str
    model_size: str | None
    last_error: str | None

    async def transcribe_file(
        self,
        path: Path,
        *,
        initial_prompt: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        ...

    async def unload(self) -> None:
        ...


class FasterWhisperFileReviewAsr:
    def __init__(self, transcriber, settings: Settings) -> None:
        self.transcriber = transcriber
        self.settings = settings

    @property
    def backend_name(self) -> str:
        return str(getattr(self.transcriber, "backend_name", "faster_whisper"))

    @property
    def model_size(self) -> str | None:
        return getattr(self.transcriber, "model_size", None)

    @property
    def last_error(self) -> str | None:
        return getattr(self.transcriber, "last_error", None)

    async def transcribe_file(
        self,
        path: Path,
        *,
        initial_prompt: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        if progress:
            await progress("Loading Faster-Whisper.", 20)
        await self.transcriber.load()
        if progress:
            await progress("Reading conditioned audio.", 38)
        pcm = await asyncio.to_thread(_read_pcm16_wav, path)
        if progress:
            await progress("Running Faster-Whisper batch ASR.", 58)
        return await self.transcriber.transcribe_pcm16(pcm, 0.0, initial_prompt=initial_prompt)

    async def unload(self) -> None:
        unload = getattr(self.transcriber, "unload", None)
        if unload is not None:
            result = unload()
            if asyncio.iscoroutine(result):
                await result


class NemoReviewAsr:
    backend_name = "nemo"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_size: str | None = None
        self.last_error: str | None = None
        self._model = None
        self._loaded_device: str | None = None
        self._load_lock = asyncio.Lock()

    async def transcribe_file(
        self,
        path: Path,
        *,
        initial_prompt: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        if progress:
            await progress("Loading high-accuracy NeMo ASR model.", 14)
        await self.load()
        chunk_dir = Path(tempfile.mkdtemp(prefix=f"{path.stem}-chunks-", dir=str(path.parent)))
        chunks: list[AudioChunk] = []
        try:
            chunks = await asyncio.to_thread(
                _chunk_pcm16_wav,
                path,
                chunk_dir,
                self.settings.review_asr_chunk_seconds,
                self.settings.review_asr_chunk_overlap_seconds,
            )
            if progress:
                await progress(f"Prepared {len(chunks)} ASR chunk{'s' if len(chunks) != 1 else ''}.", 22)
            spans = await self._transcribe_chunks(chunks, progress=progress)
            if progress:
                await progress(f"{len(spans)} ASR spans ready for review.", 92)
            return TranscriptionResult(
                model=self.model_size or self.settings.review_asr_model,
                language="en",
                spans=spans,
                audio_rms=None,
            )
        finally:
            await asyncio.to_thread(shutil.rmtree, chunk_dir, ignore_errors=True)

    async def _transcribe_chunks(
        self,
        chunks: list[AudioChunk],
        *,
        progress: ProgressCallback | None = None,
    ) -> list[TranscribedSpan]:
        if not chunks:
            return []
        spans: list[TranscribedSpan] = []
        batch_size = max(1, self.settings.review_asr_batch_size)
        batches = _batched(chunks, batch_size)
        total_batches = len(batches)
        total_chunks = len(chunks)
        completed_chunks = 0
        for batch_index, chunk_batch in enumerate(batches, start=1):
            first = completed_chunks + 1
            last = completed_chunks + len(chunk_batch)
            if progress:
                percent = 24 + round(62 * ((batch_index - 1) / max(1, total_batches)))
                await progress(f"Transcribing chunks {first}-{last}/{total_chunks}.", percent)
            try:
                batch_spans = await asyncio.to_thread(
                    self._transcribe_chunk_batch_sync,
                    chunk_batch,
                    len(chunk_batch),
                )
            except Exception as exc:
                if not cuda_out_of_memory(exc):
                    raise
                if len(chunk_batch) <= 1:
                    if self._loaded_device != "cuda":
                        raise
                    if progress:
                        await progress(
                            "GPU memory was tight; reloading NeMo ASR before retrying this chunk.",
                            24 + round(62 * (batch_index / max(1, total_batches))),
                        )
                    batch_spans = await self._retry_chunk_batch_after_cuda_oom(chunk_batch, len(chunk_batch))
                    spans.extend(batch_spans)
                    completed_chunks = last
                    continue
                if progress:
                    await progress(
                        "GPU memory was tight; retrying the current ASR batch one chunk at a time.",
                        24 + round(62 * (batch_index / max(1, total_batches))),
                    )
                batch_spans = []
                for chunk in chunk_batch:
                    try:
                        batch_spans.extend(
                            await asyncio.to_thread(self._transcribe_chunk_batch_sync, [chunk], 1)
                        )
                    except Exception as chunk_exc:
                        if self._loaded_device != "cuda" or not cuda_out_of_memory(chunk_exc):
                            raise
                        if progress:
                            await progress(
                                "GPU memory was tight; reloading NeMo ASR before retrying this chunk.",
                                24 + round(62 * (batch_index / max(1, total_batches))),
                            )
                        batch_spans.extend(await self._retry_chunk_batch_after_cuda_oom([chunk], 1))
            spans.extend(batch_spans)
            completed_chunks = last
        return _dedupe_overlapping_spans(spans, self.settings.review_asr_chunk_overlap_seconds)

    async def _retry_chunk_batch_after_cuda_oom(
        self,
        chunks: list[AudioChunk],
        batch_size: int,
    ) -> list[TranscribedSpan]:
        await self._reload_after_cuda_oom()
        return await asyncio.to_thread(self._transcribe_chunk_batch_sync, chunks, batch_size)

    def _transcribe_chunk_batch_sync(self, chunks: list[AudioChunk], batch_size: int) -> list[TranscribedSpan]:
        if self._model is None:
            raise RuntimeError("NeMo review ASR model is not loaded.")
        paths = [str(chunk.path) for chunk in chunks]
        try:
            raw = self._model.transcribe(
                paths,
                batch_size=max(1, batch_size),
                return_hypotheses=True,
                timestamps=True,
                num_workers=0,
                verbose=False,
            )
        except TypeError:
            raw = self._model.transcribe(
                paths,
                batch_size=max(1, batch_size),
                return_hypotheses=True,
                num_workers=0,
                verbose=False,
            )
        items = _transcription_items(raw, len(chunks))
        spans: list[TranscribedSpan] = []
        for chunk, item in zip(chunks, items):
            text = _hypothesis_text(item)
            chunk_spans = _spans_from_hypothesis(item, chunk.duration_s)
            if not chunk_spans:
                chunk_spans = _spans_from_plain_text(text, chunk.duration_s)
            spans.extend(_offset_spans(chunk_spans, chunk))
        return spans

    def _transcribe_sync(self, path: Path) -> TranscriptionResult:
        """Compatibility path for tests and direct callers that bypass async chunking."""
        spans = self._transcribe_chunk_batch_sync([AudioChunk(path, 0.0, _wav_duration_seconds(path))], 1)
        return TranscriptionResult(
            model=self.model_size or self.settings.review_asr_model,
            language="en",
            spans=spans,
            audio_rms=None,
        )

    async def load(self) -> None:
        async with self._load_lock:
            if self._model is not None:
                return
            await asyncio.to_thread(self._load_sync)

    async def unload(self) -> None:
        async with self._load_lock:
            if self._model is None:
                return
            await asyncio.to_thread(self._unload_sync)

    async def _reload_after_cuda_oom(self) -> None:
        async with self._load_lock:
            await asyncio.to_thread(self._unload_sync)
            await asyncio.to_thread(self._load_sync, True)

    def _load_sync(self, force_gpu_cleanup: bool = False) -> None:
        try:
            import torch
            from nemo.collections.asr.models import ASRModel
        except Exception as exc:
            self.last_error = str(exc)
            raise RuntimeError(f"NeMo ASR is unavailable: {exc}") from exc

        device = self._device(torch)
        model_name = self.settings.review_asr_model
        if device == "cuda":
            try:
                prepare_asr_gpu(self.settings, force_unload=force_gpu_cleanup)
            except RuntimeError as exc:
                self.last_error = str(exc)
                raise RuntimeError(
                    f"Could not prepare GPU for NeMo review ASR model {model_name}: {exc}"
                ) from exc

        try:
            self._load_model_sync(ASRModel, model_name, device)
        except Exception as exc:
            if device == "cuda" and not force_gpu_cleanup and cuda_out_of_memory(exc):
                self._unload_sync()
                try:
                    prepare_asr_gpu(self.settings, force_unload=True)
                    self._load_model_sync(ASRModel, model_name, device)
                    return
                except Exception as retry_exc:
                    self.last_error = str(retry_exc)
                    raise RuntimeError(
                        f"Could not load NeMo review ASR model {model_name} after GPU cleanup: {retry_exc}"
                    ) from retry_exc
            self.last_error = str(exc)
            raise RuntimeError(f"Could not load NeMo review ASR model {model_name}: {exc}") from exc

    def _load_model_sync(self, asr_model_class, model_name: str, device: str) -> None:
        self._model = asr_model_class.from_pretrained(model_name=model_name, map_location=device)
        if hasattr(self._model, "to"):
            self._model.to(device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        self.model_size = model_name
        self._loaded_device = device
        self.last_error = None

    def _unload_sync(self) -> None:
        self._model = None
        self._loaded_device = None
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    def _device(self, torch) -> str:
        configured = self.settings.review_asr_device
        if configured == "cuda":
            return "cuda"
        if configured == "cpu":
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"


def create_review_asr_backend(settings: Settings, live_transcriber) -> ReviewAsrBackend:
    backend = settings.review_asr_backend.strip().lower()
    if backend == "faster_whisper":
        return FasterWhisperFileReviewAsr(live_transcriber, settings)
    if backend == "nemo":
        return NemoReviewAsr(settings)
    raise ValueError(f"Unsupported BRAIN_SIDECAR_REVIEW_ASR_BACKEND={settings.review_asr_backend!r}.")


def _transcription_items(raw: Any, expected_count: int) -> list[Any]:
    value = raw
    if isinstance(value, tuple) and value:
        value = value[0]
    if isinstance(value, list):
        if len(value) == expected_count:
            return value
        if expected_count == 1 and value:
            return [value[0]]
        return value[:expected_count]
    if expected_count == 1:
        return [value]
    return [value]


def _hypothesis_text(item: Any) -> str:
    if isinstance(item, str):
        return clean_transcript_text(item)
    for name in ("text", "transcript", "pred_text"):
        value = getattr(item, name, None)
        if isinstance(value, str):
            return clean_transcript_text(value)
    if isinstance(item, dict):
        for name in ("text", "transcript", "pred_text"):
            value = item.get(name)
            if isinstance(value, str):
                return clean_transcript_text(value)
    return clean_transcript_text(str(item or ""))


def _spans_from_hypothesis(item: Any, duration: float) -> list[TranscribedSpan]:
    timestamp = getattr(item, "timestamp", None)
    if timestamp is None and isinstance(item, dict):
        timestamp = item.get("timestamp")
    if timestamp is None:
        timestamp = getattr(item, "timestep", None)
    if timestamp is None and isinstance(item, dict):
        timestamp = item.get("timestep")
    if not isinstance(timestamp, dict):
        return []

    for key in ("segment", "segments"):
        spans = _spans_from_timestamp_items(timestamp.get(key), duration)
        if spans:
            return spans

    word_items = timestamp.get("word") or timestamp.get("words")
    words = _word_items(word_items)
    if words:
        return _spans_from_words(words, duration)
    return []


def _spans_from_timestamp_items(items: Any, duration: float) -> list[TranscribedSpan]:
    if not isinstance(items, list):
        return []
    spans: list[TranscribedSpan] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = clean_transcript_text(item.get("segment") or item.get("text") or item.get("word") or "")
        if not text:
            continue
        start = _timestamp_float(item, "start_offset", "start", "start_s")
        end = _timestamp_float(item, "end_offset", "end", "end_s")
        if start is None:
            start = spans[-1].end_s if spans else 0.0
        if end is None or end <= start:
            end = min(duration, start + max(1.0, duration / max(1, len(items))))
        start = max(0.0, min(max(0.0, duration - 0.1), start))
        end = min(duration, max(start + 0.1, end))
        spans.append(TranscribedSpan(start_s=start, end_s=end, text=text))
    return spans


def _word_items(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and clean_transcript_text(item.get("word") or item.get("text") or "")]


def _spans_from_words(words: list[dict[str, Any]], duration: float) -> list[TranscribedSpan]:
    spans: list[TranscribedSpan] = []
    current: list[dict[str, Any]] = []
    for item in words:
        current.append(item)
        word = str(item.get("word") or item.get("text") or "")
        if len(current) >= 28 or word.endswith((".", "?", "!")):
            spans.append(_span_from_word_group(current, duration))
            current = []
    if current:
        spans.append(_span_from_word_group(current, duration))
    return [span for span in spans if span.text]


def _span_from_word_group(words: list[dict[str, Any]], duration: float) -> TranscribedSpan:
    text = clean_transcript_text(" ".join(str(item.get("word") or item.get("text") or "") for item in words))
    start = _timestamp_float(words[0], "start_offset", "start", "start_s") or 0.0
    end = _timestamp_float(words[-1], "end_offset", "end", "end_s") or min(duration, start + 4.0)
    start = max(0.0, min(max(0.0, duration - 0.1), start))
    end = min(duration, max(end, start + 0.1))
    return TranscribedSpan(start_s=start, end_s=end, text=text)


def _spans_from_plain_text(text: str, duration: float) -> list[TranscribedSpan]:
    clean = clean_transcript_text(text)
    if not clean:
        return []
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", clean) if sentence.strip()]
    if not sentences:
        sentences = [clean]
    units: list[str] = []
    for sentence in sentences:
        units.extend(_split_plain_text_unit(sentence, max_words=28))
    sentences = units or sentences
    weights = [max(1, len(sentence.split())) for sentence in sentences]
    total_weight = sum(weights)
    spans: list[TranscribedSpan] = []
    cursor = 0.0
    for index, (sentence, weight) in enumerate(zip(sentences, weights)):
        if index == len(sentences) - 1:
            end = duration
        else:
            end = cursor + (duration * (weight / total_weight))
        spans.append(TranscribedSpan(start_s=cursor, end_s=max(cursor + 0.1, end), text=sentence))
        cursor = end
    return spans


def _split_plain_text_unit(text: str, *, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    units: list[str] = []
    for index in range(0, len(words), max_words):
        units.append(" ".join(words[index:index + max_words]))
    return units


def _offset_spans(spans: list[TranscribedSpan], chunk: AudioChunk) -> list[TranscribedSpan]:
    result: list[TranscribedSpan] = []
    for span in spans:
        text = clean_transcript_text(span.text)
        if not text:
            continue
        start = chunk.start_s + max(0.0, span.start_s)
        end = chunk.start_s + max(span.start_s + 0.1, span.end_s)
        result.append(
            TranscribedSpan(
                start_s=max(chunk.start_s, start),
                end_s=min(chunk.end_s, max(start + 0.1, end)),
                text=text,
            )
        )
    return result


def _dedupe_overlapping_spans(spans: list[TranscribedSpan], overlap_seconds: float) -> list[TranscribedSpan]:
    result: list[TranscribedSpan] = []
    recent_window = max(2.0, overlap_seconds * 2.0 + 1.0)
    for span in sorted(spans, key=lambda item: (item.start_s, item.end_s, item.text)):
        text = clean_transcript_text(span.text)
        if not text:
            continue
        normalized = _normalize_for_dedupe(text)
        midpoint = (span.start_s + span.end_s) / 2.0
        duplicate = False
        for existing in reversed(result[-8:]):
            existing_midpoint = (existing.start_s + existing.end_s) / 2.0
            if midpoint - existing_midpoint > recent_window:
                break
            if normalized and normalized == _normalize_for_dedupe(existing.text):
                duplicate = True
                break
        if duplicate:
            continue
        result.append(TranscribedSpan(start_s=span.start_s, end_s=span.end_s, text=text))
    return result


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _batched(items: list[AudioChunk], size: int) -> list[list[AudioChunk]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _chunk_pcm16_wav(path: Path, chunk_dir: Path, chunk_seconds: float, overlap_seconds: float) -> list[AudioChunk]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frame_rate = wav.getframerate()
        total_frames = wav.getnframes()
        if channels != 1 or sample_width != 2:
            raise RuntimeError("Conditioned review audio must be mono PCM16 WAV.")
        if total_frames <= 0:
            return []
        chunk_frames = max(1, int(round(chunk_seconds * frame_rate)))
        overlap_frames = min(max(0, int(round(overlap_seconds * frame_rate))), max(0, chunk_frames - 1))
        step_frames = max(1, chunk_frames - overlap_frames)
        chunks: list[AudioChunk] = []
        start_frame = 0
        index = 0
        while start_frame < total_frames:
            end_frame = min(total_frames, start_frame + chunk_frames)
            wav.setpos(start_frame)
            frames = wav.readframes(end_frame - start_frame)
            chunk_path = chunk_dir / f"chunk_{index:04d}.wav"
            with wave.open(str(chunk_path), "wb") as chunk_wav:
                chunk_wav.setnchannels(channels)
                chunk_wav.setsampwidth(sample_width)
                chunk_wav.setframerate(frame_rate)
                chunk_wav.writeframes(frames)
            chunks.append(
                AudioChunk(
                    path=chunk_path,
                    start_s=start_frame / float(frame_rate),
                    end_s=end_frame / float(frame_rate),
                )
            )
            if end_frame >= total_frames:
                break
            start_frame += step_frames
            index += 1
    return chunks


def _timestamp_float(item: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = item.get(name)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            return number
    return None


def _read_pcm16_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise RuntimeError("Conditioned review audio must be mono PCM16 WAV.")
        return wav.readframes(wav.getnframes())


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())
