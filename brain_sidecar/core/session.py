from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

from brain_sidecar.config import Settings
from brain_sidecar.core.audio import AudioCapture, BrowserAudioCapture, FFmpegAudioCapture, FixtureWavAudioCapture
from brain_sidecar.core.dedupe import TranscriptDeduplicator
from brain_sidecar.core.devices import find_device
from brain_sidecar.core.event_bus import EventBus
from brain_sidecar.core.events import (
    EVENT_AUDIO_STATUS,
    EVENT_ERROR,
    EVENT_GPU_STATUS,
    EVENT_NOTE_UPDATE,
    EVENT_RECALL_HIT,
    EVENT_SPEAKER_PROFILE_UPDATE,
    EVENT_TRANSCRIPT_FINAL,
    SidecarEvent,
)
from brain_sidecar.core.gpu import GpuStatus, prepare_asr_gpu, read_gpu_status
from brain_sidecar.core.models import TranscriptSegment, new_id
from brain_sidecar.core.notes import NoteSynthesizer
from brain_sidecar.core.ollama import OllamaClient
from brain_sidecar.core.recall import RecallIndex
from brain_sidecar.core.speaker_identity import SpeakerIdentityService
from brain_sidecar.core.storage import Storage
from brain_sidecar.core.transcription import FasterWhisperTranscriber
from brain_sidecar.core.web_context import (
    BraveSearchClient,
    WebContextCandidate,
    WebContextSynthesizer,
    WebTriggerDetector,
)
from brain_sidecar.core.work_memory import WorkMemoryService


@dataclass(frozen=True)
class AudioWindow:
    pcm: bytes
    start_offset_s: float
    final: bool = False


@dataclass(frozen=True)
class SegmentBatch:
    segments: list[TranscriptSegment]
    refresh_notes: bool = False


@dataclass
class ActiveSession:
    id: str
    capture: AudioCapture
    window_queue: asyncio.Queue[AudioWindow]
    postprocess_queue: asyncio.Queue[SegmentBatch]
    tasks: list[asyncio.Task]
    deduper: TranscriptDeduplicator
    web_context_queue: asyncio.Queue[WebContextCandidate] | None = None
    audio_source: str = "server_device"
    save_transcript: bool = True
    recent_segments: list[TranscriptSegment] = field(default_factory=list)
    final_segment_count: int = 0
    next_note_segment_count: int = 3
    dropped_windows: int = 0
    web_context_pending_queries: set[str] = field(default_factory=set)
    web_context_seen_queries: set[str] = field(default_factory=set)
    web_context_last_at: float = 0.0


class SessionManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bus = EventBus()
        self.storage = Storage(settings.data_dir)
        self.storage.connect()
        self.ollama = OllamaClient(settings)
        self.recall = RecallIndex(self.storage, self.ollama)
        self.work_memory = WorkMemoryService(self.storage, self.recall)
        self.notes = NoteSynthesizer(self.ollama)
        self.speaker_identity = SpeakerIdentityService(self.storage, settings)
        self.transcriber = FasterWhisperTranscriber(settings)
        self.web_trigger_detector = WebTriggerDetector()
        self.web_search = BraveSearchClient(
            settings.brave_search_api_key,
            timeout_s=settings.web_context_timeout_seconds,
            max_results=settings.web_context_max_results,
        )
        self.web_context = WebContextSynthesizer()
        self._active: dict[str, ActiveSession] = {}
        self._speaker_captures: dict[str, BrowserAudioCapture] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, title: str | None = None) -> dict:
        record = await asyncio.to_thread(
            self.storage.create_session,
            title or time.strftime("Sidecar session %Y-%m-%d %H:%M"),
        )
        return {
            "id": record.id,
            "title": record.title,
            "status": record.status,
            "started_at": record.started_at,
            "ended_at": record.ended_at,
        }

    async def start_session(
        self,
        session_id: str,
        *,
        device_id: str | None = None,
        fixture_wav: Path | None = None,
        audio_source: str = "server_device",
        save_transcript: bool = True,
    ) -> None:
        async with self._lock:
            if session_id in self._active:
                return
            if self.transcriber.model_size is None:
                await self._publish_asr_start_status(session_id)
                await asyncio.to_thread(prepare_asr_gpu, self.settings)
                await self._publish_audio_status(session_id, "loading_asr")
            await self.transcriber.load()
            capture = self._build_capture(device_id=device_id, fixture_wav=fixture_wav, audio_source=audio_source)
            window_queue: asyncio.Queue[AudioWindow] = asyncio.Queue(maxsize=self.settings.transcription_queue_size)
            postprocess_queue: asyncio.Queue[SegmentBatch] = asyncio.Queue(maxsize=self.settings.postprocess_queue_size)
            web_context_queue: asyncio.Queue[WebContextCandidate] = asyncio.Queue(maxsize=4)
            active = ActiveSession(
                id=session_id,
                capture=capture,
                window_queue=window_queue,
                postprocess_queue=postprocess_queue,
                tasks=[],
                deduper=TranscriptDeduplicator(
                    max_recent=self.settings.dedupe_recent_segments,
                    similarity_threshold=self.settings.dedupe_similarity_threshold,
                ),
                web_context_queue=web_context_queue,
                audio_source=audio_source,
                save_transcript=save_transcript,
                next_note_segment_count=self.settings.notes_every_segments,
            )
            self._active[session_id] = active
            active.tasks = [
                asyncio.create_task(self._run_capture_loop(session_id, capture, window_queue)),
                asyncio.create_task(self._run_transcription_loop(session_id, window_queue, postprocess_queue)),
                asyncio.create_task(self._run_postprocess_loop(session_id, postprocess_queue)),
            ]
            if self._web_context_configured():
                active.tasks.append(asyncio.create_task(self._run_web_context_loop(session_id, web_context_queue)))
            for task in active.tasks:
                task.add_done_callback(lambda done_task, sid=session_id: self._surface_task_exception(sid, done_task))
            self.storage.set_session_status(session_id, "running")

        await self.bus.publish(
            SidecarEvent(
                type=EVENT_GPU_STATUS,
                session_id=session_id,
                payload={
                    **read_gpu_status().to_dict(),
                    "asr_model": self.transcriber.model_size,
                    "asr_compute_type": self.settings.asr_compute_type,
                    "asr_beam_size": self.settings.asr_beam_size,
                    "queue_size": self.settings.transcription_queue_size,
                    "speaker_identity": self.speaker_identity.status(),
                    "save_transcript": active.save_transcript,
                    "raw_audio_retained": False,
                },
            )
        )

    async def _publish_asr_start_status(self, session_id: str) -> None:
        status = read_gpu_status()
        next_status = "freeing_gpu" if self._should_free_gpu_before_asr(status) else "loading_asr"
        await self._publish_audio_status(session_id, next_status, status)

    async def _publish_audio_status(
        self,
        session_id: str,
        status: str,
        gpu_status: GpuStatus | None = None,
    ) -> None:
        gpu_status = gpu_status or read_gpu_status()
        await self.bus.publish(
            SidecarEvent(
                type=EVENT_AUDIO_STATUS,
                session_id=session_id,
                payload={
                    "status": status,
                    "memory_free_mb": gpu_status.memory_free_mb,
                    "memory_total_mb": gpu_status.memory_total_mb,
                    "gpu_pressure": gpu_status.gpu_pressure,
                    "ollama_gpu_models": gpu_status.ollama_gpu_models,
                    "asr_min_free_vram_mb": self.settings.asr_min_free_vram_mb,
                },
            )
        )

    def _should_free_gpu_before_asr(self, status: GpuStatus) -> bool:
        if self.transcriber.model_size is not None:
            return False
        if not self.settings.asr_unload_ollama_on_start:
            return False
        if status.memory_free_mb is None:
            return False
        if status.memory_free_mb >= self.settings.asr_min_free_vram_mb:
            return False
        return bool(status.ollama_gpu_models)

    async def stop_session(self, session_id: str) -> None:
        async with self._lock:
            active = self._active.pop(session_id, None)
        if active is None:
            return
        await active.capture.stop()
        for task in active.tasks:
            task.cancel()
        await asyncio.gather(*active.tasks, return_exceptions=True)
        self.storage.set_session_status(session_id, "stopped", ended_at=time.time())
        await self.bus.publish(
            SidecarEvent(
                type=EVENT_AUDIO_STATUS,
                session_id=session_id,
                payload={
                    "status": "stopped",
                    "raw_audio_retained": False,
                    "save_transcript": active.save_transcript,
                    "dropped_windows": active.dropped_windows,
                },
            )
        )

    async def add_library_root(self, path: Path) -> dict:
        root_id = await asyncio.to_thread(self.storage.add_library_root, path.expanduser().resolve())
        return {"id": root_id, "path": str(path.expanduser().resolve())}

    async def search_web_context(self, query: str, *, session_id: str | None = None) -> dict:
        cleaned_query = self.web_trigger_detector.build_query(query)
        enabled = self.settings.web_context_enabled
        configured = bool(self.settings.brave_search_api_key.strip())
        if not enabled or not configured or not cleaned_query:
            return {
                "query": cleaned_query,
                "enabled": enabled,
                "configured": configured,
                "note": None,
            }

        candidate = WebContextCandidate(
            query=cleaned_query,
            normalized_query=self.web_trigger_detector.normalize_query(cleaned_query),
            freshness=self.web_trigger_detector.freshness_for(query),
            source_segment_ids=[],
        )
        try:
            results = await self.web_search.search(candidate.query, freshness=candidate.freshness)
        except Exception:
            results = []
        note = self.web_context.synthesize(session_id or "", candidate, results)
        if note is not None:
            note["session_id"] = session_id
        return {
            "query": cleaned_query,
            "enabled": enabled,
            "configured": configured,
            "note": note,
        }

    def browser_audio_capture(self, session_id: str) -> BrowserAudioCapture:
        active = self._active.get(session_id)
        if active is None:
            raise KeyError(f"Session is not active: {session_id}")
        if not isinstance(active.capture, BrowserAudioCapture):
            raise RuntimeError("Session is not using browser microphone audio.")
        return active.capture

    async def wait_for_browser_speaker_capture(
        self,
        enrollment_id: str,
        *,
        timeout_s: float = 3.0,
    ) -> BrowserAudioCapture:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            capture = self._speaker_captures.get(enrollment_id)
            if capture is not None:
                return capture
            await asyncio.sleep(0.05)
        raise RuntimeError("Speaker enrollment is not waiting for browser microphone audio.")

    async def speaker_identity_status(self) -> dict:
        return self.speaker_identity.status()

    async def create_speaker_enrollment(self) -> dict:
        enrollment = self.speaker_identity.start_enrollment()
        await self._publish_speaker_profile_update(enrollment["profile"])
        return enrollment

    async def speaker_enrollment(self, enrollment_id: str) -> dict:
        return self.speaker_identity.enrollment(enrollment_id)

    async def record_speaker_enrollment_sample(
        self,
        enrollment_id: str,
        *,
        device_id: str | None = None,
        fixture_wav: Path | None = None,
        audio_source: str = "server_device",
    ) -> dict:
        if audio_source == "browser_stream":
            capture: AudioCapture = BrowserAudioCapture()
            self._speaker_captures[enrollment_id] = capture
        else:
            capture = self._build_capture(
                device_id=device_id,
                fixture_wav=fixture_wav,
                audio_source="fixture" if fixture_wav else "server_device",
            )
        try:
            try:
                pcm = await asyncio.wait_for(
                    self._record_enrollment_pcm(capture),
                    timeout=self.settings.speaker_enrollment_sample_seconds + 12.0,
                )
            except asyncio.TimeoutError as exc:
                raise RuntimeError("Timed out waiting for speaker enrollment audio.") from exc
        finally:
            self._speaker_captures.pop(enrollment_id, None)
        payload = self.speaker_identity.add_enrollment_sample(
            enrollment_id,
            pcm,
            sample_rate=self.settings.audio_sample_rate,
        )
        await self._publish_speaker_profile_update(payload["profile"])
        return payload

    async def finalize_speaker_enrollment(self, enrollment_id: str) -> dict:
        payload = self.speaker_identity.finalize_enrollment(enrollment_id)
        await self._publish_speaker_profile_update(payload["profile"])
        return payload

    async def recalibrate_speaker_profile(self) -> dict:
        payload = self.speaker_identity.recalibrate()
        await self._publish_speaker_profile_update(payload["profile"])
        return payload

    async def reset_speaker_profile(self) -> dict:
        payload = self.speaker_identity.reset_profile()
        await self._publish_speaker_profile_update(payload["profile"])
        return payload

    async def apply_speaker_feedback(
        self,
        *,
        session_id: str,
        segment_id: str | None,
        old_label: str,
        new_label: str,
        feedback_type: str,
    ) -> dict:
        payload = self.speaker_identity.apply_feedback(
            session_id=session_id,
            segment_id=segment_id,
            old_label=old_label,
            new_label=new_label,
            feedback_type=feedback_type,
        )
        await self._publish_speaker_profile_update(payload["profile"])
        return payload

    def _build_capture(self, *, device_id: str | None, fixture_wav: Path | None, audio_source: str) -> AudioCapture:
        if audio_source == "browser_stream":
            return BrowserAudioCapture()
        if fixture_wav is not None or audio_source == "fixture":
            if fixture_wav is None:
                raise RuntimeError("Fixture audio source requires fixture_wav.")
            return FixtureWavAudioCapture(
                fixture_wav,
                sample_rate=self.settings.audio_sample_rate,
                chunk_ms=self.settings.audio_chunk_ms,
            )
        device = find_device(device_id)
        if device is None:
            raise RuntimeError("No capture device found. Connect a USB microphone or provide fixture_wav.")
        return FFmpegAudioCapture(
            device,
            sample_rate=self.settings.audio_sample_rate,
            chunk_ms=self.settings.audio_chunk_ms,
        )

    async def _record_enrollment_pcm(self, capture: AudioCapture) -> bytes:
        target_bytes = int(
            self.settings.audio_sample_rate
            * 2
            * self.settings.speaker_enrollment_sample_seconds
        )
        buffer = bytearray()
        try:
            async for chunk in capture.chunks():
                buffer.extend(chunk)
                if len(buffer) >= target_bytes:
                    break
        finally:
            await capture.stop()
        if not buffer:
            raise RuntimeError("No speaker enrollment audio was captured.")
        return bytes(buffer[:target_bytes])

    async def _run_capture_loop(
        self,
        session_id: str,
        capture: AudioCapture,
        window_queue: asyncio.Queue[AudioWindow],
    ) -> None:
        bytes_per_second = self.settings.audio_sample_rate * 2
        window_bytes = int(bytes_per_second * self.settings.transcription_window_seconds)
        overlap_bytes = int(bytes_per_second * self.settings.transcription_overlap_seconds)
        total_bytes = 0
        buffer = bytearray()

        await self.bus.publish(
            SidecarEvent(
                type=EVENT_AUDIO_STATUS,
                session_id=session_id,
                payload={"status": "listening", "raw_audio_retained": False},
            )
        )

        try:
            async for chunk in capture.chunks():
                total_bytes += len(chunk)
                buffer.extend(chunk)
                if len(buffer) < window_bytes:
                    continue

                buffer_seconds = len(buffer) / bytes_per_second
                start_offset_s = max(0.0, (total_bytes / bytes_per_second) - buffer_seconds)
                await self._enqueue_window(session_id, window_queue, AudioWindow(bytes(buffer), start_offset_s))

                if overlap_bytes > 0:
                    buffer = bytearray(buffer[-overlap_bytes:])
                else:
                    buffer.clear()

            if buffer:
                start_offset_s = max(0.0, (total_bytes / bytes_per_second) - (len(buffer) / bytes_per_second))
                await self._enqueue_window(session_id, window_queue, AudioWindow(bytes(buffer), start_offset_s, final=True))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self._publish_error(session_id, str(exc), fatal=True)

    async def _enqueue_window(
        self,
        session_id: str,
        queue: asyncio.Queue[AudioWindow],
        window: AudioWindow,
    ) -> None:
        active = self._active.get(session_id)
        if active is None:
            return
        if queue.full():
            try:
                queue.get_nowait()
                queue.task_done()
                active.dropped_windows += 1
                await self.bus.publish(
                    SidecarEvent(
                        type=EVENT_AUDIO_STATUS,
                        session_id=session_id,
                        payload={
                            "status": "catching_up",
                            "dropped_windows": active.dropped_windows,
                            "queue_depth": queue.qsize(),
                        },
                    )
                )
            except asyncio.QueueEmpty:
                pass
        await queue.put(window)

    async def _run_transcription_loop(
        self,
        session_id: str,
        window_queue: asyncio.Queue[AudioWindow],
        postprocess_queue: asyncio.Queue[SegmentBatch],
    ) -> None:
        while True:
            window = await window_queue.get()
            try:
                batch = await self._transcribe_window(session_id, window)
                if batch.segments:
                    await self._put_latest(postprocess_queue, batch)
            finally:
                window_queue.task_done()

    async def _transcribe_window(self, session_id: str, window: AudioWindow) -> SegmentBatch:
        active = self._active.get(session_id)
        if active is None:
            return SegmentBatch(segments=[])
        result = await self.transcriber.transcribe_pcm16(
            window.pcm,
            window.start_offset_s,
            initial_prompt=self.settings.asr_initial_prompt,
        )

        segments: list[TranscriptSegment] = []
        refresh_notes = False
        for span in result.spans:
            if not active.deduper.accept(span.text, span.start_s, span.end_s):
                continue
            segment = TranscriptSegment(
                id=new_id("seg"),
                session_id=session_id,
                start_s=span.start_s,
                end_s=span.end_s,
                text=span.text,
            )
            active.recent_segments.append(segment)
            active.recent_segments = active.recent_segments[-48:]
            if active.save_transcript:
                self.storage.add_transcript_segment(segment)
            speaker_payload = self._speaker_payload_for_span(
                active,
                segment,
                window,
                persist=active.save_transcript,
            )
            segments.append(segment)
            active.final_segment_count += 1
            if active.final_segment_count >= active.next_note_segment_count:
                refresh_notes = True
                active.next_note_segment_count += self.settings.notes_every_segments
            await self.bus.publish(
                SidecarEvent(
                    type=EVENT_TRANSCRIPT_FINAL,
                    session_id=session_id,
                    payload={
                        **segment.to_dict(),
                        "asr_model": result.model,
                        "queue_depth": active.window_queue.qsize(),
                        "speaker_identity_active": self.speaker_identity.status()["ready"],
                        "save_transcript": active.save_transcript,
                        "transcript_retention": "saved" if active.save_transcript else "temporary",
                        "raw_audio_retained": False,
                        **speaker_payload,
                    },
                )
            )
        return SegmentBatch(segments=segments, refresh_notes=refresh_notes)

    def _speaker_payload_for_span(
        self,
        active: ActiveSession,
        segment: TranscriptSegment,
        window: AudioWindow,
        *,
        persist: bool,
    ) -> dict:
        bytes_per_second = self.settings.audio_sample_rate * 2
        relative_start_s = max(0.0, segment.start_s - window.start_offset_s)
        relative_end_s = max(relative_start_s, segment.end_s - window.start_offset_s)
        start_byte = max(0, int(relative_start_s * bytes_per_second))
        end_byte = min(len(window.pcm), int(relative_end_s * bytes_per_second))
        start_byte -= start_byte % 2
        end_byte -= end_byte % 2
        pcm = window.pcm[start_byte:end_byte]
        try:
            result = self.speaker_identity.label_segment(
                session_id=active.id,
                segment_id=segment.id,
                pcm=pcm,
                start_ms=int(segment.start_s * 1000),
                end_ms=int(segment.end_s * 1000),
                transcript_text=segment.text,
                persist=persist,
            )
            return result.transcript_payload()
        except Exception as exc:
            return {
                "speaker_role": "unknown",
                "speaker_label": "Unknown speaker",
                "speaker_confidence": 0.0,
                "speaker_match_reason": f"speaker_identity_unavailable: {exc}",
                "speaker_low_confidence": True,
            }

    async def _put_latest(self, queue: asyncio.Queue[SegmentBatch], batch: SegmentBatch) -> None:
        if queue.full():
            try:
                stale = queue.get_nowait()
                queue.task_done()
                if stale.refresh_notes and not batch.refresh_notes:
                    batch = SegmentBatch(segments=batch.segments, refresh_notes=True)
            except asyncio.QueueEmpty:
                pass
        await queue.put(batch)

    async def _run_postprocess_loop(
        self,
        session_id: str,
        postprocess_queue: asyncio.Queue[SegmentBatch],
    ) -> None:
        while True:
            batch = await postprocess_queue.get()
            try:
                active = self._active.get(session_id)
                save_transcript = bool(active and active.save_transcript)
                for segment in batch.segments:
                    if save_transcript and not self.settings.disable_live_embeddings:
                        await self._safe_embed_transcript(segment)
                if active is not None and batch.refresh_notes:
                    await self._refresh_notes(session_id)
            finally:
                postprocess_queue.task_done()

    async def _safe_embed_transcript(self, segment: TranscriptSegment) -> None:
        try:
            await self.recall.add_text(
                "transcript_segment",
                segment.id,
                segment.text,
                {"session_id": segment.session_id, "start_s": segment.start_s, "end_s": segment.end_s},
            )
        except Exception as exc:
            return

    async def _refresh_notes(self, session_id: str) -> None:
        active = self._active.get(session_id)
        save_transcript = True if active is None else active.save_transcript
        recent_segments = (
            active.recent_segments[-10:]
            if active is not None and active.recent_segments
            else self.storage.recent_segments(session_id, limit=10)
        )
        source_segment_ids = [segment.id for segment in recent_segments]
        query = " ".join(segment.text for segment in recent_segments[-4:])
        self._maybe_enqueue_web_context(session_id, recent_segments)
        recall_hits = []
        if not self.settings.disable_live_embeddings:
            try:
                recall_hits = await self.recall.search(query, limit=5)
                for hit in recall_hits:
                    payload = hit.to_dict()
                    payload["source_segment_ids"] = source_segment_ids
                    await self.bus.publish(
                        SidecarEvent(type=EVENT_RECALL_HIT, session_id=session_id, payload=payload)
                    )
            except Exception:
                recall_hits = []

        try:
            work_cards = self.work_memory.search(query, limit=3)
            for card in work_cards:
                if save_transcript:
                    self.work_memory.record_recall_event(session_id, card, query)
                hit = card.to_search_hit()
                recall_hits.append(hit)
                payload = hit.to_dict()
                payload["source_segment_ids"] = source_segment_ids
                await self.bus.publish(
                    SidecarEvent(type=EVENT_RECALL_HIT, session_id=session_id, payload=payload)
                )
        except Exception:
            pass

        try:
            result = await self.notes.synthesize(session_id, recent_segments, recall_hits)
        except Exception as exc:
            await self._publish_error(session_id, f"Note synthesis failed: {exc}")
            return

        for note in result.notes:
            if save_transcript:
                self.storage.add_note(note)
            await self.bus.publish(SidecarEvent(type=EVENT_NOTE_UPDATE, session_id=session_id, payload=note.to_dict()))

    def _web_context_configured(self) -> bool:
        return self.settings.web_context_enabled and bool(self.settings.brave_search_api_key.strip())

    def _maybe_enqueue_web_context(self, session_id: str, recent_segments: list[TranscriptSegment]) -> None:
        if not self._web_context_configured():
            return
        active = self._active.get(session_id)
        if active is None or active.web_context_queue is None:
            return
        candidate = self.web_trigger_detector.detect(recent_segments)
        if candidate is None:
            return
        if (
            candidate.normalized_query in active.web_context_pending_queries
            or candidate.normalized_query in active.web_context_seen_queries
        ):
            return
        if active.web_context_queue.full():
            try:
                stale = active.web_context_queue.get_nowait()
                active.web_context_queue.task_done()
                active.web_context_pending_queries.discard(stale.normalized_query)
            except asyncio.QueueEmpty:
                pass
        try:
            active.web_context_queue.put_nowait(candidate)
            active.web_context_pending_queries.add(candidate.normalized_query)
        except asyncio.QueueFull:
            return

    async def _run_web_context_loop(
        self,
        session_id: str,
        web_context_queue: asyncio.Queue[WebContextCandidate],
    ) -> None:
        while True:
            candidate = await web_context_queue.get()
            try:
                active = self._active.get(session_id)
                if active is not None:
                    active.web_context_pending_queries.discard(candidate.normalized_query)
                if active is None or not self._web_context_configured():
                    continue
                if candidate.normalized_query in active.web_context_seen_queries:
                    continue
                now = time.monotonic()
                if (
                    active.web_context_last_at
                    and now - active.web_context_last_at < self.settings.web_context_min_interval_seconds
                ):
                    continue

                active.web_context_seen_queries.add(candidate.normalized_query)
                active.web_context_last_at = now
                try:
                    results = await self.web_search.search(candidate.query, freshness=candidate.freshness)
                except Exception:
                    results = []
                payload = self.web_context.synthesize(session_id, candidate, results)
                if payload is not None:
                    await self.bus.publish(SidecarEvent(type=EVENT_NOTE_UPDATE, session_id=session_id, payload=payload))
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            finally:
                web_context_queue.task_done()

    def _surface_task_exception(self, session_id: str, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        asyncio.create_task(self._publish_error(session_id, f"Pipeline task failed: {exc}", fatal=True))

    async def _publish_error(self, session_id: str, message: str, *, fatal: bool = False) -> None:
        if fatal:
            self.storage.set_session_status(session_id, "error", ended_at=time.time())
        await self.bus.publish(
            SidecarEvent(type=EVENT_ERROR, session_id=session_id, payload={"message": message, "fatal": fatal})
        )

    async def _publish_speaker_profile_update(self, profile: dict) -> None:
        await self.bus.publish(
            SidecarEvent(
                type=EVENT_SPEAKER_PROFILE_UPDATE,
                session_id=None,
                payload=profile,
            )
        )
