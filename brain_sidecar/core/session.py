from __future__ import annotations

import asyncio
import base64
import io
import inspect
import time
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path

from brain_sidecar.config import Settings
from brain_sidecar.core.audio import AudioCapture, FFmpegAudioCapture, FixtureWavAudioCapture, MAX_INPUT_GAIN_DB, MIN_INPUT_GAIN_DB
from brain_sidecar.core.asr import ASR_BACKEND_FASTER_WHISPER
from brain_sidecar.core.asr_factory import create_asr_backend
from brain_sidecar.core.dedupe import TranscriptDeduplicator, TranscriptFinalConsolidator
from brain_sidecar.core.devices import find_device
from brain_sidecar.core.domain_keywords import EnergyConversationDetector, EnergyConversationFrame, inactive_energy_frame
from brain_sidecar.core.energy_lens import EnergyConsultingAgent
from brain_sidecar.core.event_bus import EventBus
from brain_sidecar.core.events import (
    EVENT_AUDIO_STATUS,
    EVENT_ERROR,
    EVENT_GPU_STATUS,
    EVENT_NOTE_UPDATE,
    EVENT_RECALL_HIT,
    EVENT_SIDECAR_CARD,
    EVENT_SPEAKER_PROFILE_UPDATE,
    EVENT_TRANSCRIPT_FINAL,
    EVENT_TRANSCRIPT_PARTIAL,
    SidecarEvent,
)
from brain_sidecar.core.gpu import GpuStatus, prepare_asr_gpu, read_gpu_status
from brain_sidecar.core.meeting_agents import ContractCriticAgent, diagnostics_for_cards
from brain_sidecar.core.meeting_contract import MeetingContract, normalize_meeting_contract
from brain_sidecar.core.models import SidecarCard, TranscriptSegment, new_id
from brain_sidecar.core.note_quality import NoteQualityGate
from brain_sidecar.core.notes import NoteSynthesizer, heuristic_meeting_cards, note_from_sidecar
from brain_sidecar.core.ollama import OllamaClient
from brain_sidecar.core.recall import RecallIndex
from brain_sidecar.core.speaker_identity import SpeakerIdentityService, analyze_pcm16
from brain_sidecar.core.storage import Storage
from brain_sidecar.core.web_context import (
    BraveSearchClient,
    WebContextCandidate,
    WebContextSynthesizer,
    WebTriggerDetector,
)
from brain_sidecar.core.sidecar_cards import (
    create_sidecar_card,
    note_payload_to_sidecar_card,
    recall_payload_to_sidecar_card,
    status_sidecar_card,
)
from brain_sidecar.core.work_memory import WorkMemoryService


@dataclass(frozen=True)
class AudioWindow:
    pcm: bytes
    start_offset_s: float
    final: bool = False
    preview: bool = False


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
    transcript_consolidator: TranscriptFinalConsolidator | None = None
    note_quality_gate: NoteQualityGate | None = None
    web_context_queue: asyncio.Queue[WebContextCandidate] | None = None
    audio_source: str = "server_device"
    selected_device_id: str | None = None
    input_gain_db: float = 0.0
    save_transcript: bool = True
    recent_segments: list[TranscriptSegment] = field(default_factory=list)
    final_segment_count: int = 0
    next_note_segment_count: int = 3
    dropped_windows: int = 0
    web_context_pending_queries: set[str] = field(default_factory=set)
    web_context_seen_queries: set[str] = field(default_factory=set)
    web_context_last_at: float = 0.0
    last_partial_enqueued_at: float = 0.0
    last_partial_text: str = ""
    transcriber_busy: bool = False
    preview_in_flight: bool = False
    partial_windows_enqueued: int = 0
    partial_windows_skipped_busy: int = 0
    partial_windows_skipped_queue: int = 0
    partial_windows_dropped_for_final: int = 0
    partial_asr_duration_ms: float | None = None
    last_audio_rms: float | None = None
    silent_windows: int = 0
    asr_empty_windows: int = 0
    asr_backend: str = ASR_BACKEND_FASTER_WHISPER
    asr_model: str | None = None
    note_segments: list[TranscriptSegment] = field(default_factory=list)
    final_segments_collapsed: int = 0
    final_segments_replaced: int = 0
    final_segments_seen_for_notes: int = 0
    final_segments_suppressed_for_notes: int = 0
    meeting_contract: MeetingContract = field(default_factory=normalize_meeting_contract)
    meeting_diagnostics: dict[str, object] = field(
        default_factory=lambda: diagnostics_for_cards([], accepted_count=0, suppressed_count=0).to_dict()
    )
    energy_conversation_frame: EnergyConversationFrame | None = None
    energy_lens_last_active_at: float | None = None
    energy_lens_last_card_at: float | None = None


class SessionManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.bus = EventBus()
        self.storage = Storage(settings.data_dir)
        self.storage.connect()
        self.ollama = OllamaClient(settings)
        self.recall = RecallIndex(self.storage, self.ollama, settings)
        self.work_memory = WorkMemoryService(self.storage, self.recall, settings)
        self.notes = NoteSynthesizer(self.ollama)
        self.energy_detector = EnergyConversationDetector(
            enabled=settings.energy_lens_enabled,
            min_confidence=settings.energy_lens_min_confidence,
            max_keywords=settings.energy_lens_max_keywords,
        )
        self.energy_agent = EnergyConsultingAgent()
        self.speaker_identity = SpeakerIdentityService(self.storage, settings)
        self.transcriber = create_asr_backend(settings)
        self.web_trigger_detector = WebTriggerDetector()
        self.web_search = BraveSearchClient(
            settings.brave_search_api_key,
            timeout_s=settings.web_context_timeout_seconds,
            max_results=settings.web_context_max_results,
        )
        self.web_context = WebContextSynthesizer()
        self._active: dict[str, ActiveSession] = {}
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
        mic_tuning: dict[str, object] | None = None,
        meeting_contract: object | None = None,
    ) -> None:
        async with self._lock:
            if session_id in self._active:
                return
            normalized_contract = normalize_meeting_contract(meeting_contract)
            if not self._asr_loaded():
                await self._publish_asr_start_status(session_id)
                if self.settings.asr_device == "cuda":
                    await asyncio.to_thread(prepare_asr_gpu, self.settings)
                await self._publish_audio_status(session_id, "loading_asr")
            await self.transcriber.load()
            capture = self._build_capture(
                device_id=device_id,
                fixture_wav=fixture_wav,
                audio_source=audio_source,
                mic_tuning=mic_tuning,
            )
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
                transcript_consolidator=TranscriptFinalConsolidator(max_recent=48),
                note_quality_gate=self._new_note_quality_gate(),
                web_context_queue=web_context_queue,
                audio_source=audio_source,
                selected_device_id=capture.device.id if isinstance(capture, FFmpegAudioCapture) else None,
                input_gain_db=capture.input_gain_db if isinstance(capture, FFmpegAudioCapture) else 0.0,
                save_transcript=save_transcript,
                next_note_segment_count=self.settings.notes_every_segments,
                asr_backend=self._asr_backend_name(),
                asr_model=self._asr_model_name(),
                meeting_contract=normalized_contract,
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
            self.storage.set_session_status(session_id, "running", save_transcript=save_transcript)

        await self.bus.publish(
            SidecarEvent(
                type=EVENT_GPU_STATUS,
                session_id=session_id,
                payload={
                    **read_gpu_status().to_dict(),
                    **self._asr_status_fields(active),
                    "asr_model": self._asr_model_name(),
                    "asr_device": self.settings.asr_device,
                    "asr_compute_type": self.settings.asr_compute_type,
                    "asr_beam_size": self.settings.asr_beam_size,
                    "queue_size": self.settings.transcription_queue_size,
                    "partial_transcripts_enabled": self.settings.partial_transcripts_enabled,
                    "partial_window_seconds": self.settings.partial_window_seconds,
                    "partial_min_interval_seconds": self.settings.partial_min_interval_seconds,
                    "speaker_identity": self.speaker_identity.status(),
                    "selected_device_id": active.selected_device_id,
                    "input_gain_db": active.input_gain_db,
                    "save_transcript": active.save_transcript,
                    "raw_audio_retained": False,
                    **self._meeting_status(active),
                },
            )
        )

    def _asr_backend_name(self) -> str:
        return str(getattr(self.transcriber, "backend_name", self.settings.asr_backend))

    def _asr_model_name(self) -> str | None:
        return getattr(self.transcriber, "model_size", None)

    def _asr_loaded(self) -> bool:
        return self._asr_model_name() is not None

    def _asr_status_fields(self, active: ActiveSession | None = None) -> dict[str, object]:
        return {
            "asr_backend": self._asr_backend_name(),
            "asr_model": self._asr_model_name(),
            "asr_dtype": self.settings.asr_compute_type,
            "asr_device": self.settings.asr_device,
            "asr_backend_error": getattr(self.transcriber, "last_error", None),
        }

    async def _publish_asr_start_status(self, session_id: str) -> None:
        status = read_gpu_status()
        next_status = "freeing_gpu" if self._should_free_gpu_before_asr(status) else "loading_asr"
        await self._publish_audio_status(session_id, next_status, status)

    async def _publish_audio_status(
        self,
        session_id: str,
        status: str,
        gpu_status: GpuStatus | None = None,
        extra: dict | None = None,
    ) -> None:
        gpu_status = gpu_status or read_gpu_status()
        active = self._active.get(session_id)
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
                    **self._asr_status_fields(active),
                    **(self._pipeline_metrics(active) if active is not None else {}),
                    **(self._meeting_status(active) if active is not None else {}),
                    **(extra or {}),
                },
            )
        )

    def _should_free_gpu_before_asr(self, status: GpuStatus) -> bool:
        if self._asr_loaded():
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
            active = self._active.get(session_id)
        if active is None:
            return
        async with self._lock:
            self._active.pop(session_id, None)
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
                    **self._pipeline_metrics(active),
                    **self._meeting_status(active),
                },
            )
        )
        if active.save_transcript and active.recent_segments:
            await self._safe_store_session_memory_summary(session_id, active.recent_segments)

    async def add_library_root(self, path: Path) -> dict:
        root_id = await asyncio.to_thread(self.storage.add_library_root, path.expanduser().resolve())
        return {"id": root_id, "path": str(path.expanduser().resolve())}

    async def search_web_context(self, query: str, *, session_id: str | None = None) -> dict:
        decision = self.web_trigger_detector.decision_for_manual_query(query)
        cleaned_query = decision.sanitized_query
        enabled = self.settings.web_context_enabled
        configured = bool(self.settings.brave_search_api_key.strip())
        skip_reason = decision.skip_reason
        if not enabled:
            skip_reason = "web_disabled"
        elif not configured:
            skip_reason = "brave_key_missing"
        elif decision.candidate is None:
            skip_reason = skip_reason or "query_empty_after_sanitization"
        if skip_reason:
            return {
                "query": cleaned_query,
                "enabled": enabled,
                "configured": configured,
                "note": None,
                "skip_reason": skip_reason,
                "cards": [],
            }

        candidate = decision.candidate
        assert candidate is not None
        try:
            results = await self.web_search.search(candidate.query, freshness=candidate.freshness)
        except Exception:
            results = []
        if not results:
            skip_reason = "no_results"
        note = self.web_context.synthesize(session_id or "", candidate, results)
        card = (
            note_payload_to_sidecar_card(session_id or "", note, save_transcript=False).to_dict()
            if note is not None
            else None
        )
        if not results:
            skip_reason = getattr(self.web_search, "last_error_reason", None) or skip_reason
        if note is not None:
            note["session_id"] = session_id
        return {
            "query": cleaned_query,
            "enabled": enabled,
            "configured": configured,
            "note": note,
            "skip_reason": skip_reason,
            "cards": [card] if card else [],
        }

    async def ask_sidecar(self, query: str, *, session_id: str | None = None) -> dict:
        cards: list[SidecarCard] = []
        sections: dict[str, list[dict]] = {
            "prior_transcript": [],
            "pas_past_work": [],
            "current_public_web": [],
            "suggested_meeting_contribution": [],
        }
        recall_hits = await self.recall.search(query, limit=8, manual=True)
        for hit in recall_hits:
            payload = hit.to_dict()
            payload["explicitly_requested"] = True
            card = recall_payload_to_sidecar_card(session_id or "", payload)
            card = replace(card, priority="high", card_key=f"manual:{card.card_key}")
            cards.append(card)
            sections["prior_transcript"].append(card.to_dict())
        work_cards = await asyncio.wait_for(
            asyncio.to_thread(self.work_memory.search, query, 5, manual=True),
            timeout=self.settings.work_memory_search_timeout_seconds,
        )
        for work_card in work_cards:
            payload = work_card.to_search_hit().to_dict()
            payload["explicitly_requested"] = True
            card = recall_payload_to_sidecar_card(session_id or "", payload)
            card = replace(card, priority="high", card_key=f"manual:{card.card_key}")
            cards.append(card)
            sections["pas_past_work"].append(card.to_dict())
        web_payload = await self.search_web_context(query, session_id=session_id)
        for raw_card in web_payload.get("cards", []):
            if isinstance(raw_card, dict):
                card = create_sidecar_card(
                    session_id=session_id or "",
                    category=raw_card.get("category", "web"),
                    title=raw_card.get("title", "Current public web"),
                    body=raw_card.get("body", ""),
                    suggested_say=raw_card.get("suggested_say"),
                    suggested_ask=raw_card.get("suggested_ask"),
                    why_now=raw_card.get("why_now", "Returned for the typed query."),
                    priority="high",
                    confidence=raw_card.get("confidence", 0.72),
                    source_segment_ids=raw_card.get("source_segment_ids") or [],
                    source_type=raw_card.get("source_type", "brave_web"),
                    sources=raw_card.get("sources") or [],
                    citations=raw_card.get("citations") or [],
                    card_key=f"manual:{raw_card.get('card_key') or raw_card.get('id')}",
                    ephemeral=True,
                    explicitly_requested=True,
                )
                cards.append(card)
                sections["current_public_web"].append(card.to_dict())
        if cards:
            top = cards[0]
            contribution = create_sidecar_card(
                session_id=session_id or "",
                category="contribution",
                title="Suggested meeting contribution",
                body="Use the highest-confidence source above to add a grounded point without overclaiming.",
                suggested_say=top.suggested_say or f"I found relevant context in {top.title}; it may be worth checking against the current plan.",
                why_now="BP explicitly asked Sidecar to combine local, work-memory, and public context.",
                priority="high",
                confidence=min(0.82, max(0.55, top.confidence)),
                source_segment_ids=top.source_segment_ids,
                source_type=top.source_type,
                sources=top.sources,
                citations=top.citations,
                card_key=f"manual:contribution:{new_id('query')}",
                ephemeral=True,
                explicitly_requested=True,
            )
            cards.append(contribution)
            sections["suggested_meeting_contribution"].append(contribution.to_dict())
        return {
            "query": query,
            "sanitized_web_query": web_payload.get("query"),
            "web_skip_reason": web_payload.get("skip_reason"),
            "cards": [card.to_dict() for card in cards],
            "sections": sections,
            "raw_audio_retained": False,
        }

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
        mic_tuning: dict[str, object] | None = None,
    ) -> dict:
        tuning = normalize_mic_tuning(mic_tuning)
        capture = self._build_capture(
            device_id=device_id,
            fixture_wav=fixture_wav,
            audio_source=audio_source,
            mic_tuning=tuning,
        )
        try:
            pcm = await asyncio.wait_for(
                self._record_enrollment_pcm(capture),
                timeout=self.settings.speaker_enrollment_sample_seconds + 12.0,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError("Timed out waiting for speaker enrollment audio.") from exc
        payload = self.speaker_identity.add_enrollment_sample(
            enrollment_id,
            pcm,
            sample_rate=self.settings.audio_sample_rate,
            speech_sensitivity=str(tuning["speech_sensitivity"]),
        )
        await self._publish_speaker_profile_update(payload["profile"])
        return payload

    async def test_microphone(
        self,
        *,
        device_id: str | None = None,
        fixture_wav: Path | None = None,
        audio_source: str = "server_device",
        seconds: float = 3.0,
        mic_tuning: dict[str, object] | None = None,
    ) -> dict:
        tuning = normalize_mic_tuning(mic_tuning)
        capture = self._build_capture(
            device_id=device_id,
            fixture_wav=fixture_wav,
            audio_source=audio_source,
            mic_tuning=tuning,
        )
        pcm = await asyncio.wait_for(
            self._record_pcm(capture, seconds=max(1.0, min(8.0, seconds))),
            timeout=max(4.0, min(8.0, seconds) + 8.0),
        )
        quality = analyze_pcm16(
            pcm,
            self.settings.audio_sample_rate,
            speech_sensitivity=str(tuning["speech_sensitivity"]),
        )
        device = capture.device.to_dict() if isinstance(capture, FFmpegAudioCapture) else None
        recommendation = microphone_recommendation(quality.to_dict())
        return {
            "device": device,
            "audio_source": audio_source,
            "quality": quality.to_dict(),
            "recommendation": recommendation,
            "mic_tuning": tuning,
            "suggested_tuning": suggest_microphone_tuning(quality.to_dict(), tuning),
            "playback_audio": pcm16_wav_preview(pcm, self.settings.audio_sample_rate),
            "raw_audio_retained": False,
        }

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

    def _build_capture(
        self,
        *,
        device_id: str | None,
        fixture_wav: Path | None,
        audio_source: str,
        mic_tuning: dict[str, object] | None = None,
    ) -> AudioCapture:
        tuning = normalize_mic_tuning(mic_tuning)
        if audio_source == "browser_stream":
            raise RuntimeError("Browser microphone capture has been removed. Use server_device for capture from the local server microphone.")
        if audio_source not in {"server_device", "fixture"}:
            raise RuntimeError(f"Unsupported audio source: {audio_source}")
        if fixture_wav is not None or audio_source == "fixture":
            if not self.settings.test_mode_enabled:
                raise RuntimeError("Fixture audio is only available when recorded audio test mode is enabled.")
            if fixture_wav is None:
                raise RuntimeError("Fixture audio source requires fixture_wav.")
            return FixtureWavAudioCapture(
                fixture_wav,
                sample_rate=self.settings.audio_sample_rate,
                chunk_ms=self.settings.audio_chunk_ms,
            )
        device = find_device(device_id, probe=True)
        if device is None:
            raise RuntimeError("No healthy server microphone found. Connect a microphone, then refresh.")
        return FFmpegAudioCapture(
            device,
            sample_rate=self.settings.audio_sample_rate,
            chunk_ms=self.settings.audio_chunk_ms,
            input_gain_db=float(tuning["input_gain_db"]),
        )

    async def _record_enrollment_pcm(self, capture: AudioCapture) -> bytes:
        return await self._record_pcm(capture, seconds=self.settings.speaker_enrollment_sample_seconds)

    async def _record_pcm(self, capture: AudioCapture, *, seconds: float) -> bytes:
        target_bytes = int(self.settings.audio_sample_rate * 2 * seconds)
        buffer = bytearray()
        try:
            async for chunk in capture.chunks():
                buffer.extend(chunk)
                if len(buffer) >= target_bytes:
                    break
        finally:
            await capture.stop()
        if not buffer:
            raise RuntimeError("No microphone audio was captured.")
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
        partial_bytes = int(bytes_per_second * self.settings.partial_window_seconds)
        total_bytes = 0
        buffer = bytearray()
        active = self._active.get(session_id)

        await self.bus.publish(
            SidecarEvent(
                type=EVENT_AUDIO_STATUS,
                session_id=session_id,
                payload={
                    "status": "listening",
                    "raw_audio_retained": False,
                    **(self._pipeline_metrics(active) if active is not None else {}),
                    **(self._meeting_status(active) if active is not None else {}),
                },
            )
        )

        try:
            async for chunk in capture.chunks():
                total_bytes += len(chunk)
                buffer.extend(chunk)
                if self.settings.partial_transcripts_enabled and len(buffer) >= partial_bytes:
                    buffer_seconds = len(buffer) / bytes_per_second
                    partial_start_offset_s = max(0.0, (total_bytes / bytes_per_second) - buffer_seconds)
                    partial_pcm = bytes(buffer[-partial_bytes:])
                    partial_start_offset_s += max(0.0, (len(buffer) - partial_bytes) / bytes_per_second)
                    await self._maybe_enqueue_partial_window(
                        session_id,
                        window_queue,
                        AudioWindow(partial_pcm, partial_start_offset_s, preview=True),
                    )
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
        if not window.preview:
            retained: list[AudioWindow] = []
            while True:
                try:
                    queued = queue.get_nowait()
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
                if queued.preview:
                    active.partial_windows_dropped_for_final += 1
                else:
                    retained.append(queued)
            for retained_window in retained:
                await queue.put(retained_window)
        if queue.full():
            try:
                stale = queue.get_nowait()
                queue.task_done()
                if not stale.preview:
                    active.dropped_windows += 1
                    await self.bus.publish(
                        SidecarEvent(
                            type=EVENT_AUDIO_STATUS,
                            session_id=session_id,
                            payload={
                                "status": "catching_up",
                                "dropped_windows": active.dropped_windows,
                                "queue_depth": queue.qsize(),
                                **self._pipeline_metrics(active),
                                **self._meeting_status(active),
                            },
                        )
                    )
                else:
                    active.partial_windows_dropped_for_final += 1
            except asyncio.QueueEmpty:
                pass
        await queue.put(window)

    async def _maybe_enqueue_partial_window(
        self,
        session_id: str,
        queue: asyncio.Queue[AudioWindow],
        window: AudioWindow,
    ) -> None:
        active = self._active.get(session_id)
        if active is None:
            return
        now = time.monotonic()
        if now - active.last_partial_enqueued_at < self.settings.partial_min_interval_seconds:
            return
        if active.transcriber_busy or active.preview_in_flight:
            active.partial_windows_skipped_busy += 1
            return
        if queue.qsize() > 0 or queue.full():
            active.partial_windows_skipped_queue += 1
            return
        try:
            queue.put_nowait(window)
            active.last_partial_enqueued_at = now
            active.partial_windows_enqueued += 1
        except asyncio.QueueFull:
            active.partial_windows_skipped_queue += 1
            return

    async def _run_transcription_loop(
        self,
        session_id: str,
        window_queue: asyncio.Queue[AudioWindow],
        postprocess_queue: asyncio.Queue[SegmentBatch],
    ) -> None:
        while True:
            window = await window_queue.get()
            try:
                if window.preview:
                    active = self._active.get(session_id)
                    if active is not None and window_queue.qsize() > 0:
                        active.partial_windows_dropped_for_final += 1
                        await self._publish_audio_status(session_id, "preview_dropped_for_final")
                        continue
                    await self._transcribe_partial_window(session_id, window)
                    continue
                batch = await self._transcribe_window(session_id, window)
                if batch.segments:
                    await self._put_latest(postprocess_queue, batch)
            finally:
                window_queue.task_done()

    async def _transcribe_partial_window(self, session_id: str, window: AudioWindow) -> None:
        active = self._active.get(session_id)
        if active is None:
            return
        started_at = time.monotonic()
        active.transcriber_busy = True
        active.preview_in_flight = True
        try:
            result = await self.transcriber.transcribe_pcm16(
                window.pcm,
                window.start_offset_s,
                initial_prompt=self.settings.asr_initial_prompt,
            )
        finally:
            active.transcriber_busy = False
            active.preview_in_flight = False
        asr_duration_ms = round((time.monotonic() - started_at) * 1000, 1)
        active.partial_asr_duration_ms = asr_duration_ms
        for span in result.spans:
            text_key = " ".join(span.text.lower().split())
            if not text_key or text_key == active.last_partial_text:
                continue
            active.last_partial_text = text_key
            segment = TranscriptSegment(
                id=new_id("partial"),
                session_id=session_id,
                start_s=span.start_s,
                end_s=span.end_s,
                text=span.text,
                is_final=False,
            )
            speaker_payload = self._speaker_payload_for_span(
                active,
                segment,
                window,
                persist=False,
            )
            segment = _segment_with_speaker_payload(segment, speaker_payload)
            await self.bus.publish(
                SidecarEvent(
                    type=EVENT_TRANSCRIPT_PARTIAL,
                    session_id=session_id,
                    payload=self._transcript_payload(
                        active,
                        segment,
                        asr_model=result.model,
                        is_final=False,
                        speaker_payload=speaker_payload,
                        asr_duration_ms=asr_duration_ms,
                    ),
                )
            )

    async def _transcribe_window(self, session_id: str, window: AudioWindow) -> SegmentBatch:
        active = self._active.get(session_id)
        if active is None:
            return SegmentBatch(segments=[])
        started_at = time.monotonic()
        active.transcriber_busy = True
        try:
            result = await self.transcriber.transcribe_pcm16(
                window.pcm,
                window.start_offset_s,
                initial_prompt=self.settings.asr_initial_prompt,
            )
        finally:
            active.transcriber_busy = False
        asr_duration_ms = round((time.monotonic() - started_at) * 1000, 1)
        active.last_audio_rms = result.audio_rms

        if not result.spans:
            await self._record_empty_asr_window(session_id, active, asr_duration_ms=asr_duration_ms)
            return SegmentBatch(segments=[])

        segments: list[TranscriptSegment] = []
        refresh_notes = False
        for span in result.spans:
            segment = TranscriptSegment(
                id=new_id("seg"),
                session_id=session_id,
                start_s=span.start_s,
                end_s=span.end_s,
                text=span.text,
            )
            consolidator = self._transcript_consolidator_for(active)
            if not consolidator.would_consolidate(segment):
                if not active.deduper.accept(span.text, span.start_s, span.end_s):
                    continue
            speaker_payload = self._speaker_payload_for_span(
                active,
                segment,
                window,
                persist=active.save_transcript,
            )
            segment = _segment_with_speaker_payload(segment, speaker_payload)
            batch = await self._accept_final_segment(
                session_id,
                active,
                segment,
                asr_model=result.model,
                speaker_payload=speaker_payload,
                asr_duration_ms=asr_duration_ms,
            )
            segments.extend(batch.segments)
            refresh_notes = refresh_notes or batch.refresh_notes
        return SegmentBatch(segments=segments, refresh_notes=refresh_notes)

    async def _accept_final_segment(
        self,
        session_id: str,
        active: ActiveSession,
        segment: TranscriptSegment,
        *,
        asr_model: str,
        speaker_payload: dict,
        asr_duration_ms: float | None,
    ) -> SegmentBatch:
        consolidator = self._transcript_consolidator_for(active)
        consolidation = consolidator.accept(segment)
        if consolidation.segment is None:
            active.final_segments_suppressed_for_notes += 1
            self._sync_note_evidence_segments(active)
            return SegmentBatch(segments=[])
        refresh_notes = False
        if consolidation.collapsed:
            active.final_segments_collapsed += 1
            self._sync_note_evidence_segments(active)
            if consolidation.suppressed:
                active.final_segments_suppressed_for_notes += 1
                return SegmentBatch(segments=[])
            active.final_segments_replaced += 1
            refresh_notes = True
        segment = consolidation.segment
        replaces_segment_id = consolidation.replaced_segment_id if consolidation.collapsed else None
        self._accept_transcript_segment(active, segment, replaces_segment_id=replaces_segment_id)
        if active.save_transcript:
            self.storage.upsert_transcript_segment(segment, replaces_segment_id=replaces_segment_id)
        self._update_energy_lens_frame(active)
        active.last_partial_text = ""
        if not consolidation.collapsed:
            active.final_segment_count += 1
        if active.final_segment_count >= active.next_note_segment_count:
            refresh_notes = True
            active.next_note_segment_count += self.settings.notes_every_segments
        await self.bus.publish(
            SidecarEvent(
                type=EVENT_TRANSCRIPT_FINAL,
                session_id=session_id,
                payload=self._transcript_payload(
                    active,
                    segment,
                    asr_model=asr_model,
                    is_final=True,
                    speaker_payload=speaker_payload,
                    asr_duration_ms=asr_duration_ms,
                    replaces_segment_id=replaces_segment_id,
                ),
            )
        )
        return SegmentBatch(segments=[segment], refresh_notes=refresh_notes)

    def _transcript_payload(
        self,
        active: ActiveSession,
        segment: TranscriptSegment,
        *,
        asr_model: str,
        is_final: bool,
        speaker_payload: dict,
        asr_duration_ms: float | None = None,
        replaces_segment_id: str | None = None,
    ) -> dict:
        payload = {
            **segment.to_dict(),
            "is_final": is_final,
            "asr_model": asr_model,
            "queue_depth": active.window_queue.qsize(),
            "dropped_windows": active.dropped_windows,
            "asr_duration_ms": asr_duration_ms,
            **self._asr_status_fields(active),
            **self._pipeline_metrics(active),
            "asr_model": asr_model,
            "speaker_identity_active": self.speaker_identity.status()["ready"],
            "save_transcript": active.save_transcript,
            "transcript_retention": "saved" if active.save_transcript else "temporary",
            "raw_audio_retained": False,
            **speaker_payload,
        }
        if replaces_segment_id:
            payload["replaces_segment_id"] = replaces_segment_id
        if is_final:
            payload.update(self._energy_status(active))
        return payload

    def _pipeline_metrics(self, active: ActiveSession) -> dict:
        return {
            "queue_depth": active.window_queue.qsize(),
            "dropped_windows": active.dropped_windows,
            "asr_backend": active.asr_backend,
            "asr_model": active.asr_model or self._asr_model_name(),
            "partial_windows_enqueued": active.partial_windows_enqueued,
            "partial_windows_skipped_busy": active.partial_windows_skipped_busy,
            "partial_windows_skipped_queue": active.partial_windows_skipped_queue,
            "partial_windows_dropped_for_final": active.partial_windows_dropped_for_final,
            "partial_asr_duration_ms": active.partial_asr_duration_ms,
            "last_audio_rms": active.last_audio_rms,
            "silent_windows": active.silent_windows,
            "asr_empty_windows": active.asr_empty_windows,
            "event_drops": self.bus.drop_count(active.id),
            "audio_source": active.audio_source,
            "selected_device_id": active.selected_device_id,
            "input_gain_db": active.input_gain_db,
            "final_segments_collapsed": active.final_segments_collapsed,
            "final_segments_replaced": active.final_segments_replaced,
            "final_segments_seen_for_notes": active.final_segments_seen_for_notes,
            "final_segments_suppressed_for_notes": active.final_segments_suppressed_for_notes,
            "note_evidence_window_segments": len(active.note_segments),
        }

    def _meeting_status(self, active: ActiveSession) -> dict[str, object]:
        return {
            "meeting_contract": active.meeting_contract.to_dict(),
            "meeting_diagnostics": active.meeting_diagnostics,
            **self._energy_status(active),
        }

    def _energy_status(self, active: ActiveSession) -> dict[str, object]:
        frame = active.energy_conversation_frame or inactive_energy_frame()
        payload = frame.to_dict()
        return {
            "energy_lens_active": payload["active"],
            "energy_lens_score": payload["score"],
            "energy_lens_confidence": payload["confidence"],
            "energy_lens_categories": [item["category"] for item in payload["top_categories"]],
            "energy_lens_keywords": [item["phrase"] for item in payload["top_keywords"]],
            "energy_lens_evidence_segment_ids": payload["evidence_segment_ids"],
            "energy_lens_evidence_quote": payload["evidence_quote"],
            "energy_lens_summary_label": payload["summary_label"],
            "raw_audio_retained": False,
        }

    async def _record_empty_asr_window(
        self,
        session_id: str,
        active: ActiveSession,
        *,
        asr_duration_ms: float,
    ) -> None:
        active.asr_empty_windows += 1
        if active.last_audio_rms is not None and active.last_audio_rms < self.settings.asr_min_audio_rms:
            active.silent_windows += 1
        extra: dict[str, object] = {"asr_duration_ms": asr_duration_ms}
        warning = self._capture_warning(active)
        if warning is not None:
            extra["capture_warning"] = warning
        await self._publish_audio_status(session_id, "listening", extra=extra)

    def _capture_warning(self, active: ActiveSession) -> dict[str, object] | None:
        if active.silent_windows >= 3 and (active.silent_windows <= 6 or active.silent_windows % 10 == 0):
            return {
                "reason": "audio_too_quiet",
                "message": "Capture is active, but recent ASR windows are below the speech threshold.",
            }
        if active.asr_empty_windows >= 3 and (active.asr_empty_windows <= 6 or active.asr_empty_windows % 10 == 0):
            return {
                "reason": "asr_no_spans",
                "message": "Capture is active, but ASR is not emitting transcript spans.",
            }
        return None

    def _transcript_consolidator_for(self, active: ActiveSession) -> TranscriptFinalConsolidator:
        if active.transcript_consolidator is None:
            active.transcript_consolidator = TranscriptFinalConsolidator(max_recent=48)
        return active.transcript_consolidator

    def _accept_transcript_segment(
        self,
        active: ActiveSession,
        segment: TranscriptSegment,
        *,
        replaces_segment_id: str | None = None,
    ) -> None:
        active.final_segments_seen_for_notes += 1
        replaced = False
        replacement_ids = {segment.id}
        if replaces_segment_id:
            replacement_ids.add(replaces_segment_id)
        for index, existing in enumerate(active.recent_segments):
            if existing.id in replacement_ids:
                active.recent_segments[index] = segment
                replaced = True
                break
        if not replaced:
            active.recent_segments.append(segment)
        active.recent_segments = active.recent_segments[-48:]
        self._sync_note_evidence_segments(active)

    def _sync_note_evidence_segments(self, active: ActiveSession) -> None:
        active.note_segments = self._transcript_consolidator_for(active).segments()[-48:]

    def _update_energy_lens_frame(self, active: ActiveSession) -> None:
        frame = self.energy_detector.detect(
            active.note_segments[-12:] if active.note_segments else active.recent_segments[-12:],
            meeting_contract=active.meeting_contract,
            save_transcript=active.save_transcript,
        )
        active.energy_conversation_frame = frame
        if frame.active:
            active.energy_lens_last_active_at = time.time()

    def _new_note_quality_gate(self) -> NoteQualityGate:
        return NoteQualityGate(
            min_evidence_segments=self.settings.sidecar_min_evidence_segments,
            duplicate_window_seconds=self.settings.sidecar_duplicate_window_seconds,
            generic_clarify_window_seconds=self.settings.sidecar_generic_clarify_window_seconds,
            max_cards_per_5min=self.settings.sidecar_max_cards_per_5min,
        )

    def _note_quality_gate_for(self, active: ActiveSession) -> NoteQualityGate:
        if active.note_quality_gate is None:
            active.note_quality_gate = self._new_note_quality_gate()
        return active.note_quality_gate

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
        return self._speaker_payload_for_pcm(active, segment, pcm, persist=persist)

    def _speaker_payload_for_pcm(
        self,
        active: ActiveSession,
        segment: TranscriptSegment,
        pcm: bytes,
        *,
        persist: bool,
    ) -> dict:
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
            await self._publish_error(segment.session_id, f"Embedding skipped: {exc}", fatal=False)

    async def _refresh_notes(self, session_id: str) -> None:
        active = self._active.get(session_id)
        save_transcript = True if active is None else active.save_transcript
        if active is not None and active.note_segments:
            recent_segments = active.note_segments[-24:]
        elif active is not None and active.recent_segments:
            recent_segments = active.recent_segments[-24:]
        else:
            recent_segments = self.storage.recent_segments(session_id, limit=24)
        source_segment_ids = source_ids_for_segments(recent_segments)
        query = " ".join(segment.text for segment in recent_segments[-6:])
        self._maybe_enqueue_web_context(session_id, recent_segments)
        recall_hits = []
        note_context_hits = []
        if not self.settings.disable_live_embeddings:
            try:
                try:
                    recall_hits = await self.recall.search(
                        query,
                        limit=self.settings.recall_max_live_hits,
                        recent_text=query,
                    )
                except TypeError:
                    recall_hits = await self.recall.search(query, limit=self.settings.recall_max_live_hits)
                for hit in recall_hits:
                    payload = hit.to_dict()
                    payload["source_segment_ids"] = source_segment_ids
                    await self.bus.publish(
                        SidecarEvent(type=EVENT_RECALL_HIT, session_id=session_id, payload=payload)
                    )
                    await self._publish_sidecar_card(self._sidecar_card_from_recall_payload(session_id, payload))
            except Exception as exc:
                await self._publish_error(session_id, f"Recall search skipped: {exc}", fatal=False)
                recall_hits = []
            note_context_hits = [
                hit for hit in recall_hits if hit.source_type not in {"work_memory", "work_memory_project"}
            ]

        try:
            work_cards = await asyncio.wait_for(
                asyncio.to_thread(self.work_memory.search, query, 3),
                timeout=self.settings.work_memory_search_timeout_seconds,
            )
            for card in work_cards:
                if save_transcript:
                    self.work_memory.record_recall_event(session_id, card, query)
                hit = card.to_search_hit()
                payload = hit.to_dict()
                payload["source_segment_ids"] = source_segment_ids
                await self.bus.publish(
                    SidecarEvent(type=EVENT_RECALL_HIT, session_id=session_id, payload=payload)
                )
                await self._publish_sidecar_card(self._sidecar_card_from_recall_payload(session_id, payload))
        except Exception as exc:
            await self._publish_error(session_id, f"Work memory recall skipped: {exc}", fatal=False)

        fast_cards = heuristic_meeting_cards(session_id, recent_segments)
        if active is not None and active.energy_conversation_frame is not None:
            energy_cards = self.energy_agent.cards(
                session_id,
                recent_segments,
                active.energy_conversation_frame,
                active.meeting_contract,
                max_cards=self.settings.energy_lens_max_cards_per_pass,
            )
            if energy_cards:
                active.energy_lens_last_card_at = time.time()
                fast_cards = [*energy_cards, *fast_cards]
        await self._publish_accepted_generated_cards(
            session_id,
            active,
            fast_cards,
            recent_segments,
            save_transcript=save_transcript,
        )

        try:
            synthesize_params = inspect.signature(self.notes.synthesize).parameters
            synthesize_kwargs = (
                {"meeting_contract": active.meeting_contract}
                if active is not None and "meeting_contract" in synthesize_params
                else {}
            )
            if active is not None and "energy_frame" in synthesize_params:
                synthesize_kwargs["energy_frame"] = active.energy_conversation_frame
            result = await self.notes.synthesize(session_id, recent_segments, note_context_hits, **synthesize_kwargs)
        except Exception as exc:
            await self._publish_error(session_id, f"Note synthesis failed: {exc}")
            return

        generated_cards = self._cards_from_note_result(session_id, result, save_transcript=save_transcript)
        await self._publish_accepted_generated_cards(
            session_id,
            active,
            generated_cards,
            recent_segments,
            save_transcript=save_transcript,
        )

    async def _publish_accepted_generated_cards(
        self,
        session_id: str,
        active: ActiveSession | None,
        generated_cards: list[SidecarCard],
        recent_segments: list[TranscriptSegment],
        *,
        save_transcript: bool,
    ) -> None:
        if not generated_cards:
            return
        accepted_cards = self._accepted_generated_cards(active, generated_cards, recent_segments)
        for card in accepted_cards:
            note = note_from_sidecar(card)
            if save_transcript:
                self.storage.add_note(note)
            payload = note.to_dict()
            await self.bus.publish(SidecarEvent(type=EVENT_NOTE_UPDATE, session_id=session_id, payload=payload))
            await self._publish_sidecar_card(card)

    def _cards_from_note_result(
        self,
        session_id: str,
        result: object,
        *,
        save_transcript: bool,
    ) -> list[SidecarCard]:
        preferred_cards_by_key = {
            card.card_key or f"{card.category}:{card.title}": card
            for card in getattr(result, "sidecar_cards", [])
        }
        if preferred_cards_by_key:
            return list(preferred_cards_by_key.values())
        cards: list[SidecarCard] = []
        for note in getattr(result, "notes", []):
            cards.append(
                self._sidecar_card_from_note_payload(
                    session_id,
                    note.to_dict(),
                    save_transcript=save_transcript,
                )
            )
        return cards

    def _accepted_generated_cards(
        self,
        active: ActiveSession | None,
        cards: list[SidecarCard],
        evidence_segments: list[TranscriptSegment],
    ) -> list[SidecarCard]:
        if not cards:
            if active is not None:
                active.meeting_diagnostics = diagnostics_for_cards([], accepted_count=0, suppressed_count=0).to_dict()
            return []
        if not self.settings.sidecar_quality_gate_enabled or active is None:
            accepted = cards[: self.settings.sidecar_max_cards_per_generation_pass]
            if active is not None:
                active.meeting_diagnostics = diagnostics_for_cards(
                    cards,
                    accepted_count=len(accepted),
                    suppressed_count=max(0, len(cards) - len(accepted)),
                ).to_dict()
            return accepted
        gate = self._note_quality_gate_for(active)
        speaker_status = self.speaker_identity.status()
        review = ContractCriticAgent(gate).review(
            cards,
            evidence_segments,
            speaker_status,
            max_cards=self.settings.sidecar_max_cards_per_generation_pass,
        )
        active.meeting_diagnostics = review.diagnostics.to_dict()
        return review.accepted_cards

    async def _publish_sidecar_card(self, card: SidecarCard) -> None:
        await self.bus.publish(
            SidecarEvent(type=EVENT_SIDECAR_CARD, session_id=card.session_id, payload=card.to_dict())
        )

    def _sidecar_card_from_note_payload(
        self,
        session_id: str,
        payload: dict,
        *,
        save_transcript: bool,
    ) -> SidecarCard:
        return note_payload_to_sidecar_card(session_id, payload, save_transcript=save_transcript)

    def _sidecar_card_from_recall_payload(self, session_id: str, payload: dict) -> SidecarCard:
        return recall_payload_to_sidecar_card(session_id, payload)

    def _web_context_configured(self) -> bool:
        return self.settings.web_context_enabled and bool(self.settings.brave_search_api_key.strip())

    def _maybe_enqueue_web_context(self, session_id: str, recent_segments: list[TranscriptSegment]) -> None:
        if not self._web_context_configured():
            return
        active = self._active.get(session_id)
        if active is None or active.web_context_queue is None:
            return
        decision = self.web_trigger_detector.decision_for_segments(recent_segments)
        candidate = decision.candidate
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
                    await self._publish_sidecar_card(
                        status_sidecar_card(
                            session_id=session_id,
                            title="Web lookup skipped",
                            body="Current-web search is cooling down to avoid noisy repeated lookups.",
                            why_now="A live web candidate appeared before the configured cooldown elapsed.",
                            card_key=f"web-skip:cooldown:{candidate.normalized_query}",
                        )
                    )
                    continue

                active.web_context_seen_queries.add(candidate.normalized_query)
                active.web_context_last_at = now
                try:
                    results = await self.web_search.search(candidate.query, freshness=candidate.freshness)
                except Exception as exc:
                    await self._publish_error(session_id, f"Web context search skipped: {exc}", fatal=False)
                    results = []
                payload = self.web_context.synthesize(session_id, candidate, results)
                if payload is not None:
                    await self.bus.publish(SidecarEvent(type=EVENT_NOTE_UPDATE, session_id=session_id, payload=payload))
                    await self._publish_sidecar_card(
                        self._sidecar_card_from_note_payload(session_id, payload, save_transcript=False)
                    )
                else:
                    await self._publish_sidecar_card(
                        status_sidecar_card(
                            session_id=session_id,
                            title="No web results",
                            body="The sanitized public web lookup did not return compact sources.",
                            why_now="The meeting asked for current public context, but Brave returned no usable results.",
                            card_key=f"web-skip:no-results:{candidate.normalized_query}",
                        )
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._publish_error(session_id, f"Web context worker skipped a candidate: {exc}", fatal=False)
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

    async def _safe_store_session_memory_summary(
        self,
        session_id: str,
        segments: list[TranscriptSegment],
    ) -> None:
        try:
            summary = _build_session_memory_summary(session_id, segments)
            await asyncio.to_thread(self.storage.upsert_session_memory_summary, **summary)
            await self.recall.add_text(
                "session_summary",
                session_id,
                summary["summary"],
                {
                    "session_id": session_id,
                    "title": summary["title"],
                    "source_segment_ids": summary["source_segment_ids"],
                    "summary": True,
                },
            )
        except Exception as exc:
            await self._publish_error(session_id, f"Session memory summary skipped: {exc}", fatal=False)


def _segment_with_speaker_payload(segment: TranscriptSegment, payload: dict) -> TranscriptSegment:
    return replace(
        segment,
        speaker_role=payload.get("speaker_role") or segment.speaker_role,
        speaker_label=payload.get("speaker_label") or segment.speaker_label,
        speaker_confidence=payload.get("speaker_confidence") if payload.get("speaker_confidence") is not None else segment.speaker_confidence,
        speaker_match_reason=payload.get("speaker_match_reason") or segment.speaker_match_reason,
        speaker_low_confidence=(
            bool(payload.get("speaker_low_confidence"))
            if payload.get("speaker_low_confidence") is not None
            else segment.speaker_low_confidence
        ),
    )


def source_ids_for_segments(segments: list[TranscriptSegment]) -> list[str]:
    seen: set[str] = set()
    source_ids: list[str] = []
    for segment in segments:
        for source_id in segment.source_segment_ids or [segment.id]:
            if source_id in seen:
                continue
            seen.add(source_id)
            source_ids.append(source_id)
    return source_ids


def _build_session_memory_summary(session_id: str, segments: list[TranscriptSegment]) -> dict:
    text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
    clipped = _clip_text(text, 1200)
    topics = _keyword_list(text, limit=8)
    source_segment_ids = source_ids_for_segments(segments[-24:])
    return {
        "session_id": session_id,
        "title": f"Session summary {time.strftime('%Y-%m-%d %H:%M')}",
        "summary": clipped or "Saved session with no clear transcript text.",
        "topics": topics,
        "decisions": _sentence_matches(text, ["decided", "decision", "settled", "approved"]),
        "actions": _sentence_matches(text, ["follow up", "i'll", "i will", "action", "owner"]),
        "unresolved_questions": _sentence_matches(text, ["?", "open question", "unclear", "need to know"]),
        "entities": topics[:6],
        "lessons": _sentence_matches(text, ["risk", "learned", "lesson", "watch", "dependency"]),
        "source_segment_ids": source_segment_ids,
    }


def _keyword_list(text: str, *, limit: int) -> list[str]:
    terms = [
        term
        for term in _safe_label(text, default="").replace("_", " ").split()
        if len(term) >= 4 and term not in {"that", "this", "with", "from", "will", "need", "have"}
    ]
    seen: list[str] = []
    for term in terms:
        if term not in seen:
            seen.append(term)
        if len(seen) >= limit:
            break
    return seen


def _sentence_matches(text: str, needles: list[str], *, limit: int = 5) -> list[str]:
    sentences = [sentence.strip() for sentence in re_split_sentences(text) if sentence.strip()]
    matches: list[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(needle in lower for needle in needles):
            matches.append(_clip_text(sentence, 220))
        if len(matches) >= limit:
            break
    return matches


def re_split_sentences(text: str) -> list[str]:
    import re

    return re.split(r"(?<=[?.!])\s+", text)


def _clip_text(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return clipped or text[:limit].strip()


def _optional_text(value: object, limit: int) -> str | None:
    text = _clip_text(str(value or ""), limit)
    return text or None


def _safe_label(value: object, *, default: str) -> str:
    label = str(value or default).strip().lower().replace("-", "_")
    return label if label else default


def _string_list(value: object, *, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()][:limit]


def _source_list(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    sources: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        path = str(item.get("path") or "").strip()
        if title and (url or path):
            source: dict[str, str] = {"title": title}
            if url:
                source["url"] = url
            if path:
                source["path"] = path
            sources.append(source)
    return sources[:6]


def _clamp_float(value: object, *, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(1.0, max(0.0, number))


def _note_category(kind: str, source_type: str) -> str:
    if source_type == "brave_web":
        return "web"
    if kind in {"action", "decision", "question", "risk", "clarification", "contribution"}:
        return kind
    return "note"


def _priority_for_category(category: str) -> str:
    if category in {"action", "decision", "risk", "contribution"}:
        return "high"
    if category == "note":
        return "normal"
    return "normal"


def _note_why_now(category: str, source_type: str, save_transcript: bool) -> str:
    if source_type == "brave_web":
        return "Public current-web context matched an explicit technical question."
    if category == "action":
        return "Recent speech contains a possible follow-up or owner."
    if category == "decision":
        return "Recent speech sounds like something was decided or settled."
    if category == "question":
        return "Recent speech left an open question."
    if category == "risk":
        return "Recent speech points to a risk or assumption to watch."
    if category == "clarification":
        return "A clarifying question could prevent ambiguity right now."
    if category == "contribution":
        return "This looks like a useful point BP could add to the meeting."
    if save_transcript:
        return "Generated from the current saved transcript window."
    return "Generated from the current temporary transcript window."


def _card_key(prefix: str, category: str, source_type: str, title: str, source_segment_ids: list[str]) -> str:
    normalized_title = "_".join(title.lower().split())[:80] or "untitled"
    source_tail = "_".join(source_segment_ids[-2:]) if category == "web" else ""
    return ":".join(part for part in [prefix, category, source_type, normalized_title, source_tail] if part)


def _source_title(source_type: str) -> str:
    if source_type == "work_memory_project":
        return "Relevant past work"
    if source_type in {"session", "transcript_segment"}:
        return "Relevant prior transcript"
    if source_type in {"file", "document_chunk"}:
        return "Relevant local note"
    return "Relevant memory"


def _card_source_type(source_type: str) -> str:
    if source_type == "work_memory_project":
        return "work_memory"
    if source_type in {"session", "transcript_segment"}:
        return "saved_transcript"
    if source_type in {"file", "document_chunk"}:
        return "local_file"
    return "saved_transcript"


def microphone_recommendation(quality: dict) -> dict:
    usable = float(quality.get("usable_speech_seconds") or 0.0)
    score = float(quality.get("quality_score") or 0.0)
    rms = float(quality.get("rms") or 0.0)
    peak = float(quality.get("peak") or 0.0)
    issues = list(quality.get("issues") or [])
    if "clipping" in issues or peak >= 0.98:
        return {
            "status": "too_loud",
            "title": "Input is clipping",
            "detail": "Move back from the mic or lower input gain before recording speaker samples.",
        }
    if usable < 1.0 or rms < 0.008:
        return {
            "status": "too_quiet",
            "title": "Not enough speech detected",
            "detail": "Move closer, raise input boost, or switch sensitivity to Quiet, then speak normally for the whole test.",
        }
    if score < 0.55:
        return {
            "status": "noisy",
            "title": "Mic check is marginal",
            "detail": "Try a quieter room and keep other voices out before speaker training.",
        }
    return {
        "status": "good",
        "title": "Mic check looks good",
        "detail": "Use this setup for speaker samples: one voice, normal meeting distance, steady speech.",
    }


def normalize_mic_tuning(tuning: dict[str, object] | None) -> dict[str, object]:
    tuning = tuning or {}
    sensitivity = str(tuning.get("speech_sensitivity") or "normal").strip().lower()
    if sensitivity not in {"quiet", "normal", "noisy"}:
        sensitivity = "normal"
    try:
        input_gain_db = float(tuning.get("input_gain_db", 0.0))
    except (TypeError, ValueError):
        input_gain_db = 0.0
    return {
        "auto_level": bool(tuning.get("auto_level", True)),
        "input_gain_db": round(max(MIN_INPUT_GAIN_DB, min(MAX_INPUT_GAIN_DB, input_gain_db)), 1),
        "speech_sensitivity": sensitivity,
    }


def suggest_microphone_tuning(quality: dict, current: dict[str, object] | None = None) -> dict[str, object]:
    tuning = normalize_mic_tuning(current)
    gain = float(tuning["input_gain_db"])
    sensitivity = str(tuning["speech_sensitivity"])
    usable = float(quality.get("usable_speech_seconds") or 0.0)
    rms = float(quality.get("rms") or 0.0)
    peak = float(quality.get("peak") or 0.0)
    issues = set(quality.get("issues") or [])
    reason = "Current mic tuning looks usable."

    if "clipping" in issues or peak >= 0.98:
        gain = max(MIN_INPUT_GAIN_DB, min(MAX_INPUT_GAIN_DB, gain) - 6.0)
        reason = "The recording clipped, so auto level recommends lowering input boost."
    elif usable < 1.5 or rms < 0.008:
        gain = min(MAX_INPUT_GAIN_DB, gain + 6.0)
        if rms < 0.012:
            sensitivity = "quiet"
        reason = "Only a short stretch of speech was detected, so auto level recommends more boost and quieter-room sensitivity."
    elif "too_much_silence" in issues and sensitivity != "quiet":
        sensitivity = "quiet"
        reason = "Speech was intermittent, so auto level recommends Quiet sensitivity."
    elif "noisy" in issues and sensitivity != "noisy":
        sensitivity = "noisy"
        reason = "The mic check looked noisy, so auto level recommends Noisy sensitivity."

    return {
        "auto_level": bool(tuning["auto_level"]),
        "input_gain_db": round(gain, 1),
        "speech_sensitivity": sensitivity,
        "reason": reason,
    }


def pcm16_wav_preview(pcm: bytes, sample_rate: int) -> dict:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return {
        "mime_type": "audio/wav",
        "data_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "duration_seconds": len(pcm) / float(sample_rate * 2),
    }
