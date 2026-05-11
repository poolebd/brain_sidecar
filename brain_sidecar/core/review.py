from __future__ import annotations

import asyncio
import json
import math
import re
import shutil
import subprocess
import time
import wave
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from brain_sidecar.config import Settings
from brain_sidecar.core.audio_spool import delete_audio_file
from brain_sidecar.core.meeting_agents import ContractCriticAgent, diagnostics_for_cards
from brain_sidecar.core.models import NoteCard, SidecarCard, TranscriptSegment, compact_text, new_id
from brain_sidecar.core.note_quality import NoteQualityGate
from brain_sidecar.core.notes import note_from_sidecar
from brain_sidecar.core.review_asr import create_review_asr_backend
from brain_sidecar.core.review_models import ReviewWindowExtract
from brain_sidecar.core.session import SessionManager
from brain_sidecar.core.sidecar_cards import create_sidecar_card
from brain_sidecar.core.test_mode import SUPPORTED_AUDIO_EXTENSIONS
from brain_sidecar.core.transcription import clean_transcript_text
from brain_sidecar.core.web_context import WebContextCandidate


REVIEW_STEP_ORDER = ("conditioning", "transcribing", "reviewing", "evaluating", "completed")
TERMINAL_REVIEW_STATUSES = {"completed", "error", "canceled"}
CorrectionProgressCallback = Callable[[int, int, str], Awaitable[None]]
ReviewEvaluationProgressCallback = Callable[[int, int, str], Awaitable[None]]
REVIEW_SEGMENT_TARGET_WORDS = 28
REVIEW_SEGMENT_MAX_WORDS = 45
REVIEW_SEGMENT_MAX_SECONDS = 18.0
REVIEW_CORRECTION_MAX_SEGMENTS = 6
REVIEW_CORRECTION_MAX_WORDS = 700
REVIEW_EVALUATION_WINDOW_SIZE = 18
REVIEW_EVALUATION_WINDOW_OVERLAP = 3
REVIEW_EVALUATION_MAX_CARDS = 18
REVIEW_SUMMARY_MAX_EXTRACTS = 14
REVIEW_TIMELINE_MAX_GROUPS = 60
REVIEW_CONTEXT_MAX_WEB_QUERIES = 2
REVIEW_CONTEXT_MAX_REFERENCE_HITS = 5
REVIEW_CONTEXT_MAX_ITEMS = 10
REVIEW_USEFULNESS_PASS_SCORE = 0.72
REVIEW_STANDARD_ID = "energy_consultant_v1"
REVIEW_PROJECT_TERM_RE = re.compile(r"\b(?:BESS|PGE|RFI|EPC|SPCC|gen-tie|transformer|breaker|collector|substation|permitting|utility|interconnection|one-line|load flow|short circuit|tariff|demand response|commissioning|scope 2|carbon accounting|renewable|procurement|prototype|design|requirements|budget|schedule|vendor|supplier|client|customer)\b", re.IGNORECASE)
REVIEW_EE_ANALYSIS_RE = re.compile(r"\b(?:arc flash|breaker|capacity|coordination|CT ratio|demand charge|DER|distribution|fault current|feeder|harmonic|IEEE|interconnection|inverter|load flow|NERC|one-line|power factor|protection|reactive|relay|resource adequacy|short circuit|single-line|substation|tariff|transformer|transmission|utility bill|voltage)\b", re.IGNORECASE)
REVIEW_ACTION_CUE_RE = re.compile(r"\b(?:need to|we need|we should|have to|will send|send|set up|schedule|follow up|review|confirm|draft|provide|request|coordinate|own|owns|delegate|resolve|respond|track|update)\b", re.IGNORECASE)
REVIEW_ABSTRACT_WORKSTREAM_RE = re.compile(r"\b(?:assumption|basis|collector|design|handoff|interface|requirement|routing|scope|support|workstream)\b", re.IGNORECASE)
REVIEW_STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "being", "between", "could",
    "doing", "going", "having", "meeting", "really", "should", "their", "there", "these",
    "things", "think", "those", "through", "today", "under", "where", "which", "while",
    "would", "yeah", "okay", "right", "maybe", "something", "actually", "basically",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
}
REVIEW_CORRECTION_GLOSSARY = {
    "BESS": "battery energy storage system; often misheard as best",
    "gen-tie": "generation tie line; often misheard as gentie or gen type",
    "PGE": "Portland General Electric; often transcribed as p g e",
    "RFI": "request for information; often transcribed as RF I",
    "one-line": "electrical single-line diagram",
    "breaker failure": "protection scheme term",
    "load flow": "power-system study; often misheard as low flow",
    "short circuit": "power-system study",
}


@dataclass
class ReviewStep:
    key: str
    label: str
    status: str = "pending"
    message: str = ""
    progress: int = 0
    detail: str = ""
    current: int | None = None
    total: int | None = None
    unit: str = ""
    started_at: float | None = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "label": self.label,
            "status": self.status,
            "message": self.message,
            "progress": self.progress,
        }
        if self.detail:
            payload["detail"] = self.detail
        if self.current is not None:
            payload["current"] = self.current
        if self.total is not None:
            payload["total"] = self.total
        if self.unit:
            payload["unit"] = self.unit
        if self.started_at is not None:
            payload["elapsed_seconds"] = max(0.0, self.updated_at - self.started_at)
        return payload


@dataclass
class ReviewJob:
    id: str
    title: str
    filename: str
    save_result: bool
    source_path: Path
    conditioned_path: Path
    job_dir: Path
    status: str = "queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    duration_seconds: float | None = None
    session_id: str | None = None
    clean_segments: list[dict[str, Any]] = field(default_factory=list)
    meeting_cards: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] | None = None
    raw_segment_count: int = 0
    corrected_segment_count: int = 0
    asr_backend: str | None = None
    asr_model: str | None = None
    steps: dict[str, ReviewStep] = field(default_factory=lambda: {
        "conditioning": ReviewStep("conditioning", "Conditioning"),
        "transcribing": ReviewStep("transcribing", "High-accuracy ASR"),
        "reviewing": ReviewStep("reviewing", "Segment review"),
        "evaluating": ReviewStep("evaluating", "Meeting output"),
        "completed": ReviewStep("completed", "Done"),
    })

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "title": self.title,
            "filename": self.filename,
            "status": self.status,
            "error": self.error,
            "save_result": self.save_result,
            "session_id": self.session_id if self.save_result else None,
            "duration_seconds": self.duration_seconds,
            "raw_segment_count": self.raw_segment_count,
            "corrected_segment_count": self.corrected_segment_count,
            "asr_backend": self.asr_backend,
            "asr_model": self.asr_model,
            "progress_percent": _job_progress_percent(self),
            "steps": [self.steps[key].to_dict() for key in REVIEW_STEP_ORDER],
            "clean_segments": self.clean_segments,
            "meeting_cards": self.meeting_cards,
            "summary": self.summary,
            "raw_audio_retained": False,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "elapsed_seconds": max(0.0, self.updated_at - self.created_at),
            "active": self.status not in TERMINAL_REVIEW_STATUSES,
        }


class ReviewService:
    def __init__(self, settings: Settings, manager: SessionManager) -> None:
        self.settings = settings
        self.manager = manager
        self.review_asr = create_review_asr_backend(settings, manager.transcriber)
        self.jobs: dict[str, dict[str, Any]] = {}
        self._worker_task: asyncio.Task | None = None
        self._wake_event: asyncio.Event | None = None
        self._stopping = False
        self._lock = asyncio.Lock()

    async def start_worker(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._stopping = False
        self._wake_event = asyncio.Event()
        await asyncio.to_thread(self._recover_jobs)
        self._worker_task = asyncio.create_task(self._run_worker())
        self._wake_worker()

    async def stop_worker(self) -> None:
        self._stopping = True
        self._wake_worker()
        task = self._worker_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._worker_task = None

    def _recover_jobs(self) -> None:
        for job in self.manager.storage.list_review_jobs(limit=200):
            if str(job["status"]).startswith("running_"):
                path = Path(str(job.get("temporary_audio_path") or ""))
                if path.exists():
                    self.manager.storage.update_review_job_progress(
                        job["id"],
                        status="queued",
                        phase="queued",
                        progress_pct=0,
                        message="Recovered after restart; waiting in Review queue.",
                    )
                else:
                    self.manager.storage.error_review_job(job["id"], "Review job lost its temporary audio during restart.")

    def _wake_worker(self) -> None:
        if self._wake_event is not None:
            self._wake_event.set()

    async def create_job(
        self,
        *,
        filename: str,
        data: bytes,
        save_result: bool = False,
        title: str | None = None,
    ) -> dict[str, Any]:
        safe_name = _safe_review_filename(filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise ValueError(f"Unsupported audio extension {suffix!r}. Supported: {supported}")
        if not data:
            raise ValueError("Input file is empty.")

        job_id = new_id("review")
        job_root = self.settings.data_dir / "review-spool"
        job_dir = job_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        source_path = job_dir / f"source{suffix}"
        source_path.write_bytes(data)

        clean_title = compact_text(title, limit=180) or f"Review {Path(safe_name).stem or 'audio'}"
        job = await asyncio.to_thread(
            self.manager.storage.create_review_job,
            job_id=job_id,
            source="upload",
            title=clean_title,
            status="queued",
            validation_status="pending",
            source_filename=safe_name,
            temporary_audio_path=source_path,
            message="Queued for Review.",
        )
        async with self._lock:
            self.jobs[job_id] = job
        await self.start_worker()
        self._wake_worker()
        return job

    async def create_live_handoff_job(
        self,
        *,
        live_id: str,
        title: str,
        audio_path: Path,
        filename: str | None = None,
    ) -> dict[str, Any]:
        if not audio_path.exists():
            raise ValueError("Live audio handoff is missing its temporary audio file.")
        job_id = new_id("review")
        job_root = self.settings.data_dir / "review-spool"
        job_dir = job_root / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        suffix = audio_path.suffix or ".wav"
        source_path = job_dir / f"source{suffix}"
        shutil.move(str(audio_path), source_path)
        try:
            if audio_path.parent.exists() and not any(audio_path.parent.iterdir()):
                audio_path.parent.rmdir()
        except OSError:
            pass
        safe_name = _safe_review_filename(filename or f"{live_id}.wav")
        clean_title = compact_text(title, limit=180) or "Live meeting review"
        job = await asyncio.to_thread(
            self.manager.storage.create_review_job,
            job_id=job_id,
            source="live",
            title=clean_title,
            status="queued",
            validation_status="pending",
            source_filename=safe_name,
            temporary_audio_path=source_path,
            live_id=live_id,
            message="Live audio queued for Review validation.",
        )
        async with self._lock:
            self.jobs[job_id] = job
        await self.start_worker()
        self._wake_worker()
        return job

    async def get_job(self, job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(self.manager.storage.review_job, job_id)
        async with self._lock:
            self.jobs[job_id] = job
        return job

    async def list_jobs(self, *, limit: int = 50, status: str | None = None) -> list[dict[str, Any]]:
        jobs = await asyncio.to_thread(self.manager.storage.list_review_jobs, limit=limit, status=status)
        async with self._lock:
            for job in jobs:
                self.jobs[job["id"]] = job
        return jobs

    async def latest_job(self) -> dict[str, Any] | None:
        job = await asyncio.to_thread(self.manager.storage.latest_review_job)
        if job is not None:
            async with self._lock:
                self.jobs[job["id"]] = job
        return job

    async def cancel_job(self, job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(self.manager.storage.cancel_review_job, job_id)
        await self._delete_job_audio(job)
        self._wake_worker()
        return await self.get_job(job_id)

    async def discard_job(self, job_id: str) -> dict[str, Any]:
        job = await asyncio.to_thread(self.manager.storage.discard_review_job, job_id)
        await self._delete_job_audio(job)
        return await self.get_job(job_id)

    async def approve_job(self, job_id: str) -> dict[str, Any]:
        job = await self.get_job(job_id)
        if job["status"] == "approved":
            return job
        if job["status"] != "completed_awaiting_validation":
            raise ValueError(f"Cannot approve review job in status {job['status']}.")
        result = job.get("result") or {}
        session_id = await self._persist_approved_result(job, result)
        approved = await asyncio.to_thread(self.manager.storage.approve_review_job, job_id, session_id)
        await self._delete_job_audio(approved)
        return await self.get_job(job_id)

    async def regenerate_session_review(self, session_id: str) -> dict[str, Any]:
        detail = await asyncio.to_thread(self.manager.storage.session_detail, session_id)
        raw_segments = detail.get("transcript_segments") or []
        segments: list[TranscriptSegment] = []
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, dict):
                continue
            segment = TranscriptSegment(
                id=str(raw_segment.get("id") or new_id("seg")),
                session_id=session_id,
                start_s=_safe_float(raw_segment.get("start_s"), default=0.0),
                end_s=_safe_float(raw_segment.get("end_s"), default=_safe_float(raw_segment.get("start_s"), default=0.0) + 0.1),
                text=clean_transcript_text(raw_segment.get("text") or ""),
                is_final=True,
                created_at=_safe_float(raw_segment.get("created_at"), default=time.time()),
                speaker_role=raw_segment.get("speaker_role"),
                speaker_label=raw_segment.get("speaker_label"),
                speaker_confidence=raw_segment.get("speaker_confidence"),
                speaker_match_score=raw_segment.get("speaker_match_score"),
                speaker_match_reason=raw_segment.get("speaker_match_reason"),
                speaker_low_confidence=raw_segment.get("speaker_low_confidence"),
                diarization_speaker_id=raw_segment.get("diarization_speaker_id"),
                source_segment_ids=[str(item) for item in raw_segment.get("source_segment_ids") or []] or [str(raw_segment.get("id") or "")],
            )
            if segment.text:
                segments.append(segment)
        if not segments:
            raise ValueError("Cannot regenerate a Review summary without saved transcript segments.")

        review_context = await self._build_review_context(segments)
        cards, summary = await ReviewMeetingEvaluator(self.manager.ollama).evaluate_existing_transcript(
            session_id,
            segments,
            save_result=True,
            review_context=review_context,
        )
        notes = [
            _note_from_review_card(session_id, raw_card)
            for raw_card in [card.to_dict() for card in cards]
        ]
        notes = [note for note in notes if note.body]
        await asyncio.to_thread(self.manager.storage.replace_session_notes, session_id, notes)
        summary_payload = _summary_payload_for_storage(
            session_id=session_id,
            fallback_title=detail.get("title") or "Review summary",
            summary=summary,
        )
        await asyncio.to_thread(self.manager.storage.upsert_session_memory_summary, **summary_payload)
        await self._safe_approve_embedding("session_summary", session_id, summary_payload["summary"], {
            "session_id": session_id,
            "title": summary_payload["title"],
            "source_segment_ids": summary_payload["source_segment_ids"],
            "summary": True,
            "regenerated": True,
        })
        return await asyncio.to_thread(self.manager.storage.session_detail, session_id)

    async def _run_worker(self) -> None:
        assert self._wake_event is not None
        while not self._stopping:
            if self._live_capture_active():
                await asyncio.sleep(1.0)
                continue
            job = await asyncio.to_thread(self.manager.storage.claim_next_review_job)
            if job is None:
                self._wake_event.clear()
                try:
                    await asyncio.wait_for(self._wake_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await self.process_job(job["id"])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await asyncio.to_thread(self.manager.storage.error_review_job, job["id"], str(exc))
                await self._delete_job_audio(await self.get_job(job["id"]))
                await self._unload_review_asr()

    def _live_capture_active(self) -> bool:
        checker = getattr(self.manager, "has_active_capture", None)
        if callable(checker):
            return bool(checker())
        return False

    async def process_job(self, job_id: str) -> None:
        job = await self.get_job(job_id)
        source_path = Path(str(job.get("temporary_audio_path") or ""))
        if not source_path.exists():
            await asyncio.to_thread(self.manager.storage.error_review_job, job_id, "Temporary Review audio is missing.")
            return
        conditioned_path = source_path.parent / "conditioned.wav"
        try:
            await self._update_progress(job_id, "running_asr", "conditioning", 8, "Converting audio for Review ASR.")
            await asyncio.to_thread(self._condition_audio, source_path, conditioned_path)
            duration_seconds = await asyncio.to_thread(_wav_duration_seconds, conditioned_path)

            await self._update_progress(job_id, "running_asr", "asr", 14, "Preparing high-accuracy Review ASR.")
            result = await self.review_asr.transcribe_file(
                conditioned_path,
                initial_prompt=self.settings.asr_initial_prompt,
                progress=lambda message, percent: self._set_step(
                    job_id,
                    "running_asr",
                    "asr",
                    message,
                    progress=14 + round(40 * (percent / 100.0)),
                ),
            )
            session_id = job_id
            asr_segments = _segments_from_result(session_id, result)
            raw_segments = ReviewTranscriptSegmenter().segment(
                asr_segments,
                overlap_seconds=self.settings.review_asr_chunk_overlap_seconds,
            )
            raw_segments = await self._label_review_speakers(job_id, conditioned_path, raw_segments)
            await self._unload_review_asr()
            await self._update_progress(job_id, "running_cards", "correction", 58, f"{len(raw_segments)} transcript segments. Correcting ASR errors.")

            async def review_progress(current: int, total: int, message: str) -> None:
                progress = 58 + round(20 * (current / max(1, total)))
                await self._set_step(
                    job_id,
                    "running_cards",
                    "correction",
                    message,
                    progress=progress,
                )

            clean_segments = await TranscriptReviewCorrector(
                self.manager.ollama,
                batch_size=self.settings.review_correction_batch_size,
                concurrency=self.settings.review_correction_concurrency,
            ).correct(raw_segments, progress=review_progress)
            corrected_segment_count = len([segment for segment in clean_segments if segment.text.strip()])

            await self._update_progress(job_id, "running_cards", "cards", 80, "Evaluating corrected transcript as a meeting.")
            cards, summary = await self._evaluate_meeting(job, clean_segments)
            diagnostics = {
                "asr_backend": getattr(self.review_asr, "backend_name", self.settings.review_asr_backend),
                "asr_model": result.model or getattr(self.review_asr, "model_size", None) or self.settings.review_asr_model,
                "duration_seconds": duration_seconds,
                "raw_segment_count": len(raw_segments),
                "corrected_segment_count": corrected_segment_count,
                "speaker_identity": _speaker_identity_diagnostics(clean_segments, self.manager.speaker_identity.status()),
                "card_count": len(cards),
                "evidence_quote_coverage": _coverage_ratio([card.to_dict() for card in cards], "evidence_quote"),
                "source_id_coverage": _coverage_ratio([card.to_dict() for card in cards], "source_segment_ids"),
            }
            diagnostics.update(_card_usefulness_diagnostics([card.to_dict() for card in cards]))
            summary_diagnostics = summary.get("diagnostics") if isinstance(summary.get("diagnostics"), dict) else {}
            diagnostics.update(summary_diagnostics)
            review_result = {
                "clean_transcript": [segment.to_dict() for segment in clean_segments],
                "meeting_cards": [card.to_dict() for card in cards],
                "summary": summary,
                "diagnostics": diagnostics,
            }
            await asyncio.to_thread(self.manager.storage.complete_review_job, job_id, review_result)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await asyncio.to_thread(self.manager.storage.error_review_job, job_id, str(exc))
            await self._delete_job_audio(await self.get_job(job_id))
        finally:
            await self._unload_review_asr()

    async def _evaluate_meeting(
        self,
        job: dict[str, Any],
        segments: list[TranscriptSegment],
    ) -> tuple[list[SidecarCard], dict[str, Any]]:
        async def evaluation_progress(current: int, total: int, message: str) -> None:
            await self._set_step(
                job["id"],
                "running_summary",
                "summary",
                message,
                progress=80 + round(18 * (current / max(1, total))),
            )

        review_context = await self._build_review_context(segments)
        accepted_cards, summary = await ReviewMeetingEvaluator(self.manager.ollama).evaluate(
            job["id"],
            segments,
            save_result=False,
            progress=evaluation_progress,
            review_context=review_context,
        )

        return accepted_cards, summary

    async def _label_review_speakers(
        self,
        job_id: str,
        conditioned_path: Path,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        if not segments:
            return []
        try:
            return await asyncio.to_thread(self._label_review_speakers_sync, job_id, conditioned_path, segments)
        except Exception:
            return segments

    def _label_review_speakers_sync(
        self,
        job_id: str,
        conditioned_path: Path,
        segments: list[TranscriptSegment],
    ) -> list[TranscriptSegment]:
        with wave.open(str(conditioned_path), "rb") as wav:
            sample_rate = wav.getframerate()
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            frames = wav.readframes(wav.getnframes())
        if sample_rate != self.settings.audio_sample_rate or channels != 1 or width != 2:
            return segments
        labeled: list[TranscriptSegment] = []
        bytes_per_second = sample_rate * width
        for segment in segments:
            start = max(0, int(segment.start_s * bytes_per_second))
            end = min(len(frames), int(segment.end_s * bytes_per_second))
            start -= start % width
            end -= end % width
            pcm = frames[start:end]
            result = self.manager.speaker_identity.label_segment(
                session_id=job_id,
                segment_id=segment.id,
                pcm=pcm,
                start_ms=int(segment.start_s * 1000),
                end_ms=int(segment.end_s * 1000),
                transcript_text=segment.text,
                persist=False,
            )
            payload = result.transcript_payload()
            labeled.append(_segment_with_speaker_payload(segment, payload))
        return labeled

    async def _build_review_context(self, segments: list[TranscriptSegment]) -> dict[str, Any]:
        energy_context = self._build_energy_review_context(segments)
        ee_hits = await asyncio.to_thread(self._build_ee_reference_context, segments, energy_context)
        web_context = await self._build_web_review_context(segments)
        diagnostics = {
            "energy_lens": "included" if energy_context.get("active") else "inactive",
            "ee_reference_hits": len(ee_hits),
            "web_context_hits": len(web_context.get("hits") or []),
            "web_context_skip_reason": web_context.get("skip_reason"),
        }
        return {
            "energy_lens": energy_context,
            "ee_reference_hits": ee_hits,
            "web_context_hits": web_context.get("hits") or [],
            "context_diagnostics": diagnostics,
        }

    def _build_energy_review_context(self, segments: list[TranscriptSegment]) -> dict[str, Any]:
        detector = getattr(self.manager, "energy_detector", None)
        if detector is None:
            return {"active": False, "skip_reason": "energy_detector_unavailable"}
        frames: list[dict[str, Any]] = []
        for window in _segment_windows(segments, size=12, overlap=3):
            try:
                frame = detector.detect(window, save_transcript=False)
            except Exception:
                continue
            if getattr(frame, "active", False):
                frames.append(frame.to_dict())
        if not frames:
            return {"active": False, "skip_reason": "no_energy_lens_hits"}
        categories = _merge_scored_items([item for frame in frames for item in frame.get("top_categories", [])], key="category")
        keywords = _merge_scored_items([item for frame in frames for item in frame.get("top_keywords", [])], key="phrase")
        evidence_ids = _dedupe_strings([source_id for frame in frames for source_id in frame.get("evidence_segment_ids", [])])[:24]
        evidence_quotes = _dedupe_compact([str(frame.get("evidence_quote") or "") for frame in frames], limit=4)
        confidence = max((str(frame.get("confidence") or "low") for frame in frames), key=_energy_confidence_rank)
        return {
            "active": True,
            "summary_label": _energy_summary_label(categories),
            "confidence": confidence,
            "categories": categories[:6],
            "keywords": keywords[:10],
            "source_segment_ids": evidence_ids,
            "evidence_quote": compact_text(" ".join(evidence_quotes), limit=420),
        }

    def _build_ee_reference_context(
        self,
        segments: list[TranscriptSegment],
        energy_context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        query_terms = _reference_query_terms(segments, energy_context)
        hits: dict[str, dict[str, Any]] = {}
        for term in query_terms:
            try:
                chunks = self.manager.storage.document_chunks(query=term, limit=12)
            except Exception:
                chunks = []
            for chunk in chunks:
                source_path = str(chunk.get("source_path") or "")
                if not _looks_like_ee_reference(source_path, str(chunk.get("text") or "")):
                    continue
                hit_id = str(chunk.get("id") or f"{source_path}:{chunk.get('chunk_index', 0)}")
                text = compact_text(chunk.get("text") or "", limit=520)
                current = hits.get(hit_id)
                score = (current.get("score", 0) if current else 0) + 1
                hits[hit_id] = {
                    "kind": "ee_reference",
                    "title": Path(source_path).name or "Electrical reference",
                    "body": text,
                    "source_path": source_path,
                    "citation": source_path,
                    "query": term,
                    "score": score,
                }
        ordered = sorted(hits.values(), key=lambda item: (-float(item.get("score", 0)), str(item.get("title", ""))))
        return ordered[:REVIEW_CONTEXT_MAX_REFERENCE_HITS]

    async def _build_web_review_context(self, segments: list[TranscriptSegment]) -> dict[str, Any]:
        if not self.settings.web_context_enabled:
            return {"hits": [], "skip_reason": "web_disabled"}
        if not self.settings.brave_search_api_key.strip():
            return {"hits": [], "skip_reason": "brave_key_missing"}
        candidates = _web_context_candidates_for_review(self.manager.web_trigger_detector, segments)
        if not candidates:
            return {"hits": [], "skip_reason": "no_public_current_questions"}
        hits: list[dict[str, Any]] = []
        for candidate in candidates[:REVIEW_CONTEXT_MAX_WEB_QUERIES]:
            try:
                results = await self.manager.web_search.search(candidate.query, freshness=candidate.freshness)
            except Exception:
                results = []
            for result in results[:3]:
                hits.append(
                    {
                        "kind": "brave_web",
                        "title": result.title,
                        "body": result.description,
                        "url": result.url,
                        "citation": result.url,
                        "query": candidate.query,
                        "source_segment_ids": candidate.source_segment_ids,
                    }
                )
        return {"hits": hits[:REVIEW_CONTEXT_MAX_ITEMS], "skip_reason": None if hits else "no_results"}

    def _accepted_cards(
        self,
        cards: list[SidecarCard],
        segments: list[TranscriptSegment],
    ) -> list[SidecarCard]:
        if not cards:
            return []
        max_cards = self.settings.sidecar_max_cards_per_generation_pass
        if not self.settings.sidecar_quality_gate_enabled:
            return cards[:max_cards]
        try:
            review = ContractCriticAgent(NoteQualityGate()).review(
                cards,
                segments,
                self.manager.speaker_identity.status(),
                max_cards=max_cards,
            )
            accepted = list(review.accepted_cards)
            if not any(card.category == "action" for card in accepted):
                fallback_actions = _evidence_backed_action_fallback(cards, segments, max_cards=max(1, max_cards - len(accepted)))
                accepted = [*fallback_actions, *accepted]
            return _dedupe_cards(accepted)[:max_cards]
        except Exception:
            diagnostics_for_cards(cards, accepted_count=min(len(cards), max_cards), suppressed_count=max(0, len(cards) - max_cards))
            return cards[:max_cards]

    async def _set_step(
        self,
        job_id: str,
        status: str,
        phase: str,
        message: str,
        *,
        progress: int | None = None,
    ) -> None:
        await self._update_progress(job_id, status, phase, progress or 0, message)

    async def _update_progress(
        self,
        job_id: str,
        status: str,
        phase: str,
        progress_pct: float,
        message: str,
    ) -> None:
        job = await asyncio.to_thread(
            self.manager.storage.update_review_job_progress,
            job_id,
            status=status,
            phase=phase,
            progress_pct=progress_pct,
            message=message,
        )
        async with self._lock:
            self.jobs[job_id] = job

    async def _persist_approved_result(self, job: dict[str, Any], result: dict[str, Any]) -> str:
        record = await asyncio.to_thread(self.manager.storage.create_session, job["title"])
        session_id = record.id
        await asyncio.to_thread(self.manager.storage.set_session_status, session_id, "running", save_transcript=True)
        segments: list[TranscriptSegment] = []
        for raw_segment in result.get("clean_transcript") or []:
            if not isinstance(raw_segment, dict):
                continue
            segment = TranscriptSegment(
                id=str(raw_segment.get("id") or new_id("seg")),
                session_id=session_id,
                start_s=_safe_float(raw_segment.get("start_s"), default=0.0),
                end_s=_safe_float(raw_segment.get("end_s"), default=_safe_float(raw_segment.get("start_s"), default=0.0) + 0.1),
                text=clean_transcript_text(raw_segment.get("text") or ""),
                is_final=True,
                created_at=_safe_float(raw_segment.get("created_at"), default=time.time()),
                speaker_role=raw_segment.get("speaker_role"),
                speaker_label=raw_segment.get("speaker_label"),
                speaker_confidence=raw_segment.get("speaker_confidence"),
                speaker_match_score=raw_segment.get("speaker_match_score"),
                speaker_match_reason=raw_segment.get("speaker_match_reason"),
                speaker_low_confidence=raw_segment.get("speaker_low_confidence"),
                diarization_speaker_id=raw_segment.get("diarization_speaker_id"),
                source_segment_ids=list(raw_segment.get("source_segment_ids") or []),
            )
            if segment.text:
                segments.append(segment)
                await asyncio.to_thread(self.manager.storage.upsert_transcript_segment, segment)
                if segment.speaker_role:
                    await asyncio.to_thread(
                        self.manager.storage.add_diarization_segment,
                        session_id=session_id,
                        segment_id=segment.id,
                        start_ms=int(segment.start_s * 1000),
                        end_ms=int(segment.end_s * 1000),
                        diarization_speaker_id=segment.diarization_speaker_id,
                        display_speaker_label=segment.speaker_label,
                        matched_profile_id="self_bp" if segment.speaker_role == "user" else None,
                        match_confidence=segment.speaker_confidence,
                        match_score=segment.speaker_match_score,
                        transcript_text=segment.text,
                        is_overlap=False,
                        finalized=True,
                        metadata={
                            "speaker_role": segment.speaker_role,
                            "low_confidence": bool(segment.speaker_low_confidence),
                            "reason": segment.speaker_match_reason,
                            "source": "review_approval",
                        },
                    )
                await self._safe_approve_embedding("transcript_segment", segment.id, segment.text, {
                    "session_id": session_id,
                    "start_s": segment.start_s,
                    "end_s": segment.end_s,
                })
        for raw_card in result.get("meeting_cards") or []:
            if not isinstance(raw_card, dict):
                continue
            note = _note_from_review_card(session_id, raw_card)
            if note.body:
                await asyncio.to_thread(self.manager.storage.add_note, note)
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        summary_payload = _summary_payload_for_storage(
            session_id=session_id,
            fallback_title=job["title"],
            summary=summary,
        )
        await asyncio.to_thread(self.manager.storage.upsert_session_memory_summary, **summary_payload)
        await self._safe_approve_embedding("session_summary", session_id, summary_payload["summary"], {
            "session_id": session_id,
            "title": summary_payload["title"],
            "source_segment_ids": summary_payload["source_segment_ids"],
            "summary": True,
        })
        await asyncio.to_thread(self.manager.storage.set_session_status, session_id, "stopped", ended_at=time.time(), save_transcript=True)
        return session_id

    async def _safe_approve_embedding(self, source_type: str, source_id: str, text: str, metadata: dict[str, Any]) -> None:
        try:
            await self.manager.recall.add_text(source_type, source_id, text, metadata)
        except Exception:
            return

    def _condition_audio(self, source_path: Path, conditioned_path: Path) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
            "-ac",
            "1",
            "-ar",
            str(self.settings.audio_sample_rate),
            "-sample_fmt",
            "s16",
            str(conditioned_path),
        ]
        try:
            result = subprocess.run(command, capture_output=True, check=False, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for Review audio conditioning.") from exc
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg failed to condition the audio file.")

    async def _delete_job_audio(self, job: dict[str, Any]) -> None:
        path = Path(str(job.get("temporary_audio_path") or ""))
        try:
            await asyncio.to_thread(delete_audio_file, path)
        except Exception as exc:
            await asyncio.to_thread(self.manager.storage.mark_review_audio_cleanup_error, job["id"], str(exc))
            return
        await asyncio.to_thread(self.manager.storage.mark_review_audio_deleted, job["id"])

    async def _unload_review_asr(self) -> None:
        unload = getattr(self.review_asr, "unload", None)
        if unload is None:
            return
        try:
            await unload()
        except Exception:
            return


class ReviewTranscriptSegmenter:
    def __init__(
        self,
        *,
        target_words: int = REVIEW_SEGMENT_TARGET_WORDS,
        max_words: int = REVIEW_SEGMENT_MAX_WORDS,
        max_seconds: float = REVIEW_SEGMENT_MAX_SECONDS,
    ) -> None:
        self.target_words = max(8, target_words)
        self.max_words = max(self.target_words, max_words)
        self.max_seconds = max(4.0, max_seconds)

    def segment(self, segments: list[TranscriptSegment], *, overlap_seconds: float) -> list[TranscriptSegment]:
        result: list[TranscriptSegment] = []
        for source in sorted(segments, key=lambda item: (item.start_s, item.end_s, item.id)):
            text = clean_transcript_text(source.text)
            if not text:
                continue
            if result and source.start_s <= result[-1].end_s + max(0.5, overlap_seconds):
                text = _trim_leading_text_overlap(result[-1].text, text) or text
            for start_s, end_s, part in self._split_segment_text(source, text):
                if result and _is_duplicate_segment_text(result[-1].text, part):
                    continue
                segment_id = new_id("seg")
                result.append(
                    TranscriptSegment(
                        id=segment_id,
                        session_id=source.session_id,
                        start_s=start_s,
                        end_s=end_s,
                        text=part,
                        is_final=True,
                        source_segment_ids=_dedupe_strings([segment_id, source.id, *source.source_segment_ids]),
                    )
                )
        return result

    def _split_segment_text(
        self,
        segment: TranscriptSegment,
        text: str,
    ) -> list[tuple[float, float, str]]:
        words = text.split()
        if not words:
            return []
        duration = max(0.1, segment.end_s - segment.start_s)
        expected_duration = max(duration, min(45.0, len(words) * 0.35)) if len(words) > 12 else duration
        should_split = (
            len(words) > self.max_words
            or expected_duration > self.max_seconds
        )
        if not should_split:
            return [(segment.start_s, max(segment.start_s + 0.1, segment.start_s + expected_duration), text)]

        groups = _readable_word_groups(text, target_words=self.target_words, max_words=self.max_words)
        total_words = sum(max(1, len(group.split())) for group in groups)
        cursor_words = 0
        parts: list[tuple[float, float, str]] = []
        for index, group in enumerate(groups):
            count = max(1, len(group.split()))
            start = segment.start_s + expected_duration * (cursor_words / max(1, total_words))
            cursor_words += count
            end = segment.start_s + expected_duration * (cursor_words / max(1, total_words))
            if index == len(groups) - 1:
                end = segment.start_s + expected_duration
            parts.append((start, max(start + 0.1, end), group))
        return parts


class TranscriptReviewCorrector:
    def __init__(self, ollama, *, batch_size: int = 12, concurrency: int = 2) -> None:
        self.ollama = ollama
        self.batch_size = max(1, batch_size)
        self.concurrency = max(1, concurrency)

    async def correct(
        self,
        segments: list[TranscriptSegment],
        *,
        progress: CorrectionProgressCallback | None = None,
    ) -> list[TranscriptSegment]:
        if not segments:
            return []
        batches = _correction_batches(
            segments,
            max_segments=min(self.batch_size, REVIEW_CORRECTION_MAX_SEGMENTS),
            max_words=REVIEW_CORRECTION_MAX_WORDS,
        )
        semaphore = asyncio.Semaphore(self.concurrency)
        completed = 0

        async def run_batch(index: int, batch: list[TranscriptSegment]) -> tuple[int, list[TranscriptSegment]]:
            async with semaphore:
                previous_context = _neighbor_context(_segments_before_batch(segments, batch), tail=True)
                next_context = _neighbor_context(_segments_after_batch(segments, batch), tail=False)
                return index, await self._correct_batch(
                    batch,
                    previous_context=previous_context,
                    next_context=next_context,
                )

        tasks = [asyncio.create_task(run_batch(index, batch)) for index, batch in enumerate(batches)]
        results: list[tuple[int, list[TranscriptSegment]]] = []
        for task in asyncio.as_completed(tasks):
            result = await task
            completed += 1
            if progress is not None:
                await progress(completed, len(batches), f"Corrected segment batch {completed}/{len(batches)}.")
            results.append(result)
        corrected: list[TranscriptSegment] = []
        for _, batch_result in sorted(results, key=lambda item: item[0]):
            corrected.extend(batch_result)
        return corrected

    async def _correct_batch(
        self,
        segments: list[TranscriptSegment],
        *,
        previous_context: str = "",
        next_context: str = "",
    ) -> list[TranscriptSegment]:
        system = (
            "You are a conservative transcript correction assistant for engineering and project meetings. "
            "Fix obvious ASR errors, spacing errors, missing punctuation, and capitalization only when clear. "
            "Preserve the speaker's wording, uncertainty, meaning, order, and timestamps. "
            "Use neighboring context only to resolve obvious transcription confusions. "
            "Do not summarize, extract action items, add facts, remove substantive content, or make the transcript more polished than the audio supports."
        )
        user = json.dumps(
            {
                "task": "Return corrected text for each segment id. If no correction is justified, return the original text.",
                "glossary": REVIEW_CORRECTION_GLOSSARY,
                "common_repairs": [
                    "RF I -> RFI",
                    "p g e -> PGE",
                    "best collection substation -> BESS collection substation when the project context is storage",
                    "gentie/gentie line/gen type line -> gen-tie line when discussing interconnection lines",
                    "low flow study -> load flow study when paired with short circuit study",
                ],
                "context_before": previous_context,
                "context_after": next_context,
                "segments": [
                    {
                        "id": segment.id,
                        "start_s": segment.start_s,
                        "end_s": segment.end_s,
                        "text": segment.text,
                    }
                    for segment in segments
                ],
                "return_shape": {"segments": [{"id": "same id", "text": "corrected transcript text"}]},
            },
            ensure_ascii=False,
        )
        try:
            content = await self.ollama.chat(system, user, format_json=True)
            by_id = _corrected_text_by_id(content)
        except Exception:
            by_id = {}

        return [
            _segment_with_text(segment, by_id.get(segment.id) or segment.text)
            for segment in segments
        ]


class ReviewMeetingEvaluator:
    def __init__(self, ollama) -> None:
        self.ollama = ollama

    async def evaluate(
        self,
        session_id: str,
        segments: list[TranscriptSegment],
        *,
        save_result: bool,
        progress: ReviewEvaluationProgressCallback | None = None,
        review_context: dict[str, Any] | None = None,
    ) -> tuple[list[SidecarCard], dict[str, Any]]:
        windows = _segment_windows(
            segments,
            size=REVIEW_EVALUATION_WINDOW_SIZE,
            overlap=REVIEW_EVALUATION_WINDOW_OVERLAP,
        )
        extracts: list[ReviewWindowExtract] = []
        for index, window in enumerate(windows, start=1):
            if progress is not None:
                await progress(index, max(1, len(windows) + 1), f"Evaluating transcript window {index}/{len(windows)}.")
            extracts.append(await self._extract_window(session_id, window, save_result=save_result))

        cards = _dedupe_cards([card for extract in extracts for card in extract.cards])[:REVIEW_EVALUATION_MAX_CARDS]
        summary = await self._aggregate_summary(session_id, segments, extracts, cards=cards, review_context=review_context)
        if not cards:
            cards = [_review_incomplete_card(session_id, segments, save_result=save_result)]
        if progress is not None:
            await progress(max(1, len(windows) + 1), max(1, len(windows) + 1), "Aggregated structured meeting review.")
        return cards, summary

    async def evaluate_existing_transcript(
        self,
        session_id: str,
        segments: list[TranscriptSegment],
        *,
        save_result: bool,
        review_context: dict[str, Any] | None = None,
    ) -> tuple[list[SidecarCard], dict[str, Any]]:
        extracts = _deterministic_review_extracts(segments)
        summary = await self._aggregate_summary(session_id, segments, extracts, cards=[], review_context=review_context)
        cards = _cards_from_summary(summary, session_id, segments, save_result=save_result)
        if not cards:
            cards = [_review_incomplete_card(session_id, segments, save_result=save_result)]
        return cards, summary

    async def _extract_window(
        self,
        session_id: str,
        segments: list[TranscriptSegment],
        *,
        save_result: bool,
    ) -> ReviewWindowExtract:
        system = (
            "You are a post-call review analyst for BP. Extract useful meeting intelligence from "
            "a corrected transcript window. Stay grounded only in the supplied transcript source ids. "
            "Speaker labels are BP-only: BP means the user, Other speaker and Unknown speaker are not BP. "
            "Assign BP-owned actions only to BP-labeled speech or literal BP/Brandon mentions. "
            "Prefer specific energy consulting workstreams, actions, owners, decisions, risks, open questions, "
            "technical assumptions, methods, findings, recommendations, and durable notes. "
            "Write a short window_summary that says what materially happened in this slice of the meeting. "
            "Do not create generic cards, do not echo transcript chunks, and do not invent missing details."
        )
        user = json.dumps(
            {
                "task": "Extract structured meeting review items from this transcript window.",
                "transcript": _review_transcript_lines(segments),
                "return_shape": {
                    "window_summary": "one compact sentence about the useful meeting substance in this window",
                    "priority": "high|normal|low",
                    "cards": [
                        {
                            "category": "action|decision|question|risk|clarification|note",
                            "title": "short specific title",
                            "body": "one or two useful grounded sentences",
                            "why_now": "why this matters after the call",
                            "priority": "low|normal|high",
                            "confidence": 0.0,
                            "source_segment_ids": ["ids from transcript"],
                            "evidence_quote": "short supporting quote",
                            "owner": "optional",
                            "due_date": "optional",
                            "missing_info": "optional",
                            "source_type": "transcript",
                        }
                    ],
                    "summary_points": ["specific point"],
                    "topics": ["topic"],
                    "decisions": ["decision"],
                    "actions": ["action"],
                    "unresolved_questions": ["question"],
                    "risks": ["risk"],
                    "projects": ["project or workstream"],
                    "entities": ["person/project/company"],
                    "project_workstreams": [
                        {
                            "project": "project/site/client/workstream name",
                            "status": "discussed|active|blocked|watch",
                            "decisions": ["decision"],
                            "actions": ["action"],
                            "risks": ["risk/issue/dependency"],
                            "open_questions": ["question"],
                            "owners": ["owner names if explicit"],
                            "next_checkpoint": "next milestone/checkpoint if explicit",
                            "source_segment_ids": ["ids from transcript"],
                        }
                    ],
                    "technical_findings": [
                        {
                            "topic": "technical topic",
                            "question": "technical question or analysis objective",
                            "assumptions": ["assumption or data basis"],
                            "methods": ["study/model/method mentioned"],
                            "findings": ["finding or interpretation"],
                            "recommendations": ["recommendation/follow-up"],
                            "risks": ["technical risk/data gap"],
                            "reference_context": ["standard/reference/source name if explicit"],
                            "source_segment_ids": ["ids from transcript"],
                        }
                    ],
                },
            },
            ensure_ascii=False,
        )
        try:
            content = await self.ollama.chat(system, user, format_json=True)
            return _parse_review_window_extract(content, session_id, segments, save_result=save_result)
        except Exception:
            return ReviewWindowExtract(
                window_summary=_fallback_window_summary_text(segments),
                source_segment_ids=_valid_review_source_ids(segments),
            )

    async def _aggregate_summary(
        self,
        session_id: str,
        segments: list[TranscriptSegment],
        extracts: list[ReviewWindowExtract],
        *,
        cards: list[SidecarCard] | None = None,
        review_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ranked_extracts = _rank_extracts_for_summary(extracts, segments)
        ranked_cards = _rank_cards_for_summary(cards or [], segments)
        timeline = _review_timeline_outline(segments)
        important_terms = _important_transcript_terms(segments)
        workstream_candidates = _review_workstream_candidates(segments, ranked_extracts, ranked_cards)
        system = (
            "You are a post-call review aggregation assistant. Build the main meeting brief for BP from "
            "the corrected transcript timeline, accepted evidence-backed meeting cards, and transcript-grounded window facts. The summary is the "
            f"primary review product and must follow the {REVIEW_STANDARD_ID} standard: cover what the meeting was actually about across the full call, "
            "starting with what changed, what remains open, and what BP should do next. "
            "Speaker labels are BP-only: BP means the user; Other speaker and Unknown speaker are not BP. "
            "Only assign first-person commitments to BP when the segment is BP-labeled or names BP/Brandon. "
            "For energy-consulting meetings with multiple projects, clients, sites, or technical threads, separate project workstreams instead of merging owners/actions across projects. "
            "Use workstream_candidates as a preservation checklist: every candidate with its own action, owner, risk, decision, question, or technical basis should survive as a separate project_workstream with its own source ids and substantive payload. "
            "Abstract terms such as scope, requirements, support, handoff, basis, assumptions, and design are useful only when tied to concrete evidence and follow-up. "
            "For deep electrical-engineering analysis, capture assumptions, methods/models/studies, findings, recommendations, data gaps, and reference context. "
            "Treat accepted_cards as the strongest structured evidence: synthesize them into the main brief instead of writing a generic transcript overview. "
            "Fold available Energy Lens, EE Index, and current public web context into reference_context only when it helps interpret transcript-grounded content; "
            "do not create technical_findings from references alone, and clearly separate transcript evidence from Energy Lens, EE Index, or Brave context in technical_findings.reference_context and coverage_notes. "
            "Do not let final small-talk or onboarding comments dominate a longer technical meeting. "
            "Stay grounded in supplied source ids and avoid transcript echo."
        )
        user = json.dumps(
            {
                "task": "Aggregate the full corrected transcript into a useful meeting brief. Use accepted_cards first, the timeline to preserve whole-call coverage, and ranked extracts for additional evidence.",
                "important_terms_seen": important_terms,
                "review_context": _review_context_for_prompt(review_context),
                "timeline": timeline,
                "accepted_cards": [_card_to_summary_payload(card) for card in ranked_cards],
                "workstream_candidates": workstream_candidates,
                "ranked_extracts": [_extract_to_payload(extract) for extract in ranked_extracts],
                "all_extract_count": len(extracts),
                "return_shape": {
                    "review_standard": REVIEW_STANDARD_ID,
                    "title": "short meeting title",
                    "summary": "one useful paragraph: meeting purpose, what changed, what remains open, and what BP should do next",
                    "portfolio_rollup": {
                        "bp_next_actions": ["BP-owned next action with owner/date when explicit"],
                        "open_loops": ["unresolved decision, risk, dependency, or question"],
                        "cross_project_dependencies": ["dependency or blocker spanning workstreams"],
                        "risk_posture": "clear|watch|blocked|decision_needed",
                        "source_segment_ids": ["ids from timeline or extracts"],
                    },
                    "key_points": ["important meeting point"],
                    "topics": ["topic"],
                    "projects": ["project or workstream"],
                    "project_workstreams": [
                        {
                            "project": "project/site/client/workstream name",
                            "client_site": "optional client/site if explicit",
                            "status": "discussed|active|blocked|watch|decision_needed",
                            "decisions": ["decision"],
                            "actions": ["action with owner/date when explicit"],
                            "risks": ["risk/issue/dependency"],
                            "open_questions": ["question"],
                            "owners": ["owner names if explicit"],
                            "next_checkpoint": "next milestone/checkpoint if explicit",
                            "source_segment_ids": ["ids from timeline or extracts"],
                        }
                    ],
                    "technical_findings": [
                        {
                            "topic": "technical topic",
                            "question": "technical question or analysis objective",
                            "assumptions": ["assumption, input, scenario, or data basis"],
                            "methods": ["study/model/calculation/standard mentioned"],
                            "findings": ["finding or interpretation"],
                            "recommendations": ["recommendation/follow-up"],
                            "risks": ["technical risk, issue, dependency, or data gap"],
                            "data_gaps": ["missing input or validation need"],
                            "reference_context": ["Energy Lens/EE Index/Brave item titles if relevant"],
                            "confidence": "low|medium|high",
                            "source_segment_ids": ["ids from timeline or extracts"],
                        }
                    ],
                    "technical_findings_rule": "Use an empty list unless the transcript itself contains energy/electrical/engineering analysis or another explicit technical evaluation.",
                    "decisions": ["decision"],
                    "actions": ["action"],
                    "unresolved_questions": ["question"],
                    "risks": ["risk"],
                    "entities": ["entity"],
                    "lessons": ["durable note"],
                    "coverage_notes": ["short note on what evidence/time range supports the summary"],
                    "reference_context": [
                        {
                            "kind": "energy_lens|ee_reference|brave_web",
                            "title": "context title",
                            "body": "what this context contributes",
                            "citation": "URL or local source path if available",
                            "source_segment_ids": ["transcript source ids if applicable"],
                        }
                    ],
                    "source_segment_ids": ["ids from extracts"],
                },
            },
            ensure_ascii=False,
        )
        try:
            content = await self.ollama.chat(system, user, format_json=True)
            summary = _parse_review_summary(content, session_id, ranked_extracts, segments, review_context=review_context)
        except Exception:
            summary = _fallback_review_summary(
                session_id,
                ranked_extracts,
                segments,
                review_context=review_context,
                workstream_candidates=workstream_candidates,
            )

        diagnostics = _summary_quality_diagnostics(
            summary,
            ranked_extracts,
            segments,
            review_context=review_context,
            workstream_candidates=workstream_candidates,
        )
        if _summary_needs_repair(diagnostics):
            repaired = await self._repair_summary(
                session_id,
                segments,
                ranked_extracts,
                timeline=timeline,
                important_terms=important_terms,
                review_context=review_context,
                cards=ranked_cards,
                workstream_candidates=workstream_candidates,
                previous_summary=summary,
                diagnostics=diagnostics,
            )
            repaired_diagnostics = _summary_quality_diagnostics(
                repaired,
                ranked_extracts,
                segments,
                review_context=review_context,
                workstream_candidates=workstream_candidates,
            )
            if not _summary_needs_repair(repaired_diagnostics):
                _set_usefulness_status(repaired_diagnostics, "repaired")
                return _finalize_review_summary(repaired, repaired_diagnostics)
            summary = _fallback_review_summary_from_cards(
                session_id,
                ranked_cards,
                ranked_extracts,
                segments,
                review_context=review_context,
                workstream_candidates=workstream_candidates,
            )
            diagnostics = _summary_quality_diagnostics(
                summary,
                ranked_extracts,
                segments,
                review_context=review_context,
                workstream_candidates=workstream_candidates,
            )

        if _summary_needs_repair(diagnostics):
            summary, diagnostics = _mark_summary_low_usefulness(summary, diagnostics)
        else:
            _set_usefulness_status(diagnostics, "passed")
        summary["context_diagnostics"] = (review_context or {}).get("context_diagnostics", {})
        return _finalize_review_summary(summary, diagnostics)

    async def _repair_summary(
        self,
        session_id: str,
        segments: list[TranscriptSegment],
        ranked_extracts: list[ReviewWindowExtract],
        *,
        timeline: list[dict[str, Any]],
        important_terms: list[str],
        review_context: dict[str, Any] | None,
        cards: list[SidecarCard],
        workstream_candidates: list[dict[str, Any]],
        previous_summary: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> dict[str, Any]:
        system = (
            "You are a post-call review repair assistant. Rewrite a weak meeting summary so it is useful "
            f"as BP's primary meeting review under the {REVIEW_STANDARD_ID} standard. Use broad transcript coverage, "
            "accepted evidence-backed cards, workstream_candidates, source ids, concrete actions/open loops, and project workstreams. "
            "Make the repair answer what changed, what remains open, and what BP should do next. "
            "Do not collapse separate candidate workstreams into one generic topic; every required candidate needs its own sourced workstream and concrete payload. Add technical findings only when the transcript itself supports "
            "energy, electrical-engineering, or another explicit technical evaluation. Fold Energy Lens, EE Index, and Brave/current context "
            "only as reference context tied to transcript evidence, and label it separately from transcript-derived assumptions, findings, and recommendations. Avoid generic speaker/team-member summaries unless the whole meeting was only that."
        )
        user = json.dumps(
            {
                "task": "Repair the meeting summary. Address the quality flags by using the transcript timeline and ranked extracts.",
                "quality_flags": diagnostics.get("quality_flags", []),
                "usefulness_flags": diagnostics.get("usefulness_flags", []),
                "important_terms_seen": important_terms,
                "review_context": _review_context_for_prompt(review_context),
                "previous_summary": previous_summary,
                "timeline": timeline,
                "accepted_cards": [_card_to_summary_payload(card) for card in cards],
                "workstream_candidates": workstream_candidates,
                "ranked_extracts": [_extract_to_payload(extract) for extract in ranked_extracts],
                "return_shape": {
                    "review_standard": REVIEW_STANDARD_ID,
                    "title": "short meeting title",
                    "summary": "one useful paragraph covering the whole meeting, what changed, what remains open, and what BP should do next",
                    "portfolio_rollup": {
                        "bp_next_actions": ["BP-owned next action with owner/date when explicit"],
                        "open_loops": ["unresolved decision, risk, dependency, or question"],
                        "cross_project_dependencies": ["dependency or blocker spanning workstreams"],
                        "risk_posture": "clear|watch|blocked|decision_needed",
                        "source_segment_ids": ["ids from timeline or extracts"],
                    },
                    "key_points": ["important meeting point"],
                    "topics": ["topic"],
                    "projects": ["project or workstream"],
                    "project_workstreams": [
                        {
                            "project": "project/site/client/workstream name",
                            "client_site": "optional client/site if explicit",
                            "status": "discussed|active|blocked|watch|decision_needed",
                            "decisions": ["decision"],
                            "actions": ["action with owner/date when explicit"],
                            "risks": ["risk/issue/dependency"],
                            "open_questions": ["question"],
                            "owners": ["owner names if explicit"],
                            "next_checkpoint": "next milestone/checkpoint if explicit",
                            "source_segment_ids": ["ids from timeline or extracts"],
                        }
                    ],
                    "technical_findings": [
                        {
                            "topic": "technical topic",
                            "question": "technical question or analysis objective",
                            "assumptions": ["assumption, input, scenario, or data basis"],
                            "methods": ["study/model/calculation/standard mentioned"],
                            "findings": ["finding or interpretation"],
                            "recommendations": ["recommendation/follow-up"],
                            "risks": ["technical risk, issue, dependency, or data gap"],
                            "data_gaps": ["missing input or validation need"],
                            "reference_context": ["Energy Lens/EE Index/Brave item titles if relevant"],
                            "confidence": "low|medium|high",
                            "source_segment_ids": ["ids from timeline or extracts"],
                        }
                    ],
                    "technical_findings_rule": "Use an empty list unless transcript evidence supports a technical finding.",
                    "decisions": ["decision"],
                    "actions": ["action"],
                    "unresolved_questions": ["question"],
                    "risks": ["risk"],
                    "entities": ["entity"],
                    "lessons": ["durable note"],
                    "coverage_notes": ["coverage note"],
                    "reference_context": [
                        {
                            "kind": "energy_lens|ee_reference|brave_web",
                            "title": "context title",
                            "body": "what this context contributes",
                            "citation": "URL or local source path if available",
                            "source_segment_ids": ["transcript source ids if applicable"],
                        }
                    ],
                    "source_segment_ids": ["ids from timeline or extracts"],
                },
            },
            ensure_ascii=False,
        )
        try:
            content = await self.ollama.chat(system, user, format_json=True)
            return _parse_review_summary(content, session_id, ranked_extracts, segments, review_context=review_context)
        except Exception:
            return _fallback_review_summary(
                session_id,
                ranked_extracts,
                segments,
                review_context=review_context,
                workstream_candidates=workstream_candidates,
            )


def _note_from_review_card(session_id: str, raw_card: dict[str, Any]) -> NoteCard:
    return NoteCard(
        id=new_id("note"),
        session_id=session_id,
        kind=str(raw_card.get("category") or raw_card.get("kind") or "note"),
        title=compact_text(raw_card.get("title") or "Review note", limit=140),
        body=compact_text(raw_card.get("body") or "", limit=1000),
        source_segment_ids=[str(item) for item in raw_card.get("source_segment_ids") or []],
        evidence_quote=compact_text(raw_card.get("evidence_quote") or "", limit=420),
        owner=_clean_review_owner(raw_card.get("owner")),
        due_date=raw_card.get("due_date"),
        missing_info=raw_card.get("missing_info"),
    )


def _summary_payload_for_storage(
    *,
    session_id: str,
    fallback_title: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "title": compact_text(summary.get("title") or fallback_title, limit=180),
        "summary": compact_text(summary.get("summary") or "Review approved.", limit=1800),
        "review_standard": compact_text(summary.get("review_standard") or REVIEW_STANDARD_ID, limit=80),
        "key_points": _compact_payload_list(summary.get("key_points"), limit=12),
        "topics": _compact_payload_list(summary.get("topics"), limit=12),
        "projects": _compact_payload_list(summary.get("projects"), limit=12),
        "project_workstreams": _compact_project_workstreams(summary.get("project_workstreams"), set(summary.get("source_segment_ids") or []), default_source_ids=_compact_payload_list(summary.get("source_segment_ids"), limit=36), limit=12),
        "technical_findings": _compact_technical_findings(summary.get("technical_findings"), set(summary.get("source_segment_ids") or []), default_source_ids=_compact_payload_list(summary.get("source_segment_ids"), limit=36), limit=12),
        "portfolio_rollup": _compact_portfolio_rollup(summary.get("portfolio_rollup"), summary=summary),
        "review_metrics": _review_metrics_from_summary(summary, diagnostics=summary.get("diagnostics") if isinstance(summary.get("diagnostics"), dict) else None),
        "decisions": _compact_payload_list(summary.get("decisions"), limit=12),
        "actions": _compact_payload_list(summary.get("actions"), limit=16),
        "unresolved_questions": _compact_payload_list(summary.get("unresolved_questions"), limit=12),
        "risks": _compact_payload_list(summary.get("risks"), limit=12),
        "entities": _compact_payload_list(summary.get("entities"), limit=18),
        "lessons": _compact_payload_list(summary.get("lessons"), limit=12),
        "coverage_notes": _compact_payload_list(summary.get("coverage_notes"), limit=6),
        "reference_context": _compact_reference_context(summary.get("reference_context"), limit=10),
        "context_diagnostics": _compact_context_diagnostics(summary.get("context_diagnostics")),
        "source_segment_ids": _compact_payload_list(summary.get("source_segment_ids"), limit=36),
    }


def _finalize_review_summary(summary: dict[str, Any], diagnostics: dict[str, Any]) -> dict[str, Any]:
    summary = dict(summary)
    summary["portfolio_rollup"] = _compact_portfolio_rollup(summary.get("portfolio_rollup"), summary=summary)
    summary["review_metrics"] = _review_metrics_from_summary(summary, diagnostics=diagnostics)
    summary["diagnostics"] = diagnostics
    return summary


def _compact_portfolio_rollup(value: object, *, summary: dict[str, Any]) -> dict[str, Any]:
    raw = value if isinstance(value, dict) else {}
    bp_next_actions = (
        _compact_payload_list(raw.get("bp_next_actions") or raw.get("next_actions"), limit=8, text_limit=220)
        or _rollup_bp_next_actions(summary)
    )
    open_loops = (
        _compact_payload_list(raw.get("open_loops"), limit=8, text_limit=220)
        or _rollup_open_loops(summary)
    )
    dependencies = (
        _compact_payload_list(raw.get("cross_project_dependencies") or raw.get("dependencies"), limit=8, text_limit=220)
        or _rollup_cross_project_dependencies(summary)
    )
    source_ids = _compact_payload_list(
        raw.get("source_segment_ids") or summary.get("source_segment_ids"),
        limit=24,
    )
    risk_posture = compact_text(raw.get("risk_posture") or _rollup_risk_posture(summary, open_loops), limit=80)
    return {
        key: item
        for key, item in {
            "bp_next_actions": bp_next_actions,
            "open_loops": open_loops,
            "cross_project_dependencies": dependencies,
            "risk_posture": risk_posture,
            "source_segment_ids": source_ids,
        }.items()
        if item not in ("", [], None)
    }


def _rollup_bp_next_actions(summary: dict[str, Any]) -> list[str]:
    actions = _summary_actions_from_payload(summary)
    bp_actions = [
        action
        for action in actions
        if re.search(r"\b(?:BP|Brandon)\b", action, re.IGNORECASE)
    ]
    return _dedupe_compact(bp_actions or actions, limit=6, text_limit=220)


def _rollup_open_loops(summary: dict[str, Any]) -> list[str]:
    items: list[str] = []
    items.extend(_compact_payload_list(summary.get("unresolved_questions"), limit=12, text_limit=220))
    items.extend(_compact_payload_list(summary.get("risks"), limit=12, text_limit=220))
    for workstream in summary.get("project_workstreams") or []:
        if not isinstance(workstream, dict):
            continue
        project = compact_text(workstream.get("project") or "", limit=80)
        for item in _compact_payload_list(workstream.get("open_questions"), limit=4, text_limit=220):
            items.append(f"{project}: {item}" if project else item)
        for item in _compact_payload_list(workstream.get("risks"), limit=4, text_limit=220):
            items.append(f"{project}: {item}" if project else item)
    return _dedupe_compact(items, limit=8, text_limit=220)


def _rollup_cross_project_dependencies(summary: dict[str, Any]) -> list[str]:
    dependency_re = re.compile(r"\b(?:blocked|blocker|depends|dependency|until|waiting|pending|handoff|handover|needs from|requires)\b", re.IGNORECASE)
    items: list[str] = []
    for workstream in summary.get("project_workstreams") or []:
        if not isinstance(workstream, dict):
            continue
        project = compact_text(workstream.get("project") or "", limit=80)
        candidates = [
            *_compact_payload_list(workstream.get("actions"), limit=4, text_limit=220),
            *_compact_payload_list(workstream.get("risks"), limit=4, text_limit=220),
            *_compact_payload_list(workstream.get("open_questions"), limit=4, text_limit=220),
            compact_text(workstream.get("next_checkpoint") or "", limit=220),
        ]
        status = compact_text(workstream.get("status") or "", limit=80)
        for item in candidates:
            if item and dependency_re.search(f"{status} {item}"):
                items.append(f"{project}: {item}" if project else item)
    return _dedupe_compact(items, limit=6, text_limit=220)


def _rollup_risk_posture(summary: dict[str, Any], open_loops: list[str]) -> str:
    haystack = _normalize_for_evidence(
        " ".join([
            " ".join(open_loops),
            " ".join(_compact_payload_list(summary.get("risks"), limit=12)),
            json.dumps(summary.get("project_workstreams") or [], ensure_ascii=False),
        ])
    )
    if any(term in haystack for term in ("blocked", "blocker", "cannot move", "decision needed", "pending approval")):
        return "blocked"
    if open_loops or summary.get("risks"):
        return "watch"
    return "clear"


def _summary_actions_from_payload(summary: dict[str, Any]) -> list[str]:
    items = _compact_payload_list(summary.get("actions"), limit=16, text_limit=220)
    for workstream in summary.get("project_workstreams") or []:
        if isinstance(workstream, dict):
            items.extend(_compact_payload_list(workstream.get("actions"), limit=8, text_limit=220))
    for finding in summary.get("technical_findings") or []:
        if isinstance(finding, dict):
            items.extend(_compact_payload_list(finding.get("recommendations"), limit=6, text_limit=220))
    return _dedupe_compact(items, limit=16, text_limit=220)


def _review_metrics_from_summary(summary: dict[str, Any], *, diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = summary.get("review_metrics") if isinstance(summary.get("review_metrics"), dict) else {}
    workstreams = [item for item in summary.get("project_workstreams") or [] if isinstance(item, dict)]
    findings = [item for item in summary.get("technical_findings") or [] if isinstance(item, dict)]
    reference_context = [item for item in summary.get("reference_context") or [] if isinstance(item, dict)]
    source_ids = _compact_payload_list(summary.get("source_segment_ids"), limit=60)
    context_kinds = _dedupe_compact([
        referenceContextKind
        for item in reference_context
        for referenceContextKind in [compact_text(item.get("kind") or "", limit=40)]
        if referenceContextKind
    ], limit=8)
    metrics: dict[str, Any] = {
        "workstream_count": len(workstreams),
        "action_count": len(_summary_actions_from_payload(summary)),
        "risk_count": len(_dedupe_compact([
            *_compact_payload_list(summary.get("risks"), limit=16, text_limit=220),
            *[risk for item in workstreams for risk in _compact_payload_list(item.get("risks"), limit=8, text_limit=220)],
        ], limit=24, text_limit=220)),
        "technical_finding_count": len(findings),
        "source_count": len(source_ids),
        "context_kinds": context_kinds,
    }
    if diagnostics:
        if diagnostics.get("summary_source_coverage") is not None:
            metrics["source_coverage"] = diagnostics.get("summary_source_coverage")
        if diagnostics.get("summary_time_span_coverage") is not None:
            metrics["time_span_coverage"] = diagnostics.get("summary_time_span_coverage")
    for key, item in raw.items():
        if key not in metrics and item not in ("", [], None):
            metrics[key] = item
    return metrics


def _review_context_for_prompt(review_context: dict[str, Any] | None) -> dict[str, Any]:
    if not review_context:
        return {
            "energy_lens": {"active": False},
            "ee_reference_hits": [],
            "web_context_hits": [],
            "context_diagnostics": {},
        }
    energy = review_context.get("energy_lens") if isinstance(review_context.get("energy_lens"), dict) else {"active": False}
    return {
        "energy_lens": {
            "active": bool(energy.get("active")),
            "summary_label": energy.get("summary_label"),
            "confidence": energy.get("confidence"),
            "categories": energy.get("categories") or [],
            "keywords": energy.get("keywords") or [],
            "source_segment_ids": energy.get("source_segment_ids") or [],
            "evidence_quote": energy.get("evidence_quote") or "",
        },
        "ee_reference_hits": [
            _context_item_for_prompt(item)
            for item in review_context.get("ee_reference_hits") or []
            if isinstance(item, dict) and _review_context_item_is_useful(item, kind_hint="ee_reference")
        ][:REVIEW_CONTEXT_MAX_ITEMS],
        "web_context_hits": [
            _context_item_for_prompt(item)
            for item in review_context.get("web_context_hits") or []
            if isinstance(item, dict) and _review_context_item_is_useful(item, kind_hint="brave_web")
        ][:REVIEW_CONTEXT_MAX_ITEMS],
        "context_diagnostics": review_context.get("context_diagnostics") or {},
    }


def _context_item_for_prompt(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": compact_text(item.get("kind") or "context", limit=40),
        "title": compact_text(item.get("title") or "", limit=160),
        "body": compact_text(item.get("body") or item.get("description") or "", limit=520),
        "citation": compact_text(item.get("citation") or item.get("url") or item.get("source_path") or "", limit=500),
        "query": compact_text(item.get("query") or "", limit=160),
        "source_segment_ids": _compact_payload_list(item.get("source_segment_ids"), limit=12),
    }


def _review_context_item_is_useful(item: dict[str, Any], *, kind_hint: str = "") -> bool:
    kind = _normalize_for_evidence(str(item.get("kind") or kind_hint))
    haystack = _normalize_for_evidence(" ".join([
        str(item.get("title") or ""),
        str(item.get("body") or item.get("description") or ""),
        str(item.get("citation") or item.get("url") or item.get("source_path") or ""),
    ]))
    if not haystack:
        return False
    if kind in {"brave_web", "web", "context"}:
        low_value_terms = {
            "pissedconsumer",
            "customer service reviews",
            "trustpilot",
            "yelp",
            "reviews of pge.com",
            "pge reviews",
            "oreate ai blog",
            "what's really being discussed",
        }
        if any(term in haystack for term in low_value_terms):
            return False
    return True


def _merge_scored_items(items: list[Any], *, key: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        label = compact_text(item.get(key) or item.get("phrase") or item.get("category") or "", limit=160)
        if not label:
            continue
        current = merged.get(label) or {key: label, "score": 0.0}
        try:
            score = float(item.get("score") or item.get("weight") or 1.0)
        except (TypeError, ValueError):
            score = 1.0
        current["score"] = round(float(current.get("score") or 0.0) + score, 3)
        if "phrase" in item:
            current["phrase"] = item["phrase"]
        if "category" in item:
            current["category"] = item["category"]
        merged[label] = current
    return sorted(merged.values(), key=lambda item: (-float(item.get("score") or 0.0), str(item.get(key) or item.get("phrase") or item.get("category"))))


def _energy_confidence_rank(confidence: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(confidence or "").lower(), 0)


def _energy_summary_label(categories: list[dict[str, Any]]) -> str:
    labels = [str(item.get("category") or "") for item in categories if item.get("category")]
    if labels:
        return f"Energy lens: {' + '.join(labels[:3])}"
    return "Energy lens active"


def _reference_query_terms(segments: list[TranscriptSegment], energy_context: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    if energy_context.get("active"):
        terms.extend(_labels_from_context_items(energy_context.get("keywords"), "phrase"))
        terms.extend(_labels_from_context_items(energy_context.get("categories"), "category"))
    terms.extend(_important_transcript_terms(segments, limit=18))
    text = " ".join(segment.text for segment in segments)
    terms.extend(_topical_terms_from_text(text, limit=12))
    return _dedupe_compact(terms, limit=24)


def _looks_like_ee_reference(source_path: str, text: str) -> bool:
    haystack = _normalize_for_evidence(f"{source_path} {text}")
    if "electrical" in haystack or "energy" in haystack or "engineering" in haystack:
        return True
    return bool(REVIEW_PROJECT_TERM_RE.search(text))


def _web_context_candidates_for_review(detector: Any, segments: list[TranscriptSegment]) -> list[WebContextCandidate]:
    candidates: list[WebContextCandidate] = []
    seen: set[str] = set()
    for window in _segment_windows(segments, size=8, overlap=4):
        try:
            decision = detector.decision_for_segments(window)
        except Exception:
            continue
        candidate = getattr(decision, "candidate", None)
        if candidate is None or candidate.normalized_query in seen:
            continue
        seen.add(candidate.normalized_query)
        candidates.append(candidate)
        if len(candidates) >= REVIEW_CONTEXT_MAX_WEB_QUERIES:
            break
    return candidates


def _reference_context_from_review_context(review_context: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not review_context:
        return []
    items: list[dict[str, Any]] = []
    energy = review_context.get("energy_lens") if isinstance(review_context.get("energy_lens"), dict) else {}
    if energy.get("active"):
        keywords = _labels_from_context_items(energy.get("keywords"), "phrase")
        categories = _labels_from_context_items(energy.get("categories"), "category")
        items.append(
            {
                "kind": "energy_lens",
                "title": compact_text(energy.get("summary_label") or "Energy lens", limit=140),
                "body": compact_text(
                    "Energy/electrical context detected"
                    + (f": {', '.join([*categories[:3], *keywords[:5]])}" if categories or keywords else "."),
                    limit=520,
                ),
                "citation": "",
                "source_segment_ids": _compact_payload_list(energy.get("source_segment_ids"), limit=12),
            }
        )
    for item in review_context.get("ee_reference_hits") or []:
        if isinstance(item, dict):
            if not _review_context_item_is_useful(item, kind_hint="ee_reference"):
                continue
            compact_item = _context_item_for_prompt(item)
            if compact_item["title"] or compact_item["body"]:
                items.append(compact_item)
    for item in review_context.get("web_context_hits") or []:
        if isinstance(item, dict):
            if not _review_context_item_is_useful(item, kind_hint="brave_web"):
                continue
            compact_item = _context_item_for_prompt(item)
            if compact_item["title"] or compact_item["body"]:
                items.append(compact_item)
    return _compact_reference_context(items, limit=REVIEW_CONTEXT_MAX_ITEMS)


def _labels_from_context_items(items: object, key: str) -> list[str]:
    if not isinstance(items, list):
        return []
    labels: list[str] = []
    for item in items:
        if isinstance(item, dict):
            labels.append(str(item.get(key) or item.get("phrase") or item.get("category") or ""))
        else:
            labels.append(str(item))
    return _dedupe_compact(labels, limit=12)


def _compact_reference_context(value: object, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        payload = {
            "kind": compact_text(item.get("kind") or "context", limit=40),
            "title": compact_text(item.get("title") or "", limit=160),
            "body": compact_text(item.get("body") or item.get("description") or "", limit=700),
            "citation": compact_text(item.get("citation") or item.get("url") or item.get("source_path") or "", limit=500),
            "query": compact_text(item.get("query") or "", limit=160),
            "source_segment_ids": _compact_payload_list(item.get("source_segment_ids"), limit=16),
        }
        key = _normalize_for_evidence(f"{payload['kind']} {payload['title']} {payload['citation']}")
        if not (payload["title"] or payload["body"]) or key in seen:
            continue
        seen.add(key)
        result.append(payload)
        if len(result) >= limit:
            break
    return result


def _compact_context_diagnostics(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key, item in value.items():
        clean_key = compact_text(key, limit=80)
        if clean_key:
            result[clean_key] = item if isinstance(item, (int, float, bool)) or item is None else compact_text(item, limit=180)
    return result


def _segments_from_result(session_id: str, result: TranscriptionResult) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for span in result.spans:
        text = clean_transcript_text(span.text)
        if not text:
            continue
        segment_id = new_id("seg")
        start_s = _safe_float(span.start_s, default=0.0)
        end_s = _safe_float(span.end_s, default=start_s + 0.1)
        if end_s <= start_s:
            end_s = start_s + 0.1
        segments.append(
            TranscriptSegment(
                id=segment_id,
                session_id=session_id,
                start_s=max(0.0, start_s),
                end_s=max(0.1, end_s),
                text=text,
                source_segment_ids=[segment_id],
            )
        )
    return segments


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _readable_word_groups(text: str, *, target_words: int, max_words: int) -> list[str]:
    clean = clean_transcript_text(text)
    if not clean:
        return []
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]
    if not sentences:
        sentences = [clean]
    groups: list[str] = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            groups.append(sentence)
            continue
        for index in range(0, len(words), target_words):
            groups.append(" ".join(words[index:index + target_words]))
    merged: list[str] = []
    for group in groups:
        if merged and len(group.split()) < 8 and len([*merged[-1].split(), *group.split()]) <= max_words:
            merged[-1] = f"{merged[-1]} {group}"
        else:
            merged.append(group)
    return [group for group in merged if clean_transcript_text(group)]


def _trim_leading_text_overlap(previous: str, current: str) -> str:
    previous_words = previous.split()
    current_words = current.split()
    max_overlap = min(36, len(previous_words), len(current_words))
    for size in range(max_overlap, 2, -1):
        previous_tail = [_normalize_word(word) for word in previous_words[-size:]]
        current_head = [_normalize_word(word) for word in current_words[:size]]
        if previous_tail == current_head:
            return clean_transcript_text(" ".join(current_words[size:]))
    return current


def _is_duplicate_segment_text(previous: str, current: str) -> bool:
    previous_norm = _normalize_for_evidence(previous)
    current_norm = _normalize_for_evidence(current)
    if not previous_norm or not current_norm:
        return False
    if previous_norm == current_norm:
        return True
    if len(current_norm.split()) >= 6 and current_norm in previous_norm:
        return True
    return False


def _normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", word.lower())


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _correction_batches(
    segments: list[TranscriptSegment],
    *,
    max_segments: int,
    max_words: int,
) -> list[list[TranscriptSegment]]:
    batches: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    current_words = 0
    for segment in segments:
        word_count = max(1, len(segment.text.split()))
        would_overflow = (
            current
            and (len(current) >= max_segments or current_words + word_count > max_words)
        )
        if would_overflow:
            batches.append(current)
            current = []
            current_words = 0
        current.append(segment)
        current_words += word_count
    if current:
        batches.append(current)
    return batches


def _segments_before_batch(
    segments: list[TranscriptSegment],
    batch: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    if not batch:
        return []
    try:
        index = segments.index(batch[0])
    except ValueError:
        return []
    return segments[:index]


def _segments_after_batch(
    segments: list[TranscriptSegment],
    batch: list[TranscriptSegment],
) -> list[TranscriptSegment]:
    if not batch:
        return []
    try:
        index = segments.index(batch[-1])
    except ValueError:
        return []
    return segments[index + 1:]


def _neighbor_context(segments: list[TranscriptSegment], *, tail: bool) -> str:
    if not segments:
        return ""
    nearby = segments[-2:] if tail else segments[:2]
    return "\n".join(f"{segment.start_s:.1f}-{segment.end_s:.1f}: {segment.text}" for segment in nearby)


def _segment_windows(
    segments: list[TranscriptSegment],
    *,
    size: int,
    overlap: int,
) -> list[list[TranscriptSegment]]:
    clean_segments = [segment for segment in segments if segment.text.strip()]
    if not clean_segments:
        return []
    size = max(1, size)
    overlap = max(0, min(overlap, size - 1))
    step = max(1, size - overlap)
    return [clean_segments[index:index + size] for index in range(0, len(clean_segments), step)]


def _review_transcript_lines(segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for segment in segments:
        line: dict[str, Any] = {
            "id": segment.id,
            "start_s": round(segment.start_s, 1),
            "end_s": round(segment.end_s, 1),
            "text": segment.text,
        }
        if segment.speaker_role:
            line["speaker_role"] = segment.speaker_role
        if segment.speaker_label:
            line["speaker_label"] = segment.speaker_label
        if segment.speaker_confidence is not None:
            line["speaker_confidence"] = segment.speaker_confidence
        lines.append(line)
    return lines


def _valid_review_source_ids(segments: list[TranscriptSegment]) -> list[str]:
    ids: list[str] = []
    for segment in segments:
        ids.extend([segment.id, *segment.source_segment_ids])
    return _dedupe_strings(ids)


def _parse_review_window_extract(
    content: str,
    session_id: str,
    segments: list[TranscriptSegment],
    *,
    save_result: bool,
) -> ReviewWindowExtract:
    payload = _parse_json_object(content)
    valid_source_ids = set(_valid_review_source_ids(segments))
    cards = _review_cards_from_payload(payload, session_id, valid_source_ids, save_result=save_result)
    explicit_sources = [
        str(source_id)
        for source_id in payload.get("source_segment_ids") or []
        if str(source_id) in valid_source_ids
    ]
    default_sources = _dedupe_strings([*explicit_sources, *[
        source_id
        for card in cards
        for source_id in card.source_segment_ids
        if source_id in valid_source_ids
    ]]) or _valid_review_source_ids(segments[:4])
    return ReviewWindowExtract(
        cards=cards,
        window_summary=compact_text(payload.get("window_summary") or "", limit=520) or _fallback_window_summary_text(segments),
        priority=str(payload.get("priority") or "normal").lower(),
        summary_points=_compact_payload_list(payload.get("summary_points"), limit=12),
        topics=_compact_payload_list(payload.get("topics"), limit=10),
        decisions=_compact_payload_list(payload.get("decisions"), limit=10),
        actions=_compact_payload_list(payload.get("actions"), limit=12),
        unresolved_questions=_compact_payload_list(payload.get("unresolved_questions"), limit=10),
        risks=_compact_payload_list(payload.get("risks"), limit=10),
        projects=_compact_payload_list(payload.get("projects"), limit=10),
        entities=_compact_payload_list(payload.get("entities"), limit=14),
        project_workstreams=_compact_project_workstreams(payload.get("project_workstreams"), valid_source_ids, default_source_ids=default_sources, limit=6),
        technical_findings=_compact_technical_findings(payload.get("technical_findings"), valid_source_ids, default_source_ids=default_sources, limit=6),
        source_segment_ids=default_sources,
    )


def _review_cards_from_payload(
    payload: dict[str, Any],
    session_id: str,
    valid_source_ids: set[str],
    *,
    save_result: bool,
) -> list[SidecarCard]:
    items = payload.get("cards")
    if not isinstance(items, list):
        return []
    cards: list[SidecarCard] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        source_ids = [
            str(source_id)
            for source_id in item.get("source_segment_ids") or []
            if str(source_id) in valid_source_ids
        ]
        if not source_ids:
            continue
        body = compact_text(item.get("body") or "", limit=900)
        suggested_say = item.get("suggested_say")
        suggested_ask = item.get("suggested_ask")
        if not body and not suggested_say and not suggested_ask:
            continue
        title = compact_text(item.get("title") or "Review note", limit=140)
        category = item.get("category") or item.get("kind") or "note"
        cards.append(
            create_sidecar_card(
                session_id=session_id,
                category=category,
                title=title,
                body=body,
                suggested_say=suggested_say,
                suggested_ask=suggested_ask,
                why_now=item.get("why_now") or "Extracted from the reviewed meeting transcript.",
                priority=item.get("priority") or "normal",
                confidence=item.get("confidence") or 0.72,
                source_segment_ids=_dedupe_strings(source_ids),
                source_type="transcript",
                ephemeral=not save_result,
                evidence_quote=item.get("evidence_quote") or "",
                owner=_clean_review_owner(item.get("owner")),
                due_date=item.get("due_date"),
                missing_info=item.get("missing_info"),
                card_key=compact_text(item.get("card_key") or "", limit=180) or None,
            )
        )
    return cards


def _rank_cards_for_summary(cards: list[SidecarCard], segments: list[TranscriptSegment]) -> list[SidecarCard]:
    if not cards:
        return []
    index_by_source_id = _segment_index_by_source_id(segments)
    category_rank = {"action": 5, "decision": 5, "question": 4, "risk": 4, "clarification": 3, "note": 2}

    def score(card: SidecarCard) -> tuple[int, int, float, int]:
        source_hits = len([source_id for source_id in card.source_segment_ids if source_id in index_by_source_id])
        priority = 2 if card.priority == "high" else 1 if card.priority == "normal" else 0
        return (
            category_rank.get(card.category, 1) + priority + min(source_hits, 3),
            len(card.evidence_quote or ""),
            float(card.confidence or 0.0),
            -min((index_by_source_id.get(source_id, 999999) for source_id in card.source_segment_ids), default=999999),
        )

    return sorted(cards, key=score, reverse=True)[:REVIEW_EVALUATION_MAX_CARDS]


def _card_to_summary_payload(card: SidecarCard) -> dict[str, Any]:
    return {
        "category": card.category,
        "title": card.title,
        "body": card.body,
        "why_now": card.why_now,
        "owner": _clean_review_owner(card.owner),
        "due_date": card.due_date,
        "missing_info": card.missing_info,
        "priority": card.priority,
        "confidence": card.confidence,
        "source_segment_ids": card.source_segment_ids,
        "evidence_quote": card.evidence_quote,
    }


def _extract_to_payload(extract: ReviewWindowExtract) -> dict[str, Any]:
    return {
        "window_summary": extract.window_summary,
        "priority": extract.priority,
        "cards": [
            {
                "category": card.category,
                "title": card.title,
                "body": card.body,
                "owner": card.owner,
                "due_date": card.due_date,
                "source_segment_ids": card.source_segment_ids,
                "evidence_quote": card.evidence_quote,
            }
            for card in extract.cards
        ],
        "summary_points": extract.summary_points,
        "topics": extract.topics,
        "decisions": extract.decisions,
        "actions": extract.actions,
        "unresolved_questions": extract.unresolved_questions,
        "risks": extract.risks,
        "projects": extract.projects,
        "entities": extract.entities,
        "project_workstreams": extract.project_workstreams,
        "technical_findings": extract.technical_findings,
        "source_segment_ids": extract.source_segment_ids,
    }


def _deterministic_review_extracts(segments: list[TranscriptSegment]) -> list[ReviewWindowExtract]:
    extracts: list[ReviewWindowExtract] = []
    for window in _segment_windows(
        segments,
        size=REVIEW_EVALUATION_WINDOW_SIZE,
        overlap=0,
    ):
        text = " ".join(segment.text for segment in window)
        projects = _important_transcript_terms(window, limit=8)
        actions = _sentences_matching(text, REVIEW_ACTION_CUE_RE, limit=6)
        questions = _sentences_matching(text, re.compile(r"\?|\b(?:question|confirm|clarify|figure out|determine)\b", re.IGNORECASE), limit=4)
        risks = _sentences_matching(text, re.compile(r"\b(?:risk|delay|late|issue|problem|blocked|unclear|missing)\b", re.IGNORECASE), limit=4)
        extracts.append(
            ReviewWindowExtract(
                window_summary=_fallback_window_summary_text(window),
                priority="high" if actions or projects else "normal",
                summary_points=_fallback_timeline_points(window, limit=3),
                topics=projects[:6],
                projects=projects[:6],
                decisions=_sentences_matching(text, re.compile(r"\b(?:decided|decision|settled|agreed)\b", re.IGNORECASE), limit=4),
                actions=actions,
                unresolved_questions=questions,
                risks=risks,
                entities=projects[:8],
                project_workstreams=_fallback_project_workstreams_for_extract(
                    projects=projects[:6],
                    actions=actions,
                    decisions=_sentences_matching(text, re.compile(r"\b(?:decided|decision|settled|agreed)\b", re.IGNORECASE), limit=4),
                    risks=risks,
                    questions=questions,
                    source_segment_ids=[segment.id for segment in window[:8]],
                ),
                technical_findings=_fallback_technical_findings_for_segments(window, review_context=None, source_segment_ids=[segment.id for segment in window[:8]])[:3],
                source_segment_ids=[segment.id for segment in window[:8]],
            )
        )
    if extracts:
        return extracts
    return [ReviewWindowExtract(window_summary=_fallback_window_summary_text(segments), source_segment_ids=[segment.id for segment in segments[:8]])]


def _sentences_matching(text: str, pattern: re.Pattern[str], *, limit: int) -> list[str]:
    candidates = re.split(r"(?<=[.!?])\s+", clean_transcript_text(text))
    if len(candidates) <= 1:
        candidates = re.split(r"\s+(?=(?:and|but|so|then|we|I)\b)", clean_transcript_text(text))
    matches = [compact_text(sentence, limit=260) for sentence in candidates if pattern.search(sentence)]
    return _dedupe_compact(matches, limit=limit)


def _cards_from_summary(
    summary: dict[str, Any],
    session_id: str,
    segments: list[TranscriptSegment],
    *,
    save_result: bool,
) -> list[SidecarCard]:
    source_ids = _compact_payload_list(summary.get("source_segment_ids"), limit=24) or [segment.id for segment in segments[:6]]
    evidence = _evidence_for_source_ids(source_ids, segments)
    specs: list[tuple[str, str, list[str]]] = [
        ("action", "Action", _compact_payload_list(summary.get("actions"), limit=8)),
        ("decision", "Decision", _compact_payload_list(summary.get("decisions"), limit=6)),
        ("question", "Open question", _compact_payload_list(summary.get("unresolved_questions"), limit=6)),
        ("risk", "Risk", _compact_payload_list(summary.get("risks"), limit=6)),
    ]
    cards: list[SidecarCard] = []
    for category, prefix, items in specs:
        for item in items:
            cards.append(
                create_sidecar_card(
                    session_id=session_id,
                    category=category,
                    title=compact_text(item, limit=90) or prefix,
                    body=item,
                    why_now="Included in the transcript-grounded Review summary.",
                    priority="high" if category == "action" else "normal",
                    confidence=0.78,
                    source_segment_ids=source_ids[:8],
                    source_type="transcript",
                    ephemeral=not save_result,
                    evidence_quote=evidence,
                    card_key=f"review-summary:{category}:{_normalize_for_evidence(item)[:80]}",
                )
            )
            if len(cards) >= REVIEW_EVALUATION_MAX_CARDS:
                return cards
    return cards


def _evidence_for_source_ids(source_ids: list[str], segments: list[TranscriptSegment]) -> str:
    wanted = set(source_ids)
    snippets = [segment.text for segment in segments if segment.id in wanted or wanted.intersection(segment.source_segment_ids)]
    return compact_text(" ".join(snippets[:3]), limit=420)


def _rank_extracts_for_summary(
    extracts: list[ReviewWindowExtract],
    segments: list[TranscriptSegment],
) -> list[ReviewWindowExtract]:
    if len(extracts) <= REVIEW_SUMMARY_MAX_EXTRACTS:
        return extracts
    index_by_source_id = _segment_index_by_source_id(segments)
    scored = [(_extract_summary_score(extract, index_by_source_id), order, extract) for order, extract in enumerate(extracts)]
    selected: list[ReviewWindowExtract] = []
    selected_ids: set[int] = set()

    for bucket in ("early", "middle", "late"):
        candidates = [
            item
            for item in scored
            if _extract_bucket(item[2], index_by_source_id, len(segments)) == bucket
        ]
        if candidates:
            _, order, extract = max(candidates, key=lambda item: (item[0], -item[1]))
            selected.append(extract)
            selected_ids.add(order)

    for _, order, extract in sorted(scored, key=lambda item: (-item[0], item[1])):
        if order in selected_ids:
            continue
        selected.append(extract)
        selected_ids.add(order)
        if len(selected) >= REVIEW_SUMMARY_MAX_EXTRACTS:
            break

    return sorted(selected, key=lambda extract: _extract_min_index(extract, index_by_source_id))


def _extract_summary_score(extract: ReviewWindowExtract, index_by_source_id: dict[str, int]) -> float:
    text = " ".join([
        extract.window_summary,
        *extract.summary_points,
        *extract.topics,
        *extract.projects,
        *extract.decisions,
        *extract.actions,
        *extract.unresolved_questions,
        *extract.risks,
        *extract.entities,
        *[card.title for card in extract.cards],
        *[card.body for card in extract.cards],
    ])
    domain_hits = len(REVIEW_PROJECT_TERM_RE.findall(text))
    action_hits = len(REVIEW_ACTION_CUE_RE.findall(text))
    priority_bonus = 8 if extract.priority == "high" else 0 if extract.priority == "normal" else -2
    return (
        priority_bonus
        + len(extract.actions) * 7
        + len(extract.decisions) * 6
        + len(extract.unresolved_questions) * 5
        + len(extract.risks) * 5
        + len(extract.projects) * 4
        + len(extract.cards) * 2
        + min(domain_hits, 10) * 1.5
        + min(action_hits, 10)
        + (2 if extract.window_summary else 0)
        + min(len([source_id for source_id in extract.source_segment_ids if source_id in index_by_source_id]), 8) * 0.25
    )


def _extract_bucket(
    extract: ReviewWindowExtract,
    index_by_source_id: dict[str, int],
    segment_count: int,
) -> str:
    if segment_count <= 0:
        return "middle"
    index = _extract_min_index(extract, index_by_source_id)
    ratio = index / max(1, segment_count - 1)
    if ratio < 0.34:
        return "early"
    if ratio > 0.67:
        return "late"
    return "middle"


def _extract_min_index(extract: ReviewWindowExtract, index_by_source_id: dict[str, int]) -> int:
    indexes = [index_by_source_id[source_id] for source_id in extract.source_segment_ids if source_id in index_by_source_id]
    return min(indexes) if indexes else 999999


def _review_timeline_outline(segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
    clean_segments = [segment for segment in segments if segment.text.strip()]
    if not clean_segments:
        return []
    group_size = max(4, min(16, (len(clean_segments) + REVIEW_TIMELINE_MAX_GROUPS - 1) // REVIEW_TIMELINE_MAX_GROUPS))
    outline: list[dict[str, Any]] = []
    for group in _batched(clean_segments, group_size):
        text = compact_text(" ".join(segment.text for segment in group), limit=620)
        speaker_labels = _dedupe_strings([
            segment.speaker_label or segment.speaker_role or ""
            for segment in group
            if segment.speaker_label or segment.speaker_role
        ])
        outline.append(
            {
                "start_s": round(group[0].start_s, 1),
                "end_s": round(group[-1].end_s, 1),
                "source_segment_ids": [segment.id for segment in group[:10]],
                "text": text,
                "speakers": speaker_labels,
            }
        )
    return outline[:REVIEW_TIMELINE_MAX_GROUPS]


def _fallback_window_summary_text(segments: list[TranscriptSegment]) -> str:
    text = compact_text(" ".join(segment.text for segment in segments if segment.text.strip()), limit=360)
    return text


def _fallback_timeline_points(segments: list[TranscriptSegment], *, limit: int) -> list[str]:
    outline = _review_timeline_outline(segments)
    points = [str(item.get("text") or "") for item in outline if item.get("text")]
    return _dedupe_compact(points, limit=limit, text_limit=180)


def _topical_terms_from_text(text: str, *, limit: int) -> list[str]:
    normalized = re.sub(r"[^A-Za-z0-9+.#/\- ]+", " ", text)
    capitalized_terms = []
    for match in re.findall(r"\b[A-Z][A-Za-z0-9&+#/\-]*(?:\s+[A-Z][A-Za-z0-9&+#/\-]*){0,2}\b", text):
        clean_match = compact_text(match, limit=80)
        clean_key = clean_match.lower()
        clean_tokens = clean_key.split()
        if (
            clean_key
            and clean_key not in REVIEW_STOPWORDS
            and clean_key not in {"the", "this", "that"}
            and not any(token in REVIEW_STOPWORDS or token in {"i"} for token in clean_tokens)
            and any(len(token) >= 3 or token.isupper() for token in clean_match.split())
        ):
            capitalized_terms.append(clean_match)
    words = [
        word.strip("-/.").lower()
        for word in normalized.split()
        if len(word.strip("-/.")) >= 4 and word.lower() not in REVIEW_STOPWORDS
    ]
    counts: dict[str, int] = {}
    for size in (3, 2):
        for index in range(0, max(0, len(words) - size + 1)):
            phrase_words = words[index:index + size]
            if any(word in REVIEW_STOPWORDS for word in phrase_words):
                continue
            phrase = " ".join(phrase_words)
            counts[phrase] = counts.get(phrase, 0) + 1
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    ranked = sorted(
        counts.items(),
        key=lambda item: (-(item[1] * (1.4 if " " in item[0] else 1.0)), item[0]),
    )
    return _dedupe_compact(
        [*capitalized_terms, *[phrase for phrase, count in ranked if count >= 2]],
        limit=limit,
    )


def _heuristic_meeting_brief(segments: list[TranscriptSegment]) -> dict[str, list[str] | str]:
    terms = _important_transcript_terms(segments, limit=18)
    transcript_text = " ".join(segment.text for segment in segments)
    topical_terms = _topical_terms_from_text(transcript_text, limit=10)
    workstreams = _dedupe_compact([*terms, *topical_terms], limit=8)
    actions = _dedupe_compact(_sentences_matching(transcript_text, REVIEW_ACTION_CUE_RE, limit=12), limit=12)
    risks = _dedupe_compact(
        _sentences_matching(transcript_text, re.compile(r"\b(?:risk|delay|late|issue|problem|blocked|unclear|missing|dependency|concern)\b", re.IGNORECASE), limit=8),
        limit=8,
    )
    key_points = _fallback_timeline_points(segments, limit=5)
    if not key_points and workstreams:
        key_points = [f"The meeting covered {_human_join(workstreams[:3])}."]
    if workstreams:
        summary = (
            "The meeting covered "
            f"{_human_join(workstreams[:5])}. "
            "The useful review output is the follow-up trail: actions, decisions, open questions, risks, "
            "and any outside context needed to interpret the discussion."
        )
        title = "Meeting review"
    else:
        summary = ""
        title = ""

    return {
        "title": title,
        "summary": summary,
        "key_points": _dedupe_compact(key_points, limit=8),
        "topics": _dedupe_compact([*workstreams, *terms], limit=12),
        "projects": _dedupe_compact(terms, limit=12),
        "actions": _dedupe_compact(actions, limit=12),
        "risks": _dedupe_compact(risks, limit=8),
    }


def _human_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def _compact_project_workstreams(
    value: object,
    valid_source_ids: set[str],
    *,
    default_source_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        project = compact_text(item.get("project") or item.get("name") or item.get("workstream") or "", limit=160)
        if not project:
            continue
        source_ids = _source_ids_from_payload(item.get("source_segment_ids"), valid_source_ids, default_source_ids=default_source_ids, limit=16)
        payload = {
            "project": project,
            "client_site": compact_text(item.get("client_site") or item.get("site") or item.get("client") or "", limit=160),
            "status": compact_text(item.get("status") or "discussed", limit=80),
            "decisions": _compact_payload_list(item.get("decisions"), limit=8, text_limit=180),
            "actions": _compact_payload_list(item.get("actions"), limit=10, text_limit=180),
            "risks": _compact_payload_list(item.get("risks") or item.get("issues") or item.get("dependencies"), limit=10, text_limit=180),
            "open_questions": _compact_payload_list(item.get("open_questions") or item.get("unresolved_questions"), limit=8, text_limit=180),
            "owners": [_clean_review_owner(owner) for owner in _compact_payload_list(item.get("owners"), limit=8) if _clean_review_owner(owner)],
            "next_checkpoint": compact_text(item.get("next_checkpoint") or item.get("next_step") or "", limit=220),
            "source_segment_ids": source_ids,
        }
        key = _normalize_for_evidence(project)
        if key in seen:
            continue
        seen.add(key)
        result.append({key: value for key, value in payload.items() if value not in ("", [], None)})
        if len(result) >= limit:
            break
    return result


def _compact_technical_findings(
    value: object,
    valid_source_ids: set[str],
    *,
    default_source_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        topic = compact_text(item.get("topic") or item.get("technical_topic") or item.get("question") or "", limit=180)
        if not topic:
            continue
        confidence = str(item.get("confidence") or "").lower()
        if confidence not in {"low", "medium", "high"}:
            confidence = ""
        payload = {
            "topic": topic,
            "question": compact_text(item.get("question") or "", limit=260),
            "assumptions": _compact_payload_list(item.get("assumptions"), limit=8, text_limit=180),
            "methods": _compact_payload_list(item.get("methods") or item.get("models") or item.get("studies"), limit=8, text_limit=180),
            "findings": _compact_payload_list(item.get("findings"), limit=10, text_limit=180),
            "recommendations": _compact_payload_list(item.get("recommendations") or item.get("actions"), limit=10, text_limit=180),
            "risks": _compact_payload_list(item.get("risks") or item.get("issues"), limit=8, text_limit=180),
            "data_gaps": _compact_payload_list(item.get("data_gaps") or item.get("missing_inputs"), limit=8, text_limit=180),
            "reference_context": _compact_payload_list(item.get("reference_context") or item.get("references"), limit=8),
            "confidence": confidence,
            "source_segment_ids": _source_ids_from_payload(item.get("source_segment_ids"), valid_source_ids, default_source_ids=default_source_ids, limit=16),
        }
        key = _normalize_for_evidence(f"{topic} {' '.join(payload['findings'])}")
        if key in seen:
            continue
        seen.add(key)
        result.append({key: value for key, value in payload.items() if value not in ("", [], None)})
        if len(result) >= limit:
            break
    return result


def _source_ids_from_payload(
    value: object,
    valid_source_ids: set[str],
    *,
    default_source_ids: list[str],
    limit: int,
) -> list[str]:
    source_ids: list[str] = []
    if isinstance(value, list):
        source_ids = [str(item) for item in value if str(item) in valid_source_ids]
    if not source_ids:
        source_ids = [str(item) for item in default_source_ids if not valid_source_ids or str(item) in valid_source_ids]
    return _dedupe_strings(source_ids)[:limit]


def _review_workstream_candidates(
    segments: list[TranscriptSegment],
    extracts: list[ReviewWindowExtract],
    cards: list[SidecarCard],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    valid_source_ids = set(_valid_review_source_ids(segments))
    default_source_ids = _summary_source_ids(None, extracts, segments)

    for extract in extracts:
        for item in extract.project_workstreams:
            if not isinstance(item, dict):
                continue
            candidate = _candidate_from_workstream_payload(item, valid_source_ids, default_source_ids, source="window_extract")
            if candidate:
                candidates.append(candidate)
        for finding in extract.technical_findings:
            if not isinstance(finding, dict):
                continue
            candidate = _candidate_from_technical_finding_payload(finding, valid_source_ids, default_source_ids)
            if candidate:
                candidates.append(candidate)

    for name, group in _card_workstream_groups(cards):
        workstream = _workstream_from_card_group(name, group, default_source_ids)
        candidate = _candidate_from_workstream_payload(workstream, valid_source_ids, default_source_ids, source="accepted_cards")
        if candidate:
            candidates.append(candidate)

    candidates.extend(_labeled_workstream_candidates_from_segments(segments))

    text_by_term = _important_transcript_terms(segments, limit=18)
    for term in text_by_term:
        if not _term_should_seed_workstream_candidate(term):
            continue
        related = [segment for segment in segments if _mentions_term(segment.text, term)]
        if not related:
            continue
        candidate = _candidate_from_segments(term, related[:10], source="term_scan")
        if candidate:
            candidates.append(candidate)

    for window in _segment_windows(segments, size=8, overlap=4):
        text = " ".join(segment.text for segment in window)
        if not REVIEW_ABSTRACT_WORKSTREAM_RE.search(text):
            continue
        if not (REVIEW_ACTION_CUE_RE.search(text) or REVIEW_EE_ANALYSIS_RE.search(text) or _question_signal_re().search(text)):
            continue
        name = _abstract_workstream_name(window)
        candidate = _candidate_from_segments(name, window, source="abstract_scan")
        if candidate:
            candidates.append(candidate)

    return _merge_workstream_candidates(candidates, limit=12)


def _labeled_workstream_candidates_from_segments(segments: list[TranscriptSegment]) -> list[dict[str, Any]]:
    grouped: dict[str, list[TranscriptSegment]] = {}
    order: list[str] = []
    for segment in segments:
        label = _segment_workstream_label(segment.text)
        if not label:
            continue
        key = _workstream_match_key(label)
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(segment)
    candidates: list[dict[str, Any]] = []
    for key in order:
        group = grouped[key]
        if not group:
            continue
        candidate = _candidate_from_segments(_clean_workstream_candidate_name(_segment_workstream_label(group[0].text) or ""), group[:18], source="segment_label")
        if candidate:
            candidates.append(candidate)
    return candidates


def _segment_workstream_label(text: str) -> str | None:
    clean = compact_text(clean_transcript_text(text), limit=320)
    if not clean:
        return None

    segment_match = re.match(
        r"^\s*([A-Z][A-Za-z0-9&+#/.\-]*(?:[\s-]+[A-Za-z][A-Za-z0-9&+#/.\-]*){0,4})\s+segment\s+\d+\s*:",
        clean,
    )
    if segment_match:
        label = _clean_workstream_candidate_name(segment_match.group(1))
        return label if not _weak_workstream_name(label) else None

    for_match = re.match(
        r"^\s*For\s+([A-Z][A-Za-z0-9&+#/.\-]*(?:\s+[A-Z][A-Za-z0-9&+#/.\-]*){0,3})\b[:,]?\s*(?P<body>.*)",
        clean,
    )
    if for_match:
        base = _clean_workstream_candidate_name(for_match.group(1))
        if _weak_workstream_name(base):
            return None
        suffix = _technical_workstream_label_suffix(for_match.group("body") or "")
        label = f"{base} {suffix}" if suffix and suffix.lower() not in base.lower() else base
        return compact_text(label, limit=160)

    heading_match = re.match(
        r"^\s*([A-Z][A-Za-z0-9&+#/.\-]*(?:\s+[A-Z][A-Za-z0-9&+#/.\-]*){1,4})\s*:",
        clean,
    )
    if heading_match:
        label = _clean_workstream_candidate_name(heading_match.group(1))
        return label if not _weak_workstream_name(label) else None
    return None


def _technical_workstream_label_suffix(text: str) -> str:
    terms = _dedupe_compact([_canonical_review_term(match) for match in REVIEW_EE_ANALYSIS_RE.findall(text)], limit=4)
    normalized = {_normalize_for_evidence(term) for term in terms}
    if {"transformer", "protection"} <= normalized:
        return "transformer protection"
    if {"relay", "coordination"} <= normalized:
        return "relay coordination"
    if {"load flow", "short circuit"} <= normalized:
        return "load flow and short circuit"
    if len(terms) >= 2:
        return " ".join(terms[:2])
    return terms[0] if terms else ""


def _candidate_from_workstream_payload(
    item: dict[str, Any],
    valid_source_ids: set[str],
    default_source_ids: list[str],
    *,
    source: str,
) -> dict[str, Any] | None:
    name = compact_text(item.get("project") or item.get("name") or item.get("workstream") or "", limit=160)
    name = _clean_workstream_candidate_name(name)
    if _weak_workstream_name(name):
        return None
    source_ids = _source_ids_from_payload(item.get("source_segment_ids"), valid_source_ids, default_source_ids=default_source_ids, limit=16)
    payload = {
        "project": name,
        "signal_source": source,
        "status": compact_text(item.get("status") or "discussed", limit=80),
        "actions": _compact_payload_list(item.get("actions"), limit=8, text_limit=180),
        "decisions": _compact_payload_list(item.get("decisions"), limit=6, text_limit=180),
        "risks": _compact_payload_list(item.get("risks") or item.get("issues") or item.get("dependencies"), limit=6, text_limit=180),
        "open_questions": _compact_payload_list(item.get("open_questions") or item.get("unresolved_questions"), limit=6, text_limit=180),
        "owners": [_clean_review_owner(owner) for owner in _compact_payload_list(item.get("owners"), limit=6) if _clean_review_owner(owner)],
        "next_checkpoint": compact_text(item.get("next_checkpoint") or item.get("next_step") or "", limit=220),
        "source_segment_ids": source_ids,
    }
    return {key: value for key, value in payload.items() if value not in ("", [], None)}


def _candidate_from_technical_finding_payload(
    item: dict[str, Any],
    valid_source_ids: set[str],
    default_source_ids: list[str],
) -> dict[str, Any] | None:
    topic = compact_text(item.get("topic") or item.get("technical_topic") or item.get("question") or "", limit=180)
    topic = _clean_workstream_candidate_name(topic)
    if _weak_workstream_name(topic):
        return None
    source_ids = _source_ids_from_payload(item.get("source_segment_ids"), valid_source_ids, default_source_ids=default_source_ids, limit=16)
    payload = {
        "project": topic,
        "signal_source": "technical_finding",
        "status": "decision_needed",
        "actions": _compact_payload_list(item.get("recommendations") or item.get("actions"), limit=6, text_limit=180),
        "risks": _compact_payload_list(item.get("risks") or item.get("data_gaps") or item.get("missing_inputs"), limit=6, text_limit=180),
        "open_questions": _compact_payload_list([item.get("question")] if item.get("question") else [], limit=3, text_limit=180),
        "technical_terms": _dedupe_compact([_canonical_review_term(match) for match in REVIEW_EE_ANALYSIS_RE.findall(json.dumps(item, ensure_ascii=False))], limit=8),
        "source_segment_ids": source_ids,
    }
    return {key: value for key, value in payload.items() if value not in ("", [], None)}


def _candidate_from_segments(name: str, segments: list[TranscriptSegment], *, source: str) -> dict[str, Any] | None:
    clean_name = compact_text(name, limit=160)
    clean_name = _clean_workstream_candidate_name(clean_name)
    if _weak_workstream_name(clean_name):
        return None
    text = " ".join(segment.text for segment in segments if segment.text.strip())
    actions = _sentences_matching(text, REVIEW_ACTION_CUE_RE, limit=5)
    questions = _sentences_matching(text, _question_signal_re(), limit=4)
    risks = _sentences_matching(text, re.compile(r"\b(?:risk|delay|late|issue|problem|blocked|unclear|missing|dependency|concern|constraint|dispute)\b", re.IGNORECASE), limit=4)
    decisions = _sentences_matching(text, re.compile(r"\b(?:decided|decision|settled|agreed|approved|selected)\b", re.IGNORECASE), limit=3)
    technical_terms = _dedupe_compact([_canonical_review_term(match) for match in REVIEW_EE_ANALYSIS_RE.findall(text)], limit=8)
    is_abstract = bool(REVIEW_ABSTRACT_WORKSTREAM_RE.search(clean_name) or REVIEW_ABSTRACT_WORKSTREAM_RE.search(text))
    if not (actions or questions or risks or decisions or technical_terms):
        return None
    payload = {
        "project": clean_name,
        "signal_source": source,
        "status": "active" if actions else "decision_needed" if questions or technical_terms else "watch" if risks else "discussed",
        "actions": actions,
        "decisions": decisions,
        "risks": risks,
        "open_questions": questions,
        "owners": _owners_from_items([*actions, *decisions]),
        "technical_terms": technical_terms,
        "abstract_signal": is_abstract,
        "evidence_quote": compact_text(text, limit=360),
        "source_segment_ids": _dedupe_strings([segment.id for segment in segments])[:16],
    }
    return {key: value for key, value in payload.items() if value not in ("", [], None)}


def _merge_workstream_candidates(candidates: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for candidate in candidates:
        name = compact_text(candidate.get("project") or "", limit=160)
        if _weak_workstream_name(name):
            continue
        key = _workstream_match_key(name)
        if key not in merged:
            merged[key] = {"project": name, "signal_sources": []}
            order.append(key)
        current = merged[key]
        current["signal_sources"] = _dedupe_strings([*current.get("signal_sources", []), str(candidate.get("signal_source") or "")])
        for field in ("actions", "decisions", "risks", "open_questions", "owners", "technical_terms", "source_segment_ids"):
            current[field] = _dedupe_compact([*current.get(field, []), *candidate.get(field, [])], limit=12, text_limit=220)
        for field in ("status", "next_checkpoint", "evidence_quote"):
            if not current.get(field) and candidate.get(field):
                current[field] = candidate[field]
        if candidate.get("abstract_signal"):
            current["abstract_signal"] = True
    ranked = sorted(
        [payload for payload in merged.values() if _candidate_has_substance(payload)],
        key=lambda item: (-_candidate_score(item), order.index(_workstream_match_key(str(item.get("project") or "")))),
    )
    return [{key: value for key, value in item.items() if value not in ("", [], None)} for item in ranked[:limit]]


def _candidate_score(candidate: dict[str, Any]) -> int:
    sources = {
        str(source)
        for source in (candidate.get("signal_sources") or [candidate.get("signal_source")])
        if str(source or "").strip()
    }
    source_bonus = 0
    if "segment_label" in sources:
        source_bonus += 28
    if "accepted_cards" in sources:
        source_bonus += 16
    if "window_extract" in sources:
        source_bonus += 8
    if "technical_finding" in sources:
        source_bonus += 10
    return (
        len(candidate.get("actions") or []) * 5
        + len(candidate.get("decisions") or []) * 4
        + len(candidate.get("open_questions") or []) * 4
        + len(candidate.get("risks") or []) * 3
        + len(candidate.get("technical_terms") or []) * 2
        + len(candidate.get("owners") or [])
        + source_bonus
    )


def _candidate_has_substance(candidate: dict[str, Any]) -> bool:
    return bool(
        candidate.get("actions")
        or candidate.get("decisions")
        or candidate.get("risks")
        or candidate.get("open_questions")
        or candidate.get("owners")
        or candidate.get("next_checkpoint")
        or candidate.get("technical_terms")
    )


def _candidate_is_required_workstream(candidate: dict[str, Any]) -> bool:
    if not _candidate_has_substance(candidate):
        return False
    sources = {
        str(source)
        for source in (candidate.get("signal_sources") or [candidate.get("signal_source")])
        if str(source or "").strip()
    }
    if "segment_label" in sources:
        return True
    if "accepted_cards" in sources or "technical_finding" in sources:
        return bool(candidate.get("actions") or candidate.get("open_questions") or candidate.get("risks") or candidate.get("technical_terms"))
    if "window_extract" in sources:
        return _candidate_score(candidate) >= 10
    if candidate.get("abstract_signal"):
        return _candidate_score(candidate) >= 12
    return _candidate_score(candidate) >= 12


def _abstract_workstream_name(segments: list[TranscriptSegment]) -> str:
    terms = _important_transcript_terms(segments, limit=5)
    for term in terms:
        if not _weak_workstream_name(term):
            return term
    text = " ".join(segment.text for segment in segments)
    match = REVIEW_ABSTRACT_WORKSTREAM_RE.search(text)
    if match:
        return f"{_canonical_review_term(match.group(0))} workstream"
    return "Meeting workstream"


def _weak_workstream_name(name: object) -> bool:
    clean = _normalize_for_evidence(str(name or ""))
    if not clean:
        return True
    weak = {
        "bp",
        "meeting",
        "meeting review",
        "review",
        "project",
        "project workstream",
        "workstream",
        "other speaker",
        "unknown speaker",
        "speaker",
        "team",
        "ct",
        "rfi",
        "settings",
        "package",
        "protection",
        "transformer",
        "breaker",
        "customer",
        "requirements",
        "requirement",
        "design",
        "assumptions",
        "basis",
        "scope",
        "support",
        "what",
    }
    return clean in weak or clean.startswith("for ") or len(clean) < 3


def _clean_workstream_candidate_name(name: object) -> str:
    text = compact_text(name or "", limit=160)
    text = re.sub(r"^(?:for|on|about)\s+", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+segment\s+\d+\s*$", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\s+(?:scope|support|workstream)$", "", text, flags=re.IGNORECASE).strip() or text
    return compact_text(text, limit=160)


def _term_should_seed_workstream_candidate(term: str) -> bool:
    clean = _clean_workstream_candidate_name(term)
    if _weak_workstream_name(clean):
        return False
    tokens = _workstream_match_tokens(clean)
    if not tokens:
        return False
    generic_single = {
        "analysis",
        "breaker",
        "budget",
        "collector",
        "customer",
        "interconnection",
        "package",
        "permitting",
        "pricing",
        "protection",
        "requirement",
        "requirements",
        "schedule",
        "settings",
        "substation",
        "supplier",
        "tariff",
        "transformer",
        "utility",
        "vendor",
    }
    return not (len(tokens) == 1 and tokens[0] in generic_single)


def _workstream_match_key(name: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_for_evidence(name))
        if len(token) > 2 and token not in REVIEW_STOPWORDS and token not in {"project", "workstream", "review", "scope", "support"}
    ]
    return " ".join(tokens[:6]) or _normalize_for_evidence(name)


def _question_signal_re() -> re.Pattern[str]:
    return re.compile(r"\?|\b(?:question|confirm|clarify|figure out|determine|what are|which|whether|need from)\b", re.IGNORECASE)


def _fallback_project_workstreams_for_extract(
    *,
    projects: list[str],
    actions: list[str],
    decisions: list[str],
    risks: list[str],
    questions: list[str],
    source_segment_ids: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "project": project,
            "status": "active" if actions else "watch" if risks or questions else "discussed",
            "decisions": [item for item in decisions if _mentions_term(item, project)][:4],
            "actions": [item for item in actions if _mentions_term(item, project)][:5] or actions[:3],
            "risks": [item for item in risks if _mentions_term(item, project)][:5] or risks[:2],
            "open_questions": [item for item in questions if _mentions_term(item, project)][:4] or questions[:2],
            "owners": _owners_from_items([*actions, *decisions]),
            "source_segment_ids": source_segment_ids[:8],
        }
        for project in projects[:6]
    ]


def _fallback_project_workstreams(
    extracts: list[ReviewWindowExtract],
    segments: list[TranscriptSegment],
    *,
    projects: list[str],
    actions: list[str],
    decisions: list[str],
    risks: list[str],
    questions: list[str],
    workstream_candidates: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    explicit = [item for extract in extracts for item in extract.project_workstreams]
    if explicit:
        compacted = _compact_project_workstreams(explicit, set(_valid_review_source_ids(segments)), default_source_ids=_summary_source_ids(None, extracts, segments), limit=12)
        if compacted:
            return compacted
    if workstream_candidates:
        candidate_workstreams = _project_workstreams_from_candidates(workstream_candidates)
        if candidate_workstreams:
            return candidate_workstreams
    project_terms = projects or _important_transcript_terms(segments, limit=8)
    if not project_terms:
        return []
    source_ids = _summary_source_ids(None, extracts, segments)
    return _fallback_project_workstreams_for_extract(
        projects=project_terms[:8],
        actions=actions,
        decisions=decisions,
        risks=risks,
        questions=questions,
        source_segment_ids=source_ids,
    )


def _project_workstreams_from_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        name = compact_text(candidate.get("project") or "", limit=160)
        if _weak_workstream_name(name):
            continue
        payload = {
            "project": name,
            "status": compact_text(candidate.get("status") or "active", limit=80),
            "decisions": _compact_payload_list(candidate.get("decisions"), limit=6, text_limit=180),
            "actions": _compact_payload_list(candidate.get("actions"), limit=8, text_limit=180),
            "risks": _compact_payload_list(candidate.get("risks"), limit=8, text_limit=180),
            "open_questions": _compact_payload_list(candidate.get("open_questions"), limit=6, text_limit=180),
            "owners": [_clean_review_owner(owner) for owner in _compact_payload_list(candidate.get("owners"), limit=6) if _clean_review_owner(owner)],
            "next_checkpoint": compact_text(candidate.get("next_checkpoint") or "", limit=220),
            "source_segment_ids": _compact_payload_list(candidate.get("source_segment_ids"), limit=16),
        }
        if not _workstream_has_meeting_substance(payload) and not candidate.get("technical_terms"):
            continue
        result.append({key: value for key, value in payload.items() if value not in ("", [], None)})
        if len(result) >= 12:
            break
    return result


def _fallback_technical_findings_for_segments(
    segments: list[TranscriptSegment],
    *,
    review_context: dict[str, Any] | None,
    source_segment_ids: list[str],
) -> list[dict[str, Any]]:
    text = " ".join(segment.text for segment in segments if segment.text.strip())
    if not text:
        return []
    technical_terms = _dedupe_compact(
        [_canonical_review_term(match) for match in REVIEW_EE_ANALYSIS_RE.findall(text)],
        limit=8,
    )
    reference_context = _reference_context_from_review_context(review_context)
    reference_titles = _dedupe_compact([str(item.get("title") or item.get("kind") or "") for item in reference_context], limit=8)
    findings = _sentences_matching(text, REVIEW_EE_ANALYSIS_RE, limit=6)
    assumptions = _sentences_matching(text, re.compile(r"\b(?:assum|basis|data|input|scenario|forecast|model|reference|utility bill|tariff)\b", re.IGNORECASE), limit=5)
    methods = _sentences_matching(text, re.compile(r"\b(?:study|model|analysis|calculate|compare|load flow|short circuit|coordination|standard|IEEE|NERC|FERC)\b", re.IGNORECASE), limit=5)
    recommendations = _sentences_matching(text, REVIEW_ACTION_CUE_RE, limit=6)
    risks = _sentences_matching(text, re.compile(r"\b(?:risk|issue|gap|missing|unclear|constraint|blocked|dependency|concern)\b", re.IGNORECASE), limit=5)
    questions = _sentences_matching(text, re.compile(r"\?|\b(?:question|confirm|clarify|determine|what are|which)\b", re.IGNORECASE), limit=5)
    if not (technical_terms or findings or reference_titles):
        return []
    return [
        {
            "topic": _human_join(technical_terms[:3]) or "Technical analysis",
            "question": questions[0] if questions else "",
            "assumptions": assumptions,
            "methods": methods,
            "findings": findings[:6] or ([f"The technical discussion covered {_human_join(technical_terms[:4])}."] if technical_terms else []),
            "recommendations": recommendations,
            "risks": risks,
            "data_gaps": questions[1:5] if len(questions) > 1 else [],
            "reference_context": reference_titles,
            "confidence": "medium" if source_segment_ids else "low",
            "source_segment_ids": source_segment_ids[:16],
        }
    ]


def _technical_review_needed(segments: list[TranscriptSegment], review_context: dict[str, Any] | None) -> bool:
    text = " ".join(segment.text for segment in segments)
    technical_hits = len(REVIEW_EE_ANALYSIS_RE.findall(text))
    if technical_hits >= 2:
        return True
    if not review_context:
        return False
    diagnostics = review_context.get("context_diagnostics") if isinstance(review_context.get("context_diagnostics"), dict) else {}
    context_available = (
        (review_context.get("energy_lens") or {}).get("active") is True
        or int(diagnostics.get("ee_reference_hits") or 0) > 0
        or int(diagnostics.get("web_context_hits") or 0) > 0
    )
    return technical_hits >= 1 and context_available


def _mentions_term(text: str, term: str) -> bool:
    normalized_term = _normalize_for_evidence(term)
    if not normalized_term:
        return False
    return normalized_term in _normalize_for_evidence(text)


def _owners_from_items(items: list[str]) -> list[str]:
    owners: list[str] = []
    for item in items:
        lowered = item.lower()
        if "bp" in lowered or "brandon" in lowered:
            owners.append("BP")
        for match in re.findall(r"\b([A-Z][a-z]{2,})\s+(?:owns|will|should|needs|to)\b", item):
            owners.append(match)
    return _dedupe_strings([owner for owner in (_clean_review_owner(owner) for owner in owners) if owner])[:6]


def _clean_review_owner(value: object) -> str | None:
    text = compact_text(value or "", limit=120)
    if not text:
        return None
    normalized = _normalize_for_evidence(text)
    if normalized in {
        "unknown",
        "unknown speaker",
        "other speaker",
        "speaker",
        "unspecified",
        "none",
        "n/a",
        "na",
    }:
        return None
    return text


def _important_transcript_terms(segments: list[TranscriptSegment], *, limit: int = 20) -> list[str]:
    counts: dict[str, int] = {}
    for segment in segments:
        for match in REVIEW_PROJECT_TERM_RE.findall(segment.text):
            normalized = _canonical_review_term(match)
            counts[normalized] = counts.get(normalized, 0) + 1
    for term in REVIEW_CORRECTION_GLOSSARY:
        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        count = sum(1 for segment in segments if pattern.search(segment.text))
        if count:
            counts[term] = max(counts.get(term, 0), count)
    topical_terms = _topical_terms_from_text(" ".join(segment.text for segment in segments), limit=max(limit, 12))
    for term in topical_terms:
        counts.setdefault(term, 1)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))
    return [term for term, _ in ordered[:limit]]


def _canonical_review_term(term: str) -> str:
    value = str(term or "").strip()
    lower = value.lower()
    fixed = {
        "pge": "PGE",
        "rfi": "RFI",
        "bess": "BESS",
        "epc": "EPC",
        "spcc": "SPCC",
        "gen-tie": "gen-tie",
        "one-line": "one-line",
        "load flow": "load flow",
        "short circuit": "short circuit",
    }
    return fixed.get(lower, value)


def _summary_quality_diagnostics(
    summary: dict[str, Any],
    extracts: list[ReviewWindowExtract],
    segments: list[TranscriptSegment],
    *,
    review_context: dict[str, Any] | None = None,
    workstream_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    workstream_candidates = workstream_candidates or _review_workstream_candidates(segments, extracts, [])
    source_ids = [str(source_id) for source_id in summary.get("source_segment_ids") or []]
    index_by_source_id = _segment_index_by_source_id(segments)
    covered_indexes = {index_by_source_id[source_id] for source_id in source_ids if source_id in index_by_source_id}
    segment_count = len([segment for segment in segments if segment.text.strip()])
    source_coverage = round(len(covered_indexes) / max(1, segment_count), 3)
    buckets = _covered_time_buckets(covered_indexes, segment_count)
    time_span_coverage = round(len(buckets) / 3, 3) if segment_count >= REVIEW_EVALUATION_WINDOW_SIZE * 2 else 1.0
    text = " ".join(segment.text for segment in segments)
    action_signal_count = len(REVIEW_ACTION_CUE_RE.findall(text))
    important_terms = _important_transcript_terms(segments)
    summary_text = _normalize_for_evidence(" ".join([
        str(summary.get("title") or ""),
        str(summary.get("summary") or ""),
        " ".join(str(item) for item in summary.get("topics") or []),
        " ".join(str(item) for item in summary.get("projects") or []),
        " ".join(str(item) for item in summary.get("actions") or []),
        " ".join(str(item) for item in summary.get("key_points") or []),
        json.dumps(summary.get("project_workstreams") or []),
        json.dumps(summary.get("technical_findings") or []),
    ]))
    matched_terms = [
        term
        for term in important_terms
        if _normalize_for_evidence(term) in summary_text
    ]
    flags: list[str] = []
    if segment_count >= REVIEW_EVALUATION_WINDOW_SIZE * 2 and len(buckets) < 2:
        flags.append("narrow_time_coverage")
    if segment_count >= REVIEW_EVALUATION_WINDOW_SIZE * 3 and len(buckets) < 3:
        flags.append("incomplete_time_coverage")
    if segment_count >= REVIEW_EVALUATION_WINDOW_SIZE * 3 and source_coverage < 0.025:
        flags.append("thin_source_coverage")
    if action_signal_count >= 4 and not summary.get("actions"):
        flags.append("missing_obvious_actions")
    if len(important_terms) >= 3 and not matched_terms:
        flags.append("missing_project_terms")
    if len(important_terms) >= 2 and not summary.get("project_workstreams"):
        flags.append("missing_project_workstreams")
    if _summary_has_too_few_workstreams(summary, segment_count=segment_count, action_signal_count=action_signal_count, important_terms=important_terms):
        flags.append("thin_project_workstreams")
    required_candidates = [candidate for candidate in workstream_candidates if _candidate_is_required_workstream(candidate)]
    missing_candidates = _missing_candidate_workstream_names(summary, required_candidates)
    required_candidate_count = len(required_candidates)
    if missing_candidates and len(missing_candidates) >= max(2, math.ceil(required_candidate_count / 2)):
        flags.append("missing_candidate_workstream")
    if _summary_collapses_multiple_workstreams(summary, required_candidates):
        flags.append("collapsed_multiple_workstreams")
    if _summary_has_abstract_topic_without_payload(summary):
        flags.append("abstract_topic_without_actionable_payload")
    if _summary_has_abstract_payload_too_thin(summary, required_candidates):
        flags.append("abstract_payload_too_thin")
    technical_signal_count = len(REVIEW_EE_ANALYSIS_RE.findall(text))
    if technical_signal_count >= 2 and not summary.get("technical_findings"):
        flags.append("missing_technical_findings")
    reference_context = summary.get("reference_context") if isinstance(summary.get("reference_context"), list) else []
    if reference_context and technical_signal_count >= 1 and not summary.get("technical_findings"):
        flags.append("context_not_folded_into_technical_review")
    if _summary_has_weak_technical_basis(summary):
        flags.append("weak_technical_basis")
    if _summary_has_thin_technical_finding(summary):
        flags.append("technical_finding_too_thin")
    if _summary_has_context_not_tied_to_transcript(summary, segments, review_context=review_context):
        flags.append("context_not_tied_to_transcript")
    if _summary_has_reference_context_low_value(summary, segments, review_context=review_context):
        flags.append("reference_context_low_value")
    if _looks_generic_review_summary(str(summary.get("summary") or "")) and len(important_terms) >= 2:
        flags.append("generic_summary")
    if _summary_looks_like_transcript_echo(str(summary.get("summary") or ""), segments):
        flags.append("transcript_echo")
    usefulness_flags = _summary_usefulness_flags(
        summary,
        segments,
        important_terms=important_terms,
        action_signal_count=action_signal_count,
        technical_signal_count=technical_signal_count,
        review_context=review_context,
    )
    score = _summary_usefulness_score(flags, usefulness_flags, source_coverage=source_coverage, time_span_coverage=time_span_coverage)
    usefulness_status = "passed" if score >= REVIEW_USEFULNESS_PASS_SCORE and not _usefulness_has_critical_flags([*flags, *usefulness_flags]) else "needs_repair"
    return {
        "summary_source_coverage": source_coverage,
        "summary_time_span_coverage": time_span_coverage,
        "summary_time_buckets": sorted(buckets),
        "summary_quality_flags": flags,
        "summary_usefulness_flags": usefulness_flags,
        "usefulness_flags": usefulness_flags,
        "quality_flags": _dedupe_strings([*flags, *usefulness_flags]),
        "usefulness_score": score,
        "usefulness_status": usefulness_status,
        "action_signal_count": action_signal_count,
        "technical_signal_count": technical_signal_count,
        "important_terms_seen": important_terms,
        "important_terms_covered": matched_terms,
        "workstream_candidate_count": required_candidate_count,
        "missing_workstream_candidates": missing_candidates,
        "ranked_extract_count": len(extracts),
    }


def _summary_needs_repair(diagnostics: dict[str, Any]) -> bool:
    flags = set(diagnostics.get("quality_flags") or [])
    flags.update(diagnostics.get("summary_quality_flags") or [])
    flags.update(diagnostics.get("usefulness_flags") or [])
    flags.update(diagnostics.get("summary_usefulness_flags") or [])
    critical = {
        "narrow_time_coverage",
        "incomplete_time_coverage",
        "missing_obvious_actions",
        "missing_project_terms",
        "missing_project_workstreams",
        "thin_project_workstreams",
        "missing_candidate_workstream",
        "collapsed_multiple_workstreams",
        "abstract_topic_without_actionable_payload",
        "abstract_payload_too_thin",
        "missing_technical_findings",
        "weak_technical_basis",
        "technical_finding_too_thin",
        "context_not_folded_into_technical_review",
        "context_not_tied_to_transcript",
        "reference_context_low_value",
        "generic_summary",
        "transcript_echo",
        "transcript_echo_key_points",
        "transcript_echo_workstreams",
        "transcript_echo_technical_findings",
        "generic_topic_soup",
        "unsupported_technical_findings",
        "unsupported_external_context",
        "weak_project_workstreams",
        "vague_abstract_workstream",
        "missing_bp_next_action",
        "weak_meeting_followup",
        "low_usefulness",
    }
    return bool(flags & critical) or str(diagnostics.get("usefulness_status") or "") == "needs_repair"


def _summary_usefulness_flags(
    summary: dict[str, Any],
    segments: list[TranscriptSegment],
    *,
    important_terms: list[str],
    action_signal_count: int,
    technical_signal_count: int,
    review_context: dict[str, Any] | None,
) -> list[str]:
    flags: list[str] = []
    summary_text = str(summary.get("summary") or "")
    if _summary_looks_like_topic_soup(summary, important_terms, action_signal_count=action_signal_count):
        flags.append("generic_topic_soup")
    if _list_contains_transcript_echo(summary.get("key_points"), segments):
        flags.append("transcript_echo_key_points")
    if _list_contains_transcript_echo(summary.get("actions"), segments):
        flags.append("transcript_echo_actions")
    if _structured_items_contain_transcript_echo(summary.get("project_workstreams"), segments):
        flags.append("transcript_echo_workstreams")
    if _structured_items_contain_transcript_echo(summary.get("technical_findings"), segments):
        flags.append("transcript_echo_technical_findings")
    if _summary_has_unsupported_technical_findings(summary, segments, technical_signal_count=technical_signal_count):
        flags.append("unsupported_technical_findings")
    if _summary_forces_external_context(summary, segments, review_context=review_context):
        flags.append("unsupported_external_context")
    if _summary_has_weak_project_workstreams(summary, action_signal_count=action_signal_count, important_terms=important_terms):
        flags.append("weak_project_workstreams")
    if _summary_has_vague_abstract_workstream(summary):
        flags.append("vague_abstract_workstream")
    if _segments_have_bp_action_signal(segments) and not _summary_has_bp_next_action(summary):
        flags.append("missing_bp_next_action")
    if _summary_has_weak_meeting_followup(summary, action_signal_count=action_signal_count):
        flags.append("weak_meeting_followup")
    if not summary_text.strip() or "did not return a reliable structured meeting summary" in summary_text.lower():
        flags.append("low_usefulness")
    return _dedupe_strings(flags)


def _segments_have_bp_action_signal(segments: list[TranscriptSegment]) -> bool:
    pattern = re.compile(
        r"\b(?:BP|Brandon)\b.{0,80}\b(?:will|needs? to|should|owns?|send|request|confirm|review|follow|coordinate|provide)\b|"
        r"\b(?:will|needs? to|should|owns?|send|request|confirm|review|follow|coordinate|provide)\b.{0,80}\b(?:BP|Brandon)\b",
        re.IGNORECASE,
    )
    return any(pattern.search(segment.text or "") for segment in segments)


def _summary_has_bp_next_action(summary: dict[str, Any]) -> bool:
    raw_rollup = summary.get("portfolio_rollup") if isinstance(summary.get("portfolio_rollup"), dict) else {}
    candidates = [
        *_compact_payload_list(raw_rollup.get("bp_next_actions") or raw_rollup.get("next_actions"), limit=12, text_limit=260),
        *_summary_actions_from_payload(summary),
    ]
    return any(re.search(r"\b(?:BP|Brandon)\b", item, re.IGNORECASE) for item in candidates)


def _summary_has_weak_meeting_followup(summary: dict[str, Any], *, action_signal_count: int) -> bool:
    workstreams = [item for item in summary.get("project_workstreams") or [] if isinstance(item, dict)]
    if action_signal_count < 2 and not workstreams:
        return False
    raw_rollup = summary.get("portfolio_rollup") if isinstance(summary.get("portfolio_rollup"), dict) else {}
    has_next = bool(raw_rollup.get("bp_next_actions") or raw_rollup.get("next_actions") or _summary_actions_from_payload(summary))
    has_open = bool(
        raw_rollup.get("open_loops")
        or summary.get("unresolved_questions")
        or summary.get("risks")
        or any(item.get("risks") or item.get("open_questions") for item in workstreams)
    )
    return not (has_next or has_open)


def _summary_has_vague_abstract_workstream(summary: dict[str, Any]) -> bool:
    for workstream in summary.get("project_workstreams") or []:
        if not isinstance(workstream, dict):
            continue
        name = str(workstream.get("project") or workstream.get("name") or "")
        if not REVIEW_ABSTRACT_WORKSTREAM_RE.search(name):
            continue
        if not _workstream_payload_has_specific_anchor(workstream, None):
            return True
    return False


def _summary_usefulness_score(
    quality_flags: list[str],
    usefulness_flags: list[str],
    *,
    source_coverage: float,
    time_span_coverage: float,
) -> float:
    score = 1.0
    penalties = {
        "narrow_time_coverage": 0.24,
        "incomplete_time_coverage": 0.2,
        "thin_source_coverage": 0.1,
        "missing_obvious_actions": 0.2,
        "missing_project_terms": 0.22,
        "missing_project_workstreams": 0.2,
        "thin_project_workstreams": 0.18,
        "missing_candidate_workstream": 0.24,
        "collapsed_multiple_workstreams": 0.24,
        "abstract_topic_without_actionable_payload": 0.22,
        "abstract_payload_too_thin": 0.22,
        "missing_technical_findings": 0.16,
        "weak_technical_basis": 0.18,
        "technical_finding_too_thin": 0.18,
        "context_not_folded_into_technical_review": 0.12,
        "context_not_tied_to_transcript": 0.18,
        "reference_context_low_value": 0.14,
        "generic_summary": 0.28,
        "transcript_echo": 0.34,
        "transcript_echo_key_points": 0.28,
        "transcript_echo_actions": 0.12,
        "transcript_echo_workstreams": 0.18,
        "transcript_echo_technical_findings": 0.2,
        "generic_topic_soup": 0.3,
        "unsupported_technical_findings": 0.26,
        "unsupported_external_context": 0.24,
        "weak_project_workstreams": 0.18,
        "vague_abstract_workstream": 0.22,
        "missing_bp_next_action": 0.24,
        "weak_meeting_followup": 0.18,
        "low_usefulness": 0.45,
    }
    for flag in _dedupe_strings([*quality_flags, *usefulness_flags]):
        score -= penalties.get(flag, 0.08)
    if time_span_coverage < 0.66:
        score -= 0.08
    if source_coverage < 0.025:
        score -= 0.04
    return round(max(0.0, min(1.0, score)), 2)


def _usefulness_has_critical_flags(flags: list[str]) -> bool:
    critical = {
        "transcript_echo_key_points",
        "transcript_echo_workstreams",
        "transcript_echo_technical_findings",
        "generic_topic_soup",
        "missing_candidate_workstream",
        "collapsed_multiple_workstreams",
        "abstract_topic_without_actionable_payload",
        "abstract_payload_too_thin",
        "weak_technical_basis",
        "technical_finding_too_thin",
        "context_not_tied_to_transcript",
        "reference_context_low_value",
        "unsupported_technical_findings",
        "unsupported_external_context",
        "weak_project_workstreams",
        "vague_abstract_workstream",
        "missing_bp_next_action",
        "weak_meeting_followup",
        "low_usefulness",
    }
    return bool(set(flags) & critical)


def _set_usefulness_status(diagnostics: dict[str, Any], status: str) -> None:
    diagnostics["usefulness_status"] = status
    if status == "low_usefulness":
        diagnostics["usefulness_score"] = min(float(diagnostics.get("usefulness_score") or 0.0), 0.45)


def _mark_summary_low_usefulness(
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    diagnostics = dict(diagnostics)
    flags = _dedupe_strings([*(diagnostics.get("usefulness_flags") or []), "low_usefulness"])
    diagnostics["usefulness_flags"] = flags
    diagnostics["summary_usefulness_flags"] = flags
    diagnostics["quality_flags"] = _dedupe_strings([*(diagnostics.get("quality_flags") or []), "low_usefulness"])
    _set_usefulness_status(diagnostics, "low_usefulness")
    summary = dict(summary)
    summary["summary"] = (
        "Review completed, but the generated meeting summary did not pass the usefulness gate. "
        "Validate the clean transcript and supporting evidence before approval."
    )
    notes = _compact_payload_list(summary.get("coverage_notes"), limit=5)
    warning = "Usefulness gate failed; manual validation is required before saving this Review."
    summary["coverage_notes"] = _dedupe_compact([warning, *notes], limit=6)
    return summary, diagnostics


def _segment_index_by_source_id(segments: list[TranscriptSegment]) -> dict[str, int]:
    index_by_source_id: dict[str, int] = {}
    for index, segment in enumerate(segments):
        index_by_source_id.setdefault(segment.id, index)
        for source_id in segment.source_segment_ids:
            index_by_source_id.setdefault(source_id, index)
    return index_by_source_id


def _covered_time_buckets(indexes: set[int], segment_count: int) -> set[str]:
    buckets: set[str] = set()
    if segment_count <= 0:
        return buckets
    for index in indexes:
        ratio = index / max(1, segment_count - 1)
        if ratio < 0.34:
            buckets.add("early")
        elif ratio > 0.67:
            buckets.add("late")
        else:
            buckets.add("middle")
    return buckets


def _looks_generic_review_summary(summary: str) -> bool:
    clean = _normalize_for_evidence(summary)
    if not clean:
        return True
    generic_terms = {"speaker", "team member", "new team member", "keep in the loop", "appreciates", "supporting"}
    return any(term in clean for term in generic_terms) and not REVIEW_PROJECT_TERM_RE.search(summary)


def _summary_looks_like_transcript_echo(summary: str, segments: list[TranscriptSegment]) -> bool:
    clean_summary = _normalize_for_evidence(summary)
    if len(clean_summary.split()) < 45:
        return False
    transcript = _normalize_for_evidence(" ".join(segment.text for segment in segments))
    if clean_summary[:260] and clean_summary[:260] in transcript:
        return True
    summary_words = clean_summary.split()
    if len(summary_words) < 80:
        return False
    sampled_phrases = [
        " ".join(summary_words[index:index + 12])
        for index in range(0, min(len(summary_words) - 12, 120), 24)
    ]
    if not sampled_phrases:
        return False
    hits = sum(1 for phrase in sampled_phrases if phrase and phrase in transcript)
    return hits / max(1, len(sampled_phrases)) >= 0.5


def _summary_looks_like_topic_soup(
    summary: dict[str, Any],
    important_terms: list[str],
    *,
    action_signal_count: int,
) -> bool:
    clean = _normalize_for_evidence(str(summary.get("summary") or ""))
    if not clean:
        return True
    actions = _compact_payload_list(summary.get("actions"), limit=4)
    decisions = _compact_payload_list(summary.get("decisions"), limit=4)
    questions = _compact_payload_list(summary.get("unresolved_questions"), limit=4)
    risks = _compact_payload_list(summary.get("risks"), limit=4)
    workstreams = summary.get("project_workstreams") if isinstance(summary.get("project_workstreams"), list) else []
    has_output = bool(actions or decisions or questions or risks or any(_workstream_has_meeting_substance(item) for item in workstreams if isinstance(item, dict)))
    topic_soup_phrases = (
        "meeting covered",
        "review covered",
        "meeting discussed",
        "review centered on",
        "useful review output is",
    )
    if any(phrase in clean for phrase in topic_soup_phrases) and len(important_terms) >= 3 and action_signal_count >= 2 and not has_output:
        return True
    if clean.count(",") >= 4 and len(important_terms) >= 5 and not has_output:
        return True
    return False


def _list_contains_transcript_echo(value: object, segments: list[TranscriptSegment]) -> bool:
    if not isinstance(value, list):
        return False
    return any(_text_looks_like_transcript_echo(str(item), segments) for item in value)


def _structured_items_contain_transcript_echo(value: object, segments: list[TranscriptSegment]) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        for text in _texts_from_nested_payload(item):
            if _text_looks_like_transcript_echo(text, segments):
                return True
    return False


def _texts_from_nested_payload(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_texts_from_nested_payload(item))
        return texts
    if isinstance(value, dict):
        texts = []
        for key, item in value.items():
            if str(key) in {
                "actions",
                "decisions",
                "risks",
                "open_questions",
                "owners",
                "source_segment_ids",
                "citation",
                "kind",
                "confidence",
            }:
                continue
            texts.extend(_texts_from_nested_payload(item))
        return texts
    return []


def _text_looks_like_transcript_echo(text: str, segments: list[TranscriptSegment]) -> bool:
    clean = _normalize_for_evidence(text)
    words = clean.split()
    if len(words) < 18:
        return False
    transcript = _normalize_for_evidence(" ".join(segment.text for segment in segments))
    if len(clean) >= 150 and clean in transcript:
        return True
    phrases = [
        " ".join(words[index:index + 10])
        for index in range(0, max(0, len(words) - 9), 14)
    ][:6]
    if not phrases:
        return False
    hits = sum(1 for phrase in phrases if phrase and phrase in transcript)
    return hits / max(1, len(phrases)) >= 0.6


def _summary_has_unsupported_technical_findings(
    summary: dict[str, Any],
    segments: list[TranscriptSegment],
    *,
    technical_signal_count: int,
) -> bool:
    findings = summary.get("technical_findings") if isinstance(summary.get("technical_findings"), list) else []
    if not findings or technical_signal_count >= 2:
        return False
    finding_text = _normalize_for_evidence(json.dumps(findings, ensure_ascii=False))
    if not finding_text:
        return False
    forced_context_terms = {
        "ee index",
        "energy lens",
        "electrical engineering",
        "electrical-engineering",
        "breaker coordination",
        "relay timing",
        "transformer protection",
        "utility tariff",
        "demand charge",
    }
    if any(term in finding_text for term in forced_context_terms):
        return True
    transcript_text = _normalize_for_evidence(" ".join(segment.text for segment in segments))
    if REVIEW_EE_ANALYSIS_RE.search(finding_text) and not REVIEW_EE_ANALYSIS_RE.search(transcript_text):
        return True
    return False


def _summary_forces_external_context(
    summary: dict[str, Any],
    segments: list[TranscriptSegment],
    *,
    review_context: dict[str, Any] | None,
) -> bool:
    if not review_context or _technical_review_needed(segments, None):
        return False
    primary_payload = {
        "title": summary.get("title"),
        "summary": summary.get("summary"),
        "key_points": summary.get("key_points"),
        "project_workstreams": summary.get("project_workstreams"),
        "technical_findings": summary.get("technical_findings"),
        "coverage_notes": summary.get("coverage_notes"),
    }
    primary_text = _normalize_for_evidence(json.dumps(primary_payload, ensure_ascii=False))
    forced_terms = {
        "energy lens",
        "ee index",
        "electrical engineering",
        "electrical-engineering",
        "breaker coordination",
        "relay timing",
        "utility tariff",
        "transformer protection",
    }
    return any(term in primary_text for term in forced_terms)


def _missing_candidate_workstream_names(summary: dict[str, Any], candidates: list[dict[str, Any]]) -> list[str]:
    workstreams = [item for item in summary.get("project_workstreams") or [] if isinstance(item, dict)]
    missing: list[str] = []
    for candidate in candidates:
        if not _candidate_is_required_workstream(candidate):
            continue
        if not _candidate_has_evidence_backed_workstream(candidate, workstreams):
            name = compact_text(candidate.get("project") or "", limit=120)
            if name:
                missing.append(name)
    return _dedupe_compact(missing, limit=8, text_limit=120)


def _candidate_has_evidence_backed_workstream(candidate: dict[str, Any], workstreams: list[dict[str, Any]]) -> bool:
    candidate_sources = set(_compact_payload_list(candidate.get("source_segment_ids"), limit=64))
    for workstream in workstreams:
        if not _workstream_matches_candidate(candidate, workstream):
            continue
        if not _workstream_has_meeting_substance(workstream):
            continue
        workstream_sources = _workstream_source_ids(workstream)
        if candidate_sources and (not workstream_sources or not candidate_sources.intersection(workstream_sources)):
            continue
        if candidate.get("abstract_signal") and not _workstream_payload_has_specific_anchor(workstream, candidate):
            continue
        return True
    return False


def _workstream_matches_candidate(candidate: dict[str, Any], workstream: dict[str, Any]) -> bool:
    candidate_name = str(candidate.get("project") or "")
    workstream_name = str(workstream.get("project") or workstream.get("name") or "")
    if _workstream_name_matches(candidate_name, workstream_name):
        return True
    candidate_sources = set(_compact_payload_list(candidate.get("source_segment_ids"), limit=64))
    workstream_sources = _workstream_source_ids(workstream)
    return bool(candidate_sources and workstream_sources and candidate_sources.intersection(workstream_sources))


def _workstream_source_ids(workstream: dict[str, Any]) -> set[str]:
    return set(_compact_payload_list(workstream.get("source_segment_ids"), limit=64))


def _summary_collapses_multiple_workstreams(summary: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    required = [candidate for candidate in candidates if _candidate_is_required_workstream(candidate)]
    if len(required) < 3:
        return False
    workstreams = [item for item in summary.get("project_workstreams") or [] if isinstance(item, dict)]
    substantive = [item for item in workstreams if _workstream_has_meeting_substance(item)]
    if len(substantive) >= 2:
        return False
    named_count = _summary_distinct_workstream_name_count(summary, required)
    if named_count >= min(3, len(required)):
        return False
    if len(substantive) <= 1:
        return True
    covered = len(required) - len(_missing_candidate_workstream_names(summary, required))
    return covered < min(2, len(required))


def _candidate_is_covered_in_summary_fields(candidate: dict[str, Any], summary: dict[str, Any]) -> bool:
    candidate_name = str(candidate.get("project") or "")
    names = [
        *[str(item) for item in summary.get("projects") or []],
        *[str(item) for item in summary.get("topics") or []],
    ]
    if not any(_workstream_name_matches(candidate_name, name) for name in names):
        return False
    payload_text = _normalize_for_evidence(
        " ".join([
            str(summary.get("summary") or ""),
            " ".join(str(item) for item in summary.get("actions") or []),
            " ".join(str(item) for item in summary.get("unresolved_questions") or []),
            " ".join(str(item) for item in summary.get("risks") or []),
        ])
    )
    candidate_terms = _workstream_match_tokens(candidate_name)
    return bool(candidate_terms and all(term in payload_text for term in candidate_terms[:2]))


def _summary_distinct_workstream_name_count(summary: dict[str, Any], candidates: list[dict[str, Any]]) -> int:
    names: list[str] = []
    for item in summary.get("project_workstreams") or []:
        if isinstance(item, dict):
            names.append(str(item.get("project") or item.get("name") or ""))
    names.extend(str(item) for item in summary.get("projects") or [])
    names.extend(str(item) for item in summary.get("topics") or [])
    keys = {
        _workstream_match_key(name)
        for name in names
        if name and not _weak_workstream_name(name)
    }
    if not keys:
        return 0
    candidate_keys = {_workstream_match_key(str(candidate.get("project") or "")) for candidate in candidates}
    if not candidate_keys:
        return len(keys)
    return len({key for key in keys if any(_workstream_name_matches(key, candidate_key) for candidate_key in candidate_keys)})


def _summary_has_abstract_topic_without_payload(summary: dict[str, Any]) -> bool:
    for item in summary.get("project_workstreams") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("project") or "")
        if REVIEW_ABSTRACT_WORKSTREAM_RE.search(name) and not _workstream_has_meeting_substance(item):
            return True
    return False


def _summary_has_abstract_payload_too_thin(summary: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    workstreams = [item for item in summary.get("project_workstreams") or [] if isinstance(item, dict)]
    for candidate in candidates:
        if not candidate.get("abstract_signal"):
            continue
        for workstream in workstreams:
            if not _workstream_matches_candidate(candidate, workstream):
                continue
            if not _workstream_has_meeting_substance(workstream):
                continue
            candidate_sources = set(_compact_payload_list(candidate.get("source_segment_ids"), limit=64))
            workstream_sources = _workstream_source_ids(workstream)
            if candidate_sources and workstream_sources and not candidate_sources.intersection(workstream_sources):
                continue
            if not _workstream_payload_has_specific_anchor(workstream, candidate):
                return True

    for workstream in workstreams:
        name = str(workstream.get("project") or workstream.get("name") or "")
        payload_text = _workstream_payload_text(workstream)
        if not _workstream_has_meeting_substance(workstream):
            continue
        if REVIEW_ABSTRACT_WORKSTREAM_RE.search(f"{name} {payload_text}") and not _workstream_payload_has_specific_anchor(workstream, None):
            return True
    return False


def _workstream_payload_has_specific_anchor(workstream: dict[str, Any], candidate: dict[str, Any] | None) -> bool:
    payload_text = _workstream_payload_text(workstream)
    if not payload_text:
        return False
    if _payload_has_domain_specific_signal(payload_text):
        return True
    payload_tokens = set(_specific_review_tokens(payload_text))
    if candidate:
        candidate_tokens = set(_specific_review_tokens(_candidate_payload_text(candidate)))
        if len(payload_tokens.intersection(candidate_tokens)) >= 2:
            return True
    return len(payload_tokens) >= 4


def _workstream_payload_text(workstream: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("actions", "decisions", "risks", "open_questions", "owners"):
        value = workstream.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    for key in ("next_checkpoint", "status"):
        if workstream.get(key):
            parts.append(str(workstream.get(key)))
    return " ".join(parts)


def _candidate_payload_text(candidate: dict[str, Any]) -> str:
    parts = [str(candidate.get("project") or "")]
    for key in ("actions", "decisions", "risks", "open_questions", "owners", "technical_terms", "source_segment_ids"):
        value = candidate.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    if candidate.get("next_checkpoint"):
        parts.append(str(candidate["next_checkpoint"]))
    if candidate.get("evidence_quote"):
        parts.append(str(candidate["evidence_quote"]))
    return " ".join(parts)


def _payload_has_domain_specific_signal(text: str) -> bool:
    if REVIEW_EE_ANALYSIS_RE.search(text):
        return True
    project_terms = {
        _normalize_for_evidence(_canonical_review_term(match))
        for match in REVIEW_PROJECT_TERM_RE.findall(text)
    }
    generic_project_terms = {
        "budget",
        "client",
        "customer",
        "design",
        "procurement",
        "requirements",
        "schedule",
        "supplier",
        "vendor",
    }
    if any(term and term not in generic_project_terms for term in project_terms):
        return True
    return bool(re.search(r"\b(?:BESS|EPC|IEEE|LCD|NERC|PGE|RFI|SPCC|CT)\b|\b\d+(?:\.\d+)?\b", text))


def _specific_review_tokens(text: str) -> list[str]:
    generic = {
        "action",
        "actions",
        "active",
        "basis",
        "check",
        "confirm",
        "coordinate",
        "design",
        "discussed",
        "follow",
        "handoff",
        "item",
        "items",
        "next",
        "open",
        "project",
        "requirement",
        "requirements",
        "resolve",
        "review",
        "routing",
        "scope",
        "step",
        "steps",
        "support",
        "track",
        "update",
        "workstream",
    }
    return [
        token
        for token in re.findall(r"[a-z0-9]+", _normalize_for_evidence(text))
        if len(token) > 2 and token not in REVIEW_STOPWORDS and token not in generic
    ]


def _summary_has_weak_technical_basis(summary: dict[str, Any]) -> bool:
    findings = summary.get("technical_findings") if isinstance(summary.get("technical_findings"), list) else []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        if not finding.get("source_segment_ids"):
            return True
        has_basis = bool(
            finding.get("assumptions")
            or finding.get("methods")
            or finding.get("findings")
            or finding.get("recommendations")
            or finding.get("risks")
            or finding.get("data_gaps")
        )
        if not has_basis:
            return True
    return False


def _summary_has_thin_technical_finding(summary: dict[str, Any]) -> bool:
    findings = summary.get("technical_findings") if isinstance(summary.get("technical_findings"), list) else []
    for finding in findings:
        if not isinstance(finding, dict):
            continue
        finding_text = _normalize_for_evidence(json.dumps(finding, ensure_ascii=False))
        has_technical_signal = bool(
            REVIEW_EE_ANALYSIS_RE.search(finding_text)
            or any(term in finding_text for term in ("ee index", "energy lens", "brave", "electrical engineering", "electrical-engineering"))
        )
        basis_count = _technical_finding_basis_count(finding)
        has_open_question_with_reference = bool(finding.get("question") and finding.get("reference_context"))
        finding_payload = " ".join(str(item) for item in _compact_payload_list(finding.get("findings"), limit=4, text_limit=260))
        has_specific_single_finding = bool(
            finding.get("source_segment_ids")
            and _payload_has_domain_specific_signal(finding_payload)
            and len(_specific_review_tokens(finding_payload)) >= 4
        )
        if has_technical_signal and basis_count < 2 and not (basis_count == 1 and (has_open_question_with_reference or has_specific_single_finding)):
            return True
    return False


def _technical_finding_basis_count(finding: dict[str, Any]) -> int:
    count = 0
    for key in ("assumptions", "methods", "findings", "recommendations", "risks", "data_gaps"):
        value = finding.get(key)
        if isinstance(value, list) and any(str(item).strip() for item in value):
            count += 1
        elif isinstance(value, str) and value.strip():
            count += 1
    return count


def _summary_has_context_not_tied_to_transcript(
    summary: dict[str, Any],
    segments: list[TranscriptSegment],
    *,
    review_context: dict[str, Any] | None,
) -> bool:
    if not review_context or not _technical_review_needed(segments, review_context):
        return False
    reference_context = summary.get("reference_context") if isinstance(summary.get("reference_context"), list) else []
    if not reference_context:
        return False
    reference_titles = {
        _normalize_for_evidence(str(item.get("title") or item.get("kind") or ""))
        for item in reference_context
        if isinstance(item, dict)
    }
    if not any(reference_titles):
        return False
    for finding in summary.get("technical_findings") or []:
        if not isinstance(finding, dict):
            continue
        finding_refs = _normalize_for_evidence(" ".join(str(item) for item in finding.get("reference_context") or []))
        if finding_refs and not finding.get("source_segment_ids"):
            return True
    return False


def _summary_has_reference_context_low_value(
    summary: dict[str, Any],
    segments: list[TranscriptSegment],
    *,
    review_context: dict[str, Any] | None,
) -> bool:
    if not review_context or not _technical_review_needed(segments, review_context):
        return False
    reference_context = summary.get("reference_context") if isinstance(summary.get("reference_context"), list) else []
    if not reference_context:
        return False
    terms = _reference_context_citation_terms(reference_context)
    if not terms:
        return False
    technical_refs: list[Any] = []
    for finding in summary.get("technical_findings") or []:
        if not isinstance(finding, dict):
            continue
        technical_refs.append(
            {
                "methods": finding.get("methods"),
                "findings": finding.get("findings"),
                "recommendations": finding.get("recommendations"),
                "reference_context": finding.get("reference_context"),
            }
        )
    cited_text = _normalize_for_evidence(
        json.dumps(
            {
                "technical_findings": technical_refs,
                "coverage_notes": summary.get("coverage_notes") or [],
            },
            ensure_ascii=False,
        )
    )
    if not cited_text:
        return True
    return not any(term in cited_text for term in terms)


def _reference_context_citation_terms(reference_context: list[Any]) -> set[str]:
    terms: set[str] = set()
    for item in reference_context:
        if not isinstance(item, dict):
            continue
        kind = _normalize_for_evidence(str(item.get("kind") or ""))
        title = _normalize_for_evidence(str(item.get("title") or ""))
        body = _normalize_for_evidence(str(item.get("body") or ""))
        combined = f"{kind} {title} {body}"
        if "energy" in combined:
            terms.add("energy lens")
        if "ee" in combined or "electrical" in combined:
            terms.update({"ee index", "ee reference", "electrical reference"})
        if "brave" in combined or "web" in kind or "current public" in combined or "public guidance" in combined:
            terms.update({"brave", "current public", "web context", "public guidance"})
        if title and len(title.split()) <= 8:
            terms.add(title)
        for acronym in re.findall(r"\b[A-Z]{3,}\b", str(item.get("title") or "")):
            terms.add(_normalize_for_evidence(acronym))
    return {term for term in terms if term}


def _workstream_is_covered(candidate: dict[str, Any], workstreams: list[dict[str, Any]]) -> bool:
    candidate_name = str(candidate.get("project") or "")
    candidate_sources = set(str(item) for item in candidate.get("source_segment_ids") or [])
    for workstream in workstreams:
        workstream_name = str(workstream.get("project") or workstream.get("name") or "")
        if _workstream_name_matches(candidate_name, workstream_name) and _workstream_has_meeting_substance(workstream):
            return True
        workstream_sources = set(str(item) for item in workstream.get("source_segment_ids") or [])
        if candidate_sources and workstream_sources and candidate_sources.intersection(workstream_sources):
            if _workstream_name_matches(candidate_name, workstream_name) or _workstream_has_meeting_substance(workstream):
                return True
    return False


def _workstream_name_matches(left: str, right: str) -> bool:
    left_norm = _normalize_for_evidence(left)
    right_norm = _normalize_for_evidence(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm or left_norm in right_norm or right_norm in left_norm:
        return True
    left_tokens = set(_workstream_match_tokens(left_norm))
    right_tokens = set(_workstream_match_tokens(right_norm))
    if not left_tokens or not right_tokens:
        return False
    overlap = left_tokens & right_tokens
    return len(overlap) >= 2 or len(overlap) / max(1, min(len(left_tokens), len(right_tokens))) >= 0.6


def _workstream_match_tokens(text: str) -> list[str]:
    generic = {"project", "workstream", "review", "meeting", "scope", "support", "requirements", "requirement", "design"}
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text)
        if len(token) > 2 and token not in REVIEW_STOPWORDS and token not in generic
    ]


def _summary_has_weak_project_workstreams(
    summary: dict[str, Any],
    *,
    action_signal_count: int,
    important_terms: list[str],
) -> bool:
    workstreams = summary.get("project_workstreams") if isinstance(summary.get("project_workstreams"), list) else []
    if not workstreams or not (action_signal_count >= 2 or len(important_terms) >= 3):
        return False
    return not any(_workstream_has_meeting_substance(item) for item in workstreams if isinstance(item, dict))


def _summary_has_too_few_workstreams(
    summary: dict[str, Any],
    *,
    segment_count: int,
    action_signal_count: int,
    important_terms: list[str],
) -> bool:
    workstreams = summary.get("project_workstreams") if isinstance(summary.get("project_workstreams"), list) else []
    if segment_count < REVIEW_EVALUATION_WINDOW_SIZE * 3 or action_signal_count < 8 or len(important_terms) < 6:
        return False
    substantive = [item for item in workstreams if isinstance(item, dict) and _workstream_has_meeting_substance(item)]
    return len(substantive) < 3


def _workstream_has_meeting_substance(item: dict[str, Any]) -> bool:
    return bool(
        item.get("actions")
        or item.get("decisions")
        or item.get("risks")
        or item.get("open_questions")
        or item.get("next_checkpoint")
        or item.get("owners")
    )


def _parse_review_summary(
    content: str,
    session_id: str,
    extracts: list[ReviewWindowExtract],
    segments: list[TranscriptSegment],
    *,
    review_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = _parse_json_object(content)
    summary = compact_text(payload.get("summary") or "", limit=1800)
    if not summary:
        return _fallback_review_summary(session_id, extracts, segments, review_context=review_context)
    reference_context = _compact_reference_context(payload.get("reference_context"), limit=10)
    if not reference_context:
        reference_context = _reference_context_from_review_context(review_context)
    valid_source_ids = set(_valid_review_source_ids(segments))
    source_segment_ids = _summary_source_ids(payload.get("source_segment_ids"), extracts, segments)
    projects = _compact_payload_list(payload.get("projects"), limit=12) or _dedupe_compact([project for extract in extracts for project in extract.projects], limit=12)
    actions = _compact_payload_list(payload.get("actions"), limit=16) or _dedupe_compact([action for extract in extracts for action in extract.actions], limit=16)
    decisions = _compact_payload_list(payload.get("decisions"), limit=12) or _dedupe_compact([decision for extract in extracts for decision in extract.decisions], limit=12)
    questions = _compact_payload_list(payload.get("unresolved_questions"), limit=12) or _dedupe_compact([question for extract in extracts for question in extract.unresolved_questions], limit=12)
    risks = _compact_payload_list(payload.get("risks"), limit=12) or _dedupe_compact([risk for extract in extracts for risk in extract.risks], limit=12)
    project_workstreams = _compact_project_workstreams(payload.get("project_workstreams"), valid_source_ids, default_source_ids=source_segment_ids, limit=12)
    if not project_workstreams:
        project_workstreams = _fallback_project_workstreams(extracts, segments, projects=projects, actions=actions, decisions=decisions, risks=risks, questions=questions)
    technical_findings = _compact_technical_findings(payload.get("technical_findings"), valid_source_ids, default_source_ids=source_segment_ids, limit=12)
    if not technical_findings and _technical_review_needed(segments, review_context):
        technical_findings = _fallback_technical_findings_for_segments(segments, review_context=review_context, source_segment_ids=source_segment_ids)
    return {
        "session_id": session_id,
        "review_standard": compact_text(payload.get("review_standard") or REVIEW_STANDARD_ID, limit=80),
        "title": compact_text(payload.get("title") or "Review summary", limit=180) or "Review summary",
        "summary": summary,
        "key_points": _compact_payload_list(payload.get("key_points"), limit=12, text_limit=180) or _dedupe_compact(
            [extract.window_summary for extract in extracts if extract.window_summary] + [point for extract in extracts for point in extract.summary_points],
            limit=12,
            text_limit=180,
        ),
        "topics": _compact_payload_list(payload.get("topics"), limit=12) or _dedupe_compact([topic for extract in extracts for topic in extract.topics], limit=12),
        "projects": projects,
        "project_workstreams": project_workstreams,
        "technical_findings": technical_findings,
        "decisions": decisions,
        "actions": actions,
        "unresolved_questions": questions,
        "risks": risks,
        "entities": _compact_payload_list(payload.get("entities"), limit=18) or _dedupe_compact([entity for extract in extracts for entity in extract.entities], limit=18),
        "lessons": _compact_payload_list(payload.get("lessons"), limit=12) or _dedupe_compact([risk for extract in extracts for risk in extract.risks], limit=12),
        "coverage_notes": _compact_payload_list(payload.get("coverage_notes"), limit=6),
        "reference_context": reference_context,
        "context_diagnostics": _compact_context_diagnostics((review_context or {}).get("context_diagnostics")),
        "source_segment_ids": source_segment_ids,
    }


def _fallback_review_summary(
    session_id: str,
    extracts: list[ReviewWindowExtract],
    segments: list[TranscriptSegment],
    *,
    review_context: dict[str, Any] | None = None,
    workstream_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    workstream_candidates = workstream_candidates or _review_workstream_candidates(segments, extracts, [])
    heuristic = _heuristic_meeting_brief(segments)
    actions = _dedupe_compact([*heuristic["actions"], *[action for extract in extracts for action in extract.actions]], limit=16)
    decisions = _dedupe_compact([decision for extract in extracts for decision in extract.decisions], limit=12)
    questions = _dedupe_compact([question for extract in extracts for question in extract.unresolved_questions], limit=12)
    candidate_names = [str(candidate.get("project") or "") for candidate in workstream_candidates]
    topics = _dedupe_compact([*candidate_names, *heuristic["topics"], *[topic for extract in extracts for topic in extract.topics]], limit=12)
    projects = _dedupe_compact([*candidate_names, *heuristic["projects"], *[project for extract in extracts for project in extract.projects]], limit=12)
    entities = _dedupe_compact([entity for extract in extracts for entity in extract.entities], limit=18)
    risks = _dedupe_compact([*heuristic["risks"], *[risk for extract in extracts for risk in extract.risks]], limit=12)
    window_points = _dedupe_compact([extract.window_summary for extract in extracts if extract.window_summary], limit=10, text_limit=180)
    summary_parts = heuristic["key_points"] or window_points or _dedupe_compact([point for extract in extracts for point in extract.summary_points], limit=10, text_limit=180)
    source_segment_ids = _summary_source_ids(None, extracts, segments)
    project_workstreams = _fallback_project_workstreams(
        extracts,
        segments,
        projects=projects,
        actions=actions,
        decisions=decisions,
        risks=risks,
        questions=questions,
        workstream_candidates=workstream_candidates,
    )
    technical_findings = _fallback_technical_findings_for_segments(segments, review_context=review_context, source_segment_ids=source_segment_ids) if _technical_review_needed(segments, review_context) else []
    if heuristic["summary"]:
        summary = heuristic["summary"]
    elif summary_parts:
        summary = " ".join(summary_parts[:5])
    else:
        summary = "Review completed, but the model did not return a reliable structured meeting summary."
    return {
        "session_id": session_id,
        "review_standard": REVIEW_STANDARD_ID,
        "title": heuristic["title"] or "Review summary",
        "summary": compact_text(summary, limit=1800),
        "key_points": summary_parts[:12],
        "topics": topics,
        "projects": projects,
        "project_workstreams": project_workstreams,
        "technical_findings": technical_findings,
        "decisions": decisions,
        "actions": actions,
        "unresolved_questions": questions,
        "risks": risks,
        "entities": entities,
        "lessons": risks,
        "coverage_notes": ["Fallback summary built from ranked transcript windows."],
        "reference_context": _reference_context_from_review_context(review_context),
        "context_diagnostics": _compact_context_diagnostics((review_context or {}).get("context_diagnostics")),
        "source_segment_ids": source_segment_ids,
    }


def _fallback_review_summary_from_cards(
    session_id: str,
    cards: list[SidecarCard],
    extracts: list[ReviewWindowExtract],
    segments: list[TranscriptSegment],
    *,
    review_context: dict[str, Any] | None = None,
    workstream_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    workstream_candidates = workstream_candidates or _review_workstream_candidates(segments, extracts, cards)
    if not cards:
        return _fallback_review_summary(
            session_id,
            extracts,
            segments,
            review_context=review_context,
            workstream_candidates=workstream_candidates,
        )
    groups = _card_workstream_groups(cards)
    source_segment_ids = _ensure_time_bucket_source_ids(
        _dedupe_strings([
            *[source_id for card in cards for source_id in card.source_segment_ids],
            *[source_id for extract in extracts for source_id in extract.source_segment_ids],
            *[source_id for candidate in workstream_candidates for source_id in candidate.get("source_segment_ids", [])],
        ]),
        segments,
    )[:36]
    workstreams = [
        _workstream_from_card_group(name, group, source_segment_ids)
        for name, group in groups[:8]
    ]
    for candidate_workstream in _project_workstreams_from_candidates(workstream_candidates):
        if not _workstream_is_covered(candidate_workstream, workstreams):
            workstreams.append(candidate_workstream)
        if len(workstreams) >= 12:
            break
    actions = _dedupe_compact([
        _card_meeting_text(card)
        for card in cards
        if card.category == "action"
    ], limit=16, text_limit=220)
    decisions = _dedupe_compact([
        _card_meeting_text(card)
        for card in cards
        if card.category == "decision"
    ], limit=12, text_limit=220)
    questions = _dedupe_compact([
        _card_meeting_text(card)
        for card in cards
        if card.category in {"question", "clarification"} or card.missing_info
    ], limit=12, text_limit=220)
    risks = _dedupe_compact([
        _card_meeting_text(card)
        for card in cards
        if card.category == "risk" or _card_mentions_risk(card)
    ], limit=12, text_limit=220)
    topics = _dedupe_compact([item.get("project", "") for item in workstreams] + [name for name, _ in groups] + [card.title for card in cards], limit=12)
    projects = [item["project"] for item in workstreams if item.get("project")]
    workstream_names = projects or topics
    summary = (
        f"The review centered on {_human_join(workstream_names[:4])}. "
        "The useful outputs are the evidence-backed workstreams, action owners where explicit, open technical questions, and risks below."
    )
    technical_findings = _technical_findings_from_cards(cards, groups, segments, review_context=review_context, default_source_ids=source_segment_ids)
    reference_context = _reference_context_from_review_context(review_context)
    return {
        "session_id": session_id,
        "review_standard": REVIEW_STANDARD_ID,
        "title": compact_text(f"{_human_join(workstream_names[:2])} review", limit=180) if workstream_names else "Meeting review",
        "summary": compact_text(summary, limit=1800),
        "key_points": _dedupe_compact([
            f"{item['project']}: {_human_join([*(item.get('actions') or [])[:2], *(item.get('open_questions') or [])[:1], *(item.get('risks') or [])[:1]])}"
            for item in workstreams
            if item.get("project")
        ], limit=8, text_limit=220),
        "topics": topics,
        "projects": projects,
        "project_workstreams": workstreams,
        "technical_findings": technical_findings,
        "decisions": decisions,
        "actions": actions,
        "unresolved_questions": questions,
        "risks": risks,
        "entities": _dedupe_compact([owner for item in workstreams for owner in item.get("owners", [])], limit=18),
        "lessons": [],
        "coverage_notes": ["Fallback summary synthesized from accepted evidence-backed Review cards after the model summary failed usefulness checks."],
        "reference_context": reference_context,
        "context_diagnostics": _compact_context_diagnostics((review_context or {}).get("context_diagnostics")),
        "source_segment_ids": source_segment_ids,
    }


def _card_workstream_groups(cards: list[SidecarCard]) -> list[tuple[str, list[SidecarCard]]]:
    groups: dict[str, list[SidecarCard]] = {}
    order: list[str] = []
    for card in cards:
        name = _infer_card_workstream(card)
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(card)
    return sorted(
        [(name, groups[name]) for name in order],
        key=lambda item: (-len(item[1]), order.index(item[0])),
    )


def _infer_card_workstream(card: SidecarCard) -> str:
    text = _normalize_for_evidence(f"{card.title} {card.body} {card.why_now} {card.evidence_quote}")
    rules = [
        (("westwood", "pge", "breaker", "relay", "coordination"), "PGE / Westwood relay coordination"),
        (("gen-tie", "gentie", "lark", "pge", "portland general", "interconnection"), "Gen-tie / PGE interconnection"),
        (("phase a", "phase b", "authorization", "task authorization", "budget"), "Phase A/B authorization and budget"),
        (("epc", "skid", "oil containment", "secondary containment", "spcc"), "EPC skid and transformer containment"),
        (("transformer loss", "losses", "225 mva", "round-trip efficiency", "loss guarantee"), "Transformer loss data"),
        (("load flow", "short circuit", "drawings", "studies", "client drawing"), "Client drawing and study review"),
        (("engineer time", "fars", "pnc", "anthony", "tao"), "PNC engineer support requests"),
    ]
    best_name = ""
    best_score = 0
    for terms, name in rules:
        score = sum(1 for term in terms if term in text)
        if score > best_score:
            best_name = name
            best_score = score
    if best_name:
        return best_name
    return compact_text(card.title, limit=120) or "Meeting follow-up"


def _workstream_from_card_group(name: str, cards: list[SidecarCard], fallback_source_ids: list[str]) -> dict[str, Any]:
    actions = _dedupe_compact([_card_meeting_text(card) for card in cards if card.category == "action"], limit=6, text_limit=180)
    decisions = _dedupe_compact([_card_meeting_text(card) for card in cards if card.category == "decision"], limit=5, text_limit=180)
    questions = _dedupe_compact([_card_meeting_text(card) for card in cards if card.category in {"question", "clarification"} or card.missing_info], limit=5, text_limit=180)
    risks = _dedupe_compact([_card_meeting_text(card) for card in cards if card.category == "risk" or _card_mentions_risk(card)], limit=5, text_limit=180)
    owners = _dedupe_strings([owner for owner in (_clean_review_owner(card.owner) for card in cards) if owner])[:6]
    source_ids = _dedupe_strings([source_id for card in cards for source_id in card.source_segment_ids]) or fallback_source_ids
    payload = {
        "project": name,
        "status": "active" if actions else "decision_needed" if questions else "watch" if risks else "discussed",
        "decisions": decisions,
        "actions": actions,
        "risks": risks,
        "open_questions": questions,
        "owners": owners,
        "source_segment_ids": source_ids[:12],
    }
    return {key: value for key, value in payload.items() if value not in ("", [], None)}


def _card_meeting_text(card: SidecarCard) -> str:
    return compact_text(card.body or card.title, limit=260)


def _card_mentions_risk(card: SidecarCard) -> bool:
    text = _normalize_for_evidence(f"{card.title} {card.body} {card.why_now} {card.missing_info or ''}")
    return any(term in text for term in ("risk", "delay", "blocked", "concern", "failure", "missing", "unclear", "dispute", "overrun"))


def _technical_findings_from_cards(
    cards: list[SidecarCard],
    groups: list[tuple[str, list[SidecarCard]]],
    segments: list[TranscriptSegment],
    *,
    review_context: dict[str, Any] | None,
    default_source_ids: list[str],
) -> list[dict[str, Any]]:
    if not _technical_review_needed(segments, review_context):
        return []
    reference_titles = _dedupe_compact([
        str(item.get("title") or item.get("kind") or "")
        for item in _reference_context_from_review_context(review_context)
    ], limit=6)
    findings: list[dict[str, Any]] = []
    for name, group in groups:
        technical_cards = [
            card
            for card in group
            if REVIEW_EE_ANALYSIS_RE.search(f"{card.title} {card.body} {card.evidence_quote}")
        ]
        if not technical_cards:
            continue
        actions = _dedupe_compact([_card_meeting_text(card) for card in technical_cards if card.category == "action"], limit=4, text_limit=180)
        questions = _dedupe_compact([_card_meeting_text(card) for card in technical_cards if card.category in {"question", "clarification"}], limit=3, text_limit=180)
        risks = _dedupe_compact([_card_meeting_text(card) for card in technical_cards if card.category == "risk" or _card_mentions_risk(card)], limit=3, text_limit=180)
        evidence_notes = _dedupe_compact([
            _card_meeting_text(card)
            for card in technical_cards
            if card.category in {"note", "decision", "question"}
        ], limit=4, text_limit=180)
        source_ids = _dedupe_strings([source_id for card in technical_cards for source_id in card.source_segment_ids]) or default_source_ids
        findings.append({
            "topic": name,
            "question": questions[0] if questions else "",
            "assumptions": [],
            "methods": [],
            "findings": evidence_notes or [_card_meeting_text(technical_cards[0])],
            "recommendations": actions,
            "risks": risks,
            "data_gaps": questions[1:],
            "reference_context": reference_titles,
            "confidence": "medium",
            "source_segment_ids": source_ids[:12],
        })
        if len(findings) >= 5:
            break
    return findings


def _ensure_time_bucket_source_ids(source_ids: list[str], segments: list[TranscriptSegment]) -> list[str]:
    if not segments:
        return _dedupe_strings(source_ids)
    index_by_source_id = _segment_index_by_source_id(segments)
    indexes = {index_by_source_id[source_id] for source_id in source_ids if source_id in index_by_source_id}
    buckets = _covered_time_buckets(indexes, len(segments))
    additions: list[str] = []
    bucket_targets = {
        "early": 0,
        "middle": len(segments) // 2,
        "late": max(0, len(segments) - 1),
    }
    for bucket, index in bucket_targets.items():
        if bucket not in buckets and 0 <= index < len(segments):
            additions.append(segments[index].id)
    return _dedupe_strings([*additions, *source_ids])


def _summary_source_ids(
    raw: object,
    extracts: list[ReviewWindowExtract],
    segments: list[TranscriptSegment],
) -> list[str]:
    valid_ids = set(_valid_review_source_ids(segments))
    if isinstance(raw, list):
        ids = [str(item) for item in raw if str(item) in valid_ids]
        if ids:
            return _dedupe_strings(ids)[:36]
    ids = [source_id for extract in extracts for source_id in extract.source_segment_ids if source_id in valid_ids]
    if ids:
        return _dedupe_strings(ids)[:36]
    return [segment.id for segment in segments[:24]]


def _compact_payload_list(value: object, *, limit: int, text_limit: int = 320) -> list[str]:
    if not isinstance(value, list):
        return []
    return _dedupe_compact([_payload_item_text(item) for item in value], limit=limit, text_limit=text_limit)


def _payload_item_text(item: object) -> str:
    if isinstance(item, dict):
        for key in ("description", "text", "body", "title", "action", "question", "risk", "decision", "summary"):
            value = item.get(key)
            if value:
                return str(value)
        return ""
    return str(item)


def _dedupe_compact(items: list[str], *, limit: int, text_limit: int = 320) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        text = compact_text(item, limit=text_limit)
        key = _normalize_for_evidence(text)
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


def _review_incomplete_card(
    session_id: str,
    segments: list[TranscriptSegment],
    *,
    save_result: bool,
) -> SidecarCard:
    source_ids = [segment.id for segment in segments[:6]]
    evidence = compact_text(" ".join(segment.text for segment in segments[:2]), limit=320)
    return create_sidecar_card(
        session_id=session_id,
        category="status",
        title="Review incomplete",
        body="The transcript was saved, but the model did not return reliable structured meeting output.",
        why_now="Review needs a manual pass or a rerun with a stronger local model response.",
        priority="normal",
        confidence=0.35,
        source_segment_ids=source_ids,
        source_type="model_fallback",
        ephemeral=not save_result,
        evidence_quote=evidence,
        card_key="review:status:incomplete",
    )


def _segment_with_text(segment: TranscriptSegment, text: str) -> TranscriptSegment:
    clean_text = clean_transcript_text(text)
    if not clean_text:
        clean_text = segment.text
    return TranscriptSegment(
        id=segment.id,
        session_id=segment.session_id,
        start_s=segment.start_s,
        end_s=segment.end_s,
        text=clean_text,
        is_final=True,
        created_at=segment.created_at,
        speaker_role=segment.speaker_role,
        speaker_label=segment.speaker_label,
        speaker_confidence=segment.speaker_confidence,
        speaker_match_score=segment.speaker_match_score,
        speaker_match_reason=segment.speaker_match_reason,
        speaker_low_confidence=segment.speaker_low_confidence,
        diarization_speaker_id=segment.diarization_speaker_id,
        source_segment_ids=segment.source_segment_ids or [segment.id],
    )


def _segment_with_speaker_payload(segment: TranscriptSegment, payload: dict[str, Any]) -> TranscriptSegment:
    return replace(
        segment,
        speaker_role=payload.get("speaker_role") or segment.speaker_role,
        speaker_label=payload.get("speaker_label") or segment.speaker_label,
        speaker_confidence=(
            payload.get("speaker_confidence")
            if payload.get("speaker_confidence") is not None
            else segment.speaker_confidence
        ),
        speaker_match_score=(
            payload.get("speaker_match_score")
            if payload.get("speaker_match_score") is not None
            else segment.speaker_match_score
        ),
        speaker_match_reason=payload.get("speaker_match_reason") or segment.speaker_match_reason,
        speaker_low_confidence=(
            bool(payload.get("speaker_low_confidence"))
            if payload.get("speaker_low_confidence") is not None
            else segment.speaker_low_confidence
        ),
        diarization_speaker_id=payload.get("diarization_speaker_id") or segment.diarization_speaker_id,
    )


def _speaker_identity_diagnostics(segments: list[TranscriptSegment], status: dict[str, Any]) -> dict[str, Any]:
    scores = [
        float(segment.speaker_match_score)
        for segment in segments
        if segment.speaker_match_score is not None
    ]
    roles: dict[str, int] = {}
    for segment in segments:
        role = segment.speaker_role or "unlabeled"
        roles[role] = roles.get(role, 0) + 1
    profile = status.get("profile") if isinstance(status.get("profile"), dict) else {}
    backend = status.get("backend") if isinstance(status.get("backend"), dict) else {}
    return {
        "ready": bool(status.get("ready")),
        "enrollment_status": status.get("enrollment_status"),
        "profile_label": profile.get("display_name") or "BP",
        "backend_available": bool(backend.get("available")),
        "bp_segment_count": roles.get("user", 0),
        "other_segment_count": roles.get("other", 0),
        "unknown_segment_count": roles.get("unknown", 0),
        "unlabeled_segment_count": roles.get("unlabeled", 0),
        "low_confidence_count": sum(1 for segment in segments if segment.speaker_low_confidence),
        "match_score_min": round(min(scores), 4) if scores else None,
        "match_score_mean": round(sum(scores) / len(scores), 4) if scores else None,
        "match_score_max": round(max(scores), 4) if scores else None,
    }


def _corrected_text_by_id(content: str) -> dict[str, str]:
    payload = _parse_json_object(content)
    items = payload.get("segments")
    if not isinstance(items, list):
        return {}
    result: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        segment_id = str(item.get("id") or "").strip()
        text = clean_transcript_text(str(item.get("text") or ""))
        if segment_id and text:
            result[segment_id] = text
    return result


def _evidence_backed_action_fallback(
    cards: list[SidecarCard],
    segments: list[TranscriptSegment],
    *,
    max_cards: int,
) -> list[SidecarCard]:
    if max_cards <= 0:
        return []
    segment_text_by_id = {segment.id: _normalize_for_evidence(segment.text) for segment in segments}
    result: list[SidecarCard] = []
    for card in cards:
        if card.category != "action" or not card.source_segment_ids or not card.evidence_quote:
            continue
        evidence = _normalize_for_evidence(card.evidence_quote)
        if not evidence:
            continue
        if any(evidence in segment_text_by_id.get(source_id, "") for source_id in card.source_segment_ids):
            result.append(card)
        if len(result) >= max_cards:
            break
    return result


def _dedupe_cards(cards: list[SidecarCard]) -> list[SidecarCard]:
    seen: set[str] = set()
    result: list[SidecarCard] = []
    for card in cards:
        key = card.card_key or f"{card.category}:{card.title}:{','.join(card.source_segment_ids)}"
        if key in seen:
            continue
        seen.add(key)
        result.append(card)
    return result


def _coverage_ratio(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    covered = 0
    for item in items:
        value = item.get(key)
        if isinstance(value, list):
            if value:
                covered += 1
        elif value:
            covered += 1
    return round(covered / max(1, len(items)), 3)


def _card_usefulness_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"card_usefulness_flags": []}
    missing_evidence = 0
    missing_consequence = 0
    for item in items:
        if not item.get("evidence_quote") or not item.get("source_segment_ids"):
            missing_evidence += 1
        body = _normalize_for_evidence(f"{item.get('body') or ''} {item.get('why_now') or ''} {item.get('missing_info') or ''}")
        if not any(term in body for term in ("because", "owner", "due", "risk", "blocked", "decision", "follow", "next", "missing", "depends")):
            missing_consequence += 1
    flags: list[str] = []
    if missing_evidence / max(1, len(items)) > 0.35:
        flags.append("cards_missing_evidence")
    if missing_consequence / max(1, len(items)) > 0.5:
        flags.append("cards_missing_consequence")
    return {
        "card_usefulness_flags": flags,
        "card_missing_evidence_count": missing_evidence,
        "card_missing_consequence_count": missing_consequence,
    }


def _normalize_for_evidence(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
        if fenced:
            return json.loads(fenced.group(1))
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def _batched(items: list[TranscriptSegment], size: int) -> list[list[TranscriptSegment]]:
    return [items[index:index + size] for index in range(0, len(items), size)]


def _job_progress_percent(job: ReviewJob) -> int:
    if job.status == "completed":
        return 100
    weights = {
        "conditioning": 15,
        "transcribing": 45,
        "reviewing": 25,
        "evaluating": 10,
        "completed": 5,
    }
    total = sum(weights.values())
    weighted_progress = sum(weights[key] * (job.steps[key].progress / 100.0) for key in REVIEW_STEP_ORDER)
    return _clamp_progress(round((weighted_progress / total) * 100))


def _clamp_progress(value: float | int) -> int:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, number))


def _safe_review_filename(filename: str | None) -> str:
    original = Path(filename or "review-audio").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", original).strip("._")
    return (safe or "review-audio")[:160]


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())
