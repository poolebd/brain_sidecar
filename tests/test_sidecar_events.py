from __future__ import annotations

import asyncio
from pathlib import Path

from brain_sidecar.config import Settings
from brain_sidecar.core.audio import AudioCapture
from brain_sidecar.core.dedupe import TranscriptDeduplicator
from brain_sidecar.core.events import (
    EVENT_AUDIO_STATUS,
    EVENT_SIDECAR_CARD,
    EVENT_TRANSCRIPT_FINAL,
    EVENT_TRANSCRIPT_PARTIAL,
    SidecarEvent,
)
from brain_sidecar.core.models import NoteCard, TranscriptSegment
from brain_sidecar.core.notes import NoteSynthesisResult
from brain_sidecar.core.session import ActiveSession, AudioWindow, SessionManager, normalize_mic_tuning, suggest_microphone_tuning
from brain_sidecar.core.transcription import TranscribedSpan, TranscriptionResult


class DummyCapture(AudioCapture):
    async def chunks(self):
        if False:
            yield b""

    async def stop(self) -> None:
        return None


class FakeTranscriber:
    model_size = "fake-asr"

    async def load(self) -> None:
        return None

    async def transcribe_pcm16(
        self,
        pcm: bytes,
        start_offset_s: float,
        *,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            model="fake-asr",
            language="en",
            spans=[
                TranscribedSpan(
                    start_s=start_offset_s,
                    end_s=start_offset_s + 1.25,
                    text="We should ask about rollback risk",
                )
            ],
        )


class SequencedTranscriber:
    model_size = "fake-asr"

    def __init__(self, results: list[TranscriptionResult]) -> None:
        self.results = results

    async def load(self) -> None:
        return None

    async def transcribe_pcm16(
        self,
        pcm: bytes,
        start_offset_s: float,
        *,
        initial_prompt: str | None = None,
    ) -> TranscriptionResult:
        if not self.results:
            return TranscriptionResult(model="fake-asr", language="en", spans=[], audio_rms=0.0)
        return self.results.pop(0)


class StaticNotes:
    async def synthesize(self, session_id, recent_segments, recall_hits):
        return NoteSynthesisResult(
            notes=[
                NoteCard(
                    id="note-action",
                    session_id=session_id,
                    kind="action",
                    title="Rollback owner",
                    body="Confirm who owns the rollback plan before release.",
                    source_segment_ids=[segment.id for segment in recent_segments],
                    evidence_quote=recent_segments[-1].text,
                )
            ]
        )


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        host="127.0.0.1",
        port=8765,
        asr_primary_model="tiny.en",
        asr_fallback_model="tiny.en",
        asr_compute_type="float16",
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="qwen3.5:9b",
        ollama_embed_model="embeddinggemma",
        transcription_queue_size=2,
        postprocess_queue_size=1,
        disable_live_embeddings=True,
    )


def test_partial_transcript_event_contract_and_no_persistence(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    manager.transcriber = FakeTranscriber()  # type: ignore[assignment]
    session = manager.storage.create_session("partial")
    manager._active[session.id] = active_session(session.id, save_transcript=False)

    collector = event_loop.create_task(collect_event(manager, session.id, EVENT_TRANSCRIPT_PARTIAL))
    event_loop.run_until_complete(
        manager._transcribe_partial_window(session.id, AudioWindow(b"\0" * 64000, 2.0, preview=True))
    )
    event = event_loop.run_until_complete(asyncio.wait_for(collector, timeout=1.0))

    assert event.payload["text"] == "We should ask about rollback risk"
    assert event.payload["is_final"] is False
    assert event.payload["asr_model"] == "fake-asr"
    assert event.payload["queue_depth"] == 0
    assert event.payload["transcript_retention"] == "temporary"
    assert event.payload["raw_audio_retained"] is False
    assert "speaker_role" in event.payload
    assert manager.storage.recent_segments(session.id) == []


def test_final_transcript_event_uses_authoritative_contract(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    manager.transcriber = FakeTranscriber()  # type: ignore[assignment]
    session = manager.storage.create_session("final")
    manager._active[session.id] = active_session(session.id, save_transcript=True)

    collector = event_loop.create_task(collect_event(manager, session.id, EVENT_TRANSCRIPT_FINAL))
    event_loop.run_until_complete(manager._transcribe_window(session.id, AudioWindow(b"\0" * 64000, 4.0)))
    event = event_loop.run_until_complete(asyncio.wait_for(collector, timeout=1.0))

    assert event.payload["text"] == "We should ask about rollback risk"
    assert event.payload["is_final"] is True
    assert event.payload["asr_model"] == "fake-asr"
    assert event.payload["transcript_retention"] == "saved"
    assert event.payload["raw_audio_retained"] is False
    assert manager.storage.recent_segments(session.id)[0].text == "We should ask about rollback risk"
    assert manager.storage.recent_segments(session.id)[0].speaker_role == "unknown"


def test_overlapping_cleaner_final_replaces_clipped_transcript_row(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    manager.transcriber = SequencedTranscriber([
        TranscriptionResult(
            model="fake-asr",
            language="en",
            audio_rms=0.02,
            spans=[
                TranscribedSpan(
                    start_s=0.0,
                    end_s=3.4,
                    text="We should review the relay settings.",
                )
            ],
        ),
        TranscriptionResult(
            model="fake-asr",
            language="en",
            audio_rms=0.02,
            spans=[
                TranscribedSpan(
                    start_s=1.1,
                    end_s=4.6,
                    text="We should review the relay settings before Friday.",
                )
            ],
        ),
    ])  # type: ignore[assignment]
    session = manager.storage.create_session("replacement")
    manager._active[session.id] = active_session(session.id, save_transcript=True)

    first_collector = event_loop.create_task(collect_event(manager, session.id, EVENT_TRANSCRIPT_FINAL))
    event_loop.run_until_complete(manager._transcribe_window(session.id, AudioWindow(b"\0" * 64000, 0.0)))
    first_event = event_loop.run_until_complete(asyncio.wait_for(first_collector, timeout=1.0))

    second_collector = event_loop.create_task(collect_event(manager, session.id, EVENT_TRANSCRIPT_FINAL))
    event_loop.run_until_complete(manager._transcribe_window(session.id, AudioWindow(b"\0" * 64000, 1.1)))
    second_event = event_loop.run_until_complete(asyncio.wait_for(second_collector, timeout=1.0))

    stored = manager.storage.recent_segments(session.id, limit=5)
    assert len(stored) == 1
    assert stored[0].text == "We should review the relay settings before Friday."
    assert second_event.payload["id"] == first_event.payload["id"]
    assert second_event.payload["replaces_segment_id"] == first_event.payload["id"]
    assert second_event.payload["source_segment_ids"][0] == first_event.payload["id"]
    assert len(manager._active[session.id].recent_segments) == 1
    assert manager._active[session.id].final_segments_replaced == 1


def test_empty_asr_windows_publish_capture_diagnostics(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    manager.transcriber = SequencedTranscriber([
        TranscriptionResult(model="fake-asr", language=None, spans=[], audio_rms=0.0004),
    ])  # type: ignore[assignment]
    session = manager.storage.create_session("quiet")
    active = active_session(session.id, save_transcript=True)
    manager._active[session.id] = active

    collector = event_loop.create_task(collect_event(manager, session.id, EVENT_AUDIO_STATUS))
    batch = event_loop.run_until_complete(manager._transcribe_window(session.id, AudioWindow(b"\0" * 64000, 0.0)))
    event = event_loop.run_until_complete(asyncio.wait_for(collector, timeout=1.0))

    assert batch.segments == []
    assert active.silent_windows == 1
    assert active.asr_empty_windows == 1
    assert event.payload["last_audio_rms"] == 0.0004
    assert event.payload["silent_windows"] == 1
    assert event.payload["asr_empty_windows"] == 1
    assert manager.storage.recent_segments(session.id) == []


def test_note_updates_also_publish_normalized_sidecar_cards(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    manager.notes = StaticNotes()  # type: ignore[assignment]
    session = manager.storage.create_session("notes")
    active = active_session(session.id, save_transcript=False)
    active.recent_segments = [
        TranscriptSegment(
            id="seg-current",
            session_id=session.id,
            start_s=0.0,
            end_s=2.0,
            text="Please confirm who owns the rollback plan before release.",
        )
    ]
    manager._active[session.id] = active

    collector = event_loop.create_task(collect_event(manager, session.id, EVENT_SIDECAR_CARD))
    event_loop.run_until_complete(manager._refresh_notes(session.id))
    event = event_loop.run_until_complete(asyncio.wait_for(collector, timeout=1.0))

    assert event.payload["category"] == "action"
    assert event.payload["title"] == "Rollback owner"
    assert event.payload["body"] == "Confirm who owns the rollback plan before release."
    assert event.payload["priority"] == "high"
    assert event.payload["source_segment_ids"] == ["seg-current"]
    assert event.payload["source_type"] == "transcript"
    assert event.payload["ephemeral"] is True
    assert event.payload["raw_audio_retained"] is False
    assert active.meeting_diagnostics["generated_candidate_count"] == 1
    assert active.meeting_diagnostics["accepted_count"] == 1
    assert active.meeting_diagnostics["suppressed_count"] == 0


def test_final_transcript_can_publish_company_reference_card(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    session = manager.storage.create_session("company-ref")
    active = active_session(session.id, save_transcript=False)
    manager._active[session.id] = active

    collector = event_loop.create_task(collect_event(manager, session.id, EVENT_SIDECAR_CARD))
    event_loop.run_until_complete(
        manager._accept_final_segment(
            session.id,
            active,
            TranscriptSegment(
                id="seg-company",
                session_id=session.id,
                start_s=0.0,
                end_s=2.0,
                text="We need Siemens to answer the power quality question.",
            ),
            asr_model="fake-asr",
            speaker_payload={},
            asr_duration_ms=1.0,
        )
    )
    event_loop.run_until_complete(manager._publish_company_reference_cards(session.id, active))
    event = event_loop.run_until_complete(asyncio.wait_for(collector, timeout=1.0))

    assert event.payload["category"] == "reference"
    assert event.payload["source_type"] == "company_ref"
    assert event.payload["title"] == "Siemens"
    assert event.payload["priority"] == "low"
    assert event.payload["source_segment_ids"] == ["seg-company"]
    assert event.payload["evidence_quote"] == "We need Siemens to answer the power quality question."
    assert event.payload["raw_audio_retained"] is False
    assert manager.storage.session_note_cards(session.id) == []


async def collect_event(manager: SessionManager, session_id: str, event_type: str) -> SidecarEvent:
    async for event in manager.bus.subscribe(session_id):
        if event.type == event_type:
            return event
    raise AssertionError("subscription ended")


def active_session(session_id: str, *, save_transcript: bool) -> ActiveSession:
    return ActiveSession(
        id=session_id,
        capture=DummyCapture(),
        window_queue=asyncio.Queue(maxsize=2),
        postprocess_queue=asyncio.Queue(maxsize=1),
        tasks=[],
        deduper=TranscriptDeduplicator(max_recent=4, similarity_threshold=0.88),
        save_transcript=save_transcript,
    )


def test_mic_tuning_clamps_excessive_gain() -> None:
    tuning = normalize_mic_tuning({"input_gain_db": 24, "speech_sensitivity": "normal"})

    assert tuning["input_gain_db"] == 12.0


def test_mic_tuning_suggestions_never_exceed_safe_gain() -> None:
    suggested = suggest_microphone_tuning(
        {"usable_speech_seconds": 0.2, "rms": 0.002, "peak": 0.1, "issues": []},
        {"input_gain_db": 10, "speech_sensitivity": "normal", "auto_level": True},
    )

    assert suggested["input_gain_db"] == 12.0
