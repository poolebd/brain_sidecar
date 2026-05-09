from __future__ import annotations

import asyncio
from pathlib import Path

from brain_sidecar.config import Settings
from brain_sidecar.core.audio import AudioCapture
from brain_sidecar.core.dedupe import TranscriptDeduplicator
from brain_sidecar.core.events import EVENT_NOTE_UPDATE, EVENT_RECALL_HIT, SidecarEvent
from brain_sidecar.core.models import SearchHit, TranscriptSegment
from brain_sidecar.core.notes import NoteSynthesisResult
from brain_sidecar.core.session import ActiveSession, SessionManager
from brain_sidecar.core.web_context import WebContextCandidate, WebSearchResult


class DummyCapture(AudioCapture):
    async def chunks(self):
        if False:
            yield b""

    async def stop(self) -> None:
        return None


class EmptyNotes:
    async def synthesize(self, session_id, recent_segments, recall_hits):
        return NoteSynthesisResult(notes=[])


class RecordingSearch:
    def __init__(self, results: list[WebSearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, str | None]] = []

    async def search(self, query: str, *, freshness: str | None = None) -> list[WebSearchResult]:
        self.calls.append((query, freshness))
        return self.results


class RecordingRecall:
    async def search(self, query: str, limit: int = 5) -> list[SearchHit]:
        return [
            SearchHit(
                source_type="session",
                source_id="prior-session",
                text="A prior session mentioned the same rollout risk.",
                score=0.88,
                metadata={},
            )
        ]


def make_settings(
    tmp_path: Path,
    *,
    min_interval: float = 0.0,
    disable_live_embeddings: bool = True,
    web_context_enabled: bool = True,
) -> Settings:
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
        transcription_queue_size=1,
        postprocess_queue_size=1,
        disable_live_embeddings=disable_live_embeddings,
        web_context_enabled=web_context_enabled,
        brave_search_api_key="test-key",
        web_context_min_interval_seconds=min_interval,
    )


def test_refresh_notes_enqueues_web_context_without_calling_search(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    manager.notes = EmptyNotes()  # type: ignore[assignment]
    search = RecordingSearch()
    manager.web_search = search  # type: ignore[assignment]
    session = manager.storage.create_session("web context")
    active = make_active(session.id)
    manager._active[session.id] = active
    manager.storage.add_transcript_segment(
        segment(session.id, "What are current best practices for vector database indexing?")
    )

    event_loop.run_until_complete(manager._refresh_notes(session.id))

    assert active.web_context_queue is not None
    assert active.web_context_queue.qsize() == 1
    assert search.calls == []


def test_session_web_context_override_can_enable_global_off(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path, web_context_enabled=False))
    session = manager.storage.create_session("web context override")
    active = make_active(session.id, web_context_enabled=True)
    manager._active[session.id] = active

    manager._maybe_enqueue_web_context(
        session.id,
        [segment(session.id, "What are current best practices for vector database indexing?")],
    )

    assert active.web_context_queue is not None
    assert active.web_context_queue.qsize() == 1


def test_session_web_context_override_can_disable_global_on(tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path, web_context_enabled=True))
    session = manager.storage.create_session("web context override")
    active = make_active(session.id, web_context_enabled=False)
    manager._active[session.id] = active

    manager._maybe_enqueue_web_context(
        session.id,
        [segment(session.id, "What are current best practices for vector database indexing?")],
    )

    assert active.web_context_queue is not None
    assert active.web_context_queue.qsize() == 0


def test_web_context_worker_dedupes_repeated_query(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    search = RecordingSearch()
    manager.web_search = search  # type: ignore[assignment]
    active = make_active("ses_test")
    manager._active[active.id] = active
    assert active.web_context_queue is not None
    candidate = candidate_for("vector database indexing")

    task = event_loop.create_task(manager._run_web_context_loop(active.id, active.web_context_queue))
    event_loop.run_until_complete(active.web_context_queue.put(candidate))
    event_loop.run_until_complete(active.web_context_queue.put(candidate))
    event_loop.run_until_complete(asyncio.wait_for(active.web_context_queue.join(), timeout=1.0))
    task.cancel()
    event_loop.run_until_complete(asyncio.gather(task, return_exceptions=True))

    assert search.calls == [("vector database indexing", "py")]


def test_web_context_note_is_published_but_not_persisted(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    manager.web_search = RecordingSearch(
        [
            WebSearchResult(
                title="Vector search indexing guide",
                url="https://example.com/vector-indexing",
                description="Compares index types and recall tradeoffs.",
            )
        ]
    )  # type: ignore[assignment]
    session = manager.storage.create_session("web context")
    active = make_active(session.id)
    manager._active[session.id] = active
    assert active.web_context_queue is not None

    collector = event_loop.create_task(collect_note_update(manager, session.id))
    task = event_loop.create_task(manager._run_web_context_loop(session.id, active.web_context_queue))
    event_loop.run_until_complete(active.web_context_queue.put(candidate_for("vector database indexing")))
    event = event_loop.run_until_complete(asyncio.wait_for(collector, timeout=1.0))
    task.cancel()
    event_loop.run_until_complete(asyncio.gather(task, return_exceptions=True))

    assert event.payload["ephemeral"] is True
    assert event.payload["source_type"] == "brave_web"
    assert "I found a few current references" in event.payload["body"]
    assert "Live web context found" not in event.payload["body"]
    assert event.payload["sources"] == [
        {"title": "Vector search indexing guide", "url": "https://example.com/vector-indexing"}
    ]
    row = manager.storage.conn.execute("select count(*) as count from note_cards").fetchone()
    assert row["count"] == 0


def test_live_recall_events_include_source_segment_ids(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path, disable_live_embeddings=False))
    manager.notes = EmptyNotes()  # type: ignore[assignment]
    manager.recall = RecordingRecall()  # type: ignore[assignment]
    session = manager.storage.create_session("recall pairing")
    manager.storage.add_transcript_segment(segment(session.id, "We need to revisit Apollo rollout risk.", "seg_old"))
    manager.storage.add_transcript_segment(segment(session.id, "The platform team needs a short follow-up.", "seg_latest"))

    collector = event_loop.create_task(collect_recall_hit(manager, session.id))
    event_loop.run_until_complete(asyncio.sleep(0))
    event_loop.run_until_complete(manager._refresh_notes(session.id))
    event = event_loop.run_until_complete(asyncio.wait_for(collector, timeout=1.0))

    assert event.payload["source_segment_ids"] == ["seg_old", "seg_latest"]


async def collect_note_update(manager: SessionManager, session_id: str) -> SidecarEvent:
    async for event in manager.bus.subscribe(session_id):
        if event.type == EVENT_NOTE_UPDATE:
            return event
    raise AssertionError("subscription ended")


async def collect_recall_hit(manager: SessionManager, session_id: str) -> SidecarEvent:
    async for event in manager.bus.subscribe(session_id):
        if event.type == EVENT_RECALL_HIT:
            return event
    raise AssertionError("subscription ended")


def make_active(session_id: str, *, web_context_enabled: bool = True) -> ActiveSession:
    return ActiveSession(
        id=session_id,
        capture=DummyCapture(),
        window_queue=asyncio.Queue(maxsize=1),
        postprocess_queue=asyncio.Queue(maxsize=1),
        tasks=[],
        deduper=TranscriptDeduplicator(max_recent=4, similarity_threshold=0.88),
        web_context_queue=asyncio.Queue(maxsize=4),
        web_context_enabled=web_context_enabled,
    )


def segment(session_id: str, text: str, segment_id: str = "seg_test") -> TranscriptSegment:
    return TranscriptSegment(
        id=segment_id,
        session_id=session_id,
        start_s=0.0,
        end_s=1.0,
        text=text,
    )


def candidate_for(query: str) -> WebContextCandidate:
    return WebContextCandidate(
        query=query,
        normalized_query=query,
        freshness="py",
        source_segment_ids=["seg_test"],
    )
