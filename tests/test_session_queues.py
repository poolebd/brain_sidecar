import asyncio
from pathlib import Path

from brain_sidecar.config import Settings
from brain_sidecar.core.audio import AudioCapture
from brain_sidecar.core.dedupe import TranscriptDeduplicator
from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.session import ActiveSession, AudioWindow, SegmentBatch, SessionManager


class DummyCapture(AudioCapture):
    async def chunks(self):
        if False:
            yield b""

    async def stop(self) -> None:
        return None


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
        transcription_queue_size=1,
        postprocess_queue_size=1,
    )


def test_put_latest_replaces_stale_postprocess_batch(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    queue: asyncio.Queue[SegmentBatch] = asyncio.Queue(maxsize=1)
    first = SegmentBatch([segment("first")])
    latest = SegmentBatch([segment("latest")])

    event_loop.run_until_complete(queue.put(first))
    event_loop.run_until_complete(manager._put_latest(queue, latest))

    assert queue.qsize() == 1
    assert queue.get_nowait().segments[0].id == "latest"


def test_put_latest_preserves_pending_note_refresh(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    queue: asyncio.Queue[SegmentBatch] = asyncio.Queue(maxsize=1)
    first = SegmentBatch([segment("first")], refresh_notes=True)
    latest = SegmentBatch([segment("latest")], refresh_notes=False)

    event_loop.run_until_complete(queue.put(first))
    event_loop.run_until_complete(manager._put_latest(queue, latest))

    queued = queue.get_nowait()
    assert queued.segments[0].id == "latest"
    assert queued.refresh_notes is True


def test_enqueue_window_drops_stale_audio_window(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    queue: asyncio.Queue[AudioWindow] = asyncio.Queue(maxsize=1)
    active = ActiveSession(
        id="ses_test",
        capture=DummyCapture(),
        window_queue=queue,
        postprocess_queue=asyncio.Queue(maxsize=1),
        tasks=[],
        deduper=TranscriptDeduplicator(max_recent=4, similarity_threshold=0.88),
    )
    manager._active[active.id] = active

    event_loop.run_until_complete(queue.put(AudioWindow(b"old", 0.0)))
    event_loop.run_until_complete(manager._enqueue_window(active.id, queue, AudioWindow(b"new", 1.0)))

    assert active.dropped_windows == 1
    assert queue.qsize() == 1
    assert queue.get_nowait().pcm == b"new"


def test_enqueue_window_replaces_stale_preview_without_counting_drop(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    queue: asyncio.Queue[AudioWindow] = asyncio.Queue(maxsize=1)
    active = ActiveSession(
        id="ses_test",
        capture=DummyCapture(),
        window_queue=queue,
        postprocess_queue=asyncio.Queue(maxsize=1),
        tasks=[],
        deduper=TranscriptDeduplicator(max_recent=4, similarity_threshold=0.88),
    )
    manager._active[active.id] = active

    event_loop.run_until_complete(queue.put(AudioWindow(b"preview", 0.0, preview=True)))
    event_loop.run_until_complete(manager._enqueue_window(active.id, queue, AudioWindow(b"final", 1.0)))

    assert active.dropped_windows == 0
    assert queue.qsize() == 1
    assert queue.get_nowait().pcm == b"final"


def test_balanced_default_queue_holds_normal_live_burst(event_loop, tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        host="127.0.0.1",
        port=8765,
        asr_primary_model="tiny.en",
        asr_fallback_model="tiny.en",
        asr_compute_type="float16",
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="qwen3.5:9b",
        ollama_embed_model="embeddinggemma",
    )
    manager = SessionManager(settings)
    queue: asyncio.Queue[AudioWindow] = asyncio.Queue(maxsize=settings.transcription_queue_size)
    active = ActiveSession(
        id="ses_test",
        capture=DummyCapture(),
        window_queue=queue,
        postprocess_queue=asyncio.Queue(maxsize=1),
        tasks=[],
        deduper=TranscriptDeduplicator(max_recent=4, similarity_threshold=0.88),
    )
    manager._active[active.id] = active

    for index in range(settings.transcription_queue_size):
        event_loop.run_until_complete(manager._enqueue_window(active.id, queue, AudioWindow(bytes([index]), float(index))))

    assert active.dropped_windows == 0
    assert queue.qsize() == settings.transcription_queue_size


def segment(segment_id: str) -> TranscriptSegment:
    return TranscriptSegment(
        id=segment_id,
        session_id="ses_test",
        start_s=0.0,
        end_s=1.0,
        text=segment_id,
    )
