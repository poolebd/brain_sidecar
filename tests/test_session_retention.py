from __future__ import annotations

import asyncio
from pathlib import Path

from brain_sidecar.config import Settings
from brain_sidecar.core.audio import AudioCapture
from brain_sidecar.core.dedupe import TranscriptDeduplicator
from brain_sidecar.core.session import ActiveSession, AudioWindow, SessionManager
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
            spans=[TranscribedSpan(start_s=start_offset_s, end_s=start_offset_s + 1.0, text="Temporary launch notes")],
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
        transcription_queue_size=1,
        postprocess_queue_size=1,
    )


def test_listen_only_session_keeps_transcript_out_of_storage(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    manager.transcriber = FakeTranscriber()  # type: ignore[assignment]
    session = manager.storage.create_session("listen only")
    active = active_session(session.id, save_transcript=False)
    manager._active[session.id] = active

    batch = event_loop.run_until_complete(manager._transcribe_window(session.id, AudioWindow(b"\0" * 32000, 0.0)))

    assert batch.segments[0].text == "Temporary launch notes"
    assert active.recent_segments[0].text == "Temporary launch notes"
    assert manager.storage.recent_segments(session.id) == []
    assert list(manager.storage.embedding_records()) == []
    row = manager.storage.conn.execute("select count(*) as count from diarization_segments").fetchone()
    assert row["count"] == 0


def test_record_transcript_session_persists_transcript_text(event_loop, tmp_path: Path) -> None:
    manager = SessionManager(make_settings(tmp_path))
    manager.transcriber = FakeTranscriber()  # type: ignore[assignment]
    session = manager.storage.create_session("record transcript")
    active = active_session(session.id, save_transcript=True)
    manager._active[session.id] = active

    event_loop.run_until_complete(manager._transcribe_window(session.id, AudioWindow(b"\0" * 32000, 0.0)))

    assert manager.storage.recent_segments(session.id)[0].text == "Temporary launch notes"
    assert "speaker_role" in manager.storage.recent_segments(session.id)[0].to_dict()


def active_session(session_id: str, *, save_transcript: bool) -> ActiveSession:
    return ActiveSession(
        id=session_id,
        capture=DummyCapture(),
        window_queue=asyncio.Queue(maxsize=1),
        postprocess_queue=asyncio.Queue(maxsize=1),
        tasks=[],
        deduper=TranscriptDeduplicator(max_recent=4, similarity_threshold=0.88),
        save_transcript=save_transcript,
    )
