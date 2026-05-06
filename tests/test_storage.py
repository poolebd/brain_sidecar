from pathlib import Path

from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.storage import Storage


def test_storage_persists_text_artifacts_only(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    storage.connect()
    session = storage.create_session("test")
    segment = TranscriptSegment(
        id="seg_test",
        session_id=session.id,
        start_s=0.0,
        end_s=1.0,
        text="hello local memory",
    )

    storage.add_transcript_segment(segment)
    storage.upsert_embedding(
        "transcript_segment",
        segment.id,
        segment.text,
        {"session_id": session.id},
        [1.0, 0.0, 0.0],
    )

    assert storage.recent_segments(session.id)[0].text == "hello local memory"
    records = list(storage.embedding_records())
    assert records[0]["source_type"] == "transcript_segment"
    assert not list(tmp_path.glob("*.wav"))
    assert not list(tmp_path.glob("*.pcm"))


def test_storage_upserts_consolidated_transcript_without_duplicate_rows(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    storage.connect()
    session = storage.create_session("upsert")
    first = TranscriptSegment(
        id="seg_1",
        session_id=session.id,
        start_s=0.0,
        end_s=3.0,
        text="review the relay settings",
    )
    replacement = TranscriptSegment(
        id="seg_1",
        session_id=session.id,
        start_s=0.0,
        end_s=4.2,
        text="review the relay settings before Friday",
    )

    storage.add_transcript_segment(first)
    storage.upsert_transcript_segment(replacement, replaces_segment_id="seg_1")

    segments = storage.recent_segments(session.id, limit=5)
    assert len(segments) == 1
    assert segments[0].text == "review the relay settings before Friday"
    assert segments[0].end_s == 4.2


def test_storage_uses_wal_busy_timeout_and_speaker_migration(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    storage.connect()

    settings = storage.sqlite_runtime_settings()
    assert settings["journal_mode"].lower() == "wal"
    assert settings["busy_timeout"] == 5000

    columns = {row["name"] for row in storage.conn.execute("pragma table_info(transcript_segments)").fetchall()}
    assert "speaker_role" in columns
    assert "speaker_low_confidence" in columns


def test_session_memory_summary_round_trip(tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    storage.connect()
    session = storage.create_session("summary")

    storage.upsert_session_memory_summary(
        session_id=session.id,
        title="Summary",
        summary="A saved meeting discussed rollback risk.",
        topics=["rollback"],
        decisions=["Use staged rollout."],
        actions=["BP will send checklist."],
        unresolved_questions=["Who owns validation?"],
        entities=["Apollo"],
        lessons=["Make owner explicit."],
        source_segment_ids=["seg_1"],
    )

    summary = storage.session_memory_summaries()[0]
    assert summary["session_id"] == session.id
    assert summary["topics"] == ["rollback"]
