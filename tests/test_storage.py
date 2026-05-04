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
