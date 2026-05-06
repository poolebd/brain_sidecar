from __future__ import annotations

from brain_sidecar.core.dedupe import TranscriptFinalConsolidator
from brain_sidecar.core.models import TranscriptSegment


def segment(segment_id: str, start: float, end: float, text: str) -> TranscriptSegment:
    return TranscriptSegment(id=segment_id, session_id="ses_1", start_s=start, end_s=end, text=text)


def test_collapses_overlapping_final_segments_by_time_and_text() -> None:
    consolidator = TranscriptFinalConsolidator(max_recent=8)

    first = consolidator.accept(segment("seg_1", 0.0, 3.4, "We need to review the Siemens client agreement."))
    second = consolidator.accept(
        segment("seg_2", 1.1, 4.2, "We need to review the Siemens client agreement before answering.")
    )

    assert first.segment is not None
    assert second.collapsed is True
    assert second.suppressed is False
    assert len(consolidator.segments()) == 1
    assert "before answering" in consolidator.segments()[0].text


def test_preserves_raw_source_ids_for_consolidated_evidence() -> None:
    consolidator = TranscriptFinalConsolidator(max_recent=8)

    consolidator.accept(segment("seg_1", 0.0, 3.4, "The harmonics study needs to line up with the SLD."))
    consolidator.accept(segment("seg_2", 0.8, 3.8, "The harmonics study needs to align with the SLD."))

    consolidated = consolidator.segments()[0]

    assert consolidated.source_segment_ids == ["seg_1", "seg_2"]


def test_consolidated_segment_keeps_display_id_for_replacement() -> None:
    consolidator = TranscriptFinalConsolidator(max_recent=8)

    consolidator.accept(segment("seg_1", 0.0, 3.4, "We need to review the Siemens client agreement."))
    result = consolidator.accept(
        segment("seg_2", 1.1, 4.2, "We need to review the Siemens client agreement before answering.")
    )

    consolidated = consolidator.segments()[0]

    assert result.replaced_segment_id == "seg_1"
    assert consolidated.id == "seg_1"
    assert consolidated.source_segment_ids == ["seg_1", "seg_2"]


def test_does_not_merge_distinct_adjacent_topics() -> None:
    consolidator = TranscriptFinalConsolidator(max_recent=8)

    consolidator.accept(segment("seg_1", 0.0, 2.0, "We need to review the Siemens client agreement."))
    consolidator.accept(segment("seg_2", 2.2, 4.1, "Tomorrow at eleven Eastern should work for the follow-up."))

    assert len(consolidator.segments()) == 2
