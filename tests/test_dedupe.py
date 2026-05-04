from brain_sidecar.core.dedupe import TranscriptDeduplicator, jaccard, token_containment


def test_deduplicator_rejects_near_duplicate_overlap() -> None:
    deduper = TranscriptDeduplicator(max_recent=4, similarity_threshold=0.82)

    assert deduper.accept("Glue the sheet to the dark blue background.", 1.0, 4.0)
    assert not deduper.accept("Glue the sheet to dark blue background", 2.0, 4.5)


def test_deduplicator_accepts_distinct_text() -> None:
    deduper = TranscriptDeduplicator(max_recent=4, similarity_threshold=0.88)

    assert deduper.accept("The birch canoe slid on smooth planks.", 1.0, 3.0)
    assert deduper.accept("A meeting note should capture the next action.", 3.2, 5.0)


def test_token_similarity_helpers() -> None:
    left = {"a", "b", "c"}
    right = {"a", "b", "c", "d"}

    assert jaccard(left, right) == 0.75
    assert token_containment(left, right) == 1.0
