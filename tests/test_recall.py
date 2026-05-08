from pathlib import Path

from brain_sidecar.core.recall import (
    RecallIndex,
    chunk_text,
    cosine_similarity,
    is_electrical_reference_query,
    iter_supported_files,
    normalize_text,
    rank_recall_hits,
    search_with_faiss,
)
from brain_sidecar.core.models import SearchHit
from brain_sidecar.core.storage import Storage


class FakeOllama:
    def __init__(self) -> None:
        self.embeds = 0

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        self.embeds += 1
        vectors = []
        for text in inputs:
            lower = text.lower()
            if "summary" in lower or "apollo" in lower or "rollout" in lower:
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


def test_chunk_text_preserves_overlap_shape() -> None:
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, max_chars=240, overlap=40)

    assert len(chunks) > 1
    assert all(len(chunk) <= 240 for chunk in chunks)
    assert chunks[0] != chunks[1]


def test_normalize_text_compacts_whitespace() -> None:
    assert normalize_text("  alpha\n\n beta\tgamma  ") == "alpha beta gamma"


def test_cosine_similarity_orders_related_vectors() -> None:
    assert cosine_similarity([1, 0, 0], [1, 0, 0]) == 1
    assert cosine_similarity([1, 0, 0], [0, 1, 0]) == 0


def test_iter_supported_files_filters_extensions(tmp_path: Path) -> None:
    keep = tmp_path / "notes.md"
    drop = tmp_path / "image.png"
    keep.write_text("hello", encoding="utf-8")
    drop.write_text("nope", encoding="utf-8")

    assert list(iter_supported_files(tmp_path)) == [keep]


def test_search_with_faiss_returns_nearest_record() -> None:
    records = [
        {
            "source_type": "document_chunk",
            "source_id": "a",
            "text": "alpha",
            "metadata": {},
            "vector": [1.0, 0.0],
        },
        {
            "source_type": "document_chunk",
            "source_id": "b",
            "text": "beta",
            "metadata": {},
            "vector": [0.0, 1.0],
        },
    ]

    hits = search_with_faiss([0.9, 0.1], records, limit=1)

    assert hits is not None
    assert hits[0].source_id == "a"


def test_recall_cache_reuses_records_until_embedding_marker_changes(event_loop, tmp_path: Path) -> None:
    storage = Storage(tmp_path)
    storage.connect()
    ollama = FakeOllama()
    recall = RecallIndex(storage, ollama)  # type: ignore[arg-type]

    event_loop.run_until_complete(recall.add_text("session_summary", "s1", "Apollo rollout summary", {}))
    first = event_loop.run_until_complete(recall.search("Apollo rollout", limit=2, manual=True))
    first_cache = recall._cache
    second = event_loop.run_until_complete(recall.search("Apollo rollout", limit=2, manual=True))
    assert recall._cache is first_cache
    assert first[0].source_id == second[0].source_id == "s1"

    event_loop.run_until_complete(recall.add_text("document_chunk", "d1", "Unrelated document", {}))
    event_loop.run_until_complete(recall.search("Apollo rollout", limit=2, manual=True))
    assert recall._cache is not first_cache


def test_recall_ranking_prefers_summary_and_suppresses_echo() -> None:
    hits = [
        SearchHit("transcript_segment", "raw", "Apollo rollout risk needs owner", 0.91, {}),
        SearchHit("session_summary", "summary", "Summary: Apollo rollout had a staged validation decision.", 0.89, {}),
        SearchHit("document_chunk", "weak", "Generic planning note", 0.4, {}),
    ]

    ranked = rank_recall_hits(
        hits,
        query="Apollo rollout risk needs owner",
        limit=3,
        min_score=0.5,
        prefer_summaries=True,
        manual=False,
    )

    assert [hit.source_id for hit in ranked] == ["summary"]


def test_manual_recall_override_still_respects_manual_floor() -> None:
    hits = [
        SearchHit("document_chunk", "too_weak", "Barely related filler.", 0.1, {}),
        SearchHit("document_chunk", "useful_lower", "Lower than live threshold but still useful.", 0.3, {}),
    ]

    ranked = rank_recall_hits(
        hits,
        query="manual query",
        limit=3,
        min_score=0.25,
        prefer_summaries=True,
        manual=True,
    )

    assert [hit.source_id for hit in ranked] == ["useful_lower"]


def test_electrical_reference_ranking_beats_past_work_for_technical_queries() -> None:
    hits = [
        SearchHit(
            "work_memory_project",
            "past_breaker_project",
            "Past project memory about breaker replacement and outage planning.",
            0.66,
            {},
        ),
        SearchHit(
            "document_chunk",
            "doe_breaker_reference",
            "DOE Electrical Science reference text about circuit breakers, relays, and protection.",
            0.50,
            {"path": "/home/bp/project/brain-sidecar/runtime/reference/electrical-engineering/doe-electrical-science/doe.pdf"},
        ),
    ]

    ranked = rank_recall_hits(
        hits,
        query="Need circuit breaker relay protection guidance for the transformer trip path.",
        limit=2,
        min_score=0.58,
        prefer_summaries=True,
        manual=False,
    )

    assert [hit.source_id for hit in ranked] == ["doe_breaker_reference", "past_breaker_project"]


def test_electrical_reference_floor_does_not_apply_to_generic_queries() -> None:
    hits = [
        SearchHit(
            "document_chunk",
            "generic_reference",
            "DOE Electrical Science reference text about circuit breakers.",
            0.50,
            {"path": "/home/bp/project/brain-sidecar/runtime/reference/electrical-engineering/doe-electrical-science/doe.pdf"},
        ),
    ]

    ranked = rank_recall_hits(
        hits,
        query="meeting agenda owner date and next steps",
        limit=2,
        min_score=0.58,
        prefer_summaries=True,
        manual=False,
    )

    assert ranked == []


def test_assumed_technical_conversation_prioritizes_ee_reference_without_trigger_terms() -> None:
    hits = [
        SearchHit(
            "work_memory_project",
            "past_project",
            "Past project memory about agenda ownership and closeout.",
            0.66,
            {},
        ),
        SearchHit(
            "document_chunk",
            "ee_reference",
            "DOE Electrical Science reference text about circuit breakers.",
            0.50,
            {"path": "/home/bp/project/brain-sidecar/runtime/reference/electrical-engineering/doe-electrical-science/doe.pdf"},
        ),
    ]

    ranked = rank_recall_hits(
        hits,
        query="meeting agenda owner date and next steps",
        limit=2,
        min_score=0.58,
        prefer_summaries=True,
        manual=False,
        assume_technical=True,
    )

    assert [hit.source_id for hit in ranked] == ["ee_reference", "past_project"]


def test_electrical_reference_query_ignores_current_by_itself() -> None:
    assert is_electrical_reference_query("What is the current meeting status?") is False
    assert is_electrical_reference_query("Current transformer protection settings need review.") is True
