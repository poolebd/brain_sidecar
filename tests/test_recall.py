from pathlib import Path

from brain_sidecar.core.recall import (
    chunk_text,
    cosine_similarity,
    iter_supported_files,
    normalize_text,
    search_with_faiss,
)


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
