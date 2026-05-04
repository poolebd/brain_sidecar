from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from brain_sidecar.config import Settings
from brain_sidecar.core.models import SearchHit
from brain_sidecar.core.ollama import OllamaClient
from brain_sidecar.core.storage import Storage


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".pdf", ".docx"}


@dataclass(frozen=True)
class IndexReport:
    roots: int
    files_seen: int
    chunks_indexed: int
    skipped: list[str]

    def to_dict(self) -> dict:
        return {
            "roots": self.roots,
            "files_seen": self.files_seen,
            "chunks_indexed": self.chunks_indexed,
            "skipped": self.skipped,
        }


@dataclass
class _VectorCache:
    marker: tuple[int, float]
    records: list[dict[str, Any]]
    faiss_index: Any | None = None


class RecallIndex:
    def __init__(self, storage: Storage, ollama: OllamaClient, settings: Settings | None = None) -> None:
        self.storage = storage
        self.ollama = ollama
        self.settings = settings
        self._cache: _VectorCache | None = None

    async def add_text(self, source_type: str, source_id: str, text: str, metadata: dict) -> None:
        clean = normalize_text(text)
        if not clean:
            return
        vector = (await self.ollama.embed([clean]))[0]
        self.storage.upsert_embedding(source_type, source_id, clean, metadata, vector)
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        self._cache = None

    async def search(
        self,
        query: str,
        limit: int = 8,
        *,
        manual: bool = False,
        recent_text: str | None = None,
    ) -> list[SearchHit]:
        clean = normalize_text(query)
        if not clean:
            return []
        query_vector = (await self.ollama.embed([clean]))[0]
        cache = self._cached_records()
        faiss_hits = search_cached_faiss(query_vector, cache.records, cache.faiss_index, max(limit * 3, limit))
        if faiss_hits is not None:
            return rank_recall_hits(
                faiss_hits,
                query=recent_text or clean,
                limit=limit,
                min_score=self._manual_min_score() if manual else self._min_score(),
                prefer_summaries=self._prefer_summaries(),
                manual=manual,
            )
        scored = [
            SearchHit(
                source_type=record["source_type"],
                source_id=record["source_id"],
                text=record["text"],
                score=cosine_similarity(query_vector, record["vector"]),
                metadata=record["metadata"],
            )
            for record in cache.records
        ]
        return rank_recall_hits(
            scored,
            query=recent_text or clean,
            limit=limit,
            min_score=self._manual_min_score() if manual else self._min_score(),
            prefer_summaries=self._prefer_summaries(),
            manual=manual,
        )

    def _cached_records(self) -> _VectorCache:
        marker = self.storage.embedding_marker()
        if self._cache is not None and self._cache.marker == marker:
            return self._cache
        records = list(self.storage.embedding_records())
        self._cache = _VectorCache(marker=marker, records=records, faiss_index=build_faiss_index(records))
        return self._cache

    def _min_score(self) -> float:
        return float(getattr(self.settings, "recall_min_score", 0.58))

    def _manual_min_score(self) -> float:
        # Manual queries should broaden live recall, but not turn every weak
        # embedding neighbor into a meeting card.
        return min(0.42, max(0.25, self._min_score() - 0.18))

    def _prefer_summaries(self) -> bool:
        return bool(getattr(self.settings, "recall_prefer_summaries", True))

    async def reindex_roots(self) -> IndexReport:
        roots = self.storage.library_roots()
        files_seen = 0
        chunks_indexed = 0
        skipped: list[str] = []

        for root in roots:
            if not root.exists():
                skipped.append(f"{root}: missing")
                continue
            for path in iter_supported_files(root):
                files_seen += 1
                try:
                    text = await asyncio.to_thread(read_document_text, path)
                except Exception as exc:
                    skipped.append(f"{path}: {exc}")
                    continue
                chunks = chunk_text(text)
                if not chunks:
                    continue
                vectors = await self.ollama.embed(chunks)
                for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=False)):
                    chunk_id = self.storage.upsert_document_chunk(
                        path,
                        index,
                        chunk,
                        {"path": str(path), "chunk_index": index},
                    )
                    self.storage.upsert_embedding(
                        "document_chunk",
                        chunk_id,
                        chunk,
                        {"path": str(path), "chunk_index": index},
                        vector,
                    )
                    self.invalidate_cache()
                    chunks_indexed += 1

        return IndexReport(
            roots=len(roots),
            files_seen=files_seen,
            chunks_indexed=chunks_indexed,
            skipped=skipped,
        )


def iter_supported_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield root
        return
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def read_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".rst"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except Exception as exc:
            raise RuntimeError("pypdf is required for PDF indexing") from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        try:
            import docx
        except Exception as exc:
            raise RuntimeError("python-docx is required for DOCX indexing") from exc
        document = docx.Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)
    return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, max_chars: int = 1400, overlap: int = 180) -> list[str]:
    clean = normalize_text(text)
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + max_chars)
        if end < len(clean):
            boundary = clean.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        chunk = clean[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def search_with_faiss(query_vector: list[float], records: list[dict], limit: int) -> list[SearchHit] | None:
    if not records:
        return []
    try:
        import faiss
        import numpy as np
    except Exception:
        return None

    vectors = np.array([record["vector"] for record in records], dtype="float32")
    query = np.array([query_vector], dtype="float32")
    if vectors.ndim != 2 or query.ndim != 2 or vectors.shape[1] != query.shape[1]:
        return None

    faiss.normalize_L2(vectors)
    faiss.normalize_L2(query)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    scores, indexes = index.search(query, min(limit, len(records)))

    hits: list[SearchHit] = []
    for score, record_index in zip(scores[0], indexes[0], strict=False):
        if record_index < 0:
            continue
        record = records[int(record_index)]
        hits.append(
            SearchHit(
                source_type=record["source_type"],
                source_id=record["source_id"],
                text=record["text"],
                score=float(score),
                metadata=record["metadata"],
            )
        )
    return hits


def build_faiss_index(records: list[dict]) -> Any | None:
    if not records:
        return None
    try:
        import faiss
        import numpy as np
    except Exception:
        return None
    vectors = np.array([record["vector"] for record in records], dtype="float32")
    if vectors.ndim != 2:
        return None
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search_cached_faiss(
    query_vector: list[float],
    records: list[dict],
    index: Any | None,
    limit: int,
) -> list[SearchHit] | None:
    if not records:
        return []
    if index is None:
        return None
    try:
        import numpy as np
        import faiss
    except Exception:
        return None
    query = np.array([query_vector], dtype="float32")
    if query.ndim != 2 or index.d != query.shape[1]:
        return None
    faiss.normalize_L2(query)
    scores, indexes = index.search(query, min(limit, len(records)))
    hits: list[SearchHit] = []
    for score, record_index in zip(scores[0], indexes[0], strict=False):
        if record_index < 0:
            continue
        record = records[int(record_index)]
        hits.append(
            SearchHit(
                source_type=record["source_type"],
                source_id=record["source_id"],
                text=record["text"],
                score=float(score),
                metadata=record["metadata"],
            )
        )
    return hits


def rank_recall_hits(
    hits: list[SearchHit],
    *,
    query: str,
    limit: int,
    min_score: float,
    prefer_summaries: bool,
    manual: bool,
) -> list[SearchHit]:
    query_norm = normalize_text(query).lower()
    ranked: list[tuple[float, SearchHit]] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        key = (hit.source_type, hit.source_id)
        if key in seen:
            continue
        seen.add(key)
        if hit.score < min_score:
            continue
        if is_transcript_echo(query_norm, hit.text):
            continue
        boost = source_type_boost(hit.source_type, prefer_summaries=prefer_summaries)
        ranked.append((hit.score + boost, hit))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [hit for _, hit in ranked[:limit]]


def source_type_boost(source_type: str, *, prefer_summaries: bool) -> float:
    if source_type == "session_summary":
        return 0.08 if prefer_summaries else 0.02
    if source_type == "work_memory_project":
        return 0.06
    if source_type == "transcript_segment":
        return -0.04 if prefer_summaries else 0.0
    if source_type == "document_chunk":
        return -0.01
    return 0.0


def is_transcript_echo(query: str, candidate: str) -> bool:
    query_norm = normalize_text(query).lower()
    candidate_norm = normalize_text(candidate).lower()
    if not query_norm or not candidate_norm:
        return False
    if candidate_norm in query_norm and len(candidate_norm) > 24:
        return True
    query_tokens = set(re.findall(r"[a-z0-9]{3,}", query_norm))
    candidate_tokens = set(re.findall(r"[a-z0-9]{3,}", candidate_norm))
    if len(query_tokens) < 6 or len(candidate_tokens) < 6:
        return False
    overlap = len(query_tokens & candidate_tokens) / max(1, min(len(query_tokens), len(candidate_tokens)))
    return overlap >= 0.86
