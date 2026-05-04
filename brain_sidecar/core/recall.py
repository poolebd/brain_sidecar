from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


class RecallIndex:
    def __init__(self, storage: Storage, ollama: OllamaClient) -> None:
        self.storage = storage
        self.ollama = ollama

    async def add_text(self, source_type: str, source_id: str, text: str, metadata: dict) -> None:
        clean = normalize_text(text)
        if not clean:
            return
        vector = (await self.ollama.embed([clean]))[0]
        self.storage.upsert_embedding(source_type, source_id, clean, metadata, vector)

    async def search(self, query: str, limit: int = 8) -> list[SearchHit]:
        clean = normalize_text(query)
        if not clean:
            return []
        query_vector = (await self.ollama.embed([clean]))[0]
        records = list(self.storage.embedding_records())
        faiss_hits = search_with_faiss(query_vector, records, limit)
        if faiss_hits is not None:
            return faiss_hits
        scored = [
            SearchHit(
                source_type=record["source_type"],
                source_id=record["source_id"],
                text=record["text"],
                score=cosine_similarity(query_vector, record["vector"]),
                metadata=record["metadata"],
            )
            for record in records
        ]
        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:limit]

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
