from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


SIDECAR_CATEGORIES = {
    "action",
    "decision",
    "question",
    "risk",
    "clarification",
    "contribution",
    "memory",
    "work_memory",
    "web",
    "status",
    "note",
}
SIDECAR_PRIORITIES = {"low", "normal", "high"}
SIDECAR_SOURCE_TYPES = {
    "transcript",
    "saved_transcript",
    "work_memory",
    "brave_web",
    "local_file",
    "model_fallback",
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def normalize_sidecar_category(value: object, *, default: str = "note") -> str:
    category = str(value or default).strip().lower().replace("-", "_")
    if category == "work":
        category = "work_memory"
    return category if category in SIDECAR_CATEGORIES else default


def normalize_sidecar_priority(value: object, *, default: str = "normal") -> str:
    priority = str(value or default).strip().lower()
    return priority if priority in SIDECAR_PRIORITIES else default


def normalize_sidecar_source_type(value: object, *, default: str = "transcript") -> str:
    source_type = str(value or default).strip().lower().replace("-", "_")
    aliases = {
        "work_memory_project": "work_memory",
        "document_chunk": "local_file",
        "file": "local_file",
        "session": "saved_transcript",
        "transcript_segment": "saved_transcript",
        "session_summary": "saved_transcript",
    }
    source_type = aliases.get(source_type, source_type)
    return source_type if source_type in SIDECAR_SOURCE_TYPES else default


def clamp_confidence(value: object, *, default: float = 0.6) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(1.0, max(0.0, number))


def compact_text(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0].strip()
    return clipped or text[:limit].strip()


def compact_optional_text(value: object, *, limit: int) -> str | None:
    text = compact_text(value, limit=limit)
    return text or None


def compact_string_list(value: object, *, limit: int = 24) -> list[str]:
    if not isinstance(value, list):
        return []
    return [compact_text(item, limit=260) for item in value if compact_text(item, limit=260)][:limit]


def compact_sources(value: object, *, limit: int = 6) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    sources: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = compact_text(item.get("title"), limit=180)
        url = compact_text(item.get("url"), limit=500)
        path = compact_text(item.get("path"), limit=500)
        if not title or not (url or path):
            continue
        source = {"title": title}
        if url:
            source["url"] = url
        if path:
            source["path"] = path
        sources.append(source)
    return sources[:limit]


@dataclass(frozen=True)
class SessionRecord:
    id: str
    title: str
    status: str
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None


@dataclass(frozen=True)
class TranscriptSegment:
    id: str
    session_id: str
    start_s: float
    end_s: float
    text: str
    is_final: bool = True
    created_at: float = field(default_factory=time.time)
    speaker_role: str | None = None
    speaker_label: str | None = None
    speaker_confidence: float | None = None
    speaker_match_reason: str | None = None
    speaker_low_confidence: bool | None = None
    source_segment_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "session_id": self.session_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "text": self.text,
            "is_final": self.is_final,
            "created_at": self.created_at,
        }
        if self.speaker_role:
            payload["speaker_role"] = self.speaker_role
        if self.speaker_label:
            payload["speaker_label"] = self.speaker_label
        if self.speaker_confidence is not None:
            payload["speaker_confidence"] = self.speaker_confidence
        if self.speaker_match_reason:
            payload["speaker_match_reason"] = self.speaker_match_reason
        if self.speaker_low_confidence is not None:
            payload["speaker_low_confidence"] = self.speaker_low_confidence
        if self.source_segment_ids:
            payload["source_segment_ids"] = self.source_segment_ids
        return payload


@dataclass(frozen=True)
class NoteCard:
    id: str
    session_id: str
    kind: str
    title: str
    body: str
    source_segment_ids: list[str]
    created_at: float = field(default_factory=time.time)
    evidence_quote: str = ""
    owner: str | None = None
    due_date: str | None = None
    missing_info: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "session_id": self.session_id,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "source_segment_ids": self.source_segment_ids,
            "created_at": self.created_at,
        }
        if self.evidence_quote:
            payload["evidence_quote"] = self.evidence_quote
        if self.owner:
            payload["owner"] = self.owner
        if self.due_date:
            payload["due_date"] = self.due_date
        if self.missing_info:
            payload["missing_info"] = self.missing_info
        return payload


@dataclass(frozen=True)
class SearchHit:
    source_type: str
    source_id: str
    text: str
    score: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class SidecarCard:
    id: str
    session_id: str
    category: str
    title: str
    body: str
    why_now: str
    priority: str
    confidence: float
    source_segment_ids: list[str]
    source_type: str
    sources: list[dict[str, str]] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    suggested_say: str | None = None
    suggested_ask: str | None = None
    ephemeral: bool = True
    expires_at: float | None = None
    supersedes_id: str | None = None
    card_key: str | None = None
    explicitly_requested: bool = False
    created_at: float = field(default_factory=time.time)
    raw_audio_retained: bool = False
    evidence_quote: str = ""
    owner: str | None = None
    due_date: str | None = None
    missing_info: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", normalize_sidecar_category(self.category))
        object.__setattr__(self, "priority", normalize_sidecar_priority(self.priority))
        object.__setattr__(self, "source_type", normalize_sidecar_source_type(self.source_type))
        object.__setattr__(self, "title", compact_text(self.title, limit=140) or "Sidecar")
        object.__setattr__(self, "body", compact_text(self.body, limit=1000))
        object.__setattr__(self, "why_now", compact_text(self.why_now, limit=300))
        object.__setattr__(self, "suggested_say", compact_optional_text(self.suggested_say, limit=280))
        object.__setattr__(self, "suggested_ask", compact_optional_text(self.suggested_ask, limit=280))
        object.__setattr__(self, "confidence", clamp_confidence(self.confidence))
        object.__setattr__(self, "source_segment_ids", compact_string_list(self.source_segment_ids, limit=24))
        object.__setattr__(self, "sources", compact_sources(self.sources, limit=6))
        object.__setattr__(self, "citations", compact_string_list(self.citations, limit=12))
        object.__setattr__(self, "raw_audio_retained", False)
        object.__setattr__(self, "evidence_quote", compact_text(self.evidence_quote, limit=420))
        object.__setattr__(self, "owner", compact_optional_text(self.owner, limit=120))
        object.__setattr__(self, "due_date", compact_optional_text(self.due_date, limit=120))
        object.__setattr__(self, "missing_info", compact_optional_text(self.missing_info, limit=240))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "session_id": self.session_id,
            "category": self.category,
            "title": self.title,
            "body": self.body,
            "suggested_say": self.suggested_say,
            "suggested_ask": self.suggested_ask,
            "why_now": self.why_now,
            "priority": self.priority,
            "confidence": self.confidence,
            "source_segment_ids": self.source_segment_ids,
            "source_type": self.source_type,
            "sources": self.sources,
            "citations": self.citations,
            "ephemeral": self.ephemeral,
            "expires_at": self.expires_at,
            "supersedes_id": self.supersedes_id,
            "card_key": self.card_key,
            "explicitly_requested": self.explicitly_requested,
            "created_at": self.created_at,
            "raw_audio_retained": self.raw_audio_retained,
        }
        if self.evidence_quote:
            payload["evidence_quote"] = self.evidence_quote
        if self.owner:
            payload["owner"] = self.owner
        if self.due_date:
            payload["due_date"] = self.due_date
        if self.missing_info:
            payload["missing_info"] = self.missing_info
        return payload
