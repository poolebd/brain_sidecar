from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "text": self.text,
            "is_final": self.is_final,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class NoteCard:
    id: str
    session_id: str
    kind: str
    title: str
    body: str
    source_segment_ids: list[str]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "kind": self.kind,
            "title": self.title,
            "body": self.body,
            "source_segment_ids": self.source_segment_ids,
            "created_at": self.created_at,
        }


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
    created_at: float = field(default_factory=time.time)
    raw_audio_retained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "created_at": self.created_at,
            "raw_audio_retained": self.raw_audio_retained,
        }
