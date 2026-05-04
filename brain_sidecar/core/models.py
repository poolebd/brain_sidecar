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
