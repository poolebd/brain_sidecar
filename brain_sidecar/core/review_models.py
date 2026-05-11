from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brain_sidecar.core.models import SidecarCard


@dataclass
class ReviewWindowExtract:
    cards: list[SidecarCard] = field(default_factory=list)
    window_summary: str = ""
    priority: str = "normal"
    summary_points: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    project_workstreams: list[dict[str, Any]] = field(default_factory=list)
    technical_findings: list[dict[str, Any]] = field(default_factory=list)
    source_segment_ids: list[str] = field(default_factory=list)
