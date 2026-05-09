from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

from brain_sidecar.core.models import SidecarCard, TranscriptSegment
from brain_sidecar.core.sidecar_cards import create_sidecar_card


@dataclass(frozen=True)
class EvalTranscriptEvent:
    id: str
    session_id: str
    start_s: float
    end_s: float
    text: str
    is_final: bool = True
    speaker_label: str | None = None
    speaker_role: str | None = None
    speaker_confidence: float | None = None
    speaker_low_confidence: bool | None = None
    expected_topics: list[str] = field(default_factory=list)

    def to_segment(self) -> TranscriptSegment:
        return TranscriptSegment(
            id=self.id,
            session_id=self.session_id,
            start_s=self.start_s,
            end_s=self.end_s,
            text=self.text,
            is_final=self.is_final,
            speaker_label=self.speaker_label,
            speaker_role=self.speaker_role,
            speaker_confidence=self.speaker_confidence,
            speaker_low_confidence=self.speaker_low_confidence,
        )

    def to_json(self) -> str:
        payload = {"type": "transcript", **asdict(self)}
        return json.dumps(payload, sort_keys=True)


@dataclass(frozen=True)
class EvalMemoryEvent:
    id: str
    title: str
    body: str
    expected_behavior: str = "memory_only"
    forbidden_in_current_notes: list[str] = field(default_factory=list)

    def to_card(self, session_id: str) -> SidecarCard:
        return create_sidecar_card(
            session_id=session_id,
            category="memory",
            title=self.title,
            body=self.body,
            why_now="Injected offline evaluation memory distractor.",
            priority="normal",
            confidence=0.72,
            source_type="saved_transcript",
            source_segment_ids=[],
            citations=[f"eval-memory:{self.id}"],
            card_key=f"eval:memory:{self.id}",
        )

    def to_json(self) -> str:
        payload = {"type": "memory", **asdict(self)}
        return json.dumps(payload, sort_keys=True)


@dataclass(frozen=True)
class EvalExpectedTopic:
    topic: str
    required_terms_any: list[str]
    max_count: int | None = None


@dataclass(frozen=True)
class EvalExpectedBehavior:
    session_id: str
    should_accept_any: list[EvalExpectedTopic] = field(default_factory=list)
    should_reject_terms: list[str] = field(default_factory=list)
    min_accepted_current_cards: int | None = None
    max_accepted_current_cards: int | None = None


@dataclass(frozen=True)
class EvalSuppressedCard:
    card: SidecarCard
    reason: str
    normalized_fingerprint: str | None


@dataclass(frozen=True)
class EvalReplayResult:
    session_id: str
    input_segments: list[TranscriptSegment]
    consolidated_segments: list[TranscriptSegment]
    collapsed_segment_count: int
    generated_cards: list[SidecarCard]
    accepted_cards: list[SidecarCard]
    suppressed_cards: list[EvalSuppressedCard]
    memory_cards: list[SidecarCard]
    expectation: EvalExpectedBehavior | None = None


@dataclass(frozen=True)
class EvalReport:
    session_id: str
    input_segment_count: int
    consolidated_segment_count: int
    collapsed_segment_count: int
    generated_card_count: int
    accepted_card_count: int
    suppressed_card_count: int
    memory_card_count: int
    current_note_count: int
    cards_per_5min: float
    cards_per_100_segments: float
    percent_cards_with_source_ids: float
    percent_cards_with_evidence_quote: float
    unsupported_topic_violations: list[dict[str, Any]]
    memory_leakage_violations: list[dict[str, Any]]
    identity_claim_violations: list[dict[str, Any]]
    duplicate_card_count: int
    generic_clarification_count: int
    generic_clarification_rate_limited_count: int
    expected_topic_hits: list[str]
    expected_topic_misses: list[str]
    suppression_reasons: dict[str, int]
    report_passed: bool
    accepted_cards: list[dict[str, Any]] = field(default_factory=list)
    suppressed_cards: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def transcript_event_from_dict(payload: dict[str, Any], *, fallback_session_id: str) -> EvalTranscriptEvent:
    fallback_id = hashlib.sha1(str(payload.get("text") or "").encode("utf-8")).hexdigest()[:12]
    return EvalTranscriptEvent(
        id=str(payload.get("id") or f"seg_{fallback_id}"),
        session_id=str(payload.get("session_id") or fallback_session_id),
        start_s=float(payload.get("start_s") or 0.0),
        end_s=float(payload.get("end_s") or float(payload.get("start_s") or 0.0) + 3.0),
        text=str(payload.get("text") or ""),
        is_final=bool(payload.get("is_final", True)),
        speaker_label=payload.get("speaker_label"),
        speaker_role=payload.get("speaker_role"),
        speaker_confidence=_float_or_none(payload.get("speaker_confidence")),
        speaker_low_confidence=payload.get("speaker_low_confidence"),
        expected_topics=[str(item) for item in payload.get("expected_topics") or []],
    )


def memory_event_from_dict(payload: dict[str, Any]) -> EvalMemoryEvent:
    return EvalMemoryEvent(
        id=str(payload.get("id") or "memory"),
        title=str(payload.get("title") or "Memory distractor"),
        body=str(payload.get("body") or ""),
        expected_behavior=str(payload.get("expected_behavior") or "memory_only"),
        forbidden_in_current_notes=[str(item) for item in payload.get("forbidden_in_current_notes") or []],
    )


def expectation_from_dict(payload: dict[str, Any], *, fallback_session_id: str) -> EvalExpectedBehavior:
    topics = []
    for item in payload.get("should_accept_any") or []:
        if not isinstance(item, dict):
            continue
        topics.append(
            EvalExpectedTopic(
                topic=str(item.get("topic") or ""),
                required_terms_any=[str(term) for term in item.get("required_terms_any") or []],
                max_count=_int_or_none(item.get("max_count")),
            )
        )
    return EvalExpectedBehavior(
        session_id=str(payload.get("session_id") or fallback_session_id),
        should_accept_any=topics,
        should_reject_terms=[str(term) for term in payload.get("should_reject_terms") or []],
        min_accepted_current_cards=_int_or_none(payload.get("min_accepted_current_cards")),
        max_accepted_current_cards=_int_or_none(payload.get("max_accepted_current_cards")),
    )


def candidate_card_from_dict(payload: dict[str, Any], *, session_id: str) -> SidecarCard:
    return create_sidecar_card(
        session_id=session_id,
        category=payload.get("category") or payload.get("kind") or "note",
        title=payload.get("title") or "Candidate",
        body=payload.get("body") or "",
        suggested_say=payload.get("suggested_say"),
        suggested_ask=payload.get("suggested_ask"),
        why_now=payload.get("why_now") or "Fixture candidate for offline evaluation.",
        priority=payload.get("priority") or "normal",
        confidence=payload.get("confidence", 0.72),
        source_segment_ids=payload.get("source_segment_ids") or [],
        source_type=payload.get("source_type") or "transcript",
        sources=payload.get("sources") or [],
        citations=payload.get("citations") or [],
        card_key=payload.get("card_key"),
        evidence_quote=payload.get("evidence_quote") or "",
        owner=payload.get("owner"),
        due_date=payload.get("due_date"),
        missing_info=payload.get("missing_info"),
        created_at=payload.get("created_at"),
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
