from __future__ import annotations

import hashlib
import time
from typing import Any

from brain_sidecar.core.models import (
    SidecarCard,
    clamp_confidence,
    compact_optional_text,
    compact_sources,
    compact_string_list,
    compact_text,
    new_id,
    normalize_sidecar_category,
    normalize_sidecar_priority,
    normalize_sidecar_source_type,
)


LIVE_CARD_TTL_SECONDS = 15 * 60


def create_sidecar_card(
    *,
    session_id: str,
    category: object,
    title: object,
    body: object,
    why_now: object,
    source_type: object = "transcript",
    source_segment_ids: object = None,
    suggested_say: object = None,
    suggested_ask: object = None,
    priority: object = "normal",
    confidence: object = 0.6,
    sources: object = None,
    citations: object = None,
    ephemeral: bool = True,
    expires_at: float | None = None,
    supersedes_id: str | None = None,
    card_key: str | None = None,
    created_at: float | None = None,
    explicitly_requested: bool = False,
    provisional: bool = False,
    evidence_quote: object = "",
    owner: object = None,
    due_date: object = None,
    missing_info: object = None,
) -> SidecarCard:
    normalized_category = normalize_sidecar_category(category)
    normalized_source_type = normalize_sidecar_source_type(source_type)
    normalized_priority = normalize_sidecar_priority(priority)
    source_ids = compact_string_list(source_segment_ids or [], limit=24)
    title_text = compact_text(title, limit=140) or default_title(normalized_category)
    body_text = compact_text(body, limit=1000)
    why_now_text = compact_text(why_now, limit=300) or default_why_now(normalized_category, normalized_source_type)
    confidence_value = clamp_confidence(confidence, default=default_confidence(normalized_category))
    now = time.time() if created_at is None else float(created_at)
    if expires_at is None and ephemeral:
        expires_at = now + LIVE_CARD_TTL_SECONDS
    metadata_sources = compact_sources(sources or [], limit=6)
    citation_list = compact_string_list(citations or [], limit=12)
    if not card_key:
        card_key = stable_card_key(
            normalized_category,
            normalized_source_type,
            title_text,
            source_ids or citation_list,
        )
    return SidecarCard(
        id=new_id("card"),
        session_id=session_id,
        category=normalized_category,
        title=title_text,
        body=body_text,
        suggested_say=compact_optional_text(suggested_say, limit=280),
        suggested_ask=compact_optional_text(suggested_ask, limit=280),
        why_now=why_now_text,
        priority=normalized_priority,
        confidence=confidence_value,
        source_segment_ids=source_ids,
        source_type=normalized_source_type,
        sources=metadata_sources,
        citations=citation_list,
        ephemeral=bool(ephemeral),
        expires_at=expires_at,
        supersedes_id=supersedes_id,
        card_key=card_key,
        explicitly_requested=bool(explicitly_requested),
        created_at=now,
        raw_audio_retained=False,
        evidence_quote=compact_text(evidence_quote, limit=420),
        owner=compact_optional_text(owner, limit=120),
        due_date=compact_optional_text(due_date, limit=120),
        missing_info=compact_optional_text(missing_info, limit=240),
    )


def note_payload_to_sidecar_card(session_id: str, payload: dict[str, Any], *, save_transcript: bool) -> SidecarCard:
    kind = str(payload.get("kind") or "note")
    source_type = normalize_sidecar_source_type(payload.get("source_type") or "transcript")
    category = "web" if source_type == "brave_web" else normalize_sidecar_category(kind)
    title = payload.get("title") or default_title(category)
    source_ids = compact_string_list(payload.get("source_segment_ids"), limit=24)
    priority = payload.get("priority") or default_priority(category)
    confidence = payload.get("confidence") or default_confidence(category)
    created_at = _float_or_none(payload.get("created_at"))
    card_key = str(payload.get("card_key") or "").strip() or note_card_key(category, title)
    return create_sidecar_card(
        session_id=session_id,
        category=category,
        title=title,
        body=payload.get("body") or "",
        suggested_say=payload.get("suggested_say"),
        suggested_ask=payload.get("suggested_ask"),
        why_now=payload.get("why_now") or default_why_now(category, source_type),
        priority=priority,
        confidence=confidence,
        source_segment_ids=source_ids,
        source_type=source_type,
        sources=payload.get("sources") or [],
        citations=payload.get("citations") or [],
        ephemeral=bool(payload.get("ephemeral")) or not save_transcript,
        card_key=card_key,
        created_at=created_at,
        provisional=bool(payload.get("provisional")),
        evidence_quote=payload.get("evidence_quote") or "",
        owner=payload.get("owner"),
        due_date=payload.get("due_date"),
        missing_info=payload.get("missing_info"),
    )


def recall_payload_to_sidecar_card(session_id: str, payload: dict[str, Any]) -> SidecarCard:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    raw_source_type = str(payload.get("source_type") or "saved_transcript")
    source_type = normalize_sidecar_source_type(raw_source_type)
    category = "work_memory" if source_type == "work_memory" else "memory"
    source_id = str(payload.get("source_id") or "")
    score = clamp_confidence(payload.get("score"), default=0.65)
    reason = str(metadata.get("why_now") or metadata.get("reason") or metadata.get("retrieval_reason") or "").strip()
    if not reason:
        reason = "Retrieved because it overlaps with the current discussion."
    title = metadata.get("title") or source_title(raw_source_type)
    suggested_say = metadata.get("suggested_say") or metadata.get("suggested_contribution")
    priority = metadata.get("priority")
    if not priority:
        if source_type == "work_memory" and score < 0.45:
            priority = "low"
        elif score >= 0.88:
            priority = "high"
        else:
            priority = "normal"
    return create_sidecar_card(
        session_id=session_id,
        category=category,
        title=title,
        body=payload.get("text") or "",
        suggested_say=suggested_say,
        suggested_ask=metadata.get("suggested_ask"),
        why_now=reason,
        priority=priority,
        confidence=score,
        source_segment_ids=payload.get("source_segment_ids") or [],
        source_type=source_type,
        sources=metadata.get("sources") or [],
        citations=metadata.get("citations") or [],
        ephemeral=True,
        card_key=f"recall:{raw_source_type}:{source_id}",
        explicitly_requested=bool(payload.get("explicitly_requested") or metadata.get("explicitly_requested")),
    )


def status_sidecar_card(
    *,
    session_id: str,
    title: str,
    body: str,
    why_now: str,
    card_key: str,
    priority: str = "low",
    confidence: float = 0.5,
) -> SidecarCard:
    return create_sidecar_card(
        session_id=session_id,
        category="status",
        title=title,
        body=body,
        why_now=why_now,
        priority=priority,
        confidence=confidence,
        source_type="model_fallback",
        ephemeral=True,
        card_key=card_key,
    )


def note_card_key(category: object, title: object) -> str:
    normalized_category = normalize_sidecar_category(category)
    title_text = " ".join(str(title or "").lower().split())[:120] or "untitled"
    return f"note:{normalized_category}:{title_text}"


def stable_card_key(category: str, source_type: str, title: str, sources: list[str]) -> str:
    material = "|".join([category, source_type, title.lower(), *sources])
    digest = hashlib.sha1(material.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"{category}:{source_type}:{digest}"


def default_title(category: str) -> str:
    return {
        "action": "Follow-up",
        "decision": "Decision",
        "question": "Open question",
        "risk": "Risk to watch",
        "clarification": "Clarify",
        "contribution": "Say this",
        "memory": "Relevant prior transcript",
        "work_memory": "Relevant past work",
        "web": "Current public web",
        "status": "Sidecar status",
    }.get(category, "Sidecar note")


def default_priority(category: str) -> str:
    if category in {"action", "decision", "risk", "contribution"}:
        return "high"
    if category == "status":
        return "low"
    return "normal"


def default_confidence(category: str) -> float:
    if category in {"status", "note"}:
        return 0.55
    if category in {"web", "work_memory", "memory"}:
        return 0.72
    return 0.68


def default_why_now(category: str, source_type: str) -> str:
    if source_type == "brave_web":
        return "Sanitized public/current web context matched the meeting question."
    if source_type == "work_memory":
        return "Past-work context overlaps with the current discussion."
    if source_type == "saved_transcript":
        return "Saved transcript history overlaps with the current discussion."
    if category == "action":
        return "Recent speech contains a possible follow-up or owner."
    if category == "decision":
        return "Recent speech sounds like something was decided or settled."
    if category == "question":
        return "Recent speech left an open question."
    if category == "risk":
        return "Recent speech points to a risk or assumption to watch."
    if category == "clarification":
        return "A clarifying question could prevent ambiguity right now."
    if category == "contribution":
        return "This looks like a useful point BP could add to the meeting."
    return "Generated from the current discussion."


def source_title(source_type: str) -> str:
    if source_type == "work_memory_project":
        return "Relevant past work"
    if source_type in {"session", "transcript_segment", "session_summary"}:
        return "Relevant prior transcript"
    if source_type in {"file", "document_chunk"}:
        return "Relevant local note"
    return "Relevant memory"


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
