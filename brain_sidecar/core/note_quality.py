from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Literal

from brain_sidecar.core.asr_aliases import evidence_terms_with_aliases, normalize_for_evidence_match
from brain_sidecar.core.models import SidecarCard, TranscriptSegment


NoteQualityAction = Literal["accept", "suppress", "merge"]


@dataclass(frozen=True)
class NoteQualityDecision:
    action: NoteQualityAction
    reason: str
    normalized_fingerprint: str | None
    priority_override: str | None = None


@dataclass(frozen=True)
class AcceptedNote:
    fingerprint: str
    category: str
    title: str
    evidence_entities: frozenset[str]
    created_at: float
    generic_clarification: bool


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+-]*")

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "and",
    "are",
    "ask",
    "before",
    "being",
    "card",
    "clear",
    "clarify",
    "confirm",
    "current",
    "during",
    "from",
    "have",
    "help",
    "into",
    "issue",
    "meeting",
    "need",
    "needs",
    "next",
    "note",
    "open",
    "owner",
    "owners",
    "owns",
    "please",
    "point",
    "question",
    "review",
    "should",
    "speaker",
    "that",
    "the",
    "their",
    "them",
    "there",
    "this",
    "through",
    "time",
    "weekly",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "work",
    "would",
    "you",
}

STRUCTURE_TERMS = {
    "action",
    "align",
    "alignment",
    "answer",
    "answers",
    "appropriate",
    "area",
    "assignment",
    "available",
    "communication",
    "consulting",
    "coordinate",
    "coordination",
    "decision",
    "determine",
    "discuss",
    "follow",
    "follow-up",
    "material",
    "matter",
    "path",
    "reviewer",
    "schedule",
    "stakeholder",
    "subject",
    "technical",
    "unclear",
}

MATERIAL_ALLOW_IF_ALIAS = {
    "diagram": "sld",
    "expert": "expert",
    "reviewer": "reviewer",
    "siemens": "siemens",
    "nerc": "nerc",
    "gba": "gba",
}

EXPLICIT_SIGNAL_RE = re.compile(
    r"\b("
    r"review|reply|send|forward|confirm|ask|meet|meeting|due|tomorrow|"
    r"by next monday|6 to 8 hours|six to eight hours|one day|"
    r"you are going to|you're going to|please reply|please send"
    r")\b",
    re.IGNORECASE,
)

GENERIC_CLARIFY_RE = re.compile(
    r"\b("
    r"owner unclear|who owns|time confirmation|what is due|next step unclear|"
    r"clarify task assignments|confirm owner|owner|unclear"
    r")\b",
    re.IGNORECASE,
)

NEW_EVIDENCE_RE = re.compile(
    r"\b("
    r"sunil|greg|craig|manish|siemens|client|tomorrow|monday|tuesday|"
    r"wednesday|thursday|friday|\d{1,2}\s*(?:am|pm|o'clock)|"
    r"due|deadline|decided|decision|owner|reviewer"
    r")\b",
    re.IGNORECASE,
)

IDENTITY_CLAIM_RE = re.compile(
    r"\b(bp|brandon|brandon poole)\b.{0,80}\b("
    r"needs?|owns?|responsible|must|owes?|pay|assigned|will|should"
    r")\b|"
    r"\b(needs?|owns?|responsible|must|owes?|pay|assigned|will|should)\b.{0,80}\b(bp|brandon|brandon poole)\b",
    re.IGNORECASE,
)

IDENTITY_EVIDENCE_RE = re.compile(
    r"\b(bp|brandon|brandon poole)\b.{0,80}\b("
    r"i will|i'll|i can|own|owns|responsible|assigned|send|review|confirm|follow"
    r")\b|"
    r"\b(i will|i'll|i can|own|owns|responsible|assigned|send|review|confirm|follow)\b.{0,80}\b"
    r"(bp|brandon|brandon poole)\b",
    re.IGNORECASE,
)

UNSUPPORTED_TOPIC_PATTERNS = [
    (re.compile(r"\b(owed|debt|payable|tax|compensation|investment return)\b", re.IGNORECASE), re.compile(r"\b(owed|debt|payable|tax|compensation|investment return|invoice|payment|pay)\b", re.IGNORECASE)),
    (re.compile(r"\b(remote work|company policy)\b", re.IGNORECASE), re.compile(r"\b(remote work|company policy|policy)\b", re.IGNORECASE)),
    (re.compile(r"\bproject alpha\b", re.IGNORECASE), re.compile(r"\bproject alpha\b", re.IGNORECASE)),
    (re.compile(r"\bgrass maintenance\b", re.IGNORECASE), re.compile(r"\bgrass maintenance|maintenance\b", re.IGNORECASE)),
    (re.compile(r"\b(ct fees|current transformer fees)\b", re.IGNORECASE), re.compile(r"\b(ct fees|current transformer fees|fee|fees)\b", re.IGNORECASE)),
    (
        re.compile(
            r"\b(ct/pt|ct\s*/\s*pt|relay replacement|relay settings|trip path|breaker failure|interlocks?|workforce planning phase)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(ct|pt|c\.t\.|p\.t\.|relay|protection|trip path|breaker|settings|interlock|workforce planning phase)\b",
            re.IGNORECASE,
        ),
    ),
]


class NoteQualityGate:
    def __init__(
        self,
        *,
        min_evidence_segments: int = 2,
        duplicate_window_seconds: float = 120.0,
        generic_clarify_window_seconds: float = 300.0,
        max_cards_per_5min: int = 8,
    ) -> None:
        self.min_evidence_segments = max(1, int(min_evidence_segments))
        self.duplicate_window_seconds = max(1.0, float(duplicate_window_seconds))
        self.generic_clarify_window_seconds = max(1.0, float(generic_clarify_window_seconds))
        self.max_cards_per_5min = max(1, int(max_cards_per_5min))
        self._accepted: deque[AcceptedNote] = deque(maxlen=128)

    def evaluate(
        self,
        card_or_candidate: SidecarCard,
        evidence_segments: list[TranscriptSegment],
        speaker_identity_status: dict,
        now: float | None = None,
    ) -> NoteQualityDecision:
        now = time.time() if now is None else float(now)
        self._prune(now)
        card = card_or_candidate
        fingerprint = normalized_fingerprint(card, evidence_segments)

        if card.source_type not in {"transcript", "model_fallback"}:
            return NoteQualityDecision("suppress", "generated_note_must_use_live_transcript_evidence", fingerprint)
        if not card.title.strip():
            return NoteQualityDecision("suppress", "empty_title", fingerprint)
        if not card.body.strip() and not (card.suggested_say or card.suggested_ask):
            return NoteQualityDecision("suppress", "empty_body", fingerprint)
        source_ids = set(card.source_segment_ids)
        if not source_ids:
            return NoteQualityDecision("suppress", "missing_source_segment_ids", fingerprint)
        evidence_by_id = evidence_segment_map(evidence_segments)
        if not source_ids <= set(evidence_by_id):
            return NoteQualityDecision("suppress", "source_segment_ids_not_in_recent_evidence", fingerprint)
        evidence_quote = str(getattr(card, "evidence_quote", "") or "").strip()
        if not evidence_quote:
            return NoteQualityDecision("suppress", "missing_evidence_quote", fingerprint)

        selected_segments = [evidence_by_id[source_id] for source_id in card.source_segment_ids if source_id in evidence_by_id]
        evidence_text = " ".join(segment.text for segment in selected_segments)
        if len({segment.id for segment in selected_segments}) < self.min_evidence_segments and not EXPLICIT_SIGNAL_RE.search(evidence_text):
            return NoteQualityDecision("suppress", "insufficient_evidence_segments", fingerprint)
        if not quote_supported(evidence_quote, evidence_text):
            return NoteQualityDecision("suppress", "evidence_quote_not_supported", fingerprint)

        output_text = " ".join(
            part
            for part in [
                card.title,
                card.body,
                card.suggested_say or "",
                card.suggested_ask or "",
            ]
            if part
        )
        topic_reason = unsupported_topic_reason(output_text, evidence_text)
        if topic_reason:
            return NoteQualityDecision("suppress", topic_reason, fingerprint)
        unsupported = unsupported_material_terms(output_text, evidence_text)
        if unsupported:
            return NoteQualityDecision("suppress", f"unsupported_material_terms:{','.join(unsupported[:5])}", fingerprint)
        if identity_claim_unsupported(output_text, evidence_text, speaker_identity_status):
            return NoteQualityDecision("suppress", "identity_claim_not_supported", fingerprint)
        if duplicate_recent(fingerprint, self._accepted, now, self.duplicate_window_seconds):
            return NoteQualityDecision("suppress", "duplicate_recent_card", fingerprint)
        if generic_clarification_limited(output_text, evidence_text, self._accepted, now, self.generic_clarify_window_seconds):
            return NoteQualityDecision("suppress", "generic_clarification_rate_limited", fingerprint)
        if not high_priority_bypass(card) and count_recent(self._accepted, now, 300.0) >= self.max_cards_per_5min:
            return NoteQualityDecision("suppress", "rolling_volume_cap", fingerprint)

        priority_override = "low" if card.category in {"status", "note"} and card.priority == "normal" else None
        return NoteQualityDecision("accept", "accepted", fingerprint, priority_override=priority_override)

    def remember_accepted(self, card: SidecarCard, evidence_segments: list[TranscriptSegment] | None = None, now: float | None = None) -> None:
        now = time.time() if now is None else float(now)
        evidence_segments = evidence_segments or []
        evidence_text = " ".join(segment.text for segment in evidence_segments)
        self._accepted.append(
            AcceptedNote(
                fingerprint=normalized_fingerprint(card, evidence_segments),
                category=card.category,
                title=card.title,
                evidence_entities=frozenset(important_terms(evidence_text)[:12]),
                created_at=now,
                generic_clarification=is_generic_clarification(card.title + " " + card.body),
            )
        )
        self._prune(now)

    def _prune(self, now: float) -> None:
        keep_after = now - max(300.0, self.duplicate_window_seconds, self.generic_clarify_window_seconds)
        while self._accepted and self._accepted[0].created_at < keep_after:
            self._accepted.popleft()


def evidence_segment_map(segments: list[TranscriptSegment]) -> dict[str, TranscriptSegment]:
    mapped: dict[str, TranscriptSegment] = {}
    for segment in segments:
        mapped[segment.id] = segment
        for source_id in getattr(segment, "source_segment_ids", []) or []:
            mapped[source_id] = segment
    return mapped


def normalized_fingerprint(card: SidecarCard, evidence_segments: list[TranscriptSegment]) -> str:
    output_text = " ".join([card.title, card.body, card.suggested_ask or "", card.suggested_say or ""])
    entities = "-".join(important_terms(output_text)[:6])
    verb = core_action_verb(output_text)
    title = normalize_for_evidence_match(card.title)
    title = re.sub(r"\b(owner unclear|confirm owner|clarify|question|action|risk)\b", "", title).strip()
    return "|".join([card.category, title[:80], verb, entities])


def quote_supported(quote: str, evidence_text: str) -> bool:
    quote_norm = normalize_for_evidence_match(quote)
    evidence_norm = normalize_for_evidence_match(evidence_text)
    if not quote_norm:
        return False
    if quote_norm in evidence_norm:
        return True
    quote_terms = [term for term in _TOKEN_RE.findall(quote_norm) if term not in STOPWORDS]
    if not quote_terms:
        return False
    evidence_terms = evidence_terms_with_aliases(evidence_text)
    overlap = sum(1 for term in quote_terms if term in evidence_terms)
    return overlap / max(1, len(quote_terms)) >= 0.75


def unsupported_topic_reason(output_text: str, evidence_text: str) -> str | None:
    for output_pattern, evidence_pattern in UNSUPPORTED_TOPIC_PATTERNS:
        if output_pattern.search(output_text) and not evidence_pattern.search(evidence_text):
            return f"unsupported_topic:{output_pattern.pattern[:48]}"
    return None


def unsupported_material_terms(output_text: str, evidence_text: str) -> list[str]:
    output_terms = important_terms(output_text)
    evidence_terms = evidence_terms_with_aliases(evidence_text)
    unsupported: list[str] = []
    for term in output_terms:
        if term in evidence_terms or plural_variant_supported(term, evidence_terms) or term in STRUCTURE_TERMS:
            continue
        alias = MATERIAL_ALLOW_IF_ALIAS.get(term)
        if alias and alias in evidence_terms:
            continue
        if len(term) <= 3 and term not in {"sld", "gba", "nerc"}:
            continue
        unsupported.append(term)
    return unsupported


def plural_variant_supported(term: str, evidence_terms: set[str]) -> bool:
    if term.endswith("ies") and f"{term[:-3]}y" in evidence_terms:
        return True
    if term.endswith("s") and term[:-1] in evidence_terms:
        return True
    return f"{term}s" in evidence_terms or f"{term}es" in evidence_terms


def important_terms(text: str) -> list[str]:
    clean = normalize_for_evidence_match(text)
    terms = []
    for token in _TOKEN_RE.findall(clean):
        if token in STOPWORDS or token in STRUCTURE_TERMS:
            continue
        if len(token) < 3 and token not in {"bp"}:
            continue
        terms.append(token)
    return dedupe(terms)


def core_action_verb(text: str) -> str:
    for verb in [
        "review",
        "reply",
        "send",
        "forward",
        "confirm",
        "ask",
        "meet",
        "align",
        "schedule",
        "clarify",
    ]:
        if re.search(rf"\b{re.escape(verb)}\w*\b", text, re.IGNORECASE):
            return verb
    return "note"


def identity_claim_unsupported(output_text: str, evidence_text: str, speaker_identity_status: dict) -> bool:
    if not IDENTITY_CLAIM_RE.search(output_text):
        return False
    ready = bool(speaker_identity_status.get("ready"))
    enrolled = speaker_identity_status.get("enrollment_status") == "enrolled"
    if ready and enrolled:
        return False
    return IDENTITY_EVIDENCE_RE.search(evidence_text) is None


def duplicate_recent(fingerprint: str, accepted: deque[AcceptedNote], now: float, window_s: float) -> bool:
    return any(item.fingerprint == fingerprint and now - item.created_at <= window_s for item in accepted)


def generic_clarification_limited(
    output_text: str,
    evidence_text: str,
    accepted: deque[AcceptedNote],
    now: float,
    window_s: float,
) -> bool:
    if not is_generic_clarification(output_text):
        return False
    if NEW_EVIDENCE_RE.search(evidence_text):
        return False
    return any(item.generic_clarification and now - item.created_at <= window_s for item in accepted)


def is_generic_clarification(text: str) -> bool:
    return GENERIC_CLARIFY_RE.search(text) is not None


def count_recent(accepted: deque[AcceptedNote], now: float, window_s: float) -> int:
    return sum(1 for item in accepted if now - item.created_at <= window_s)


def high_priority_bypass(card: SidecarCard) -> bool:
    return card.priority == "high" and card.category in {"action", "decision"} and bool(card.evidence_quote)


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
