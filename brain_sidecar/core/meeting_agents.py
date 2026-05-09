from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from brain_sidecar.core.models import SearchHit, SidecarCard, TranscriptSegment, compact_text
from brain_sidecar.core.note_quality import NoteQualityDecision, NoteQualityGate, evidence_segment_map, quote_supported
from brain_sidecar.core.sidecar_cards import create_sidecar_card


COMMITMENT_RE = re.compile(r"\b(i'll|i will|i can|i'm going to|i am going to|let me)\b", re.IGNORECASE)
DUE_RE = re.compile(r"\b(by|before|after|on)\s+(monday|tuesday|wednesday|thursday|friday|tomorrow|next week|eod|end of day)\b", re.IGNORECASE)
DECISION_CRITERIA_RE = re.compile(r"\b(criteria|threshold|decide|decision|approved|approval|sign[- ]?off)\b", re.IGNORECASE)
BLOCKED_RE = re.compile(r"\b(blocked|blocking|dependency|depends on|waiting on|risk)\b", re.IGNORECASE)
REVIEW_PATH_RE = re.compile(r"\b(review path|reviewer|under review|for review|sign[- ]?off|approval path)\b", re.IGNORECASE)
CONTRIBUTION_RE = re.compile(r"\b(we should|we need to|it sounds like|worth|maybe ask|question is|risk is)\b", re.IGNORECASE)


@dataclass(frozen=True)
class CandidateEvidence:
    source_segment_ids: list[str]
    evidence_quote: str


@dataclass(frozen=True)
class MeetingDiagnostics:
    generated_candidate_count: int = 0
    accepted_count: int = 0
    suppressed_count: int = 0
    top_suppression_reasons: list[dict[str, object]] = field(default_factory=list)
    evidence_quote_coverage: float = 1.0
    source_id_coverage: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_candidate_count": self.generated_candidate_count,
            "accepted_count": self.accepted_count,
            "suppressed_count": self.suppressed_count,
            "top_suppression_reasons": self.top_suppression_reasons,
            "evidence_quote_coverage": self.evidence_quote_coverage,
            "source_id_coverage": self.source_id_coverage,
        }


@dataclass(frozen=True)
class ContractCriticResult:
    accepted_cards: list[SidecarCard]
    diagnostics: MeetingDiagnostics


class EvidenceGroundingAgent:
    def source_id_map(self, segments: list[TranscriptSegment]) -> dict[str, TranscriptSegment]:
        return evidence_segment_map(segments)

    def normalize_candidate_evidence(
        self,
        card: SidecarCard,
        segments: list[TranscriptSegment],
    ) -> CandidateEvidence | None:
        evidence_by_id = self.source_id_map(segments)
        source_ids = [source_id for source_id in card.source_segment_ids if source_id in evidence_by_id]
        if not source_ids:
            return None
        selected_text = " ".join(evidence_by_id[source_id].text for source_id in source_ids)
        quote = compact_text(card.evidence_quote or selected_text, limit=420)
        if not quote or not quote_supported(quote, selected_text):
            return None
        return CandidateEvidence(source_ids, quote)


class CommitmentAgent:
    def cards(self, session_id: str, recent_segments: list[TranscriptSegment]) -> list[SidecarCard]:
        cards: list[SidecarCard] = []
        for segment in recent_segments[-4:]:
            text = segment.text.strip()
            if not text or not COMMITMENT_RE.search(text):
                continue
            if _is_confident_bp_speaker(segment):
                cards.append(
                    create_sidecar_card(
                        session_id=session_id,
                        category="action",
                        title="BP follow-up",
                        body=f"BP appears to own this follow-up: {compact_text(text, limit=220)}",
                        suggested_say="I'll own that follow-up and confirm the next step.",
                        why_now="Speaker identity labels this commitment as BP with enough confidence.",
                        priority="high",
                        confidence=max(0.74, float(segment.speaker_confidence or 0.0)),
                        source_segment_ids=[segment.id],
                        source_type="transcript",
                        card_key=f"action:bp:{segment.id}",
                        ephemeral=True,
                        evidence_quote=compact_text(text, limit=260),
                        owner="BP",
                    )
                )
            elif segment.speaker_role == "other":
                cards.append(
                    create_sidecar_card(
                        session_id=session_id,
                        category="clarification",
                        title="Confirm owner",
                        body="Another speaker used first-person commitment language; do not assign this to BP without confirmation.",
                        suggested_ask="Can we confirm who owns that follow-up?",
                        why_now="Speaker identity indicates the commitment came from someone other than BP.",
                        priority="normal",
                        confidence=0.68,
                        source_segment_ids=[segment.id],
                        source_type="transcript",
                        card_key=f"clarify:owner:{segment.id}",
                        ephemeral=True,
                        evidence_quote=compact_text(text, limit=260),
                    )
                )
            else:
                cards.append(
                    create_sidecar_card(
                        session_id=session_id,
                        category="clarification",
                        title="Owner unclear",
                        body="A follow-up may have been promised, but the speaker was unknown or low confidence.",
                        suggested_ask="Who should own that follow-up?",
                        why_now="Speaker identity is not confident enough to assign the action to BP.",
                        priority="normal",
                        confidence=0.52,
                        source_segment_ids=[segment.id],
                        source_type="transcript",
                        card_key=f"clarify:unknown-owner:{segment.id}",
                        ephemeral=True,
                        evidence_quote=compact_text(text, limit=260),
                    )
                )
        return cards


class ClarificationAgent:
    def cards(self, session_id: str, recent_segments: list[TranscriptSegment]) -> list[SidecarCard]:
        cards: list[SidecarCard] = []
        recent = recent_segments[-4:]
        if not recent:
            return []
        text = " ".join(segment.text for segment in recent)
        source_ids = [segment.id for segment in recent]
        quote = compact_text(text, limit=260)
        lower = text.lower()

        if "owner" in lower and any(term in lower for term in ["unclear", "confirm", "who owns", "who owner"]):
            cards.append(self._card(session_id, "Confirm owner", "Confirm who owns the follow-up.", "Who owns that follow-up?", source_ids, quote, "owner"))
        if DUE_RE.search(text) and any(term in lower for term in ["unclear", "confirm", "which", "when", "timing"]):
            cards.append(self._card(session_id, "Confirm timing", "The due date or timing should be confirmed.", "What timing should we use for that follow-up?", source_ids, quote, "due-date"))
        if DECISION_CRITERIA_RE.search(text) and any(term in lower for term in ["unclear", "criteria", "threshold", "decide"]):
            cards.append(self._card(session_id, "Decision criteria", "The decision criteria need to be made explicit.", "What criteria decide this?", source_ids, quote, "decision-criteria"))
        if BLOCKED_RE.search(text):
            cards.append(self._card(session_id, "Blocked dependency", "A dependency or blocker may need an owner.", "What dependency blocks this, and who can clear it?", source_ids, quote, "blocked-dependency"))
        if REVIEW_PATH_RE.search(text):
            cards.append(self._card(session_id, "Review path", "The review or approval path may need confirmation.", "Who is the reviewer or approval path for this?", source_ids, quote, "review-path"))
        return _dedupe_by_key(cards)[:3]

    def _card(
        self,
        session_id: str,
        title: str,
        body: str,
        suggested_ask: str,
        source_ids: list[str],
        evidence_quote: str,
        key_suffix: str,
    ) -> SidecarCard:
        return create_sidecar_card(
            session_id=session_id,
            category="clarification",
            title=title,
            body=body,
            suggested_ask=suggested_ask,
            why_now="Recent speech contains concrete ambiguity.",
            priority="normal",
            confidence=0.66,
            source_segment_ids=source_ids,
            source_type="transcript",
            card_key=f"clarify:{key_suffix}:{':'.join(source_ids[-2:])}",
            ephemeral=True,
            evidence_quote=evidence_quote,
            missing_info=key_suffix.replace("-", " "),
        )


class ContributionAgent:
    def cards(self, session_id: str, recent_segments: list[TranscriptSegment]) -> list[SidecarCard]:
        cards: list[SidecarCard] = []
        for segment in recent_segments[-4:]:
            text = segment.text.strip()
            if not text or not CONTRIBUTION_RE.search(text):
                continue
            if len(text.split()) < 7:
                continue
            cards.append(
                create_sidecar_card(
                    session_id=session_id,
                    category="contribution",
                    title="Say this",
                    body="There may be a concise meeting contribution to make from the current thread.",
                    suggested_say=_suggested_say(text),
                    why_now="The current transcript contains a directly grounded point BP could voice.",
                    priority="normal",
                    confidence=0.63,
                    source_segment_ids=[segment.id],
                    source_type="transcript",
                    card_key=f"contribution:say-this:{segment.id}",
                    ephemeral=True,
                    evidence_quote=compact_text(text, limit=260),
                )
            )
        return cards[:2]


class MemoryBoundaryAgent:
    def filter_current_meeting_cards(
        self,
        cards: list[SidecarCard],
        recent_segments: list[TranscriptSegment],
        recall_hits: list[SearchHit],
    ) -> list[SidecarCard]:
        if not cards:
            return []
        transcript_text = " ".join(segment.text for segment in recent_segments)
        recall_text = " ".join(hit.text for hit in recall_hits[:6])
        if not transcript_text or not recall_text:
            return cards
        kept: list[SidecarCard] = []
        transcript_terms = _terms(transcript_text)
        recall_only_terms = _terms(recall_text) - transcript_terms
        for card in cards:
            if card.category in {"memory", "web", "status"}:
                kept.append(card)
                continue
            output_terms = _terms(" ".join([card.title, card.body, card.suggested_say or "", card.suggested_ask or ""]))
            unsupported_recall_terms = [term for term in output_terms if term in recall_only_terms and len(term) > 4]
            if unsupported_recall_terms:
                continue
            kept.append(card)
        return kept


class ContractCriticAgent:
    def __init__(self, gate: NoteQualityGate) -> None:
        self.gate = gate

    def review(
        self,
        cards: list[SidecarCard],
        evidence_segments: list[TranscriptSegment],
        speaker_identity_status: dict,
        *,
        max_cards: int,
    ) -> ContractCriticResult:
        accepted: list[SidecarCard] = []
        reasons: Counter[str] = Counter()
        reviewed = 0
        for card in cards:
            reviewed += 1
            decision = self.gate.evaluate(card, evidence_segments, speaker_identity_status)
            if decision.action != "accept":
                reasons[decision.reason] += 1
                continue
            if decision.priority_override:
                card = _replace_priority(card, decision)
            self.gate.remember_accepted(card, evidence_segments)
            accepted.append(card)
            if len(accepted) >= max_cards:
                break
        if reviewed < len(cards):
            reasons["generation_pass_cap"] += len(cards) - reviewed
        suppressed_count = sum(reasons.values())
        return ContractCriticResult(
            accepted_cards=accepted,
            diagnostics=diagnostics_for_cards(
                cards,
                accepted_count=len(accepted),
                suppressed_count=suppressed_count,
                reasons=reasons,
            ),
        )


def diagnostics_for_cards(
    cards: list[SidecarCard],
    *,
    accepted_count: int,
    suppressed_count: int,
    reasons: Counter[str] | None = None,
) -> MeetingDiagnostics:
    total = len(cards)
    if total == 0:
        return MeetingDiagnostics()
    reason_counts = reasons or Counter()
    return MeetingDiagnostics(
        generated_candidate_count=total,
        accepted_count=accepted_count,
        suppressed_count=suppressed_count,
        top_suppression_reasons=[
            {"reason": reason, "count": count}
            for reason, count in reason_counts.most_common(4)
        ],
        evidence_quote_coverage=round(sum(1 for card in cards if card.evidence_quote) / total, 3),
        source_id_coverage=round(sum(1 for card in cards if card.source_segment_ids) / total, 3),
    )


def deterministic_meeting_cards(
    session_id: str,
    recent_segments: list[TranscriptSegment],
) -> list[SidecarCard]:
    cards: list[SidecarCard] = []
    cards.extend(CommitmentAgent().cards(session_id, recent_segments))
    cards.extend(ClarificationAgent().cards(session_id, recent_segments))
    cards.extend(ContributionAgent().cards(session_id, recent_segments))
    cards.extend(_risk_cards(session_id, recent_segments))
    return _dedupe_by_key(cards)[:5]


def _risk_cards(session_id: str, recent_segments: list[TranscriptSegment]) -> list[SidecarCard]:
    risk_text = " ".join(segment.text for segment in recent_segments[-4:])
    if not any(term in risk_text.lower() for term in ["risk", "rollback", "blocked", "deadline", "dependency"]):
        return []
    return [
        create_sidecar_card(
            session_id=session_id,
            category="risk",
            title="Watch the risk",
            body="Recent speech points to a risk or dependency that may need an owner or mitigation.",
            suggested_ask="What would make this plan fail, and who owns that mitigation?",
            why_now="Risk language appeared in the latest transcript window.",
            priority="high",
            confidence=0.64,
            source_segment_ids=[segment.id for segment in recent_segments[-4:]],
            source_type="transcript",
            card_key="risk:recent-window",
            ephemeral=True,
            evidence_quote=compact_text(risk_text, limit=260),
        )
    ]


def _is_confident_bp_speaker(segment: TranscriptSegment) -> bool:
    return (
        segment.speaker_role == "user"
        and not segment.speaker_low_confidence
        and (segment.speaker_confidence is None or segment.speaker_confidence >= 0.82)
    )


def _suggested_say(text: str) -> str:
    clean = compact_text(text, limit=180)
    if clean.endswith("?"):
        return clean
    return f"It sounds like {clean[0].lower() + clean[1:]}" if clean else ""


def _replace_priority(card: SidecarCard, decision: NoteQualityDecision) -> SidecarCard:
    from dataclasses import replace

    return replace(card, priority=decision.priority_override or card.priority)


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9][a-z0-9+-]*", text.lower())
        if len(term) >= 3
    }


def _dedupe_by_key(cards: list[SidecarCard]) -> list[SidecarCard]:
    seen: set[str] = set()
    result: list[SidecarCard] = []
    for card in cards:
        key = card.card_key or f"{card.category}:{card.title}:{','.join(card.source_segment_ids)}"
        if key in seen or not card.source_segment_ids or not card.evidence_quote:
            continue
        seen.add(key)
        result.append(card)
    return result
