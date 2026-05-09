from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

from brain_sidecar.core.models import SidecarCard, TranscriptSegment
from brain_sidecar.core.note_quality import identity_claim_unsupported, is_generic_clarification
from brain_sidecar.eval.models import EvalMemoryEvent, EvalReplayResult, EvalReport


DEFAULT_REJECT_TERMS = [
    "tax",
    "owed",
    "payable",
    "compensation",
    "investment return",
    "Project Alpha",
    "grass maintenance",
    "CT/PT",
    "relay settings",
    "trip path",
    "breaker failure",
    "remote work policy",
]


def build_report(result: EvalReplayResult, memory_events: list[EvalMemoryEvent] | None = None) -> EvalReport:
    memory_events = memory_events or []
    current_cards = [card for card in result.accepted_cards if is_current_note_card(card)]
    duration_s = transcript_duration(result.input_segments)
    suppression_reasons = Counter(normalize_suppression_reason(item.reason) for item in result.suppressed_cards)
    reject_terms = list(DEFAULT_REJECT_TERMS)
    if result.expectation:
        reject_terms.extend(result.expectation.should_reject_terms)
    unsupported = unsupported_topic_violations(current_cards, reject_terms)
    memory_leaks = memory_leakage_violations(current_cards, memory_events)
    identity = identity_claim_violations(current_cards, result.consolidated_segments)
    expected_hits, expected_misses = expected_topic_results(current_cards, result.expectation)
    duplicate_count = duplicate_card_count(current_cards)
    generic_count = sum(1 for card in current_cards if is_generic_clarification(card.title + " " + card.body))
    generic_rate_limited = suppression_reasons.get("generic_clarification_rate_limited", 0)
    source_coverage = percent_with(current_cards, lambda card: bool(card.source_segment_ids))
    quote_coverage = percent_with(current_cards, lambda card: bool(card.evidence_quote.strip()))
    report_passed, recommendations = evaluate_pass_fail(
        current_cards=current_cards,
        duration_s=duration_s,
        source_coverage=source_coverage,
        quote_coverage=quote_coverage,
        unsupported=unsupported,
        memory_leaks=memory_leaks,
        identity=identity,
        expected_misses=expected_misses,
        expectation=result.expectation,
        generic_count=generic_count,
    )
    return EvalReport(
        session_id=result.session_id,
        input_segment_count=len(result.input_segments),
        consolidated_segment_count=len(result.consolidated_segments),
        collapsed_segment_count=result.collapsed_segment_count,
        generated_card_count=len(result.generated_cards),
        accepted_card_count=len(current_cards),
        suppressed_card_count=len(result.suppressed_cards),
        memory_card_count=len(result.memory_cards),
        current_note_count=len(current_cards),
        cards_per_5min=round(len(current_cards) / max(1.0, duration_s / 300.0), 3),
        cards_per_100_segments=round(len(current_cards) / max(1, len(result.input_segments)) * 100, 3),
        percent_cards_with_source_ids=source_coverage,
        percent_cards_with_evidence_quote=quote_coverage,
        unsupported_topic_violations=unsupported,
        memory_leakage_violations=memory_leaks,
        identity_claim_violations=identity,
        duplicate_card_count=duplicate_count,
        generic_clarification_count=generic_count,
        generic_clarification_rate_limited_count=generic_rate_limited,
        expected_topic_hits=expected_hits,
        expected_topic_misses=expected_misses,
        suppression_reasons=dict(sorted(suppression_reasons.items())),
        report_passed=report_passed,
        accepted_cards=[card.to_dict() for card in current_cards],
        suppressed_cards=[
            {
                "title": item.card.title,
                "category": item.card.category,
                "reason": item.reason,
                "source_segment_ids": item.card.source_segment_ids,
            }
            for item in result.suppressed_cards
        ],
        recommendations=recommendations,
    )


def is_current_note_card(card: SidecarCard) -> bool:
    if card.category in {"memory", "web"}:
        return False
    return card.source_type in {"transcript", "model_fallback"}


def transcript_duration(segments: list[TranscriptSegment]) -> float:
    if not segments:
        return 0.0
    return max(segment.end_s for segment in segments) - min(segment.start_s for segment in segments)


def percent_with(cards: list[SidecarCard], predicate) -> float:
    if not cards:
        return 1.0
    return round(sum(1 for card in cards if predicate(card)) / len(cards), 6)


def unsupported_topic_violations(cards: list[SidecarCard], terms: Iterable[str]) -> list[dict]:
    violations: list[dict] = []
    for card in cards:
        text = card_text(card)
        for term in dedupe_terms(terms):
            if term_present(term, text):
                violations.append({"term": term, "card_title": card.title, "card_id": card.id})
    return violations


def memory_leakage_violations(cards: list[SidecarCard], memory_events: list[EvalMemoryEvent]) -> list[dict]:
    violations: list[dict] = []
    for memory in memory_events:
        if memory.expected_behavior != "memory_only":
            continue
        for term in memory.forbidden_in_current_notes:
            for card in cards:
                if term_present(term, card_text(card)):
                    violations.append(
                        {
                            "term": term,
                            "memory_id": memory.id,
                            "card_title": card.title,
                            "card_id": card.id,
                        }
                    )
    return violations


def identity_claim_violations(cards: list[SidecarCard], evidence_segments: list[TranscriptSegment]) -> list[dict]:
    evidence_text = " ".join(segment.text for segment in evidence_segments)
    violations: list[dict] = []
    for card in cards:
        output_text = " ".join([card.title, card.body, card.suggested_say or "", card.suggested_ask or ""])
        if identity_claim_unsupported(output_text, evidence_text, {"ready": False, "enrollment_status": "not_enrolled"}):
            violations.append({"reason": "identity_claim_not_supported", "card_title": card.title, "card_id": card.id})
    return violations


def expected_topic_results(cards: list[SidecarCard], expectation) -> tuple[list[str], list[str]]:
    if expectation is None:
        return [], []
    hits: list[str] = []
    misses: list[str] = []
    for expected in expectation.should_accept_any:
        count = sum(1 for card in cards if any(term_present(term, card_text(card)) for term in expected.required_terms_any))
        if count > 0 and (expected.max_count is None or count <= expected.max_count):
            hits.append(expected.topic)
        else:
            misses.append(expected.topic)
    return hits, misses


def duplicate_card_count(cards: list[SidecarCard]) -> int:
    seen: set[str] = set()
    duplicates = 0
    for card in cards:
        key = card.card_key or re.sub(r"\s+", " ", f"{card.category} {card.title}".lower()).strip()
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def normalize_suppression_reason(reason: str) -> str:
    if reason.startswith("unsupported_material_terms"):
        return "unsupported_material_terms"
    if reason.startswith("unsupported_topic"):
        return "unsupported_topic"
    return reason


def evaluate_pass_fail(
    *,
    current_cards: list[SidecarCard],
    duration_s: float,
    source_coverage: float,
    quote_coverage: float,
    unsupported: list[dict],
    memory_leaks: list[dict],
    identity: list[dict],
    expected_misses: list[str],
    expectation,
    generic_count: int,
) -> tuple[bool, list[str]]:
    recommendations: list[str] = []
    min_cards = expectation.min_accepted_current_cards if expectation else None
    max_cards = expectation.max_accepted_current_cards if expectation else None
    if min_cards is not None and len(current_cards) < min_cards:
        recommendations.append(f"Accepted card count is below fixture minimum {min_cards}.")
    if max_cards is not None and len(current_cards) > max_cards:
        recommendations.append(f"Accepted card count exceeds fixture maximum {max_cards}.")
    if source_coverage < 1.0:
        recommendations.append("Some accepted current cards are missing source_segment_ids.")
    if quote_coverage < 1.0:
        recommendations.append("Some accepted current cards are missing evidence_quote.")
    if unsupported:
        recommendations.append("Accepted current cards contain unsupported-topic terms.")
    if memory_leaks:
        recommendations.append("Memory-only distractor content leaked into current-meeting cards.")
    if identity:
        recommendations.append("Accepted current cards contain unsupported BP/Brandon ownership claims.")
    if expected_misses:
        recommendations.append("Expected useful topics were missed: " + ", ".join(expected_misses))
    allowed_generic = max(1, math.ceil(max(0.1, duration_s) / 300.0))
    if generic_count > allowed_generic:
        recommendations.append("Generic clarification cards exceed one per simulated five-minute window.")
    return not recommendations, recommendations


def term_present(term: str, text: str) -> bool:
    clean_term = term.strip().lower()
    clean_text = text.lower()
    if not clean_term:
        return False
    if re.search(r"^[a-z0-9 ]+$", clean_term):
        pattern = r"(?<![a-z0-9])" + re.escape(clean_term) + r"(?![a-z0-9])"
        return re.search(pattern, clean_text) is not None
    return clean_term in clean_text


def card_text(card: SidecarCard) -> str:
    return " ".join(
        part
        for part in [
            card.title,
            card.body,
            card.suggested_say or "",
            card.suggested_ask or "",
            card.evidence_quote,
        ]
        if part
    )


def dedupe_terms(terms: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for term in terms:
        clean = term.strip()
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        result.append(clean)
    return result
