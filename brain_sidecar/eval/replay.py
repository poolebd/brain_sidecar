from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from brain_sidecar.core.dedupe import TranscriptFinalConsolidator
from brain_sidecar.core.models import SidecarCard, TranscriptSegment, compact_text
from brain_sidecar.core.note_quality import NoteQualityGate
from brain_sidecar.core.sidecar_cards import create_sidecar_card
from brain_sidecar.eval.metrics import build_report
from brain_sidecar.eval.models import (
    EvalExpectedBehavior,
    EvalMemoryEvent,
    EvalReplayResult,
    EvalSuppressedCard,
    EvalTranscriptEvent,
    candidate_card_from_dict,
    expectation_from_dict,
    memory_event_from_dict,
    transcript_event_from_dict,
)
from brain_sidecar.eval.noise import apply_default_noise
from brain_sidecar.eval.report import write_json_report, write_markdown_report


def load_fixture(path: Path) -> tuple[list[EvalTranscriptEvent], list[EvalMemoryEvent], list[SidecarCard], EvalExpectedBehavior | None]:
    if path.suffix.lower() == ".txt":
        transcript_events = load_plain_text_fixture(path)
        return transcript_events, [], [], None
    transcript_events: list[EvalTranscriptEvent] = []
    memory_events: list[EvalMemoryEvent] = []
    candidate_payloads: list[dict[str, Any]] = []
    expectation: EvalExpectedBehavior | None = None
    fallback_session_id = path.stem
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
        event_type = payload.get("type")
        if event_type == "transcript":
            transcript_events.append(transcript_event_from_dict(payload, fallback_session_id=fallback_session_id))
        elif event_type == "memory":
            memory_events.append(memory_event_from_dict(payload))
        elif event_type == "candidate_card":
            candidate_payloads.append(payload)
        elif event_type == "expectation":
            expectation = expectation_from_dict(payload, fallback_session_id=fallback_session_id)
        else:
            raise ValueError(f"{path}:{line_number}: unsupported eval event type {event_type!r}")
    session_id = expectation.session_id if expectation else (transcript_events[0].session_id if transcript_events else fallback_session_id)
    candidate_cards = [candidate_card_from_dict(payload, session_id=session_id) for payload in candidate_payloads]
    return transcript_events, memory_events, candidate_cards, expectation


def load_plain_text_fixture(path: Path) -> list[EvalTranscriptEvent]:
    events: list[EvalTranscriptEvent] = []
    session_id = path.stem
    start_s = 0.0
    for index, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = raw_line.strip()
        if not text:
            continue
        end_s = start_s + max(2.0, min(8.0, len(text.split()) * 0.45))
        events.append(
            EvalTranscriptEvent(
                id=f"seg_{index:03d}",
                session_id=session_id,
                start_s=start_s,
                end_s=end_s,
                text=text,
            )
        )
        start_s = end_s + 0.2
    return events


def replay_fixture(
    path: Path,
    *,
    mode: str = "pipeline",
    noise: str = "none",
    include_memory_distractors: bool = True,
) -> tuple[EvalReplayResult, list[EvalMemoryEvent]]:
    transcript_events, memory_events, candidate_cards, expectation = load_fixture(path)
    if noise == "default":
        transcript_events = apply_default_noise(transcript_events, seed=7)
    input_segments = [event.to_segment() for event in transcript_events if event.is_final]
    consolidated_segments, collapsed_count = consolidate_segments(input_segments)
    session_id = expectation.session_id if expectation else (input_segments[0].session_id if input_segments else path.stem)
    memory_cards = [memory.to_card(session_id) for memory in memory_events] if include_memory_distractors else []
    generated_cards = list(candidate_cards)
    if mode == "pipeline":
        generated_cards = [*deterministic_candidate_cards(session_id, consolidated_segments), *generated_cards]
    elif mode != "gate-only":
        raise ValueError(f"unsupported replay mode {mode!r}")
    accepted_cards, suppressed_cards = apply_quality_gate(generated_cards, consolidated_segments)
    return (
        EvalReplayResult(
            session_id=session_id,
            input_segments=input_segments,
            consolidated_segments=consolidated_segments,
            collapsed_segment_count=collapsed_count,
            generated_cards=generated_cards,
            accepted_cards=accepted_cards,
            suppressed_cards=suppressed_cards,
            memory_cards=memory_cards,
            expectation=expectation,
        ),
        memory_events,
    )


def consolidate_segments(segments: list[TranscriptSegment]) -> tuple[list[TranscriptSegment], int]:
    consolidator = TranscriptFinalConsolidator(max_recent=96)
    collapsed_count = 0
    for segment in segments:
        result = consolidator.accept(segment)
        if result.collapsed:
            collapsed_count += 1
    return consolidator.segments(), collapsed_count


def deterministic_candidate_cards(session_id: str, segments: list[TranscriptSegment]) -> list[SidecarCard]:
    cards: list[SidecarCard] = []
    cards.extend(card_if_terms(session_id, segments, "harmonics_alignment"))
    cards.extend(card_if_terms(session_id, segments, "siemens_power_quality"))
    cards.extend(card_if_terms(session_id, segments, "sunil_path"))
    cards.extend(card_if_terms(session_id, segments, "manish_sme"))
    cards.extend(card_if_terms(session_id, segments, "hours_confirmation"))
    cards.extend(card_if_terms(session_id, segments, "schedule_clarification"))
    cards.extend(card_if_terms(session_id, segments, "generic_owner_unclear"))
    return dedupe_cards(cards)


def card_if_terms(session_id: str, segments: list[TranscriptSegment], topic: str) -> list[SidecarCard]:
    text = " ".join(segment.text for segment in segments).lower()
    if topic == "harmonics_alignment" and all(term in text for term in ["harmonics", "sld"]):
        selected = select_segments(segments, ["harmonics", "studies", "sld", "single line", "siemens"])
        return [
            eval_card(
                session_id,
                "question",
                "Harmonics and SLD alignment",
                "Align harmonics and other studies with the SLD and Siemens/client expectations.",
                selected,
                "harmonics",
            )
        ]
    if topic == "siemens_power_quality" and "siemens" in text and "power quality" in text:
        selected = select_segments(segments, ["siemens", "client agreement", "generation agreement", "power quality", "gba"])
        return [
            eval_card(
                session_id,
                "action",
                "Siemens power quality review",
                "Review the Siemens client agreement and power quality question.",
                selected,
                "siemens",
                priority="high",
            )
        ]
    if topic == "sunil_path" and "sunil" in text:
        selected = select_segments(segments, ["sunil", "reviewer", "client communication", "communication path"])
        return [
            eval_card(
                session_id,
                "clarification",
                "Sunil reviewer path",
                "Use Sunil as reviewer or clarify the client communication path if he is unavailable.",
                selected,
                "sunil",
            )
        ]
    if topic == "manish_sme" and "manish" in text and ("sme" in text or "nerc" in text or "narc" in text or "nerd side" in text):
        selected = select_segments(segments, ["manish", "sme", "nerc", "narc", "nerd side", "technical"])
        return [
            eval_card(
                session_id,
                "clarification",
                "Manish SME",
                "Manish may be the SME for the NERC/SLC technical area.",
                selected,
                "manish",
            )
        ]
    if topic == "hours_confirmation" and ("six to eight" in text or "6 to 8" in text or "one day" in text):
        selected = select_segments(segments, ["six to eight", "6 to 8", "one day", "hours", "roberto", "dan", "sunil"])
        return [
            eval_card(
                session_id,
                "action",
                "Confirm project hours",
                "Confirm six to eight hours weekly, roughly one day, for the project over the next few weeks.",
                selected,
                "hours",
                priority="high",
            )
        ]
    if topic == "schedule_clarification" and "tomorrow" in text:
        selected = select_segments(segments, ["tomorrow", "11", "10 your time", "eastern", "local time"])
        return [
            eval_card(
                session_id,
                "clarification",
                "Clarify tomorrow follow-up time",
                "Clarify the exact local time for tomorrow after the morning meeting.",
                selected,
                "tomorrow",
            )
        ]
    if topic == "generic_owner_unclear" and "i can send" in text:
        selected = select_segments(segments, ["i can send"])
        return [
            eval_card(
                session_id,
                "clarification",
                "Owner unclear",
                "Confirm owner for the follow-up.",
                selected,
                "i can send",
            ),
            eval_card(
                session_id,
                "clarification",
                "Owner unclear",
                "Confirm owner for the follow-up.",
                selected,
                "i can send",
            ),
        ]
    return []


def eval_card(
    session_id: str,
    category: str,
    title: str,
    body: str,
    segments: list[TranscriptSegment],
    evidence_hint: str,
    *,
    priority: str = "normal",
) -> SidecarCard:
    source_ids = source_ids_for_segments(segments)
    evidence_quote = evidence_quote_for(segments, evidence_hint)
    created_at = segments[0].start_s if segments else 0.0
    return create_sidecar_card(
        session_id=session_id,
        category=category,
        title=title,
        body=body,
        why_now="Offline replay candidate generated from transcript evidence.",
        priority=priority,
        confidence=0.78,
        source_segment_ids=source_ids,
        source_type="transcript",
        evidence_quote=evidence_quote,
        card_key=f"eval:{category}:{title.lower().replace(' ', '-')}",
        created_at=created_at,
    )


def select_segments(segments: list[TranscriptSegment], terms: list[str]) -> list[TranscriptSegment]:
    selected = [segment for segment in segments if any(term in segment.text.lower() for term in terms)]
    return selected[:3] or segments[:1]


def evidence_quote_for(segments: list[TranscriptSegment], evidence_hint: str) -> str:
    for segment in segments:
        if evidence_hint.lower() in segment.text.lower():
            return compact_text(segment.text, limit=260)
    return compact_text(" ".join(segment.text for segment in segments[:2]), limit=260)


def source_ids_for_segments(segments: list[TranscriptSegment]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for segment in segments:
        for source_id in segment.source_segment_ids or [segment.id]:
            if source_id in seen:
                continue
            seen.add(source_id)
            result.append(source_id)
    return result


def dedupe_cards(cards: list[SidecarCard]) -> list[SidecarCard]:
    seen: set[str] = set()
    result: list[SidecarCard] = []
    for card in cards:
        key = card.card_key or f"{card.category}:{card.title}"
        if key in seen:
            continue
        seen.add(key)
        result.append(card)
    return result


def apply_quality_gate(
    cards: list[SidecarCard],
    evidence_segments: list[TranscriptSegment],
) -> tuple[list[SidecarCard], list[EvalSuppressedCard]]:
    gate = NoteQualityGate()
    accepted: list[SidecarCard] = []
    suppressed: list[EvalSuppressedCard] = []
    status = {"ready": False, "enrollment_status": "not_enrolled"}
    for index, card in enumerate(cards):
        now = simulated_now_for(card, evidence_segments, index)
        decision = gate.evaluate(card, evidence_segments, status, now=now)
        if decision.action == "accept":
            if decision.priority_override:
                card = replace(card, priority=decision.priority_override)
            accepted.append(card)
            gate.remember_accepted(card, evidence_segments, now=now)
        else:
            suppressed.append(
                EvalSuppressedCard(
                    card=card,
                    reason=decision.reason,
                    normalized_fingerprint=decision.normalized_fingerprint,
                )
            )
    return accepted, suppressed


def simulated_now_for(card: SidecarCard, evidence_segments: list[TranscriptSegment], index: int) -> float:
    if card.created_at is not None and card.created_at < 1_000_000:
        return float(card.created_at)
    by_id: dict[str, TranscriptSegment] = {}
    for segment in evidence_segments:
        by_id[segment.id] = segment
        for source_id in segment.source_segment_ids:
            by_id[source_id] = segment
    starts = [by_id[source_id].start_s for source_id in card.source_segment_ids if source_id in by_id]
    if starts:
        return min(starts)
    return float(index)


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay Brain Sidecar offline quality fixtures.")
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--mode", choices=["pipeline", "gate-only"], default="pipeline")
    parser.add_argument("--noise", choices=["none", "default"], default="none")
    parser.add_argument("--no-memory-distractors", action="store_true")
    parser.add_argument("--fail-on-threshold", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    result, memory_events = replay_fixture(
        args.fixture,
        mode=args.mode,
        noise=args.noise,
        include_memory_distractors=not args.no_memory_distractors,
    )
    report = build_report(result, memory_events)
    write_json_report(report, args.output)
    if args.markdown:
        write_markdown_report(report, args.markdown)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    if args.fail_on_threshold and not report.report_passed:
        return 2
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
