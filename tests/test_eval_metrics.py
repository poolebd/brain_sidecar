from __future__ import annotations

from pathlib import Path

from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.sidecar_cards import create_sidecar_card
from brain_sidecar.eval.metrics import build_report
from brain_sidecar.eval.models import (
    EvalExpectedBehavior,
    EvalExpectedTopic,
    EvalMemoryEvent,
    EvalReplayResult,
    EvalSuppressedCard,
)
from brain_sidecar.eval.replay import replay_fixture


def seg(segment_id: str, text: str) -> TranscriptSegment:
    return TranscriptSegment(id=segment_id, session_id="eval", start_s=0.0, end_s=3.0, text=text)


def card(title: str, body: str, *, evidence_quote: str = "Please review this.", source_ids: list[str] | None = None):
    return create_sidecar_card(
        session_id="eval",
        category="action",
        title=title,
        body=body,
        why_now="test",
        priority="normal",
        confidence=0.8,
        source_segment_ids=source_ids or ["seg_1"],
        source_type="transcript",
        evidence_quote=evidence_quote,
    )


def result_with(cards) -> EvalReplayResult:
    evidence = [seg("seg_1", "Please review this.")]
    return EvalReplayResult(
        session_id="eval",
        input_segments=evidence,
        consolidated_segments=evidence,
        collapsed_segment_count=0,
        generated_cards=list(cards),
        accepted_cards=list(cards),
        suppressed_cards=[],
        memory_cards=[],
        expectation=None,
    )


def test_report_flags_unsupported_topics() -> None:
    report = build_report(result_with([card("Tax payable", "Clarify the owed tax payable.")]))

    assert report.unsupported_topic_violations
    assert report.report_passed is False


def test_report_flags_memory_leakage() -> None:
    memory = EvalMemoryEvent(
        id="mem_1",
        title="Relay memory",
        body="Past relay settings.",
        forbidden_in_current_notes=["relay settings"],
    )
    report = build_report(result_with([card("Relay settings", "Review relay settings.")]), [memory])

    assert report.memory_leakage_violations
    assert report.report_passed is False


def test_report_requires_evidence_quotes() -> None:
    report = build_report(result_with([card("Review", "Review this.", evidence_quote="")]))

    assert report.percent_cards_with_evidence_quote == 0.0
    assert report.report_passed is False


def test_report_counts_expected_topic_hits_and_misses() -> None:
    replay_result = result_with([card("Harmonics review", "Review harmonics and SLD alignment.")])
    replay_result = EvalReplayResult(
        **{
            **replay_result.__dict__,
            "expectation": EvalExpectedBehavior(
                session_id="eval",
                should_accept_any=[
                    EvalExpectedTopic("harmonics_alignment", ["harmonics", "SLD"]),
                    EvalExpectedTopic("hours_confirmation", ["six to eight", "hours"]),
                ],
            ),
        }
    )

    report = build_report(replay_result)

    assert report.expected_topic_hits == ["harmonics_alignment"]
    assert report.expected_topic_misses == ["hours_confirmation"]


def test_report_passes_clean_energy_fixture() -> None:
    replay_result, memory_events = replay_fixture(Path("tests/fixtures/eval/energy_consulting_seed.jsonl"))
    report = build_report(replay_result, memory_events)

    assert report.report_passed is True
    assert 4 <= report.accepted_card_count <= 12
    assert report.percent_cards_with_source_ids == 1.0
    assert report.percent_cards_with_evidence_quote == 1.0
