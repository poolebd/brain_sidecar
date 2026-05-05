from __future__ import annotations

import json
from pathlib import Path

from brain_sidecar.eval.metrics import build_report
from brain_sidecar.eval.replay import replay_fixture, run_cli


FIXTURE = Path("tests/fixtures/eval/energy_consulting_seed.jsonl")


def test_gate_only_replay_suppresses_bad_cards() -> None:
    replay_result, _ = replay_fixture(FIXTURE, mode="gate-only")

    accepted_titles = {card.title for card in replay_result.accepted_cards}
    assert "Tax contribution clarification" not in accepted_titles
    assert "CT/PT trip path" not in accepted_titles
    assert "BP review" not in accepted_titles
    assert replay_result.suppressed_cards


def test_replay_energy_fixture_hits_expected_topics() -> None:
    replay_result, memory_events = replay_fixture(FIXTURE)
    report = build_report(replay_result, memory_events)

    assert report.report_passed is True
    assert set(report.expected_topic_hits) >= {
        "harmonics_alignment",
        "siemens_power_quality_review",
        "sunil_path",
        "manish_sme",
        "hours_confirmation",
        "schedule_clarification",
    }


def test_replay_with_memory_distractors_has_zero_current_note_leakage() -> None:
    replay_result, memory_events = replay_fixture(FIXTURE)
    report = build_report(replay_result, memory_events)

    assert report.memory_card_count == 2
    assert report.memory_leakage_violations == []


def test_replay_outputs_json_and_markdown_report(tmp_path: Path) -> None:
    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"

    assert run_cli(["--fixture", str(FIXTURE), "--output", str(json_path), "--markdown", str(md_path)]) == 0

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["report_passed"] is True
    assert "Accepted Cards" in md_path.read_text(encoding="utf-8")


def test_replay_does_not_require_external_services(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_OLLAMA_HOST", "http://127.0.0.1:9")
    json_path = tmp_path / "report.json"

    assert run_cli(["--fixture", str(FIXTURE), "--output", str(json_path)]) == 0
