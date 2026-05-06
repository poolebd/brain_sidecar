from __future__ import annotations

from brain_sidecar.core.meeting_contract import contract_prompt_block, default_meeting_contract, normalize_meeting_contract
from brain_sidecar.server.app import StartSessionRequest, _meeting_contract_payload


def test_meeting_contract_defaults_and_prompt_block() -> None:
    contract = default_meeting_contract()

    assert contract.mode == "quiet"
    assert contract.goal == "Offload meeting obligations, questions, risks, follow-ups, useful context, and post-call synthesis for BP."
    assert len(contract.reminders) == 6
    assert any("Prefer silence" in reminder for reminder in contract.reminders)
    assert any("memory" in reminder.lower() and "reminder context" in reminder.lower() for reminder in contract.reminders)

    block = contract_prompt_block(contract)
    assert "Prefer silence" in block
    assert "Memory is reminder context only" in block


def test_meeting_contract_sanitizes_input() -> None:
    contract = normalize_meeting_contract(
        {
            "goal": "  " + ("Track follow-ups. " * 80),
            "mode": "loud",
            "reminders": [" Owners ", "", "Owners", *[f"item {index}" for index in range(20)]],
        }
    )

    assert contract.mode == "quiet"
    assert contract.goal.startswith("Track follow-ups.")
    assert len(contract.goal) <= 420
    assert contract.reminders == ["Owners", "item 0", "item 1", "item 2", "item 3", "item 4", "item 5", "item 6"]


def test_start_session_request_accepts_and_normalizes_meeting_contract() -> None:
    request = StartSessionRequest(
        meeting_contract={
            "goal": "  Track owners and close the brief. ",
            "mode": "balanced",
            "reminders": ["Owners", "Post-call brief", "", "Owners"],
        }
    )

    assert _meeting_contract_payload(request.meeting_contract) == {
        "goal": "Track owners and close the brief.",
        "mode": "balanced",
        "reminders": ["Owners", "Post-call brief"],
    }
