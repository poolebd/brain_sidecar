from __future__ import annotations

import time

from brain_sidecar.core.models import SidecarCard, new_id
from brain_sidecar.core.sidecar_cards import create_sidecar_card, note_payload_to_sidecar_card


def test_valid_sidecar_card_serialization_clamps_contract() -> None:
    card = create_sidecar_card(
        session_id="ses_1",
        category="action",
        title="Rollback owner",
        body="Confirm who owns rollback.",
        suggested_say="I can confirm the rollback owner.",
        why_now="The current thread lacks an owner.",
        priority="high",
        confidence=1.5,
        source_segment_ids=["seg_1"],
        source_type="transcript",
        ephemeral=True,
    )

    payload = card.to_dict()

    assert payload["category"] == "action"
    assert payload["priority"] == "high"
    assert payload["source_type"] == "transcript"
    assert payload["confidence"] == 1.0
    assert payload["raw_audio_retained"] is False
    assert payload["expires_at"] > time.time()


def test_invalid_sidecar_values_normalize_deterministically() -> None:
    card = SidecarCard(
        id=new_id("card"),
        session_id="ses_1",
        category="nonsense",
        title="Bad values",
        body="Still useful enough.",
        why_now="Testing fallback normalization.",
        priority="urgent",
        confidence=-4,
        source_segment_ids=["seg_1"],
        source_type="unknown",
    )

    assert card.category == "note"
    assert card.priority == "normal"
    assert card.source_type == "transcript"
    assert card.confidence == 0.0
    assert card.raw_audio_retained is False


def test_note_payload_card_key_is_stable_for_legacy_dedupe() -> None:
    card = note_payload_to_sidecar_card(
        "ses_1",
        {
            "kind": "question",
            "title": "Clarify rollback",
            "body": "Ask when rollback cutoff happens.",
            "source_segment_ids": ["seg_1"],
        },
        save_transcript=False,
    )

    assert card.card_key == "note:question:clarify rollback"
    assert card.ephemeral is True
