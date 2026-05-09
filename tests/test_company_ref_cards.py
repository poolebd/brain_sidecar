from __future__ import annotations

from brain_sidecar.core.company_refs import CompanyMention
from brain_sidecar.core.sidecar_cards import company_mention_to_sidecar_card


def mention() -> CompanyMention:
    return CompanyMention(
        ref_id="siemens",
        canonical_name="Siemens",
        entity_type="company",
        domain="industrial technology / electrical equipment",
        description="Industrial technology company with automation and grid businesses.",
        matched_text="Siemens",
        alias="Siemens",
        alias_type="name",
        confidence=0.94,
        source_segment_ids=["seg_1"],
        evidence_quote="We need Siemens to answer the power quality question.",
        sources=[{"title": "Seed reference", "url": "https://example.com/siemens"}],
        metadata={"seed": "test"},
    )


def test_company_mention_converts_to_reference_card() -> None:
    card = company_mention_to_sidecar_card("ses_1", mention())
    payload = card.to_dict()

    assert payload["category"] == "reference"
    assert payload["source_type"] == "company_ref"
    assert payload["title"] == "Siemens"
    assert payload["priority"] == "low"
    assert payload["confidence"] == 0.94
    assert payload["source_segment_ids"] == ["seg_1"]
    assert payload["evidence_quote"] == "We need Siemens to answer the power quality question."
    assert payload["card_key"] == "company_ref:siemens"
    assert payload["ephemeral"] is True
    assert payload["raw_audio_retained"] is False


def test_manual_company_reference_card_is_explicit_context() -> None:
    card = company_mention_to_sidecar_card("ses_1", mention(), explicitly_requested=True, priority="normal")

    assert card.category == "reference"
    assert card.source_type == "company_ref"
    assert card.priority == "normal"
    assert card.explicitly_requested is True
    assert card.raw_audio_retained is False
