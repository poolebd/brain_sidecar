from __future__ import annotations

from brain_sidecar.core.asr_aliases import alias_supported, extract_supported_aliases
from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.note_quality import NoteQualityGate
from brain_sidecar.core.sidecar_cards import create_sidecar_card


def seg(segment_id: str, text: str, start: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(id=segment_id, session_id="ses_1", start_s=start, end_s=start + 2.0, text=text)


def gate(**kwargs) -> NoteQualityGate:
    return NoteQualityGate(**kwargs)


def status(*, ready: bool = False, enrolled: bool = False) -> dict:
    return {"ready": ready, "enrollment_status": "enrolled" if enrolled else "not_enrolled"}


def card(
    title: str,
    body: str,
    source_ids: list[str],
    evidence_quote: str,
    *,
    category: str = "question",
    priority: str = "normal",
    suggested_ask: str | None = None,
) -> object:
    return create_sidecar_card(
        session_id="ses_1",
        category=category,
        title=title,
        body=body,
        suggested_ask=suggested_ask,
        why_now="Recent transcript evidence supports this.",
        priority=priority,
        confidence=0.78,
        source_segment_ids=source_ids,
        source_type="transcript",
        evidence_quote=evidence_quote,
    )


def accepted(decision) -> bool:
    return decision.action == "accept"


def test_suppresses_empty_body() -> None:
    evidence = [seg("seg_1", "Review the Siemens agreement tomorrow.")]
    item = card("Siemens review", "", ["seg_1"], "Review the Siemens agreement tomorrow.")

    assert gate().evaluate(item, evidence, status()).action == "suppress"


def test_suppresses_missing_source_segment_ids() -> None:
    evidence = [seg("seg_1", "Review the Siemens agreement tomorrow.")]
    item = card("Siemens review", "Review the Siemens agreement.", [], "Review the Siemens agreement tomorrow.")

    assert gate().evaluate(item, evidence, status()).reason == "missing_source_segment_ids"


def test_requires_multi_segment_evidence_for_normal_note() -> None:
    evidence = [seg("seg_1", "The Siemens agreement has context.")]
    item = card("Siemens context", "Siemens agreement context may matter.", ["seg_1"], "The Siemens agreement has context.")

    assert gate().evaluate(item, evidence, status()).reason == "insufficient_evidence_segments"


def test_allows_one_segment_explicit_action_with_due_date() -> None:
    evidence = [seg("seg_1", "Please review the Siemens agreement tomorrow.")]
    item = card(
        "Siemens review",
        "Review the Siemens agreement tomorrow.",
        ["seg_1"],
        "Please review the Siemens agreement tomorrow.",
        category="action",
        priority="high",
    )

    assert accepted(gate().evaluate(item, evidence, status()))


def test_suppresses_financial_hallucination_from_contribute_phrase() -> None:
    evidence = [seg("seg_1", "Brandon is going to contribute on this continuous basis."), seg("seg_2", "The project needs support.")]
    item = card("Brandon owes contribution", "Brandon owes a payable contribution.", ["seg_1", "seg_2"], "Brandon is going to contribute")

    assert gate().evaluate(item, evidence, status()).action == "suppress"


def test_suppresses_tax_owed_payables_cards_without_finance_evidence() -> None:
    evidence = [seg("seg_1", "Brandon Poole here we go."), seg("seg_2", "The project needs support.")]
    item = card("Tax contribution clarification", "Clarify the owed tax payable.", ["seg_1", "seg_2"], "Brandon Poole here we go")

    assert gate().evaluate(item, evidence, status()).action == "suppress"


def test_suppresses_project_alpha_without_evidence() -> None:
    evidence = [seg("seg_1", "We need to review Siemens."), seg("seg_2", "The client has a question.")]
    item = card("Project Alpha timeline", "Analyze Project Alpha milestones.", ["seg_1", "seg_2"], "review Siemens")

    assert gate().evaluate(item, evidence, status()).action == "suppress"


def test_suppresses_remote_work_policy_without_evidence() -> None:
    evidence = [seg("seg_1", "We need to review Siemens."), seg("seg_2", "The client has a question.")]
    item = card("Company policy", "Discuss remote work company policy.", ["seg_1", "seg_2"], "client has a question")

    assert gate().evaluate(item, evidence, status()).action == "suppress"


def test_suppresses_grass_maintenance_without_actual_maintenance_context() -> None:
    evidence = [seg("seg_1", "Grass or something on the back end."), seg("seg_2", "I am trying to say this.")]
    item = card("Grass maintenance", "Decide grass maintenance responsibilities.", ["seg_1", "seg_2"], "Grass or something")

    assert gate().evaluate(item, evidence, status()).action == "suppress"


def test_suppresses_ct_pt_trip_path_from_memory_when_live_evidence_missing() -> None:
    evidence = [seg("seg_1", "The harmonics study needs the SLD."), seg("seg_2", "Siemens and the client need alignment.")]
    item = card("CT/PT trip path", "Review CT/PT relay settings and breaker failure trip path.", ["seg_1", "seg_2"], "harmonics study needs the SLD")

    assert gate().evaluate(item, evidence, status()).action == "suppress"


def test_accepts_siemens_power_quality_review_card() -> None:
    evidence = [
        seg("seg_1", "We need to review the Siemens client agreement."),
        seg("seg_2", "The Arup power quality question needs an answer."),
    ]
    item = card(
        "Siemens power quality review",
        "Review the Siemens client agreement and power quality question.",
        ["seg_1", "seg_2"],
        "review the Siemens client agreement",
        category="action",
        priority="high",
    )

    assert accepted(gate().evaluate(item, evidence, status()))


def test_accepts_harmonics_sld_alignment_card() -> None:
    evidence = [
        seg("seg_1", "The harmonics and other studies need to line up."),
        seg("seg_2", "Review the S L D before answering Siemens."),
    ]
    item = card("Harmonics SLD alignment", "Align harmonics and studies with the SLD.", ["seg_1", "seg_2"], "harmonics and other studies")

    assert accepted(gate().evaluate(item, evidence, status()))


def test_accepts_sunil_reviewer_card() -> None:
    evidence = [
        seg("seg_1", "Sunil is the reviewer for this."),
        seg("seg_2", "If Sunil is unavailable use the client communication path."),
    ]
    item = card("Sunil reviewer path", "Use Sunil as reviewer or clarify the client communication path.", ["seg_1", "seg_2"], "Sunil is the reviewer")

    assert accepted(gate().evaluate(item, evidence, status()))


def test_accepts_manish_sme_card_with_uncertainty_if_narc_nerd_asr() -> None:
    evidence = [
        seg("seg_1", "Manish is the SME on the NARC or nerd side."),
        seg("seg_2", "He can clarify the SLC technical expertise."),
    ]
    item = card("Manish SME", "Manish may be the SME for the NERC/SLC technical area.", ["seg_1", "seg_2"], "Manish is the SME")

    assert accepted(gate().evaluate(item, evidence, status()))


def test_accepts_hours_confirmation_card() -> None:
    evidence = [
        seg("seg_1", "Ask Roberto or Dan about six to eight hours each week."),
        seg("seg_2", "That is roughly one day for this project over the next few weeks."),
    ]
    item = card("Confirm project hours", "Confirm six to eight hours weekly for the project.", ["seg_1", "seg_2"], "six to eight hours")

    assert accepted(gate().evaluate(item, evidence, status()))


def test_identity_specific_bp_card_requires_literal_name_or_enrolled_identity() -> None:
    evidence = [seg("seg_1", "I will review the Siemens agreement tomorrow.")]
    item = card(
        "BP review",
        "BP needs to review the Siemens agreement tomorrow.",
        ["seg_1"],
        "I will review the Siemens agreement tomorrow.",
        category="action",
        priority="high",
    )

    assert gate().evaluate(item, evidence, status()).reason == "identity_claim_not_supported"
    assert accepted(gate().evaluate(item, evidence, status(ready=True, enrolled=True)))


def test_repeated_owner_unclear_is_rate_limited() -> None:
    quality_gate = gate()
    evidence = [seg("seg_1", "I can send it after the meeting.")]
    first = card("Owner unclear", "Confirm owner for the follow-up.", ["seg_1"], "I can send it after the meeting.", category="clarification")
    decision = quality_gate.evaluate(first, evidence, status(), now=10.0)
    assert accepted(decision)
    quality_gate.remember_accepted(first, evidence, now=10.0)

    second = card("Owner unclear", "Confirm owner for the follow-up.", ["seg_1"], "I can send it after the meeting.", category="clarification")
    assert quality_gate.evaluate(second, evidence, status(), now=60.0).action == "suppress"


def test_volume_cap_limits_cards_per_5min() -> None:
    quality_gate = gate(max_cards_per_5min=2)
    evidence = [
        seg("seg_1", "Review the Siemens agreement tomorrow."),
        seg("seg_2", "Confirm the power quality answer tomorrow."),
    ]
    for index in range(2):
        accepted_card = card(f"Siemens review {index}", "Review the Siemens agreement.", ["seg_1", "seg_2"], "Review the Siemens agreement")
        decision = quality_gate.evaluate(accepted_card, evidence, status(), now=float(index))
        assert accepted(decision)
        quality_gate.remember_accepted(accepted_card, evidence, now=float(index))

    extra = card("Siemens review 3", "Review the Siemens agreement.", ["seg_1", "seg_2"], "Review the Siemens agreement")
    assert quality_gate.evaluate(extra, evidence, status(), now=20.0).reason == "rolling_volume_cap"


def test_asr_aliases_do_not_generate_facts() -> None:
    evidence = "The demons client agreement needs power quality review."

    assert "siemens" in extract_supported_aliases(evidence)
    assert alias_supported("Siemens", evidence)

    unrelated = [seg("seg_1", evidence), seg("seg_2", "The client question needs review.")]
    unsupported = card("Siemens relay settings", "Review Siemens relay settings.", ["seg_1", "seg_2"], "demons client agreement")

    assert gate().evaluate(unsupported, unrelated, status()).action == "suppress"


def test_company_reference_context_does_not_bypass_current_meeting_gate() -> None:
    evidence = [seg("seg_1", "We need Siemens to answer the power quality question.")]
    reference = create_sidecar_card(
        session_id="ses_1",
        category="reference",
        title="Siemens",
        body="Industrial technology company with grid-related businesses.",
        why_now="Local reference context only.",
        priority="low",
        confidence=0.9,
        source_segment_ids=["seg_1"],
        source_type="company_ref",
        evidence_quote="We need Siemens to answer the power quality question.",
    )
    unsupported = card(
        "Siemens relay settings",
        "Review Siemens relay settings.",
        ["seg_1"],
        "We need Siemens to answer",
        category="action",
        priority="high",
    )

    assert gate().evaluate(reference, evidence, status()).reason == "generated_note_must_use_live_transcript_evidence"
    assert gate().evaluate(unsupported, evidence, status()).action == "suppress"
