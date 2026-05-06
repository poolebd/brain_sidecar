from __future__ import annotations

from brain_sidecar.core.meeting_agents import ContractCriticAgent, MemoryBoundaryAgent
from brain_sidecar.core.models import SearchHit, TranscriptSegment
from brain_sidecar.core.note_quality import NoteQualityGate
from brain_sidecar.core.sidecar_cards import create_sidecar_card


def seg(segment_id: str, text: str, *, speaker_role: str | None = None, confidence: float | None = None) -> TranscriptSegment:
    return TranscriptSegment(
        id=segment_id,
        session_id="ses_1",
        start_s=0.0,
        end_s=2.0,
        text=text,
        speaker_role=speaker_role,
        speaker_confidence=confidence,
    )


def test_memory_only_recall_cannot_become_current_meeting_action() -> None:
    segment = seg("seg_1", "We need to confirm the platform owner.")
    recall = [SearchHit(source_type="session", source_id="old", text="Send the tax payable report tomorrow.", score=0.91, metadata={})]
    card = create_sidecar_card(
        session_id="ses_1",
        category="action",
        title="Tax payable report",
        body="Send the tax payable report tomorrow.",
        why_now="Recall mentioned it.",
        priority="high",
        confidence=0.9,
        source_segment_ids=["seg_1"],
        source_type="transcript",
        evidence_quote="We need to confirm the platform owner.",
    )

    assert MemoryBoundaryAgent().filter_current_meeting_cards([card], [segment], recall) == []


def test_contract_critic_suppresses_missing_quote_and_invalid_source_id() -> None:
    evidence = [seg("seg_1", "Please review the Siemens agreement tomorrow.")]
    cards = [
        create_sidecar_card(
            session_id="ses_1",
            category="action",
            title="Missing quote",
            body="Review the Siemens agreement tomorrow.",
            why_now="Recent transcript evidence supports this.",
            priority="high",
            confidence=0.8,
            source_segment_ids=["seg_1"],
            source_type="transcript",
            evidence_quote="",
        ),
        create_sidecar_card(
            session_id="ses_1",
            category="action",
            title="Invalid source",
            body="Review the Siemens agreement tomorrow.",
            why_now="Recent transcript evidence supports this.",
            priority="high",
            confidence=0.8,
            source_segment_ids=["missing"],
            source_type="transcript",
            evidence_quote="Please review the Siemens agreement tomorrow.",
        ),
    ]

    result = ContractCriticAgent(NoteQualityGate()).review(
        cards,
        evidence,
        {"ready": False, "enrollment_status": "not_enrolled"},
        max_cards=3,
    )

    assert result.accepted_cards == []
    assert result.diagnostics.generated_candidate_count == 2
    assert result.diagnostics.suppressed_count == 2
    reasons = {item["reason"] for item in result.diagnostics.top_suppression_reasons}
    assert "missing_evidence_quote" in reasons
    assert "source_segment_ids_not_in_recent_evidence" in reasons
