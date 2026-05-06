from __future__ import annotations

from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.note_quality import NoteQualityGate
from brain_sidecar.core.notes import NoteSynthesizer, heuristic_project_review_cards


class FailingOllama:
    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        raise RuntimeError("timed out")


class RecordingOllama:
    def __init__(self) -> None:
        self.system = ""
        self.user = ""

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        self.system = system
        self.user = user
        assert format_json is True
        return """
        ```json
        {"cards":[{
          "category":"clarification",
          "title":"Handoff owner",
          "body":"The handoff risk needs a clearer owner.",
          "suggested_ask":"Who owns the handoff risk?",
          "why_now":"The current thread mentions handoff and operating risk.",
          "priority":"high",
          "confidence":0.74,
          "source_segment_ids":["seg_1"],
          "source_type":"transcript",
          "card_key":"clarify:handoff-owner"
        }]}
        ```
        """


def test_note_synthesizer_preserves_source_ids_and_prompts_for_sidecar_tone(event_loop) -> None:
    ollama = RecordingOllama()
    synthesizer = NoteSynthesizer(ollama)  # type: ignore[arg-type]
    segments = [
        TranscriptSegment(
            id="seg_1",
            session_id="ses_1",
            start_s=0.0,
            end_s=2.0,
            text="We are discussing a project handoff and the next operating risks.",
        )
    ]

    result = event_loop.run_until_complete(synthesizer.synthesize("ses_1", segments, []))

    assert any(note.source_segment_ids == ["seg_1"] for note in result.notes)
    assert result.sidecar_cards[0].suggested_ask == "Who owns the handoff risk?"
    assert result.sidecar_cards[0].why_now == "The current thread mentions handoff and operating risk."
    assert "meeting-intelligence assistant" in ollama.system
    assert "suggested_say" in ollama.user
    assert "decision" in ollama.user


def test_note_synthesizer_falls_back_when_llm_times_out(event_loop) -> None:
    synthesizer = NoteSynthesizer(FailingOllama())  # type: ignore[arg-type]
    segments = [
        TranscriptSegment(
            id="seg_1",
            session_id="ses_1",
            start_s=0.0,
            end_s=2.0,
            text="We are discussing a project handoff and the next operating risks.",
        )
    ]

    result = event_loop.run_until_complete(synthesizer.synthesize("ses_1", segments, []))

    assert len(result.notes) >= 1
    assert result.notes[0].source_segment_ids == ["seg_1"]
    assert result.sidecar_cards[0].category in {"risk", "status"}
    assert "project handoff" in result.notes[0].body or result.sidecar_cards[0].category == "risk"


def test_note_synthesizer_extracts_bp_owned_action_when_speaker_is_confident(event_loop) -> None:
    synthesizer = NoteSynthesizer(FailingOllama())  # type: ignore[arg-type]
    segments = [
        TranscriptSegment(
            id="seg_bp",
            session_id="ses_1",
            start_s=0.0,
            end_s=2.0,
            text="I'll send the rollback checklist after the meeting.",
            speaker_role="user",
            speaker_label="BP",
            speaker_confidence=0.93,
        )
    ]

    result = event_loop.run_until_complete(synthesizer.synthesize("ses_1", segments, []))

    assert result.sidecar_cards[0].category == "action"
    assert "BP appears to own" in result.sidecar_cards[0].body
    assert result.sidecar_cards[0].suggested_say


def test_note_synthesizer_does_not_assign_other_speaker_commitment_to_bp(event_loop) -> None:
    synthesizer = NoteSynthesizer(FailingOllama())  # type: ignore[arg-type]
    segments = [
        TranscriptSegment(
            id="seg_other",
            session_id="ses_1",
            start_s=0.0,
            end_s=2.0,
            text="I'll send the rollback checklist after the meeting.",
            speaker_role="other",
            speaker_label="Speaker 2",
            speaker_confidence=0.0,
        )
    ]

    result = event_loop.run_until_complete(synthesizer.synthesize("ses_1", segments, []))

    assert result.sidecar_cards[0].category == "clarification"
    assert "do not assign this to BP" in result.sidecar_cards[0].body
    assert result.sidecar_cards[0].suggested_ask == "Can we confirm who owns that follow-up?"


def test_note_synthesizer_extracts_project_review_cards_when_llm_times_out(event_loop) -> None:
    synthesizer = NoteSynthesizer(FailingOllama())  # type: ignore[arg-type]
    segments = [
        TranscriptSegment(
            id="seg_siemens",
            session_id="ses_1",
            start_s=0.0,
            end_s=3.0,
            text="There is a Siemens document which is under review from Greg and Sunil.",
        ),
        TranscriptSegment(
            id="seg_rfi",
            session_id="ses_1",
            start_s=3.2,
            end_s=6.4,
            text="Create an RFI log for the scope comments and deviations.",
        ),
        TranscriptSegment(
            id="seg_monday",
            session_id="ses_1",
            start_s=6.6,
            end_s=9.5,
            text="Monday is the target to send comments.",
        ),
    ]

    result = event_loop.run_until_complete(synthesizer.synthesize("ses_1", segments, []))
    titles = {card.title for card in result.sidecar_cards}

    assert "Siemens document review" in titles
    assert "RFI log review" in titles
    assert "Send by Monday" in titles
    assert all(card.source_segment_ids for card in result.sidecar_cards)
    assert all(card.evidence_quote for card in result.sidecar_cards)


def test_project_review_heuristics_pass_quality_gate(event_loop) -> None:
    synthesizer = NoteSynthesizer(FailingOllama())  # type: ignore[arg-type]
    segments = [
        TranscriptSegment(
            id="seg_siemens",
            session_id="ses_1",
            start_s=0.0,
            end_s=3.0,
            text="There is a Siemens document which is under review from Greg and Sunil.",
        ),
        TranscriptSegment(
            id="seg_rfi",
            session_id="ses_1",
            start_s=3.2,
            end_s=6.4,
            text="Create an RFI log for the scope comments and deviations.",
        ),
        TranscriptSegment(
            id="seg_monday",
            session_id="ses_1",
            start_s=6.6,
            end_s=9.5,
            text="Monday is the target to send comments.",
        ),
    ]
    gate = NoteQualityGate()

    result = event_loop.run_until_complete(synthesizer.synthesize("ses_1", segments, []))
    decisions = [gate.evaluate(card, segments, {"ready": False, "enrollment_status": "not_enrolled"}, now=card.created_at) for card in result.sidecar_cards]

    assert any(card.title == "Siemens document review" and decision.action == "accept" for card, decision in zip(result.sidecar_cards, decisions))
    assert any(card.title == "RFI log review" and decision.action == "accept" for card, decision in zip(result.sidecar_cards, decisions))
    assert any(card.title == "Send by Monday" and decision.action == "accept" for card, decision in zip(result.sidecar_cards, decisions))


def test_review_path_heuristic_uses_stable_title_for_supported_reviewer_evidence() -> None:
    segments = [
        TranscriptSegment(
            id="seg_greg",
            session_id="ses_1",
            start_s=0.0,
            end_s=2.0,
            text="The Siemens document is under review from Greg.",
        ),
        TranscriptSegment(
            id="seg_sunil",
            session_id="ses_1",
            start_s=2.1,
            end_s=4.0,
            text="Sunil is the focal point for that review.",
        ),
    ]

    cards = heuristic_project_review_cards("ses_1", segments)
    review_path_card = next(card for card in cards if card.title == "Review path")

    assert "under review" in review_path_card.evidence_quote
    assert not any(" / " in card.title and "review path" in card.title.lower() for card in cards)


def test_review_path_heuristic_does_not_treat_drive_as_review_signal() -> None:
    segments = [
        TranscriptSegment(
            id="seg_names",
            session_id="ses_1",
            start_s=0.0,
            end_s=2.0,
            text="Kyle and Greg understand the skills involved.",
        ),
        TranscriptSegment(
            id="seg_drive",
            session_id="ses_1",
            start_s=2.1,
            end_s=4.0,
            text="Can you drive that, Greg?",
        ),
    ]

    cards = heuristic_project_review_cards("ses_1", segments)

    assert not any(card.title == "Review path" for card in cards)
