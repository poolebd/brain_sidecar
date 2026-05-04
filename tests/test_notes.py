from __future__ import annotations

from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.notes import NoteSynthesizer


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
        return '{"notes":[{"kind":"context","title":"Handoff","body":"This sounds like a handoff risk to keep nearby."}]}'


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

    assert len(result.notes) == 1
    assert result.notes[0].source_segment_ids == ["seg_1"]
    assert result.notes[0].body == "This sounds like a handoff risk to keep nearby."
    assert "natural sidecar replies" in ollama.system
    assert "conversational" in ollama.user
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

    assert len(result.notes) == 1
    assert result.notes[0].kind == "context"
    assert result.notes[0].title == "Current thread"
    assert result.notes[0].source_segment_ids == ["seg_1"]
    assert result.notes[0].body.startswith("I'm hearing:")
    assert "project handoff" in result.notes[0].body
