from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from brain_sidecar.config import Settings
from brain_sidecar.core.audio import AudioCapture
from brain_sidecar.core.domain_keywords import EnergyConversationDetector, inactive_energy_frame
from brain_sidecar.core.energy_lens import EnergyConsultingAgent
from brain_sidecar.core.events import EVENT_TRANSCRIPT_PARTIAL
from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.note_quality import NoteQualityGate
from brain_sidecar.core.notes import NoteSynthesizer, energy_prompt_block
from brain_sidecar.core.session import ActiveSession, InMemoryPcmRingBuffer, SessionManager
from brain_sidecar.core.dedupe import TranscriptDeduplicator
from brain_sidecar.core.asr import ASR_BACKEND_NEMOTRON_STREAMING, StreamingAsrEvent
from brain_sidecar.server.app import create_app


class DummyCapture(AudioCapture):
    async def chunks(self):
        if False:
            yield b""

    async def stop(self) -> None:
        return None


class RecordingOllama:
    def __init__(self) -> None:
        self.user = ""

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        del system
        self.user = user
        assert format_json is True
        return '{"cards":[]}'


def settings(tmp_path, *, save_backend: str = "nemotron_streaming") -> Settings:
    return Settings(
        data_dir=tmp_path,
        host="127.0.0.1",
        port=8765,
        asr_primary_model="tiny.en",
        asr_fallback_model="tiny.en",
        asr_compute_type="float16",
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="phi3:mini",
        ollama_embed_model="embeddinggemma",
        asr_backend=save_backend,
        disable_live_embeddings=True,
    )


def seg(segment_id: str, text: str, *, final: bool = True, start: float = 0.0) -> TranscriptSegment:
    return TranscriptSegment(
        id=segment_id,
        session_id="ses_1",
        start_s=start,
        end_s=start + 2.0,
        text=text,
        is_final=final,
    )


def active_session(session_id: str, *, save_transcript: bool) -> ActiveSession:
    return ActiveSession(
        id=session_id,
        capture=DummyCapture(),
        window_queue=asyncio.Queue(maxsize=4),
        postprocess_queue=asyncio.Queue(maxsize=4),
        tasks=[],
        deduper=TranscriptDeduplicator(max_recent=4, similarity_threshold=0.88),
        save_transcript=save_transcript,
        asr_backend=ASR_BACKEND_NEMOTRON_STREAMING,
        asr_model="fake-nemotron",
        streaming_chunk_ms=160,
        pcm_ring_buffer=InMemoryPcmRingBuffer(sample_rate=16_000, seconds=5.0),
    )


def test_detector_activates_tariff_and_bill_language() -> None:
    frame = EnergyConversationDetector().detect([
        seg("seg_1", "We need utility bill analysis and tariff analysis before pricing this site."),
    ])

    assert frame.active is True
    assert frame.confidence == "high"
    assert {item["category"] for item in frame.top_categories} >= {"operations", "market"}
    assert frame.evidence_segment_ids == ["seg_1"]
    assert "utility bill analysis" in frame.evidence_quote


def test_detector_activates_sustainability_and_market_language() -> None:
    frame = EnergyConversationDetector().detect([
        seg("seg_1", "Scope 2 emissions and renewable energy certificate treatment are the issue."),
    ])

    assert frame.active is True
    assert {item["category"] for item in frame.top_categories} >= {"sustainability", "market"}


def test_detector_activates_commercial_service_scoping() -> None:
    frame = EnergyConversationDetector().detect([
        seg("seg_1", "The RFP, proposal, and scope of work are for an energy audit."),
    ])

    assert frame.active is True
    assert {item["category"] for item in frame.top_categories} >= {"commercial", "service"}


def test_detector_guards_ambiguous_acronyms_until_context_arrives() -> None:
    detector = EnergyConversationDetector()

    assert detector.detect([seg("seg_1", "PPA")]).active is False
    assert detector.detect([seg("seg_1", "ROI")]).active is False

    ppa = detector.detect([seg("seg_1", "PPA plus renewable energy procurement")])
    roi = detector.detect([seg("seg_1", "ROI for energy conservation measures")])

    assert ppa.active is True
    assert {item["category"] for item in ppa.top_categories} >= {"market"}
    assert roi.active is True
    assert {item["category"] for item in roi.top_categories} >= {"finance", "technical"}


def test_detector_uses_context_for_eui_and_ignores_partials() -> None:
    detector = EnergyConversationDetector()
    active = detector.detect([seg("seg_1", "EUI and energy benchmarking are the baseline.")])
    partial_only = detector.detect([seg("partial_1", "utility bill analysis", final=False)])

    assert active.active is True
    assert {item["category"] for item in active.top_categories} >= {"operations"}
    assert partial_only.active is False


def test_detector_inactive_without_energy_terms() -> None:
    frame = EnergyConversationDetector().detect([seg("seg_1", "The project owner will send the agenda tomorrow.")])

    assert frame.active is False
    assert frame.evidence_quote == ""


def test_energy_agent_creates_grounded_cards_that_pass_quality_gate() -> None:
    segments = [seg("seg_1", "We need utility bill analysis and tariff analysis before pricing this site.")]
    frame = EnergyConversationDetector().detect(segments)
    cards = EnergyConsultingAgent().cards("ses_1", segments, frame)

    assert len(cards) == 1
    assert cards[0].suggested_ask
    assert cards[0].evidence_quote
    decision = NoteQualityGate().evaluate(cards[0], segments, {"ready": False})
    assert decision.action == "accept"


def test_energy_agent_card_patterns_for_core_domains() -> None:
    cases = [
        ("ASHRAE Level 2 audit and energy benchmarking need a baseline.", "Assessment baseline"),
        ("Renewable energy procurement and PPA treatment are in scope.", "Procurement path"),
        ("Scope 2 emissions and market-based emissions factors are unclear.", "Scope 2 basis"),
        ("ROI for energy conservation measures and utility incentives matter.", "Decision metric"),
    ]
    agent = EnergyConsultingAgent()
    gate = NoteQualityGate()
    for index, (text, expected_title) in enumerate(cases):
        segments = [seg(f"seg_{index}", text)]
        frame = EnergyConversationDetector().detect(segments)
        cards = agent.cards("ses_1", segments, frame)
        assert cards, text
        assert cards[0].title == expected_title
        assert gate.evaluate(cards[0], segments, {"ready": False}).action == "accept"


def test_energy_agent_suppresses_finance_and_ppa_without_context() -> None:
    agent = EnergyConsultingAgent()

    assert agent.cards("ses_1", [seg("seg_1", "ROI")], inactive_energy_frame()) == []
    assert agent.cards("ses_1", [seg("seg_1", "PPA")], EnergyConversationDetector().detect([seg("seg_1", "PPA")])) == []


def test_energy_prompt_block_is_compact_and_guarded() -> None:
    inactive = energy_prompt_block(inactive_energy_frame())
    active = EnergyConversationDetector().detect([seg("seg_1", "Tariff analysis and utility bill analysis are needed.")])
    block = energy_prompt_block(active)

    assert inactive == ""
    assert "Energy consulting lens active." in block
    assert "tariff analysis" in block.lower()
    assert "Do not turn keyword matches into facts" in block


def test_note_synthesizer_includes_energy_prompt_only_when_active(event_loop) -> None:
    ollama = RecordingOllama()
    synthesizer = NoteSynthesizer(ollama)  # type: ignore[arg-type]
    segments = [seg("seg_1", "Tariff analysis and utility bill analysis are needed.")]
    frame = EnergyConversationDetector().detect(segments)

    event_loop.run_until_complete(synthesizer.synthesize("ses_1", segments, [], energy_frame=frame))
    assert "Energy consulting lens active." in ollama.user

    ollama = RecordingOllama()
    synthesizer = NoteSynthesizer(ollama)  # type: ignore[arg-type]
    event_loop.run_until_complete(synthesizer.synthesize("ses_1", [seg("seg_2", "Normal agenda item.")], []))
    assert "Energy consulting lens active." not in ollama.user


def test_session_final_updates_energy_frame_and_partial_does_not(event_loop, tmp_path) -> None:
    manager = SessionManager(settings(tmp_path))
    session = manager.storage.create_session("energy")
    active = active_session(session.id, save_transcript=True)
    manager._active[session.id] = active

    event_loop.run_until_complete(
        manager._handle_streaming_asr_event(
            session.id,
            active,
            StreamingAsrEvent(kind="partial", text="utility bill analysis", start_s=0.0, end_s=0.5, model="fake"),
            asr_duration_ms=1.0,
        )
    )
    assert active.energy_conversation_frame is None

    event_loop.run_until_complete(
        manager._handle_streaming_asr_event(
            session.id,
            active,
            StreamingAsrEvent(
                kind="final",
                text="We need utility bill analysis and tariff analysis.",
                start_s=0.0,
                end_s=2.0,
                model="fake",
            ),
            asr_duration_ms=1.0,
        )
    )

    assert active.energy_conversation_frame is not None
    assert active.energy_conversation_frame.active is True
    status = manager._energy_status(active)
    assert status["energy_lens_active"] is True
    assert status["raw_audio_retained"] is False


def test_listen_only_does_not_persist_energy_artifacts(event_loop, tmp_path) -> None:
    manager = SessionManager(settings(tmp_path))
    session = manager.storage.create_session("listen-only")
    active = active_session(session.id, save_transcript=False)
    manager._active[session.id] = active

    event_loop.run_until_complete(
        manager._handle_streaming_asr_event(
            session.id,
            active,
            StreamingAsrEvent(
                kind="final",
                text="We need utility bill analysis and tariff analysis.",
                start_s=0.0,
                end_s=2.0,
                model="fake",
            ),
            asr_duration_ms=1.0,
        )
    )

    assert active.energy_conversation_frame is not None
    assert active.energy_conversation_frame.active is True
    assert manager.storage.recent_segments(session.id) == []


def test_gpu_health_exposes_compact_energy_status_only(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        response = client.get("/api/health/gpu")

    payload = response.json()
    assert response.status_code == 200
    assert payload["energy_lens_enabled"] is True
    assert payload["energy_lens_keyword_count"] > 20
    assert "energy_keywords" not in payload
    assert "lexicon" not in str(payload).lower()
