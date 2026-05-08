from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain_sidecar.core.models import NoteCard, TranscriptSegment
from brain_sidecar.server.app import create_app


def test_session_browser_lists_details_and_updates_title(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        storage = client.app.state.manager.storage
        session = storage.create_session("Original title")
        storage.set_session_status(session.id, "stopped", ended_at=123.0, save_transcript=True)
        storage.add_transcript_segment(
            TranscriptSegment(
                id="seg_saved",
                session_id=session.id,
                start_s=0.0,
                end_s=2.0,
                text="We decided to keep Nemotron on the GPU.",
                speaker_label="BP",
            )
        )
        storage.add_note(
            NoteCard(
                id="note_saved",
                session_id=session.id,
                kind="decision",
                title="GPU residency",
                body="Keep Nemotron and Phi resident together.",
                source_segment_ids=["seg_saved"],
                evidence_quote="keep Nemotron on the GPU",
            )
        )
        storage.upsert_session_memory_summary(
            session_id=session.id,
            title="Summary",
            summary="GPU residency was the main decision.",
            topics=["gpu"],
            decisions=["Keep models resident."],
            actions=[],
            unresolved_questions=[],
            entities=["Nemotron"],
            lessons=[],
            source_segment_ids=["seg_saved"],
        )

        list_response = client.get("/api/sessions")
        detail_response = client.get(f"/api/sessions/{session.id}")
        patch_response = client.patch(f"/api/sessions/{session.id}", json={"title": "Renamed meeting"})

    assert list_response.status_code == 200
    listed = list_response.json()["sessions"][0]
    assert listed["id"] == session.id
    assert listed["transcript_count"] == 1
    assert listed["note_count"] == 1
    assert listed["summary_exists"] is True
    assert listed["retention"] == "saved"
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["transcript_segments"][0]["text"] == "We decided to keep Nemotron on the GPU."
    assert detail["note_cards"][0]["source_segment_ids"] == ["seg_saved"]
    assert detail["summary"]["summary"] == "GPU residency was the main decision."
    assert patch_response.status_code == 200
    assert patch_response.json()["title"] == "Renamed meeting"


def test_listen_only_session_detail_redacts_transcript_text(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        storage = client.app.state.manager.storage
        session = storage.create_session("Temporary")
        storage.set_session_status(session.id, "stopped", ended_at=123.0, save_transcript=False)
        storage.add_transcript_segment(
            TranscriptSegment(
                id="seg_bug",
                session_id=session.id,
                start_s=0.0,
                end_s=1.0,
                text="This should not be exposed.",
            )
        )

        response = client.get(f"/api/sessions/{session.id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["retention"] == "temporary"
    assert payload["transcript_redacted"] is True
    assert payload["transcript_segments"] == []
    assert payload["note_cards"] == []


def test_library_chunks_endpoint_lists_sources_and_filters(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        storage = client.app.state.manager.storage
        storage.upsert_document_chunk(tmp_path / "apollo.md", 0, "Apollo rollout note", {"title": "Apollo"})
        storage.upsert_document_chunk(tmp_path / "apollo.md", 1, "Nemotron timing note", {"title": "Apollo"})
        storage.upsert_document_chunk(tmp_path / "other.md", 0, "Unrelated note", {})

        all_response = client.get("/api/library/chunks")
        filtered_response = client.get("/api/library/chunks", params={"query": "nemotron"})
        source_response = client.get("/api/library/chunks", params={"source_path": str(tmp_path / "apollo.md")})

    assert all_response.status_code == 200
    assert len(all_response.json()["sources"]) == 2
    assert filtered_response.status_code == 200
    assert filtered_response.json()["sources"][0]["title"] == "apollo.md"
    assert source_response.status_code == 200
    assert [chunk["chunk_index"] for chunk in source_response.json()["chunks"]] == [0, 1]


def test_gpu_health_model_status_does_not_expose_secrets(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_NEMOTRON_HF_TOKEN", "hf_secret")
    monkeypatch.setenv("BRAIN_SIDECAR_BRAVE_SEARCH_API_KEY", "brave_secret")
    with TestClient(create_app()) as client:
        response = client.get("/api/health/gpu")

    assert response.status_code == 200
    payload_text = str(response.json())
    assert "hf_secret" not in payload_text
    assert "brave_secret" not in payload_text


def test_ollama_models_api_lists_and_persists_chat_model(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("BRAIN_SIDECAR_OLLAMA_CHAT_MODEL=phi3:mini\n", encoding="utf-8")
    monkeypatch.setenv("BRAIN_SIDECAR_ENV_PATH", str(env_path))
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_OLLAMA_CHAT_MODEL", "phi3:mini")
    monkeypatch.setattr(
        "brain_sidecar.core.ollama.OllamaClient.list_models",
        lambda _self, _host: [
            {"name": "gpt-oss:120b-cloud", "size": None},
            {"name": "phi3:mini", "size": 2_200_000_000},
        ],
    )
    monkeypatch.setattr("brain_sidecar.core.ollama.OllamaClient.host_reachable", lambda _self, _host: True)

    with TestClient(create_app()) as client:
        list_response = client.get("/api/models/ollama")
        update_response = client.post("/api/models/ollama/chat", json={"model": "gpt-oss:120b-cloud"})
        health_response = client.get("/api/health/gpu")

    assert list_response.status_code == 200
    assert [model["name"] for model in list_response.json()["models"]] == ["gpt-oss:120b-cloud", "phi3:mini"]
    assert update_response.status_code == 200
    assert update_response.json()["selected_chat_model"] == "gpt-oss:120b-cloud"
    assert health_response.json()["ollama_chat_model"] == "gpt-oss:120b-cloud"
    assert "BRAIN_SIDECAR_OLLAMA_CHAT_MODEL=gpt-oss:120b-cloud" in env_path.read_text(encoding="utf-8")


def test_ollama_chat_model_update_rejects_unknown_model(monkeypatch, tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("BRAIN_SIDECAR_OLLAMA_CHAT_MODEL=phi3:mini\n", encoding="utf-8")
    monkeypatch.setenv("BRAIN_SIDECAR_ENV_PATH", str(env_path))
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_OLLAMA_CHAT_MODEL", "phi3:mini")
    monkeypatch.setattr(
        "brain_sidecar.core.ollama.OllamaClient.list_models",
        lambda _self, _host: [{"name": "phi3:mini", "size": 2_200_000_000}],
    )

    with TestClient(create_app()) as client:
        response = client.post("/api/models/ollama/chat", json={"model": "missing:latest"})

    assert response.status_code == 400
    assert "BRAIN_SIDECAR_OLLAMA_CHAT_MODEL=phi3:mini" in env_path.read_text(encoding="utf-8")
