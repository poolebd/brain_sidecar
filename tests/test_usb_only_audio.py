from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain_sidecar.server.app import create_app


def test_browser_stream_session_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        session_id = client.post("/api/sessions", json={}).json()["id"]

        response = client.post(f"/api/sessions/{session_id}/start", json={"audio_source": "browser_stream"})

    assert response.status_code == 400
    assert "Browser microphone capture has been removed" in response.json()["detail"]


def test_browser_stream_speaker_enrollment_is_rejected(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/speaker/enrollments/spenr-test/record",
            json={"audio_source": "browser_stream"},
        )

    assert response.status_code == 400
    assert "Browser microphone capture has been removed" in response.json()["detail"]


def test_fixture_audio_requires_test_mode(monkeypatch, tmp_path: Path) -> None:
    fixture = tmp_path / "input.wav"
    fixture.write_bytes(b"RIFF")
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_TEST_MODE_ENABLED", "0")
    with TestClient(create_app()) as client:
        session_id = client.post("/api/sessions", json={}).json()["id"]

        response = client.post(
            f"/api/sessions/{session_id}/start",
            json={"audio_source": "fixture", "fixture_wav": str(fixture)},
        )

    assert response.status_code == 403
    assert "test mode" in response.json()["detail"]


def test_browser_audio_websocket_is_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        with client.websocket_connect("/api/sessions/missing/audio-stream") as websocket:
            message = websocket.receive()

    assert message["type"] == "websocket.close"
    assert message["code"] == 1008
    assert "Browser microphone capture has been removed" in message["reason"]
