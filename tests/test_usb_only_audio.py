from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain_sidecar.core.devices import DeviceInfo
import brain_sidecar.server.app as server_app
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


def test_devices_endpoint_reports_auto_selected_server_mic(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        server_app,
        "list_audio_devices",
        lambda probe=False: [
            DeviceInfo(
                id="alsa:plughw:2,0",
                label="ALSA USB Mic / USB Audio",
                driver="alsa",
                ffmpeg_input="plughw:2,0",
                hardware_id="0c76:161e",
                healthy=True,
                score=95,
                selection_reason="USB capture",
            )
        ],
    )
    with TestClient(create_app()) as client:
        response = client.get("/api/devices")

    payload = response.json()
    assert response.status_code == 200
    assert payload["server_mic_available"] is True
    assert payload["selected_device"]["id"] == "alsa:plughw:2,0"
    assert "preferred_device_configured" not in payload


def test_browser_audio_websocket_is_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        with client.websocket_connect("/api/sessions/missing/audio-stream") as websocket:
            message = websocket.receive()

    assert message["type"] == "websocket.close"
    assert message["code"] == 1008
    assert "Browser microphone capture has been removed" in message["reason"]
