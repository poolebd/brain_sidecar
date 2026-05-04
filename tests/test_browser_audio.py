from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from brain_sidecar.core.audio import BrowserAudioCapture
from brain_sidecar.server.app import create_app


class FakeTranscriber:
    model_size = "fake"

    async def load(self) -> None:
        return None

    async def transcribe_pcm16(self, pcm: bytes, start_offset_s: float, initial_prompt: str | None = None):
        return SimpleNamespace(spans=[], model="fake")


def test_browser_audio_capture_yields_queued_pcm(event_loop) -> None:
    capture = BrowserAudioCapture()

    async def read_one() -> bytes:
        await capture.feed(b"\x01\x02")
        stream = capture.chunks()
        try:
            chunk = await anext(stream)
            return chunk
        finally:
            await capture.stop()
            await stream.aclose()

    assert event_loop.run_until_complete(read_one()) == b"\x01\x02"


def test_browser_stream_session_does_not_require_server_device(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    app = create_app()
    app.state.manager.transcriber = FakeTranscriber()
    with TestClient(app) as client:
        session_id = client.post("/api/sessions", json={}).json()["id"]

        response = client.post(f"/api/sessions/{session_id}/start", json={"audio_source": "browser_stream"})

        assert response.status_code == 200
        assert response.json()["audio_source"] == "browser_stream"
        assert isinstance(app.state.manager.browser_audio_capture(session_id), BrowserAudioCapture)
        client.post(f"/api/sessions/{session_id}/stop")


def test_browser_audio_websocket_accepts_active_browser_session(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    app = create_app()
    app.state.manager.transcriber = FakeTranscriber()
    with TestClient(app) as client:
        session_id = client.post("/api/sessions", json={}).json()["id"]
        start = client.post(f"/api/sessions/{session_id}/start", json={"audio_source": "browser_stream"})
        assert start.status_code == 200

        with client.websocket_connect(f"/api/sessions/{session_id}/audio-stream") as websocket:
            websocket.send_bytes(b"\0" * 3200)

        client.post(f"/api/sessions/{session_id}/stop")


def test_browser_audio_websocket_rejects_missing_or_wrong_source(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        with client.websocket_connect("/api/sessions/missing/audio-stream") as websocket:
            message = websocket.receive()

    assert message["type"] == "websocket.close"
    assert message["code"] == 1008
