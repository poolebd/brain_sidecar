from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from brain_sidecar.core.test_mode import SUPPORTED_AUDIO_EXTENSIONS
from brain_sidecar.server.app import create_app


def test_test_mode_prepare_rejects_when_disabled(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_TEST_MODE_ENABLED", "false")
    client = TestClient(create_app())

    response = client.post("/api/test-mode/audio/prepare", json={"source_path": "/tmp/recording.wav"})

    assert response.status_code == 403


def test_test_mode_prepare_rejects_missing_source(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_TEST_MODE_ENABLED", "true")
    monkeypatch.setenv("BRAIN_SIDECAR_TEST_AUDIO_RUN_DIR", str(tmp_path / "runs"))
    client = TestClient(create_app())

    response = client.post("/api/test-mode/audio/prepare", json={"source_path": str(tmp_path / "missing.wav")})

    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]


def test_input_file_upload_writes_under_data_dir(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(data_dir))
    client = TestClient(create_app())

    response = client.post(
        "/api/input-files",
        files={"file": ("meeting notes.md", b"# Notes", "text/markdown")},
    )

    assert response.status_code == 200
    payload = response.json()
    uploaded = Path(payload["path"])
    assert uploaded.parent == data_dir / "input-files"
    assert uploaded.name.endswith("meeting_notes.md")
    assert uploaded.read_bytes() == b"# Notes"
    assert payload["size_bytes"] == 7


def test_test_mode_allows_common_audio_extensions() -> None:
    assert {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".mp4", ".webm"} <= SUPPORTED_AUDIO_EXTENSIONS


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required for conversion")
def test_test_mode_prepare_converts_to_fixture_wav(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    write_wav(source, sample_rate=8_000, seconds=1)
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_TEST_MODE_ENABLED", "true")
    monkeypatch.setenv("BRAIN_SIDECAR_TEST_AUDIO_RUN_DIR", str(tmp_path / "runs"))
    client = TestClient(create_app())

    response = client.post(
        "/api/test-mode/audio/prepare",
        json={
            "source_path": str(source),
            "max_seconds": 0.5,
            "expected_terms": ["Alpha", "alpha", "Beta"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    fixture = Path(payload["fixture_wav"])
    assert fixture.exists()
    assert payload["expected_terms"] == ["Alpha", "Beta"]
    assert 0.45 <= payload["duration_seconds"] <= 0.55
    with wave.open(str(fixture), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getframerate() == 16_000
        assert wav.getsampwidth() == 2


def test_test_mode_report_writes_under_run_dir(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "runs"
    artifact_dir = run_dir / "testrun_known"
    artifact_dir.mkdir(parents=True)
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_TEST_MODE_ENABLED", "true")
    monkeypatch.setenv("BRAIN_SIDECAR_TEST_AUDIO_RUN_DIR", str(run_dir))
    client = TestClient(create_app())

    response = client.post(
        "/api/test-mode/runs/testrun_known/report",
        json={"report": {"transcript_segments": 3, "issues": []}},
    )

    assert response.status_code == 200
    report_path = Path(response.json()["report_path"])
    assert report_path == artifact_dir / "report.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["transcript_segments"] == 3


def write_wav(path: Path, *, sample_rate: int, seconds: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0" * sample_rate * 2 * seconds)
