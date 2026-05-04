from __future__ import annotations

import pytest

from brain_sidecar.config import load_settings


def test_transcription_timing_defaults_remain_conservative(monkeypatch) -> None:
    for name in [
        "BRAIN_SIDECAR_AUDIO_CHUNK_MS",
        "BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS",
        "BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.audio_chunk_ms == 500
    assert settings.transcription_window_seconds == 5.0
    assert settings.transcription_overlap_seconds == 0.75


def test_transcription_timing_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_AUDIO_CHUNK_MS", "250")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS", "3.4")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS", "0.8")

    settings = load_settings()

    assert settings.audio_chunk_ms == 250
    assert settings.transcription_window_seconds == 3.4
    assert settings.transcription_overlap_seconds == 0.8


def test_transcription_overlap_is_clamped_below_window(monkeypatch) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS", "3.4")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS", "4.0")

    settings = load_settings()

    assert settings.transcription_window_seconds == 3.4
    assert settings.transcription_overlap_seconds == pytest.approx(3.3)
