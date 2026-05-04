from __future__ import annotations

import pytest

import brain_sidecar.config as config
from brain_sidecar.config import load_settings


def test_transcription_timing_defaults_use_balanced_live_profile(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    for name in [
        "BRAIN_SIDECAR_AUDIO_CHUNK_MS",
        "BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS",
        "BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS",
        "BRAIN_SIDECAR_TRANSCRIPTION_QUEUE_SIZE",
        "BRAIN_SIDECAR_ASR_VAD_MIN_SILENCE_MS",
        "BRAIN_SIDECAR_PARTIAL_TRANSCRIPTS_ENABLED",
        "BRAIN_SIDECAR_PARTIAL_WINDOW_SECONDS",
        "BRAIN_SIDECAR_PARTIAL_MIN_INTERVAL_SECONDS",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.audio_chunk_ms == 250
    assert settings.transcription_window_seconds == 3.4
    assert settings.transcription_overlap_seconds == 0.8
    assert settings.transcription_queue_size == 8
    assert settings.asr_vad_min_silence_ms == 300
    assert settings.partial_transcripts_enabled is True
    assert settings.partial_window_seconds == 2.0
    assert settings.partial_min_interval_seconds == 2.0


def test_load_settings_reads_repo_dotenv_without_overriding_exports(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "BRAIN_SIDECAR_AUDIO_CHUNK_MS=125",
                "BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS=3.2",
                "BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS=0.7",
                "BRAIN_SIDECAR_ASR_VAD_MIN_SILENCE_MS=275",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", env_file)
    monkeypatch.delenv("BRAIN_SIDECAR_AUDIO_CHUNK_MS", raising=False)
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS", "4.5")
    monkeypatch.delenv("BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS", raising=False)
    monkeypatch.delenv("BRAIN_SIDECAR_ASR_VAD_MIN_SILENCE_MS", raising=False)

    settings = load_settings()

    assert settings.audio_chunk_ms == 125
    assert settings.transcription_window_seconds == 4.5
    assert settings.transcription_overlap_seconds == 0.7
    assert settings.asr_vad_min_silence_ms == 275


def test_transcription_timing_env_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_AUDIO_CHUNK_MS", "250")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS", "3.4")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS", "0.8")

    settings = load_settings()

    assert settings.audio_chunk_ms == 250
    assert settings.transcription_window_seconds == 3.4
    assert settings.transcription_overlap_seconds == 0.8


def test_transcription_overlap_is_clamped_below_window(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS", "3.4")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS", "4.0")

    settings = load_settings()

    assert settings.transcription_window_seconds == 3.4
    assert settings.transcription_overlap_seconds == pytest.approx(3.3)


def test_partial_transcript_env_overrides_are_guarded(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS", "3.4")
    monkeypatch.setenv("BRAIN_SIDECAR_PARTIAL_TRANSCRIPTS_ENABLED", "false")
    monkeypatch.setenv("BRAIN_SIDECAR_PARTIAL_WINDOW_SECONDS", "10")
    monkeypatch.setenv("BRAIN_SIDECAR_PARTIAL_MIN_INTERVAL_SECONDS", "0.1")

    settings = load_settings()

    assert settings.partial_transcripts_enabled is False
    assert settings.partial_window_seconds == 3.4
    assert settings.partial_min_interval_seconds == 0.5
