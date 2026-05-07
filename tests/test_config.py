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
        "BRAIN_SIDECAR_STREAMING_MIN_FINAL_WORDS",
        "BRAIN_SIDECAR_STREAMING_MIN_FINAL_SECONDS",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.audio_chunk_ms == 250
    assert settings.transcription_window_seconds == 3.4
    assert settings.transcription_overlap_seconds == 0.8
    assert settings.transcription_queue_size == 8
    assert settings.asr_vad_min_silence_ms == 300
    assert settings.partial_transcripts_enabled is False
    assert settings.partial_window_seconds == 2.0
    assert settings.partial_min_interval_seconds == 2.0
    assert settings.streaming_min_final_words == 10
    assert settings.streaming_min_final_seconds == 2.8
    assert settings.recall_min_score == 0.58
    assert settings.recall_max_live_hits == 4
    assert settings.recall_prefer_summaries is True


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


def test_ollama_split_host_env_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_OLLAMA_HOST", "http://127.0.0.1:11434/")
    monkeypatch.setenv("BRAIN_SIDECAR_OLLAMA_CHAT_HOST", "http://192.168.86.219:11434/")
    monkeypatch.setenv("BRAIN_SIDECAR_OLLAMA_EMBED_HOST", "http://127.0.0.1:11434/")
    monkeypatch.setenv("BRAIN_SIDECAR_OLLAMA_KEEP_ALIVE", "10m")
    monkeypatch.setenv("BRAIN_SIDECAR_OLLAMA_CHAT_KEEP_ALIVE", "15m")
    monkeypatch.setenv("BRAIN_SIDECAR_OLLAMA_EMBED_KEEP_ALIVE", "0")

    settings = load_settings()

    assert settings.ollama_host == "http://127.0.0.1:11434"
    assert settings.ollama_chat_host == "http://192.168.86.219:11434"
    assert settings.ollama_embed_host == "http://127.0.0.1:11434"
    assert settings.ollama_keep_alive == "10m"
    assert settings.ollama_chat_keep_alive == "15m"
    assert settings.ollama_embed_keep_alive == "0"


def test_same_gpu_nemotron_phi_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    for name in [
        "BRAIN_SIDECAR_ASR_BACKEND",
        "BRAIN_SIDECAR_ASR_UNLOAD_OLLAMA_ON_START",
        "BRAIN_SIDECAR_OLLAMA_HOST",
        "BRAIN_SIDECAR_OLLAMA_CHAT_HOST",
        "BRAIN_SIDECAR_OLLAMA_EMBED_HOST",
        "BRAIN_SIDECAR_OLLAMA_KEEP_ALIVE",
        "BRAIN_SIDECAR_OLLAMA_CHAT_KEEP_ALIVE",
        "BRAIN_SIDECAR_OLLAMA_EMBED_KEEP_ALIVE",
        "BRAIN_SIDECAR_OLLAMA_CHAT_MODEL",
        "BRAIN_SIDECAR_OLLAMA_EMBED_MODEL",
    ]:
        monkeypatch.delenv(name, raising=False)

    settings = load_settings()

    assert settings.asr_backend == "nemotron_streaming"
    assert settings.asr_unload_ollama_on_start is False
    assert settings.ollama_host == "http://127.0.0.1:11434"
    assert settings.ollama_chat_host == "http://127.0.0.1:11434"
    assert settings.ollama_embed_host == "http://127.0.0.1:11434"
    assert settings.ollama_chat_model == "phi3:mini"
    assert settings.ollama_embed_model == "embeddinggemma"
    assert settings.ollama_keep_alive == "30m"
    assert settings.ollama_chat_keep_alive == "30m"
    assert settings.ollama_embed_keep_alive == "0"


def test_transcription_timing_env_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_AUDIO_CHUNK_MS", "250")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS", "3.4")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_OVERLAP_SECONDS", "0.8")
    monkeypatch.setenv("BRAIN_SIDECAR_STREAMING_MIN_FINAL_WORDS", "12")
    monkeypatch.setenv("BRAIN_SIDECAR_STREAMING_MIN_FINAL_SECONDS", "3.5")

    settings = load_settings()

    assert settings.audio_chunk_ms == 250
    assert settings.transcription_window_seconds == 3.4
    assert settings.transcription_overlap_seconds == 0.8
    assert settings.streaming_min_final_words == 12
    assert settings.streaming_min_final_seconds == 3.5


def test_balanced_preview_env_profile(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_TRANSCRIPTION_WINDOW_SECONDS", "3.4")
    monkeypatch.setenv("BRAIN_SIDECAR_PARTIAL_TRANSCRIPTS_ENABLED", "true")
    monkeypatch.setenv("BRAIN_SIDECAR_PARTIAL_WINDOW_SECONDS", "1.25")
    monkeypatch.setenv("BRAIN_SIDECAR_PARTIAL_MIN_INTERVAL_SECONDS", "0.8")

    settings = load_settings()

    assert settings.partial_transcripts_enabled is True
    assert settings.partial_window_seconds == 1.25
    assert settings.partial_min_interval_seconds == 0.8


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


def test_recall_and_work_memory_env_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_RECALL_MIN_SCORE", "0.7")
    monkeypatch.setenv("BRAIN_SIDECAR_RECALL_MAX_LIVE_HITS", "2")
    monkeypatch.setenv("BRAIN_SIDECAR_RECALL_PREFER_SUMMARIES", "false")
    monkeypatch.setenv("BRAIN_SIDECAR_WORK_MEMORY_JOB_HISTORY_ROOT", str(tmp_path / "job"))
    monkeypatch.setenv("BRAIN_SIDECAR_WORK_MEMORY_PAST_WORK_ROOT", str(tmp_path / "past"))
    monkeypatch.setenv("BRAIN_SIDECAR_WORK_MEMORY_PAS_ROOT", str(tmp_path / "pas"))

    settings = load_settings()

    assert settings.recall_min_score == 0.7
    assert settings.recall_max_live_hits == 2
    assert settings.recall_prefer_summaries is False
    assert settings.work_memory_job_history_root == tmp_path / "job"
    assert settings.work_memory_past_work_root == tmp_path / "past"
    assert settings.work_memory_pas_root == tmp_path / "pas"


def test_sidecar_quality_gate_env_overrides(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "_DEFAULT_ENV_PATH", tmp_path / "missing.env")
    monkeypatch.setenv("BRAIN_SIDECAR_SIDECAR_QUALITY_GATE_ENABLED", "false")
    monkeypatch.setenv("BRAIN_SIDECAR_SIDECAR_MIN_EVIDENCE_SEGMENTS", "3")
    monkeypatch.setenv("BRAIN_SIDECAR_SIDECAR_DUPLICATE_WINDOW_SECONDS", "90")
    monkeypatch.setenv("BRAIN_SIDECAR_SIDECAR_GENERIC_CLARIFY_WINDOW_SECONDS", "240")
    monkeypatch.setenv("BRAIN_SIDECAR_SIDECAR_MAX_CARDS_PER_5MIN", "5")
    monkeypatch.setenv("BRAIN_SIDECAR_SIDECAR_MAX_CARDS_PER_GENERATION_PASS", "2")

    settings = load_settings()

    assert settings.sidecar_quality_gate_enabled is False
    assert settings.sidecar_min_evidence_segments == 3
    assert settings.sidecar_duplicate_window_seconds == 90.0
    assert settings.sidecar_generic_clarify_window_seconds == 240.0
    assert settings.sidecar_max_cards_per_5min == 5
    assert settings.sidecar_max_cards_per_generation_pass == 2
