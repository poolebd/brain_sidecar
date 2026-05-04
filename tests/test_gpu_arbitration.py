from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from brain_sidecar.config import Settings
from brain_sidecar.core import gpu
from brain_sidecar.core.gpu import GpuStatus
from brain_sidecar.core.transcription import FasterWhisperTranscriber


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        host="127.0.0.1",
        port=8765,
        asr_primary_model="medium.en",
        asr_fallback_model="small.en",
        asr_compute_type="float16",
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="qwen3.5:9b",
        ollama_embed_model="embeddinggemma",
        asr_min_free_vram_mb=3500,
        asr_unload_ollama_on_start=True,
        asr_gpu_free_timeout_seconds=0,
    )


def status(free_mb: int, models: list[str] | None = None) -> GpuStatus:
    return GpuStatus(
        nvidia_available=True,
        name="NVIDIA GeForce RTX 4070",
        memory_total_mb=12282,
        memory_used_mb=12282 - free_mb,
        memory_free_mb=free_mb,
        driver_version="570.144",
        gpu_pressure=gpu.gpu_pressure(12282, free_mb),
        gpu_processes=[],
        asr_cuda_available=True,
        asr_cuda_error=None,
        ollama_gpu_models=models or [],
    )


def test_parse_ollama_gpu_models() -> None:
    output = """NAME               ID              SIZE      PROCESSOR    UNTIL
qwen3.5:9b         abc123          7.9 GB    100% GPU     4 minutes from now
embeddinggemma     def456          846 MB    100% GPU     4 minutes from now
llama3             ghi789          4.7 GB    100% CPU     4 minutes from now
"""

    assert gpu.parse_ollama_gpu_models(output) == ["qwen3.5:9b", "embeddinggemma"]


def test_prepare_asr_gpu_stops_ollama_when_free_vram_is_low(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    statuses = iter([status(1024, ["qwen3.5:9b"]), status(4200, [])])
    stop_calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(gpu, "read_gpu_status", lambda: next(statuses))

    def fake_stop(ollama_host: str, models: list[str]) -> tuple[list[str], list[str]]:
        stop_calls.append((ollama_host, models))
        return models, []

    monkeypatch.setattr(gpu, "stop_ollama_gpu_models", fake_stop)

    preparation = gpu.prepare_asr_gpu(make_settings(tmp_path))

    assert stop_calls == [("http://127.0.0.1:11434", ["qwen3.5:9b"])]
    assert preparation.attempted_unload is True
    assert preparation.stopped_ollama_models == ["qwen3.5:9b"]
    assert preparation.after.memory_free_mb == 4200


def test_prepare_asr_gpu_raises_with_actionable_vram_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gpu, "read_gpu_status", lambda: status(1200, []))

    with pytest.raises(RuntimeError) as error:
        gpu.prepare_asr_gpu(make_settings(tmp_path))

    message = str(error.value)
    assert "Need at least 3500 MB free VRAM" in message
    assert "currently free: 1200 MB" in message
    assert "No GPU-resident Ollama model was detected" in message


def test_transcriber_retries_once_after_cuda_oom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prepare_calls: list[bool] = []

    def fake_prepare(_settings: Settings, *, force_unload: bool = False):
        prepare_calls.append(force_unload)
        return None

    class FakeWhisperModel:
        attempts = 0

        def __init__(self, model_size: str, *, device: str, compute_type: str) -> None:
            FakeWhisperModel.attempts += 1
            if FakeWhisperModel.attempts <= 2:
                raise RuntimeError("CUDA failed with error out of memory")
            self.model_size = model_size
            self.device = device
            self.compute_type = compute_type

    monkeypatch.setattr("brain_sidecar.core.transcription.prepare_asr_gpu", fake_prepare)
    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))

    transcriber = FasterWhisperTranscriber(make_settings(tmp_path))
    transcriber._load_sync()

    assert prepare_calls == [False, True]
    assert transcriber.model_size == "medium.en"
    assert FakeWhisperModel.attempts == 3
