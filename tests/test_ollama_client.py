from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from brain_sidecar.config import Settings
from brain_sidecar.core.gpu import GpuStatus
from brain_sidecar.core.ollama import OllamaClient


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        host="127.0.0.1",
        port=8765,
        asr_primary_model="tiny.en",
        asr_fallback_model="tiny.en",
        asr_compute_type="float16",
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="qwen3.5:9b",
        ollama_embed_model="embeddinggemma",
        ollama_keep_alive="0s",
        ollama_chat_host="http://192.168.86.219:11434",
        ollama_embed_host="http://127.0.0.1:11434",
        ollama_chat_keep_alive="10m",
        ollama_embed_keep_alive="0",
    )


class CapturingOllama(OllamaClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.requests: list[tuple[str, str, dict[str, Any]]] = []

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        timeout_s: float | None = None,
        host: str | None = None,
    ) -> dict[str, Any]:
        self.requests.append((host or self.settings.ollama_host, path, payload))
        if path == "/api/embed":
            return {"embeddings": [[0.1, 0.2]]}
        return {"message": {"content": "ok"}}


def gpu_status(free_mb: int | None) -> GpuStatus:
    total_mb = 12282
    return GpuStatus(
        nvidia_available=True,
        name="NVIDIA GeForce RTX 4070",
        memory_total_mb=total_mb,
        memory_used_mb=None if free_mb is None else total_mb - free_mb,
        memory_free_mb=free_mb,
        driver_version="570.144",
        gpu_pressure="unknown" if free_mb is None else "ok",
        gpu_processes=[],
        asr_cuda_available=True,
        asr_cuda_error=None,
        ollama_gpu_models=[],
    )


def test_ollama_payloads_use_route_specific_keep_alive(event_loop, tmp_path: Path) -> None:
    client = CapturingOllama(make_settings(tmp_path))

    event_loop.run_until_complete(client.chat("system", "user"))
    event_loop.run_until_complete(client.embed(["hello"]))

    assert client.requests[0][2]["keep_alive"] == "10m"
    assert client.requests[1][2]["keep_alive"] == "0"


def test_ollama_routes_chat_and_embed_to_split_hosts(event_loop, tmp_path: Path) -> None:
    client = CapturingOllama(make_settings(tmp_path))

    event_loop.run_until_complete(client.chat("system", "user"))
    event_loop.run_until_complete(client.embed(["hello"]))

    assert client.requests[0][0] == "http://192.168.86.219:11434"
    assert client.requests[0][1] == "/api/chat"
    assert client.requests[1][0] == "http://127.0.0.1:11434"
    assert client.requests[1][1] == "/api/embed"


def test_ollama_chat_uses_primary_model_when_vram_guard_passes(
    event_loop,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = replace(
        make_settings(tmp_path),
        ollama_chat_model="phi3:mini",
        ollama_chat_fallback_model="smollm2:135m",
        ollama_chat_min_free_vram_mb=9000,
        ollama_chat_host="http://127.0.0.1:11434",
    )
    client = CapturingOllama(settings)
    monkeypatch.setattr("brain_sidecar.core.ollama.read_gpu_status", lambda: gpu_status(9500))

    event_loop.run_until_complete(client.chat("system", "user"))

    assert client.requests[0][2]["model"] == "phi3:mini"


def test_ollama_chat_uses_configured_fallback_model_when_vram_guard_fails(
    event_loop,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = replace(
        make_settings(tmp_path),
        ollama_chat_model="phi3:mini",
        ollama_chat_fallback_model="smollm2:135m",
        ollama_chat_min_free_vram_mb=9000,
        ollama_chat_host="http://127.0.0.1:11434",
    )
    client = CapturingOllama(settings)
    monkeypatch.setattr("brain_sidecar.core.ollama.read_gpu_status", lambda: gpu_status(7400))

    event_loop.run_until_complete(client.chat("system", "user"))

    assert client.requests[0][2]["model"] == "smollm2:135m"


def test_ollama_chat_raises_before_loading_primary_when_vram_guard_fails_without_fallback(
    event_loop,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = replace(
        make_settings(tmp_path),
        ollama_chat_model="phi3:mini",
        ollama_chat_fallback_model="",
        ollama_chat_min_free_vram_mb=9000,
        ollama_chat_host="http://127.0.0.1:11434",
    )
    client = CapturingOllama(settings)
    monkeypatch.setattr("brain_sidecar.core.ollama.read_gpu_status", lambda: gpu_status(7400))

    with pytest.raises(RuntimeError, match="Skipping Ollama chat model phi3:mini"):
        event_loop.run_until_complete(client.chat("system", "user"))

    assert client.requests == []


def test_ollama_chat_vram_guard_does_not_block_remote_chat_host(
    event_loop,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = replace(
        make_settings(tmp_path),
        ollama_chat_model="phi3:mini",
        ollama_chat_fallback_model="smollm2:135m",
        ollama_chat_min_free_vram_mb=9000,
        ollama_chat_host="http://192.168.86.219:11434",
    )
    client = CapturingOllama(settings)

    def fail_if_called() -> GpuStatus:
        raise AssertionError("remote chat host should not read local GPU status")

    monkeypatch.setattr("brain_sidecar.core.ollama.read_gpu_status", fail_if_called)

    event_loop.run_until_complete(client.chat("system", "user"))

    assert client.requests[0][2]["model"] == "phi3:mini"
