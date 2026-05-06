from __future__ import annotations

from pathlib import Path
from typing import Any

from brain_sidecar.config import Settings
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
