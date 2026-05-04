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
    )


class CapturingOllama(OllamaClient):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((path, payload))
        if path == "/api/embed":
            return {"embeddings": [[0.1, 0.2]]}
        return {"message": {"content": "ok"}}


def test_ollama_payloads_set_keep_alive_zero(event_loop, tmp_path: Path) -> None:
    client = CapturingOllama(make_settings(tmp_path))

    event_loop.run_until_complete(client.chat("system", "user"))
    event_loop.run_until_complete(client.embed(["hello"]))

    assert client.requests[0][1]["keep_alive"] == "0s"
    assert client.requests[1][1]["keep_alive"] == "0s"
