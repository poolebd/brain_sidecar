from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from brain_sidecar.config import Settings


class OllamaClient:
    def __init__(self, settings: Settings, timeout_s: float | None = None) -> None:
        self.settings = settings
        self.timeout_s = timeout_s if timeout_s is not None else settings.ollama_chat_timeout_seconds

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.ollama_chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if self.settings.ollama_keep_alive:
            payload["keep_alive"] = self.settings.ollama_keep_alive
        if format_json:
            payload["format"] = "json"
        response = await asyncio.to_thread(
            self._post_json_with_timeout,
            "/api/chat",
            payload,
            self.settings.ollama_chat_timeout_seconds,
        )
        return str(response.get("message", {}).get("content", ""))

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.settings.ollama_embed_model, "input": inputs}
        if self.settings.ollama_keep_alive:
            payload["keep_alive"] = self.settings.ollama_keep_alive
        response = await asyncio.to_thread(
            self._post_json_with_timeout,
            "/api/embed",
            payload,
            self.settings.ollama_embed_timeout_seconds,
        )
        if "embeddings" in response:
            return response["embeddings"]
        if "embedding" in response:
            return [response["embedding"]]
        raise RuntimeError("Ollama embed response did not include embeddings.")

    def _post_json(self, path: str, payload: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.settings.ollama_host}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed for {path}: {exc}") from exc

    def _post_json_with_timeout(self, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        try:
            return self._post_json(path, payload, timeout_s)
        except TypeError:
            # Test doubles and older callers may override _post_json(path, payload).
            return self._post_json(path, payload)  # type: ignore[misc]
