from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlparse

from brain_sidecar.config import Settings
from brain_sidecar.core.gpu import read_gpu_status


class OllamaClient:
    def __init__(self, settings: Settings, timeout_s: float | None = None) -> None:
        self.settings = settings
        self.timeout_s = timeout_s if timeout_s is not None else settings.ollama_chat_timeout_seconds

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self._chat_model(),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        keep_alive = self._chat_keep_alive()
        if keep_alive:
            payload["keep_alive"] = keep_alive
        if format_json:
            payload["format"] = "json"
        response = await asyncio.to_thread(
            self._post_json_with_timeout,
            self._chat_host(),
            "/api/chat",
            payload,
            self.settings.ollama_chat_timeout_seconds,
        )
        return str(response.get("message", {}).get("content", ""))

    async def embed(self, inputs: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.settings.ollama_embed_model, "input": inputs}
        keep_alive = self._embed_keep_alive()
        if keep_alive:
            payload["keep_alive"] = keep_alive
        response = await asyncio.to_thread(
            self._post_json_with_timeout,
            self._embed_host(),
            "/api/embed",
            payload,
            self.settings.ollama_embed_timeout_seconds,
        )
        if "embeddings" in response:
            return response["embeddings"]
        if "embedding" in response:
            return [response["embedding"]]
        raise RuntimeError("Ollama embed response did not include embeddings.")

    def host_reachable(self, host: str, *, timeout_s: float = 0.35) -> bool:
        host = host.rstrip("/")
        if not host:
            return False
        request = urllib.request.Request(host, method="HEAD")
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return 200 <= int(response.status) < 500
        except Exception:
            return False

    def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        timeout_s: float | None = None,
        host: str | None = None,
    ) -> dict[str, Any]:
        host = (host or self.settings.ollama_host).rstrip("/")
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{host}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s or self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama request failed for {host}{path}: {exc}") from exc

    def _post_json_with_timeout(
        self,
        host: str,
        path: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        try:
            return self._post_json(path, payload, timeout_s, host)
        except TypeError:
            # Test doubles and older callers may override _post_json(path, payload).
            return self._post_json(path, payload)  # type: ignore[misc]

    def _chat_host(self) -> str:
        return (getattr(self.settings, "ollama_chat_host", "") or self.settings.ollama_host).rstrip("/")

    def _embed_host(self) -> str:
        return (getattr(self.settings, "ollama_embed_host", "") or self.settings.ollama_host).rstrip("/")

    def _chat_model(self) -> str:
        primary = self.settings.ollama_chat_model
        min_free_mb = int(getattr(self.settings, "ollama_chat_min_free_vram_mb", 0) or 0)
        if min_free_mb <= 0 or not self._chat_uses_local_gpu():
            return primary

        status = read_gpu_status()
        free_mb = status.memory_free_mb
        if free_mb is None or free_mb >= min_free_mb:
            return primary

        fallback = str(getattr(self.settings, "ollama_chat_fallback_model", "") or "").strip()
        if fallback:
            return fallback

        raise RuntimeError(
            f"Skipping Ollama chat model {primary}: free VRAM is {free_mb} MB, "
            f"below the configured chat reserve of {min_free_mb} MB."
        )

    def _chat_uses_local_gpu(self) -> bool:
        host = self._chat_host()
        parsed = urlparse(host)
        hostname = parsed.hostname
        if hostname is None:
            hostname = host.split(":", 1)[0].strip("[]/")
        return hostname in {"", "127.0.0.1", "localhost", "::1", "0.0.0.0"}

    def _chat_keep_alive(self) -> str:
        return getattr(self.settings, "ollama_chat_keep_alive", "") or self.settings.ollama_keep_alive

    def _embed_keep_alive(self) -> str:
        return getattr(self.settings, "ollama_embed_keep_alive", "") or self.settings.ollama_keep_alive
