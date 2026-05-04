from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain_sidecar.core.web_context import WebSearchResult
from brain_sidecar.server.app import create_app


class RecordingSearch:
    def __init__(self, results: list[WebSearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, str | None]] = []

    async def search(self, query: str, *, freshness: str | None = None) -> list[WebSearchResult]:
        self.calls.append((query, freshness))
        return self.results


def test_manual_web_context_search_returns_ephemeral_note(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_WEB_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("BRAIN_SIDECAR_BRAVE_SEARCH_API_KEY", "test-key")
    app = create_app()
    search = RecordingSearch(
        [
            WebSearchResult(
                title="Vector indexing guide",
                url="https://example.com/vector-indexing",
                description="Compares index types and recall tradeoffs.",
            )
        ]
    )
    app.state.manager.web_search = search
    client = TestClient(app)

    response = client.post(
        "/api/web-context/search",
        json={
            "query": "What are current best practices for vector database indexing?",
            "session_id": "ses_manual",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["configured"] is True
    assert payload["query"] == "current best practices for vector database indexing"
    assert search.calls == [("current best practices for vector database indexing", "py")]
    assert payload["note"]["ephemeral"] is True
    assert payload["note"]["source_type"] == "brave_web"
    assert payload["note"]["session_id"] == "ses_manual"
    assert "I found a few current references" in payload["note"]["body"]
    assert "Live web context found" not in payload["note"]["body"]
    assert payload["note"]["sources"] == [
        {"title": "Vector indexing guide", "url": "https://example.com/vector-indexing"}
    ]


def test_manual_web_context_search_disabled_is_nonfatal(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_WEB_CONTEXT_ENABLED", "false")
    monkeypatch.setenv("BRAIN_SIDECAR_BRAVE_SEARCH_API_KEY", "")
    client = TestClient(create_app())

    response = client.post("/api/web-context/search", json={"query": "latest React release"})

    assert response.status_code == 200
    assert response.json() == {
        "query": "latest React release",
        "enabled": False,
        "configured": False,
        "note": None,
    }
