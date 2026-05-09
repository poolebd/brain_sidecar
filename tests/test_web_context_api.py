from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain_sidecar.core.web_context import WebSearchResult
from brain_sidecar.core.models import SearchHit
from brain_sidecar.server.app import create_app


class RecordingSearch:
    def __init__(self, results: list[WebSearchResult] | None = None) -> None:
        self.results = results or []
        self.calls: list[tuple[str, str | None]] = []

    async def search(self, query: str, *, freshness: str | None = None) -> list[WebSearchResult]:
        self.calls.append((query, freshness))
        return self.results


class ManualRecall:
    async def search(self, query: str, limit: int = 8, **kwargs) -> list[SearchHit]:
        return [
            SearchHit(
                source_type="document_chunk",
                source_id="ref_ee",
                text="DOE electrical reference notes discuss rollback validation and breaker acceptance criteria.",
                score=0.92,
                metadata={"title": "DOE Electrical Science", "path": "/tmp/reference/doe.pdf"},
            ),
            SearchHit(
                source_type="session_summary",
                source_id="ses_old",
                text="Prior transcript summary covered Apollo rollback validation.",
                score=0.9,
                metadata={"title": "Apollo rollout"},
            )
        ]


class EmptyRecall:
    async def search(self, query: str, limit: int = 8, **kwargs) -> list[SearchHit]:
        return []


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
    assert payload["skip_reason"] is None
    assert payload["cards"][0]["source_type"] == "brave_web"
    assert payload["cards"][0]["ephemeral"] is True
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
    payload = response.json()
    assert payload["query"] == "latest React release"
    assert payload["enabled"] is False
    assert payload["configured"] is False
    assert payload["note"] is None
    assert payload["skip_reason"] == "web_disabled"
    assert payload["cards"] == []


def test_manual_sidecar_query_separates_technical_references_and_web_sources(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_WEB_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("BRAIN_SIDECAR_BRAVE_SEARCH_API_KEY", "test-key")
    app = create_app()
    app.state.manager.recall = ManualRecall()  # type: ignore[assignment]
    app.state.manager.web_search = RecordingSearch(
        [
            WebSearchResult(
                title="React release notes",
                url="https://example.com/react",
                description="Current React release details.",
            )
        ]
    )

    response = TestClient(app).post(
        "/api/sidecar/query",
        json={"query": "latest React release Apollo rollback validation"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_audio_retained"] is False
    assert payload["sections"]["prior_transcript"]
    assert payload["sections"]["technical_references"]
    assert payload["sections"]["current_public_web"]
    assert payload["sections"]["suggested_meeting_contribution"]
    assert all(card["explicitly_requested"] is True for card in payload["cards"])


def test_company_refs_api_and_manual_sidecar_query_are_local(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_WEB_CONTEXT_ENABLED", "false")
    app = create_app()
    app.state.manager.recall = EmptyRecall()  # type: ignore[assignment]
    client = TestClient(app)

    status = client.get("/api/company-refs/status")
    assert status.status_code == 200
    assert status.json()["active_ref_count"] >= 1

    search = client.get("/api/company-refs/search", params={"query": "Siemens"})
    assert search.status_code == 200
    assert search.json()["company_refs"][0]["id"] == "siemens"

    query = client.post("/api/sidecar/query", json={"query": "What does Siemens do?"})
    assert query.status_code == 200
    payload = query.json()
    assert payload["raw_audio_retained"] is False
    assert payload["sections"]["company_refs"]
    assert payload["sections"]["company_refs"][0]["source_type"] == "company_ref"
    assert payload["sections"]["suggested_meeting_contribution"] == []
