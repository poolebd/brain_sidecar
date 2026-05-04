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
                source_type="session_summary",
                source_id="ses_old",
                text="Prior transcript summary covered Apollo rollback validation.",
                score=0.9,
                metadata={"title": "Apollo rollout"},
            )
        ]


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


def test_manual_sidecar_query_separates_local_work_and_web_sources(monkeypatch, tmp_path: Path) -> None:
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
    storage = app.state.manager.storage
    project_id = storage.upsert_work_memory_project(
        key="apollo_rollout",
        title="Apollo Rollout",
        organization="BP history",
        date_range="2025",
        role="Lead",
        domain="Software",
        summary="Rollback validation and owner mapping.",
        lessons=["Make rollback owner explicit before release."],
        triggers=["Apollo", "rollback", "validation"],
        source_group="pas_history",
        confidence=0.9,
    )
    storage.add_work_memory_evidence(
        project_id=project_id,
        source_id=None,
        source_path="/tmp/apollo.md",
        snippet="Rollback owner evidence.",
        artifact_type="text_supported",
        weight=1.0,
    )

    response = TestClient(app).post(
        "/api/sidecar/query",
        json={"query": "latest React release Apollo rollback validation"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_audio_retained"] is False
    assert payload["sections"]["prior_transcript"]
    assert payload["sections"]["pas_past_work"]
    assert payload["sections"]["current_public_web"]
    assert payload["sections"]["suggested_meeting_contribution"]
    assert all(card["explicitly_requested"] is True for card in payload["cards"])
