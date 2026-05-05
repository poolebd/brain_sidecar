from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from brain_sidecar.core.storage import Storage
from brain_sidecar.core.work_memory import WorkMemoryService, describe_source, normalize_lookup
from brain_sidecar.server.app import create_app
import brain_sidecar.core.work_memory as work_memory


class FakeRecallIndex:
    def __init__(self) -> None:
        self.embedded: list[tuple[str, str, str]] = []

    async def add_text(self, source_type: str, source_id: str, text: str, metadata: dict) -> None:
        self.embedded.append((source_type, source_id, text))


def test_describe_source_guards_sensitive_and_current_employer_material() -> None:
    medical = Path("/home/bp/Nextcloud2/_library/_shoalstone/past work/Navy/Medical Records/scan.pdf")
    guardrail = Path("/home/bp/Nextcloud2/Job Hunting/S&L Offer/employment agreement.txt")
    spreadsheet = Path("/home/bp/Nextcloud2/_library/_shoalstone/past work/UMS Group/PG&E model.xlsx")

    assert describe_source(medical)["status"] == "excluded"
    assert describe_source(medical)["disabled"] is True
    assert describe_source(guardrail)["status"] == "guardrail"
    assert describe_source(guardrail)["disabled"] is True
    assert describe_source(spreadsheet)["status"] == "metadata_only"
    assert describe_source(spreadsheet)["source_group"] == "consulting_history"


def test_work_memory_normalizes_spoken_domain_terms() -> None:
    text = "T A Smith, P G and E gas mains, and five hundred kilovolt breaker work used C T and P T signals."

    normalized = normalize_lookup(text)

    assert "ta smith" in normalized
    assert "pg e gas mains" in normalized
    assert "500kv breaker" in normalized
    assert "ct and pt" in normalized


def test_work_memory_reindex_builds_project_cards_without_live_model_calls(
    monkeypatch,
    event_loop,
    tmp_path: Path,
) -> None:
    job_root = tmp_path / "Job Hunting"
    pmp = job_root / "_portfolio" / "Projects" / "PMP_Experience_Summary.csv"
    pmp.parent.mkdir(parents=True)
    pmp.write_text(
        "\n".join(
            [
                "Project Title,Organization,Industry,Job Title,Functional Area,Project Role,Areas,Project Dates",
                "Online Generator Monitoring - T.A. Smith,Oglethorpe Power,Energy,Electrical Engineer,Capital Projects,Project Manager,Planning,Mar 2019 - June 2021",
            ]
        ),
        encoding="utf-8",
    )
    star_path = job_root / "Standard Interview" / "STAR PPT" / "ppts" / "stars_500kvbreaker.txt"
    star_path.parent.mkdir(parents=True)
    star_path.write_text(
        "Managed a 500kV breaker replacement with outage planning, factory acceptance testing, and commissioning.",
        encoding="utf-8",
    )
    past_root = tmp_path / "_library" / "_shoalstone" / "past work"
    ogm_dir = past_root / "OPC" / "2021 OGM"
    ogm_dir.mkdir(parents=True)
    (ogm_dir / "OGMS failure mode alarm notes.txt").write_text(
        "Flux probes and bus couplers need operator interpretation for condition monitoring.",
        encoding="utf-8",
    )

    monkeypatch.setattr(work_memory, "JOB_HISTORY_ROOT", job_root)
    monkeypatch.setattr(work_memory, "PMP_SUMMARY", pmp)

    storage = Storage(tmp_path / "runtime")
    storage.connect()
    fake_recall = FakeRecallIndex()
    service = WorkMemoryService(storage, fake_recall)

    report = event_loop.run_until_complete(service.reindex(roots=[job_root, past_root]))
    cards = service.search("generator monitoring failure mode alarm interpretation", limit=3)

    assert report.projects_indexed == 2
    assert report.evidence_indexed >= 2
    assert fake_recall.embedded
    assert cards
    assert cards[0].title.startswith("Online Generator Monitoring")
    assert "failure" in cards[0].text.lower()
    assert cards[0].suggested_say
    assert cards[0].card_key.startswith("work:")


def test_work_memory_api_searches_preindexed_cards(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path))
    app = create_app()
    storage = app.state.manager.storage
    project_id = storage.upsert_work_memory_project(
        key="pg_e_gas_mains",
        title="PG&E Gas Mains Replacement Cost Model",
        organization="PG&E / UMS Group",
        date_range="2023 - 2024",
        role="Consulting Analytics Lead",
        domain="Cost Modeling",
        summary="Predictive cost model that surfaced gas mains replacement cost drivers.",
        lessons=["When variance is high, expose the cost drivers before chasing perfect prediction."],
        triggers=["PG&E", "gas mains", "cost drivers", "scenario analysis"],
        source_group="consulting_history",
        confidence=0.9,
    )
    storage.add_work_memory_evidence(
        project_id=project_id,
        source_id=None,
        source_path="/tmp/PG&E Mains Replacement Cost Predict Model.exe",
        snippet="Filename evidence for the PG&E gas mains cost model.",
        artifact_type="metadata_only",
        weight=0.8,
    )

    client = TestClient(app)
    response = client.post("/api/work-memory/search", json={"query": "gas main cost drivers", "limit": 2})

    assert response.status_code == 200
    payload = response.json()
    assert payload["cards"][0]["title"] == "PG&E Gas Mains Replacement Cost Model"
    assert "reminiscent" in payload["cards"][0]["text"]
    assert payload["cards"][0]["suggested_say"]


def test_manual_work_memory_does_not_emit_unmatched_baseline_cards(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "runtime")
    storage.connect()
    project_id = storage.upsert_work_memory_project(
        key="apollo_rollout",
        title="Apollo Rollout",
        organization="BP history",
        date_range="2025",
        role="Lead",
        domain="Software",
        summary="Current status notes for rollback validation and owner mapping.",
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
    service = WorkMemoryService(storage)

    assert service.search("current public status of Python free threading", manual=True) == []
    assert service.search("Apollo rollback owner", manual=True)


def test_live_work_memory_ignores_generic_transcript_chatter(tmp_path: Path) -> None:
    storage = Storage(tmp_path / "runtime")
    storage.connect()
    project_id = storage.upsert_work_memory_project(
        key="online_generator_monitoring",
        title="Online Generator Monitoring",
        organization="Oglethorpe Power",
        date_range="2021",
        role="Project Manager",
        domain="Generator monitoring",
        summary="Condition monitoring work for generator failure modes and owner decisions.",
        lessons=["Monitoring helps when measurements map to a failure mode and a decision owner."],
        triggers=["generator monitoring", "failure mode", "condition monitoring"],
        source_group="opc_history",
        confidence=0.92,
    )
    storage.add_work_memory_evidence(
        project_id=project_id,
        source_id=None,
        source_path="/tmp/ogm.txt",
        snippet="Generator monitoring failure mode evidence.",
        artifact_type="text_supported",
        weight=1.0,
    )
    service = WorkMemoryService(storage)

    assert service.search("you know what time it was with her and one real thing", limit=3) == []
    assert service.search("generator monitoring failure mode decision owner", limit=3)
    assert service.search("generator monitoring", limit=3, manual=True)


def test_work_memory_uses_configured_roots_and_pas_root(tmp_path: Path) -> None:
    from brain_sidecar.config import Settings

    job_root = tmp_path / "job"
    past_root = tmp_path / "past"
    pas_root = tmp_path / "pas"
    for root in [job_root, past_root, pas_root]:
        root.mkdir()
    storage = Storage(tmp_path / "runtime")
    storage.connect()
    settings = Settings(
        data_dir=tmp_path / "runtime",
        host="127.0.0.1",
        port=8765,
        asr_primary_model="tiny.en",
        asr_fallback_model="tiny.en",
        asr_compute_type="float16",
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="qwen3.5:9b",
        ollama_embed_model="embeddinggemma",
        work_memory_job_history_root=job_root,
        work_memory_past_work_root=past_root,
        work_memory_pas_root=pas_root,
    )

    service = WorkMemoryService(storage, settings=settings)

    assert service.default_roots() == [job_root, past_root, pas_root]
