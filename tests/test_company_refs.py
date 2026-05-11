from __future__ import annotations

from pathlib import Path

from brain_sidecar.config import Settings
from brain_sidecar.core.company_refs import CompanyRefService, company_ref_from_seed_row
from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.storage import Storage


def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        host="127.0.0.1",
        port=8765,
        asr_primary_model="tiny.en",
        asr_fallback_model="tiny.en",
        asr_compute_type="int8",
        ollama_host="http://127.0.0.1:11434",
        ollama_chat_model="gemma3:12b",
        ollama_embed_model="embeddinggemma",
        company_refs_min_confidence=0.70,
        company_refs_max_live_cards=3,
    )


def service_for(tmp_path: Path, *rows: dict) -> CompanyRefService:
    storage = Storage(tmp_path)
    storage.connect()
    storage.upsert_company_refs([company_ref_from_seed_row(row) for row in rows])
    service = CompanyRefService(storage, settings(tmp_path))
    service.refresh()
    return service


def seg(text: str, segment_id: str = "seg_1") -> TranscriptSegment:
    return TranscriptSegment(id=segment_id, session_id="ses_1", start_s=0.0, end_s=2.0, text=text)


def company_row(**overrides) -> dict:
    row = {
        "id": "siemens",
        "canonical_name": "Siemens",
        "entity_type": "company",
        "domain": "industrial technology / electrical equipment",
        "description": "Industrial technology company with automation and grid businesses.",
        "aliases": ["Siemens Energy"],
        "acronyms": ["SIE"],
        "asr_variants": ["demons"],
        "context_terms": ["client", "agreement", "power quality", "grid"],
        "negative_terms": [],
        "sources": [{"title": "Seed reference", "url": "https://example.com/siemens"}],
        "metadata": {},
    }
    row.update(overrides)
    return row


def test_exact_canonical_name_match(tmp_path: Path) -> None:
    service = service_for(tmp_path, company_row())

    mentions = service.match_segments([seg("We need Siemens to answer the power quality question.")])

    assert len(mentions) == 1
    assert mentions[0].canonical_name == "Siemens"
    assert mentions[0].alias_type == "name"
    assert mentions[0].source_segment_ids == ["seg_1"]
    assert "Siemens" in mentions[0].evidence_quote


def test_alias_match(tmp_path: Path) -> None:
    service = service_for(tmp_path, company_row(aliases=["Gridworks"]))

    mentions = service.match_segments([seg("Gridworks needs the grid equipment context.")])

    assert len(mentions) == 1
    assert mentions[0].alias == "Gridworks"


def test_acronym_match_requires_context(tmp_path: Path) -> None:
    service = service_for(tmp_path, company_row())

    mentions = service.match_segments([seg("SIE has the client agreement for the grid scope.")])

    assert len(mentions) == 1
    assert mentions[0].alias_type == "acronym"
    assert mentions[0].confidence >= 0.70


def test_short_acronym_without_context_is_suppressed(tmp_path: Path) -> None:
    service = service_for(tmp_path, company_row())

    assert service.match_segments([seg("SIE is on the agenda.")]) == []


def test_asr_variant_requires_context(tmp_path: Path) -> None:
    service = service_for(tmp_path, company_row())

    assert service.match_segments([seg("The demons client agreement needs power quality review.")])
    assert service.match_segments([seg("The demons are just a noisy transcript fragment.")]) == []


def test_negative_terms_suppress_false_positive(tmp_path: Path) -> None:
    service = service_for(tmp_path, company_row(negative_terms=["mythology"]))

    mentions = service.match_segments([seg("The demons mythology note is not about the client agreement.")])

    assert mentions == []


def test_dedupe_by_ref_keeps_best_mention(tmp_path: Path) -> None:
    service = service_for(tmp_path, company_row())

    mentions = service.match_segments([
        seg("The demons client agreement needs power quality review.", "seg_asr"),
        seg("Siemens needs the same grid context.", "seg_name"),
    ])

    assert len(mentions) == 1
    assert mentions[0].ref_id == "siemens"
    assert mentions[0].alias == "Siemens"
    assert mentions[0].source_segment_ids == ["seg_name"]
