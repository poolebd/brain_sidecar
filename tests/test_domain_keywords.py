from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from brain_sidecar.core.domain_keywords import parse_energy_keyword_report


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "docs" / "research" / "energy-keyword-framework.md"
LEXICON = REPO_ROOT / "brain_sidecar" / "data" / "energy_keywords.json"


def test_research_file_is_copied_into_repo() -> None:
    assert SOURCE.exists()
    assert "not** search volume" in SOURCE.read_text(encoding="utf-8")


def test_parser_extracts_expected_report_rows() -> None:
    entries = parse_energy_keyword_report(SOURCE.read_text(encoding="utf-8"))
    phrases = {entry.phrase.lower() for entry in entries}

    for phrase in [
        "energy consulting",
        "energy efficiency",
        "energy management",
        "energy audit",
        "decarbonization",
        "utility bill analysis",
        "energy procurement",
        "decarbonization roadmap",
        "demand response",
        "tariff analysis",
        "iso 50001",
    ]:
        assert phrase in phrases


def test_synonyms_misspellings_and_acronym_flags_are_preserved() -> None:
    entries = parse_energy_keyword_report(SOURCE.read_text(encoding="utf-8"))
    synonyms = {synonym.lower() for entry in entries for synonym in entry.synonyms}

    for misspelling in [
        "engery consulting",
        "engery audit",
        "energy effeciency",
        "electrfication",
    ]:
        assert misspelling in synonyms
    assert any("commisioning" in synonym for synonym in synonyms)

    by_token = {entry.normalized_token: entry for entry in entries}
    for token in ["eui", "rcx", "chp", "ipmvp", "derms"]:
        assert by_token[token].ambiguous is True
        assert by_token[token].requires_context is True


def test_generated_lexicon_is_deterministic_and_conflict_free() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/build_energy_keyword_lexicon.py", "--check"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout

    payload = json.loads(LEXICON.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for entry in payload["entries"]:
        grouped[(entry["normalized_token"], entry["category"])].append(entry["weight"])

    assert all(len(weights) == 1 for weights in grouped.values())
