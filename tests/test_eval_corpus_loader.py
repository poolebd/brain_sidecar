from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from brain_sidecar.eval.corpus import import_corpus


def test_plain_text_import_creates_segments(tmp_path: Path) -> None:
    source = tmp_path / "meeting.txt"
    source.write_text("First line.\n\nSecond line with more words.\n", encoding="utf-8")
    output = tmp_path / "out.jsonl"

    count = import_corpus(source_type="plain-text", input_path=source, output=output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert count == 2
    assert len(lines) == 2
    assert json.loads(lines[0])["type"] == "transcript"


def test_jsonl_import_preserves_ids_and_timestamps(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"type": "transcript", "id": "seg_a", "session_id": "ses", "start_s": 2.5, "end_s": 5.0, "text": "Hello."}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "out.jsonl"

    import_corpus(source_type="jsonl", input_path=source, output=output)
    payload = json.loads(output.read_text(encoding="utf-8").strip())

    assert payload["id"] == "seg_a"
    assert payload["start_s"] == 2.5
    assert payload["end_s"] == 5.0


def test_unknown_source_type_fails_cleanly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported source type"):
        import_corpus(source_type="unknown", input_path=tmp_path / "missing.txt", output=tmp_path / "out.jsonl")


def test_loader_does_not_require_network(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "meeting.txt"
    source.write_text("Network-free import.\n", encoding="utf-8")
    output = tmp_path / "out.jsonl"

    def blocked_socket(*args, **kwargs):
        raise AssertionError("network should not be used")

    monkeypatch.setattr(socket, "socket", blocked_socket)

    assert import_corpus(source_type="plain-text", input_path=source, output=output) == 1
