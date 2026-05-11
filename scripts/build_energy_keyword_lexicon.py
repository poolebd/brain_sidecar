#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from brain_sidecar.core.domain_keywords import lexicon_payload, parse_energy_keyword_report

DEFAULT_SOURCE = REPO_ROOT / "docs" / "research" / "energy-keyword-framework.md"
DEFAULT_OUTPUT = REPO_ROOT / "brain_sidecar" / "data" / "energy_keywords.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the local energy consulting keyword lexicon.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="fail if the generated JSON differs from the committed file")
    args = parser.parse_args()

    source = args.source
    output = args.output
    if not source.exists():
        raise SystemExit(f"Missing source markdown: {source}")

    entries = parse_energy_keyword_report(source.read_text(encoding="utf-8"))
    payload = lexicon_payload(entries, source_path=str(source.relative_to(REPO_ROOT)))
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not output.exists():
            raise SystemExit(f"Generated lexicon is missing: {output}")
        existing = output.read_text(encoding="utf-8")
        if existing != text:
            raise SystemExit("Generated lexicon is stale; run scripts/build_energy_keyword_lexicon.py")
        print(f"energy lexicon is current: {len(entries)} entries")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"wrote {len(entries)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
