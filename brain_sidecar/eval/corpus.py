from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from brain_sidecar.eval.models import EvalTranscriptEvent, transcript_event_from_dict


SUPPORTED_SOURCE_TYPES = {"plain-text", "jsonl", "qmsum", "meetingbank"}


def import_corpus(
    *,
    source_type: str,
    output: Path,
    input_path: Path | None = None,
    input_dir: Path | None = None,
) -> int:
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"unsupported source type: {source_type}")
    if source_type == "plain-text":
        if input_path is None:
            raise ValueError("--input is required for plain-text imports")
        events = import_plain_text(input_path)
    elif source_type == "jsonl":
        if input_path is None:
            raise ValueError("--input is required for jsonl imports")
        events = import_jsonl(input_path)
    else:
        if input_dir is None:
            raise ValueError(f"--input-dir is required for {source_type} imports")
        events = import_directory_text(input_dir, source_type=source_type)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(event.to_json() for event in events) + ("\n" if events else ""), encoding="utf-8")
    return len(events)


def import_plain_text(path: Path) -> list[EvalTranscriptEvent]:
    session_id = path.stem
    events: list[EvalTranscriptEvent] = []
    start_s = 0.0
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        end_s = start_s + max(2.0, min(8.0, len(text.split()) * 0.45))
        events.append(
            EvalTranscriptEvent(
                id=f"{session_id}_seg_{index:04d}",
                session_id=session_id,
                start_s=start_s,
                end_s=end_s,
                text=text,
            )
        )
        start_s = end_s + 0.2
    return events


def import_jsonl(path: Path) -> list[EvalTranscriptEvent]:
    events: list[EvalTranscriptEvent] = []
    fallback_session_id = path.stem
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if payload.get("type", "transcript") != "transcript":
            continue
        events.append(transcript_event_from_dict(payload, fallback_session_id=fallback_session_id))
    return events


def import_directory_text(input_dir: Path, *, source_type: str) -> list[EvalTranscriptEvent]:
    events: list[EvalTranscriptEvent] = []
    for path in sorted(input_dir.rglob("*.txt")):
        for event in import_plain_text(path):
            events.append(
                EvalTranscriptEvent(
                    id=f"{source_type}_{event.id}",
                    session_id=f"{source_type}_{path.stem}",
                    start_s=event.start_s,
                    end_s=event.end_s,
                    text=event.text,
                    is_final=event.is_final,
                )
            )
    return events


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize local transcript corpora into Brain Sidecar eval JSONL.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--source-type", required=True)
    import_parser.add_argument("--input", type=Path)
    import_parser.add_argument("--input-dir", type=Path)
    import_parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command != "import":
        raise ValueError(f"unsupported command {args.command!r}")
    count = import_corpus(
        source_type=args.source_type,
        input_path=args.input,
        input_dir=args.input_dir,
        output=args.output,
    )
    print(json.dumps({"output": str(args.output), "segments": count, "source_type": args.source_type}, sort_keys=True))
    return 0


def main() -> None:
    try:
        raise SystemExit(run_cli())
    except ValueError as exc:
        print(str(exc))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
