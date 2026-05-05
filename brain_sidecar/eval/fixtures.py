from __future__ import annotations

import argparse
import random
from pathlib import Path

from brain_sidecar.eval.models import EvalMemoryEvent, EvalTranscriptEvent


ENERGY_TOPICS = [
    "The client asked a power quality question about Siemens equipment and the generation agreement exhibit.",
    "The harmonics study needs to line up with the SLD before the answer goes back to the client.",
    "Sunil is the reviewer and the right communication path if the client needs a response.",
    "Manish may be the SME for the NERC or NARC technical area, so ask him for context.",
    "Confirm six to eight hours a week, roughly one day, for the next few weeks of project support.",
    "Tomorrow after the morning meeting, clarify whether eleven Eastern means ten your time.",
]

DISTRACTORS = [
    "Brandon is going to contribute on this continuous basis.",
    "Grass or something on the back end came through as a noisy fragment.",
    "Project Alpha timeline was mentioned only as a generic placeholder.",
    "A memory-only relay example should not create CT/PT or breaker failure current notes.",
    "Remote work policy is not part of this discussion.",
]


def synth_energy_events(*, count: int, seed: int, session_id: str = "synthetic_energy") -> list[EvalTranscriptEvent | EvalMemoryEvent]:
    rng = random.Random(seed)
    events: list[EvalTranscriptEvent | EvalMemoryEvent] = []
    start_s = 0.0
    for index in range(count):
        pool = ENERGY_TOPICS if index % 4 != 3 else DISTRACTORS
        text = rng.choice(pool)
        end_s = start_s + max(2.0, min(7.0, len(text.split()) * 0.42))
        events.append(
            EvalTranscriptEvent(
                id=f"seg_{index + 1:03d}",
                session_id=session_id,
                start_s=start_s,
                end_s=end_s,
                text=text,
            )
        )
        start_s = end_s + 0.4
    events.append(
        EvalMemoryEvent(
            id="mem_relay",
            title="Relay replacement trip-path reminder",
            body="Past relay work required CT/PT inputs, settings, interlocks, and breaker-failure checks.",
            forbidden_in_current_notes=["CT/PT", "relay settings", "trip path", "breaker failure", "interlocks"],
        )
    )
    return events


def write_events(events: list[EvalTranscriptEvent | EvalMemoryEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(event.to_json() for event in events) + "\n", encoding="utf-8")


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic Brain Sidecar eval fixtures.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synth = subparsers.add_parser("synth-energy")
    synth.add_argument("--count", type=int, default=25)
    synth.add_argument("--output", type=Path, required=True)
    synth.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.command != "synth-energy":
        raise ValueError(f"unsupported command {args.command!r}")
    events = synth_energy_events(count=max(1, args.count), seed=args.seed)
    write_events(events, args.output)
    print(f"wrote {len(events)} events to {args.output}")
    return 0


def main() -> None:
    raise SystemExit(run_cli())


if __name__ == "__main__":
    main()
