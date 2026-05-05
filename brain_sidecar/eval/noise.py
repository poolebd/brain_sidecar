from __future__ import annotations

import random
import re
from dataclasses import replace

from brain_sidecar.eval.models import EvalTranscriptEvent


DEFAULT_CORRUPTION_MAPPING = {
    "Siemens": ["demons"],
    "NERC": ["narc", "nerd side"],
    "SLD": ["s l d", "single line"],
    "six to eight hours": ["six to eight hours", "6 to 8 hours", "photo peak"],
    "Greg": ["Craig"],
    "queue reviewer": ["cue reviewer", "q reviewer"],
}


def duplicate_overlap(segment: EvalTranscriptEvent) -> EvalTranscriptEvent:
    return replace(
        segment,
        id=f"{segment.id}_overlap",
        start_s=segment.start_s + 0.4,
        end_s=segment.end_s + 0.4,
        text=segment.text,
    )


def fragment_segment(segment: EvalTranscriptEvent) -> list[EvalTranscriptEvent]:
    words = segment.text.split()
    if len(words) < 6:
        return [segment]
    midpoint = max(2, len(words) // 2)
    split_s = segment.start_s + max(0.3, (segment.end_s - segment.start_s) / 2)
    return [
        replace(segment, id=f"{segment.id}_frag_a", end_s=split_s, text=" ".join(words[:midpoint])),
        replace(segment, id=f"{segment.id}_frag_b", start_s=split_s, text=" ".join(words[midpoint:])),
    ]


def repeat_tail(segment: EvalTranscriptEvent, *, word_count: int = 4) -> EvalTranscriptEvent:
    words = segment.text.split()
    if not words:
        return segment
    tail = " ".join(words[-word_count:])
    return replace(segment, text=f"{segment.text} {tail}")


def corrupt_terms(
    segment: EvalTranscriptEvent,
    mapping: dict[str, list[str]] | None = None,
    *,
    seed: int = 0,
) -> EvalTranscriptEvent:
    mapping = mapping or DEFAULT_CORRUPTION_MAPPING
    rng = random.Random(seed)
    text = segment.text
    for source, replacements in mapping.items():
        if not replacements:
            continue
        replacement = replacements[rng.randrange(len(replacements))]
        text = re.sub(rf"\b{re.escape(source)}\b", replacement, text, flags=re.IGNORECASE)
    return replace(segment, text=text)


def drop_words(segment: EvalTranscriptEvent, probability: float, seed: int) -> EvalTranscriptEvent:
    rng = random.Random(seed)
    words = segment.text.split()
    kept = [word for word in words if rng.random() >= probability]
    return replace(segment, text=" ".join(kept or words[:1]))


def inject_filler(segment: EvalTranscriptEvent, *, seed: int = 0) -> EvalTranscriptEvent:
    rng = random.Random(seed)
    fillers = ["um", "you know", "kind of", "sort of"]
    words = segment.text.split()
    if not words:
        return segment
    index = rng.randrange(len(words))
    words.insert(index, fillers[rng.randrange(len(fillers))])
    return replace(segment, text=" ".join(words))


def scramble_case_punctuation(segment: EvalTranscriptEvent) -> EvalTranscriptEvent:
    text = re.sub(r"[,.?!;:]", "", segment.text)
    words = [word.upper() if index % 5 == 0 else word.lower() for index, word in enumerate(text.split())]
    return replace(segment, text=" ".join(words))


def apply_default_noise(events: list[EvalTranscriptEvent], *, seed: int = 7) -> list[EvalTranscriptEvent]:
    noisy: list[EvalTranscriptEvent] = []
    for index, event in enumerate(events):
        transformed = event
        if index % 2 == 0:
            transformed = corrupt_terms(transformed, seed=seed + index)
        if index % 3 == 0:
            transformed = inject_filler(transformed, seed=seed + index)
        if index % 4 == 0:
            transformed = repeat_tail(transformed)
        if index % 5 == 0:
            noisy.append(duplicate_overlap(transformed))
        noisy.append(transformed)
    return sorted(noisy, key=lambda item: (item.start_s, item.id))
