from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from brain_sidecar.core.transcription import transcript_fingerprint


_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class SeenTranscript:
    fingerprint: str
    text: str
    tokens: set[str]
    start_s: float
    end_s: float


class TranscriptDeduplicator:
    """Suppress near-duplicates caused by overlapping transcription windows."""

    def __init__(self, max_recent: int, similarity_threshold: float) -> None:
        self.max_recent = max_recent
        self.similarity_threshold = similarity_threshold
        self._recent: deque[SeenTranscript] = deque(maxlen=max_recent)

    def accept(self, text: str, start_s: float, end_s: float) -> bool:
        normalized = normalize_for_dedupe(text)
        if not normalized:
            return False
        fingerprint = transcript_fingerprint(normalized)
        tokens = set(tokenize(normalized))
        if not tokens:
            return False

        candidate = SeenTranscript(
            fingerprint=fingerprint,
            text=normalized,
            tokens=tokens,
            start_s=start_s,
            end_s=end_s,
        )
        if any(is_duplicate(candidate, seen, self.similarity_threshold) for seen in self._recent):
            return False
        self._recent.append(candidate)
        return True


def normalize_for_dedupe(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .,!?:;\"'")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def is_duplicate(candidate: SeenTranscript, seen: SeenTranscript, threshold: float) -> bool:
    if candidate.fingerprint == seen.fingerprint:
        return True
    if candidate.end_s < seen.start_s - 1.0 or candidate.start_s > seen.end_s + 6.0:
        return False
    if candidate.text in seen.text or seen.text in candidate.text:
        return True
    similarity = jaccard(candidate.tokens, seen.tokens)
    containment = token_containment(candidate.tokens, seen.tokens)
    return similarity >= threshold or containment >= threshold


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def token_containment(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))
