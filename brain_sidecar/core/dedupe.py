from __future__ import annotations

import re
from difflib import SequenceMatcher
from collections import deque
from dataclasses import dataclass

from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.transcription import transcript_fingerprint


_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class SeenTranscript:
    fingerprint: str
    text: str
    tokens: set[str]
    start_s: float
    end_s: float


@dataclass(frozen=True)
class TranscriptConsolidationResult:
    segment: TranscriptSegment | None
    collapsed: bool = False
    suppressed: bool = False
    replaced_segment_id: str | None = None


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


class TranscriptFinalConsolidator:
    """Build a cleaner rolling evidence window from final transcript events."""

    def __init__(self, max_recent: int = 48) -> None:
        self.max_recent = max(4, int(max_recent))
        self._segments: list[TranscriptSegment] = []

    def accept(self, segment: TranscriptSegment) -> TranscriptConsolidationResult:
        normalized = normalize_for_dedupe(segment.text)
        if not normalized:
            return TranscriptConsolidationResult(segment=None, suppressed=True)
        for index in range(len(self._segments) - 1, -1, -1):
            existing = self._segments[index]
            if not should_consolidate(segment, existing):
                continue
            merged = merge_transcript_segments(existing, segment)
            if normalize_for_dedupe(merged.text) == normalize_for_dedupe(existing.text):
                self._segments[index] = merged
                return TranscriptConsolidationResult(
                    segment=merged,
                    collapsed=True,
                    suppressed=True,
                    replaced_segment_id=existing.id,
                )
            self._segments[index] = merged
            return TranscriptConsolidationResult(
                segment=merged,
                collapsed=True,
                suppressed=False,
                replaced_segment_id=existing.id,
            )
        self._segments.append(
            TranscriptSegment(
                id=segment.id,
                session_id=segment.session_id,
                start_s=segment.start_s,
                end_s=segment.end_s,
                text=segment.text,
                is_final=segment.is_final,
                created_at=segment.created_at,
                speaker_role=segment.speaker_role,
                speaker_label=segment.speaker_label,
                speaker_confidence=segment.speaker_confidence,
                speaker_match_reason=segment.speaker_match_reason,
                speaker_low_confidence=segment.speaker_low_confidence,
                source_segment_ids=segment.source_segment_ids or [segment.id],
            )
        )
        self._segments = self._segments[-self.max_recent :]
        return TranscriptConsolidationResult(segment=self._segments[-1])

    def segments(self) -> list[TranscriptSegment]:
        return list(self._segments)


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


def should_consolidate(candidate: TranscriptSegment, existing: TranscriptSegment) -> bool:
    if candidate.session_id != existing.session_id:
        return False
    if candidate.start_s > existing.end_s + 3.0 or candidate.end_s < existing.start_s - 3.0:
        return False
    candidate_text = normalize_for_dedupe(candidate.text)
    existing_text = normalize_for_dedupe(existing.text)
    if not candidate_text or not existing_text:
        return False
    if candidate_text in existing_text or existing_text in candidate_text:
        return True
    if time_overlap_ratio(candidate.start_s, candidate.end_s, existing.start_s, existing.end_s) > 0.60:
        return True
    candidate_tokens = set(tokenize(candidate_text))
    existing_tokens = set(tokenize(existing_text))
    if jaccard(candidate_tokens, existing_tokens) > 0.80:
        return True
    return SequenceMatcher(None, candidate_text, existing_text).ratio() > 0.84


def merge_transcript_segments(existing: TranscriptSegment, candidate: TranscriptSegment) -> TranscriptSegment:
    existing_text = normalize_for_dedupe(existing.text)
    candidate_text = normalize_for_dedupe(candidate.text)
    chosen = candidate if cleaner_text_score(candidate.text) > cleaner_text_score(existing.text) else existing
    if existing_text and candidate_text and candidate_text not in existing_text and existing_text not in candidate_text:
        if abs(existing.end_s - candidate.start_s) <= 1.0 and len(candidate.text) > len(existing.text) * 0.35:
            text = f"{existing.text.rstrip()} {candidate.text.strip()}"
        else:
            text = chosen.text
    else:
        text = chosen.text
    source_ids = dedupe_source_ids([*source_ids_for(existing), *source_ids_for(candidate)])
    return TranscriptSegment(
        id=chosen.id,
        session_id=existing.session_id,
        start_s=min(existing.start_s, candidate.start_s),
        end_s=max(existing.end_s, candidate.end_s),
        text=text,
        is_final=True,
        created_at=max(existing.created_at, candidate.created_at),
        speaker_role=chosen.speaker_role,
        speaker_label=chosen.speaker_label,
        speaker_confidence=chosen.speaker_confidence,
        speaker_match_reason=chosen.speaker_match_reason,
        speaker_low_confidence=chosen.speaker_low_confidence,
        source_segment_ids=source_ids,
    )


def time_overlap_ratio(left_start: float, left_end: float, right_start: float, right_end: float) -> float:
    overlap = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    shortest = max(0.001, min(max(0.001, left_end - left_start), max(0.001, right_end - right_start)))
    return overlap / shortest


def cleaner_text_score(text: str) -> tuple[int, int]:
    normalized = normalize_for_dedupe(text)
    tokens = tokenize(normalized)
    alpha_count = sum(1 for char in text if char.isalpha())
    return (len(set(tokens)), alpha_count)


def source_ids_for(segment: TranscriptSegment) -> list[str]:
    return segment.source_segment_ids or [segment.id]


def dedupe_source_ids(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def token_containment(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))
