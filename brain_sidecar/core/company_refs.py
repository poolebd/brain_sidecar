from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Iterable

from brain_sidecar.config import Settings
from brain_sidecar.core.asr_aliases import normalize_for_evidence_match
from brain_sidecar.core.models import TranscriptSegment, compact_text
from brain_sidecar.core.storage import Storage


DEFAULT_COMPANY_REF_SEED = "company_refs.seed.jsonl"
CONTEXT_WINDOW_SEGMENTS = 2


@dataclass(frozen=True)
class CompanyAlias:
    id: str
    ref_id: str
    alias: str
    normalized_alias: str
    alias_type: str
    weight: float = 1.0
    requires_context: bool = False
    context_terms: list[str] = field(default_factory=list)
    negative_terms: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompanyRef:
    id: str
    canonical_name: str
    entity_type: str = "company"
    domain: str = ""
    description: str = ""
    website: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, str]] = field(default_factory=list)
    active: bool = True
    updated_at: float = field(default_factory=time.time)
    aliases: list[CompanyAlias] = field(default_factory=list)


@dataclass(frozen=True)
class CompanyMention:
    ref_id: str
    canonical_name: str
    entity_type: str
    domain: str
    description: str
    matched_text: str
    alias: str
    alias_type: str
    confidence: float
    source_segment_ids: list[str]
    evidence_quote: str
    sources: list[dict[str, str]]
    metadata: dict[str, Any]


class CompanyRefService:
    def __init__(self, storage: Storage, settings: Settings) -> None:
        self.storage = storage
        self.settings = settings
        self._refs: dict[str, CompanyRef] | None = None
        self._aliases: list[CompanyAlias] | None = None
        self.last_import_error: str | None = None

    def ensure_seeded(self) -> dict[str, Any]:
        if not self.settings.company_refs_enabled:
            return self.status()
        status = self.storage.company_ref_status()
        if status["ref_count"] == 0:
            return self.reload()
        self.refresh()
        return self.status()

    def reload(self) -> dict[str, Any]:
        refs = load_company_ref_seed(self.settings.company_refs_seed_path)
        imported_count = self.storage.upsert_company_refs(refs)
        self.refresh()
        self.last_import_error = None
        return {**self.status(), "imported_count": imported_count}

    def refresh(self) -> None:
        refs = [company_ref_from_row(row) for row in self.storage.company_refs(active_only=True)]
        self._refs = {ref.id: ref for ref in refs}
        self._aliases = [alias for ref in refs for alias in ref.aliases]

    def status(self) -> dict[str, Any]:
        seed_path = self.settings.company_refs_seed_path
        return {
            **self.storage.company_ref_status(),
            "enabled": self.settings.company_refs_enabled,
            "seed_path": str(seed_path) if seed_path else f"package:{DEFAULT_COMPANY_REF_SEED}",
            "min_confidence": self.settings.company_refs_min_confidence,
            "max_live_cards": self.settings.company_refs_max_live_cards,
            "duplicate_window_seconds": self.settings.company_refs_duplicate_window_seconds,
            "last_import_error": self.last_import_error,
            "raw_audio_retained": False,
        }

    def search(self, query: str, limit: int = 12) -> list[dict[str, Any]]:
        return self.storage.search_company_refs(query, limit=limit)

    def mentions_for_query(self, query: str, *, limit: int = 12) -> list[CompanyMention]:
        segment = TranscriptSegment(
            id="",
            session_id="manual",
            start_s=0.0,
            end_s=0.0,
            text=query,
            is_final=True,
        )
        return self.match_segments(segment_list=[segment], max_mentions=limit)

    def match_segments(
        self,
        segment_list: Iterable[TranscriptSegment],
        *,
        min_confidence: float | None = None,
        max_mentions: int | None = None,
    ) -> list[CompanyMention]:
        if not self.settings.company_refs_enabled:
            return []
        if self._refs is None or self._aliases is None:
            self.refresh()
        refs = self._refs or {}
        aliases = self._aliases or []
        if not refs or not aliases:
            return []

        threshold = self.settings.company_refs_min_confidence if min_confidence is None else min_confidence
        limit = self.settings.company_refs_max_live_cards if max_mentions is None else max_mentions
        if limit <= 0:
            return []

        segments = [segment for segment in segment_list if segment.is_final and segment.text.strip()]
        normalized_segments = [normalize_company_text(segment.text) for segment in segments]
        best_by_ref: dict[str, tuple[CompanyMention, int]] = {}
        for index, segment in enumerate(segments):
            segment_text = normalized_segments[index]
            if not segment_text:
                continue
            context_text = _context_text(normalized_segments, index)
            for alias in aliases:
                ref = refs.get(alias.ref_id)
                if ref is None or not _alias_present(alias, segment_text):
                    continue
                if _has_any_phrase(alias.negative_terms, context_text):
                    continue
                context_matched = _has_any_phrase(alias.context_terms, context_text)
                confidence = confidence_for_alias(alias, ref, context_matched)
                if confidence < threshold:
                    continue
                mention = CompanyMention(
                    ref_id=ref.id,
                    canonical_name=ref.canonical_name,
                    entity_type=ref.entity_type,
                    domain=ref.domain,
                    description=ref.description,
                    matched_text=alias.alias,
                    alias=alias.alias,
                    alias_type=alias.alias_type,
                    confidence=confidence,
                    source_segment_ids=_source_ids_for_segment(segment),
                    evidence_quote=compact_text(segment.text, limit=420),
                    sources=ref.sources,
                    metadata={**ref.metadata, "matched_alias_type": alias.alias_type},
                )
                existing = best_by_ref.get(ref.id)
                if existing is None or _mention_sort_key(mention, index) > _mention_sort_key(existing[0], existing[1]):
                    best_by_ref[ref.id] = (mention, index)

        mentions = [item[0] for item in best_by_ref.values()]
        mentions.sort(key=lambda mention: (mention.confidence, bool(mention.source_segment_ids)), reverse=True)
        return mentions[:limit]


def normalize_company_text(text: str) -> str:
    clean = normalize_for_evidence_match(text)
    clean = clean.replace("/", " ").replace(".", " ")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def load_company_ref_seed(seed_path: Path | None = None) -> list[CompanyRef]:
    rows = _seed_lines(seed_path)
    refs: list[CompanyRef] = []
    now = time.time()
    for line_number, line in rows:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid company refs JSONL at line {line_number}: {exc}") from exc
        refs.append(company_ref_from_seed_row(payload, updated_at=now))
    return refs


def company_ref_from_seed_row(payload: dict[str, Any], *, updated_at: float | None = None) -> CompanyRef:
    ref_id = compact_text(payload.get("id"), limit=120)
    canonical_name = compact_text(payload.get("canonical_name"), limit=180)
    description = compact_text(payload.get("description"), limit=1000)
    if not ref_id or not canonical_name or not description:
        raise ValueError("Company reference seed rows require id, canonical_name, and description.")
    context_terms = _string_list(payload.get("context_terms"))
    negative_terms = _string_list(payload.get("negative_terms"))
    aliases = build_aliases_from_seed(
        ref_id=ref_id,
        canonical_name=canonical_name,
        aliases=_string_list(payload.get("aliases")),
        acronyms=_string_list(payload.get("acronyms")),
        asr_variants=_string_list(payload.get("asr_variants")),
        former_names=_string_list(payload.get("former_names")),
        shorthands=_string_list(payload.get("shorthands")),
        context_terms=context_terms,
        negative_terms=negative_terms,
    )
    return CompanyRef(
        id=ref_id,
        canonical_name=canonical_name,
        entity_type=compact_text(payload.get("entity_type") or "company", limit=80) or "company",
        domain=compact_text(payload.get("domain") or "", limit=240),
        description=description,
        website=compact_text(payload.get("website") or "", limit=500) or None,
        metadata=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
        sources=_sources(payload.get("sources")),
        active=bool(payload.get("active", True)),
        updated_at=time.time() if updated_at is None else updated_at,
        aliases=aliases,
    )


def company_ref_from_row(row: dict[str, Any]) -> CompanyRef:
    aliases = [
        CompanyAlias(
            id=str(alias["id"]),
            ref_id=str(alias["ref_id"]),
            alias=str(alias["alias"]),
            normalized_alias=str(alias["normalized_alias"]),
            alias_type=str(alias["alias_type"]),
            weight=float(alias.get("weight", 1.0)),
            requires_context=bool(alias.get("requires_context", False)),
            context_terms=_string_list(alias.get("context_terms")),
            negative_terms=_string_list(alias.get("negative_terms")),
        )
        for alias in row.get("aliases", [])
        if isinstance(alias, dict)
    ]
    return CompanyRef(
        id=str(row["id"]),
        canonical_name=str(row["canonical_name"]),
        entity_type=str(row.get("entity_type") or "company"),
        domain=str(row.get("domain") or ""),
        description=str(row.get("description") or ""),
        website=str(row["website"]) if row.get("website") else None,
        metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        sources=_sources(row.get("sources")),
        active=bool(row.get("active", True)),
        updated_at=float(row.get("updated_at") or 0.0),
        aliases=aliases,
    )


def build_aliases_from_seed(
    *,
    ref_id: str,
    canonical_name: str,
    aliases: list[str],
    acronyms: list[str],
    asr_variants: list[str],
    former_names: list[str],
    shorthands: list[str],
    context_terms: list[str],
    negative_terms: list[str],
) -> list[CompanyAlias]:
    alias_specs: list[tuple[str, str, bool]] = [(canonical_name, "name", False)]
    alias_specs.extend((alias, "name", False) for alias in aliases)
    alias_specs.extend((alias, "former_name", False) for alias in former_names)
    alias_specs.extend((alias, "shorthand", _is_short_alias(alias)) for alias in shorthands)
    alias_specs.extend((alias, "acronym", True) for alias in acronyms)
    alias_specs.extend((alias, "asr_variant", True) for alias in asr_variants)

    result: list[CompanyAlias] = []
    seen: set[tuple[str, str]] = set()
    normalized_context = [normalize_company_text(term) for term in context_terms if normalize_company_text(term)]
    normalized_negative = [normalize_company_text(term) for term in negative_terms if normalize_company_text(term)]
    for alias, alias_type, requires_context in alias_specs:
        alias_text = compact_text(alias, limit=180)
        normalized = normalize_company_text(alias_text)
        if not alias_text or not normalized:
            continue
        key = (normalized, alias_type)
        if key in seen:
            continue
        seen.add(key)
        if _is_short_alias(alias_text) and alias_type != "name":
            requires_context = True
        result.append(
            CompanyAlias(
                id=stable_alias_id(ref_id, alias_type, normalized),
                ref_id=ref_id,
                alias=alias_text,
                normalized_alias=normalized,
                alias_type=alias_type,
                weight=1.0,
                requires_context=requires_context,
                context_terms=normalized_context,
                negative_terms=normalized_negative,
            )
        )
    return result


def confidence_for_alias(alias: CompanyAlias, ref: CompanyRef, context_matched: bool) -> float:
    canonical = normalize_company_text(ref.canonical_name)
    short_alias = _is_short_normalized(alias.normalized_alias)
    requires_context = alias.requires_context or alias.alias_type == "asr_variant" or (
        alias.alias_type == "acronym" and short_alias
    )
    if requires_context and not context_matched:
        return 0.55 if alias.alias_type == "acronym" else 0.45

    if alias.alias_type == "name" and alias.normalized_alias == canonical:
        base = 0.94
    elif alias.alias_type in {"name", "former_name"}:
        base = 0.86
    elif alias.alias_type == "shorthand":
        base = 0.78
    elif alias.alias_type == "acronym":
        base = 0.76
    elif alias.alias_type == "asr_variant":
        base = 0.72
    else:
        base = 0.70
    if context_matched:
        base += 0.04
    base += max(-0.12, min(0.12, (alias.weight - 1.0) * 0.08))
    return max(0.0, min(1.0, base))


def stable_alias_id(ref_id: str, alias_type: str, normalized_alias: str) -> str:
    digest = hashlib.sha1(f"{ref_id}:{alias_type}:{normalized_alias}".encode("utf-8")).hexdigest()[:12]
    return f"calias_{digest}"


def _seed_lines(seed_path: Path | None) -> list[tuple[int, str]]:
    if seed_path is not None:
        path = seed_path.expanduser()
        return list(enumerate(path.read_text(encoding="utf-8").splitlines(), start=1))
    seed = resources.files("brain_sidecar.resources").joinpath(DEFAULT_COMPANY_REF_SEED)
    return list(enumerate(seed.read_text(encoding="utf-8").splitlines(), start=1))


def _alias_present(alias: CompanyAlias, normalized_text: str) -> bool:
    variants = [alias.normalized_alias]
    compact_alias = alias.normalized_alias.replace(" ", "")
    if alias.alias_type == "acronym" and compact_alias.isalnum() and 2 <= len(compact_alias) <= 8:
        variants.append(" ".join(compact_alias))
    return any(_phrase_present(variant, normalized_text) for variant in variants)


def _has_any_phrase(phrases: list[str], normalized_text: str) -> bool:
    return any(_phrase_present(normalize_company_text(phrase), normalized_text) for phrase in phrases)


def _phrase_present(phrase: str, normalized_text: str) -> bool:
    clean_phrase = normalize_company_text(phrase)
    if not clean_phrase or not normalized_text:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(clean_phrase) + r"(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def _context_text(normalized_segments: list[str], index: int) -> str:
    start = max(0, index - CONTEXT_WINDOW_SEGMENTS)
    end = min(len(normalized_segments), index + CONTEXT_WINDOW_SEGMENTS + 1)
    return " ".join(normalized_segments[start:end])


def _source_ids_for_segment(segment: TranscriptSegment) -> list[str]:
    result: list[str] = []
    for value in [segment.id, *segment.source_segment_ids]:
        if value and value not in result:
            result.append(value)
    return result


def _mention_sort_key(mention: CompanyMention, index: int) -> tuple[float, int, int]:
    return mention.confidence, index, len(mention.evidence_quote)


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [compact_text(item, limit=240) for item in value if compact_text(item, limit=240)]


def _sources(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    sources: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        title = compact_text(item.get("title"), limit=180)
        url = compact_text(item.get("url"), limit=500)
        path = compact_text(item.get("path"), limit=500)
        if not title:
            continue
        source = {"title": title}
        if url:
            source["url"] = url
        if path:
            source["path"] = path
        sources.append(source)
    return sources[:6]


def _is_short_alias(alias: str) -> bool:
    return _is_short_normalized(normalize_company_text(alias))


def _is_short_normalized(alias: str) -> bool:
    return len(alias.replace(" ", "")) <= 3
