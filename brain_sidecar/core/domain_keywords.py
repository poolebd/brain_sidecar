from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from brain_sidecar.core.models import TranscriptSegment, compact_text


ENERGY_KEYWORD_SOURCE = "energy_keyword_report"
ENERGY_LEXICON_PACKAGE = "brain_sidecar.data"
ENERGY_LEXICON_RESOURCE = "energy_keywords.json"
ENERGY_RESEARCH_SECTIONS = {
    "Highest-weight phrases",
    "Core advisory, audit, operations, and sales keywords",
    "Market, technology, systems, and flexibility keywords",
    "Sustainability, finance, standards, contracting, and pain-point keywords",
}
ENERGY_CATEGORIES = {
    "commercial",
    "service",
    "operations",
    "technical",
    "technology",
    "market",
    "sustainability",
    "finance",
    "regulatory",
}
INTENT_ORDER = ["informational", "transactional", "lead-gen", "support", "compliance", "complaint"]
INTENT_TAGS = set(INTENT_ORDER)
AMBIGUOUS_ACRONYMS = {
    "bas",
    "bess",
    "chp",
    "der",
    "derms",
    "dr",
    "eac",
    "ecm",
    "emis",
    "enpi",
    "espc",
    "esco",
    "eui",
    "fdd",
    "iga",
    "ipmvp",
    "irr",
    "macc",
    "npv",
    "ppa",
    "rcx",
    "rec",
    "roi",
    "rtp",
    "sem",
    "seu",
    "tou",
    "vpp",
}
COMMERCIAL_ENERGY_ANCHORS = {
    "audit",
    "bill",
    "building",
    "carbon",
    "charge",
    "cost",
    "demand",
    "electric",
    "electricity",
    "emissions",
    "energy",
    "facility",
    "load",
    "power",
    "procurement",
    "rate",
    "renewable",
    "savings",
    "scope",
    "tariff",
    "utility",
}
STRONG_DIRECT_PHRASES = {
    "energy audit",
    "utility bill analysis",
    "energy procurement",
    "decarbonization roadmap",
    "demand response",
    "tariff analysis",
    "demand charge management",
    "iso 50001",
}


@dataclass(frozen=True)
class KeywordEntry:
    phrase: str
    normalized_token: str
    category: str
    subcategory: str
    intent_tags: list[str]
    synonyms: list[str]
    weight: float
    ambiguous: bool
    requires_context: bool
    source_section: str
    source: str = ENERGY_KEYWORD_SOURCE

    def to_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.phrase,
            "normalized_token": self.normalized_token,
            "category": self.category,
            "subcategory": self.subcategory,
            "intent_tags": self.intent_tags,
            "synonyms": self.synonyms,
            "weight": self.weight,
            "ambiguous": self.ambiguous,
            "requires_context": self.requires_context,
            "source_section": self.source_section,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KeywordEntry":
        return cls(
            phrase=str(payload["phrase"]),
            normalized_token=str(payload["normalized_token"]),
            category=str(payload["category"]),
            subcategory=str(payload["subcategory"]),
            intent_tags=[str(item) for item in payload.get("intent_tags", [])],
            synonyms=[str(item) for item in payload.get("synonyms", [])],
            weight=float(payload["weight"]),
            ambiguous=bool(payload.get("ambiguous", False)),
            requires_context=bool(payload.get("requires_context", False)),
            source_section=str(payload["source_section"]),
            source=str(payload.get("source", ENERGY_KEYWORD_SOURCE)),
        )


@dataclass(frozen=True)
class KeywordHit:
    entry: KeywordEntry
    matched_text: str
    matched_variant: str
    start_char: int
    end_char: int
    segment_id: str
    segment_text: str
    weight: float
    variant_requires_context: bool = False

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "phrase": self.entry.phrase,
            "normalized_token": self.entry.normalized_token,
            "category": self.entry.category,
            "subcategory": self.entry.subcategory,
            "matched_variant": self.matched_variant,
            "weight": round(self.weight, 3),
            "segment_id": self.segment_id,
        }


EnergyConfidence = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class EnergyConversationFrame:
    active: bool
    score: float
    confidence: EnergyConfidence
    top_categories: list[dict[str, Any]]
    top_keywords: list[dict[str, Any]]
    intent_tags: list[str]
    evidence_segment_ids: list[str]
    evidence_quote: str
    summary_label: str
    raw_audio_retained: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": self.active,
            "score": round(self.score, 3),
            "confidence": self.confidence,
            "top_categories": self.top_categories,
            "top_keywords": self.top_keywords,
            "intent_tags": self.intent_tags,
            "evidence_segment_ids": self.evidence_segment_ids,
            "evidence_quote": self.evidence_quote,
            "summary_label": self.summary_label,
            "raw_audio_retained": False,
        }


def inactive_energy_frame() -> EnergyConversationFrame:
    return EnergyConversationFrame(
        active=False,
        score=0.0,
        confidence="low",
        top_categories=[],
        top_keywords=[],
        intent_tags=[],
        evidence_segment_ids=[],
        evidence_quote="",
        summary_label="Energy lens inactive",
        raw_audio_retained=False,
    )


def normalize_keyword_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"\bm\s+and\s+v\b", "mv measurement and verification", text)
    text = re.sub(r"['`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_energy_keyword_report(markdown: str) -> list[KeywordEntry]:
    later_by_token: dict[str, list[KeywordEntry]] = defaultdict(list)
    raw_entries: list[KeywordEntry] = []
    for section, rows in _section_tables(markdown):
        if section not in ENERGY_RESEARCH_SECTIONS:
            continue
        for row in rows:
            entry = _entry_from_row(section, row)
            if entry is None:
                continue
            raw_entries.append(entry)
            if section != "Highest-weight phrases":
                later_by_token[entry.normalized_token].append(entry)

    enriched: list[KeywordEntry] = []
    for entry in raw_entries:
        if entry.source_section != "Highest-weight phrases":
            enriched.append(entry)
            continue
        inferred = later_by_token.get(entry.normalized_token, [])
        if inferred:
            best = max(inferred, key=lambda candidate: candidate.weight)
            enriched.append(
                KeywordEntry(
                    phrase=entry.phrase,
                    normalized_token=entry.normalized_token,
                    category=best.category,
                    subcategory=best.subcategory,
                    intent_tags=best.intent_tags,
                    synonyms=best.synonyms,
                    weight=entry.weight,
                    ambiguous=entry.ambiguous,
                    requires_context=entry.requires_context,
                    source_section=entry.source_section,
                )
            )
        else:
            category, subcategory = _infer_category(entry.normalized_token)
            enriched.append(
                KeywordEntry(
                    phrase=entry.phrase,
                    normalized_token=entry.normalized_token,
                    category=category,
                    subcategory=subcategory,
                    intent_tags=_default_intents_for_category(category),
                    synonyms=[],
                    weight=entry.weight,
                    ambiguous=entry.ambiguous,
                    requires_context=entry.requires_context,
                    source_section=entry.source_section,
                )
            )
    return dedupe_keyword_entries(enriched)


def dedupe_keyword_entries(entries: list[KeywordEntry]) -> list[KeywordEntry]:
    grouped: dict[tuple[str, str], KeywordEntry] = {}
    for entry in entries:
        key = (entry.normalized_token, entry.category)
        existing = grouped.get(key)
        if existing is None:
            grouped[key] = entry
            continue
        synonyms = _sorted_unique([*existing.synonyms, *entry.synonyms])
        intents = _sorted_unique([*existing.intent_tags, *entry.intent_tags], order=INTENT_ORDER)
        source_section = existing.source_section
        if entry.source_section not in source_section:
            source_section = f"{source_section}; {entry.source_section}"
        keep = existing if existing.weight >= entry.weight else entry
        grouped[key] = KeywordEntry(
            phrase=keep.phrase,
            normalized_token=keep.normalized_token,
            category=keep.category,
            subcategory=keep.subcategory,
            intent_tags=intents,
            synonyms=synonyms,
            weight=max(existing.weight, entry.weight),
            ambiguous=existing.ambiguous or entry.ambiguous,
            requires_context=existing.requires_context or entry.requires_context,
            source_section=source_section,
        )
    return sorted(grouped.values(), key=lambda item: (item.category, item.normalized_token, item.phrase))


def load_energy_lexicon(path: Path | None = None) -> list[KeywordEntry]:
    if path is None:
        with resources.files(ENERGY_LEXICON_PACKAGE).joinpath(ENERGY_LEXICON_RESOURCE).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
    entries_payload = payload.get("entries", payload) if isinstance(payload, dict) else payload
    return [KeywordEntry.from_dict(item) for item in entries_payload]


def lexicon_payload(entries: list[KeywordEntry], *, source_path: str) -> dict[str, Any]:
    return {
        "source": source_path,
        "source_type": ENERGY_KEYWORD_SOURCE,
        "entry_count": len(entries),
        "entries": [entry.to_dict() for entry in sorted(entries, key=lambda item: (item.category, item.normalized_token, item.phrase))],
    }


class EnergyConversationDetector:
    def __init__(
        self,
        entries: list[KeywordEntry] | None = None,
        *,
        enabled: bool = True,
        min_confidence: str = "medium",
        max_keywords: int = 6,
        window_segments: int = 12,
        window_seconds: float = 90.0,
    ) -> None:
        self.enabled = enabled
        self.min_confidence = _normalize_min_confidence(min_confidence)
        self.max_keywords = max(1, int(max_keywords))
        self.window_segments = max(1, int(window_segments))
        self.window_seconds = max(1.0, float(window_seconds))
        self.entries = entries if entries is not None else load_energy_lexicon()
        self._variants = _compiled_variants(self.entries)

    @property
    def keyword_count(self) -> int:
        return len(self.entries)

    def detect(
        self,
        recent_segments: list[TranscriptSegment],
        *,
        manual_query: str | None = None,
        meeting_contract: object | None = None,
        save_transcript: bool | None = None,
    ) -> EnergyConversationFrame:
        del manual_query, meeting_contract, save_transcript
        if not self.enabled:
            return inactive_energy_frame()
        final_segments = [segment for segment in recent_segments if segment.is_final and segment.text.strip()]
        if not final_segments:
            return inactive_energy_frame()
        window = self._window(final_segments)
        raw_hits = self._hits(window)
        usable_hits = _usable_hits(raw_hits, window)
        if not usable_hits:
            return inactive_energy_frame()

        capped_hits = _cap_duplicate_hits(usable_hits)
        score = round(sum(hit.weight for hit in capped_hits), 3)
        top_categories = _top_categories(capped_hits)
        top_keywords = [hit.to_public_dict() for hit in sorted(capped_hits, key=lambda item: (-item.weight, item.entry.phrase))[: self.max_keywords]]
        intent_tags = _top_intents(capped_hits)
        evidence_segment_ids = _unique([hit.segment_id for hit in sorted(capped_hits, key=lambda item: -item.weight)[: self.max_keywords]])
        evidence_quote = _evidence_quote(window, evidence_segment_ids)
        confidence = _confidence(score, capped_hits, top_categories)
        active = _confidence_rank(confidence) >= _confidence_rank(self.min_confidence)
        summary_label = _summary_label(active, top_categories)
        return EnergyConversationFrame(
            active=active,
            score=score,
            confidence=confidence,
            top_categories=top_categories,
            top_keywords=top_keywords,
            intent_tags=intent_tags,
            evidence_segment_ids=evidence_segment_ids,
            evidence_quote=evidence_quote,
            summary_label=summary_label,
            raw_audio_retained=False,
        )

    def _window(self, segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
        newest = max(segment.end_s for segment in segments)
        timed = [segment for segment in segments if segment.end_s >= newest - self.window_seconds]
        return timed[-self.window_segments:]

    def _hits(self, segments: list[TranscriptSegment]) -> list[KeywordHit]:
        hits: list[KeywordHit] = []
        for segment in segments:
            normalized_text = normalize_keyword_text(segment.text)
            if not normalized_text:
                continue
            for entry, variant, normalized_variant in self._variants:
                for match in re.finditer(rf"(?<![a-z0-9]){re.escape(normalized_variant)}(?![a-z0-9])", normalized_text):
                    hits.append(
                        KeywordHit(
                            entry=entry,
                            matched_text=match.group(0),
                            matched_variant=variant,
                            start_char=match.start(),
                            end_char=match.end(),
                            segment_id=segment.id,
                            segment_text=segment.text,
                            weight=entry.weight,
                            variant_requires_context=_variant_requires_context(entry, normalized_variant),
                        )
                    )
        return hits


def _section_tables(markdown: str) -> list[tuple[str, list[dict[str, str]]]]:
    current_section = ""
    lines = markdown.splitlines()
    result: list[tuple[str, list[dict[str, str]]]] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("## "):
            current_section = _strip_markdown(line[3:].strip())
        if current_section in ENERGY_RESEARCH_SECTIONS and line.startswith("|"):
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            rows = _parse_markdown_table(table_lines)
            if rows:
                result.append((current_section, rows))
            continue
        index += 1
    return result


def _parse_markdown_table(lines: list[str]) -> list[dict[str, str]]:
    if len(lines) < 3:
        return []
    headers = [_normalize_header(cell) for cell in _split_table_row(lines[0])]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        cells = _split_table_row(line)
        if len(cells) != len(headers):
            continue
        rows.append({header: _strip_markdown(cell) for header, cell in zip(headers, cells, strict=False)})
    return rows


def _split_table_row(line: str) -> list[str]:
    clean = line.strip()
    if clean.startswith("|"):
        clean = clean[1:]
    if clean.endswith("|"):
        clean = clean[:-1]
    return [cell.strip() for cell in clean.split("|")]


def _normalize_header(value: str) -> str:
    return normalize_keyword_text(value).replace(" ", "_")


def _strip_markdown(value: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", value)
    text = re.sub(r"\*\*([^*]*)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]*)\*", r"\1", text)
    text = text.replace("\\|", "|")
    return re.sub(r"\s+", " ", text).strip()


def _entry_from_row(section: str, row: dict[str, str]) -> KeywordEntry | None:
    if section == "Highest-weight phrases":
        phrase = row.get("phrase", "")
        weight = _float_or_none(row.get("weight"))
        if not phrase or weight is None:
            return None
        normalized = normalize_keyword_text(phrase)
        return KeywordEntry(
            phrase=phrase,
            normalized_token=normalized,
            category="service",
            subcategory=row.get("primary_signal", "highest weight"),
            intent_tags=[],
            synonyms=[],
            weight=weight,
            ambiguous=normalized in AMBIGUOUS_ACRONYMS,
            requires_context=normalized in AMBIGUOUS_ACRONYMS,
            source_section=section,
        )

    phrase = row.get("exact_phrase", "")
    weight = _float_or_none(row.get("weight"))
    if not phrase or weight is None:
        return None
    category = _normalize_category(row.get("category", "service"))
    normalized = normalize_keyword_text(row.get("normalized_token") or phrase)
    synonyms = _variant_list(row.get("synonyms_variants", ""))
    ambiguous = normalized in AMBIGUOUS_ACRONYMS
    return KeywordEntry(
        phrase=phrase,
        normalized_token=normalized,
        category=category,
        subcategory=normalize_keyword_text(row.get("subcategory", "")) or "general",
        intent_tags=_intent_list(row.get("intent_tags", "")),
        synonyms=synonyms,
        weight=weight,
        ambiguous=ambiguous,
        requires_context=ambiguous,
        source_section=section,
    )


def _infer_category(normalized: str) -> tuple[str, str]:
    if any(term in normalized for term in ["procurement", "tariff", "demand response", "demand flexibility", "ppa", "charge"]):
        return "market", "inferred from highest-weight phrase"
    if any(term in normalized for term in ["carbon", "decarbon", "emissions", "sustainability", "net zero"]):
        return "sustainability", "inferred from highest-weight phrase"
    if any(term in normalized for term in ["audit", "strategy", "roadmap", "consulting"]):
        return "service", "inferred from highest-weight phrase"
    if any(term in normalized for term in ["bill", "benchmark", "management", "eui", "energy use"]):
        return "operations", "inferred from highest-weight phrase"
    if any(term in normalized for term in ["cost", "savings"]):
        return "commercial", "inferred from highest-weight phrase"
    return "technical", "inferred from highest-weight phrase"


def _default_intents_for_category(category: str) -> list[str]:
    if category in {"commercial", "service", "market", "finance"}:
        return ["informational", "transactional", "lead-gen"]
    if category in {"regulatory", "sustainability"}:
        return ["informational", "compliance", "lead-gen"]
    return ["informational", "support", "transactional"]


def _normalize_category(value: str) -> str:
    normalized = normalize_keyword_text(value)
    if normalized == "software":
        return "technology"
    return normalized if normalized in ENERGY_CATEGORIES else "service"


def _intent_list(value: str) -> list[str]:
    tags = [normalize_keyword_text(item).replace(" ", "-") for item in value.split(";")]
    return _sorted_unique([tag for tag in tags if tag in INTENT_TAGS], order=INTENT_ORDER)


def _variant_list(value: str) -> list[str]:
    return _sorted_unique([_strip_markdown(item) for item in value.split(";") if _strip_markdown(item)])


def _compiled_variants(entries: list[KeywordEntry]) -> list[tuple[KeywordEntry, str, str]]:
    compiled: list[tuple[KeywordEntry, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for entry in entries:
        variants = [entry.phrase, entry.normalized_token, *entry.synonyms]
        for variant in _expanded_variants(variants):
            normalized = normalize_keyword_text(variant)
            if not normalized:
                continue
            key = (entry.normalized_token, entry.category, normalized)
            if key in seen:
                continue
            seen.add(key)
            compiled.append((entry, variant, normalized))
    return sorted(compiled, key=lambda item: (-len(item[2]), -item[0].weight, item[2]))


def _expanded_variants(variants: list[str]) -> list[str]:
    result: list[str] = []
    for variant in variants:
        result.append(variant)
        if normalize_keyword_text(variant).endswith("measure"):
            result.append(f"{variant}s")
        normalized = normalize_keyword_text(variant)
        if normalized == "mv":
            result.append("measurement and verification")
        if normalized == "m v":
            result.extend(["mv", "measurement and verification"])
        if normalized == "roi":
            result.append("return on investment")
    return _sorted_unique(result)


def _variant_requires_context(entry: KeywordEntry, normalized_variant: str) -> bool:
    if normalized_variant in AMBIGUOUS_ACRONYMS:
        return True
    return entry.requires_context and normalized_variant == entry.normalized_token


def _usable_hits(hits: list[KeywordHit], segments: list[TranscriptSegment]) -> list[KeywordHit]:
    if not hits:
        return []
    by_segment = {segment.id: segment for segment in segments}
    segment_order = {segment.id: index for index, segment in enumerate(segments)}
    non_acronym_by_segment = {
        hit.segment_id
        for hit in hits
        if not hit.variant_requires_context and not hit.entry.requires_context
    }
    acronym_tokens = {normalize_keyword_text(hit.matched_variant) for hit in hits if hit.variant_requires_context or hit.entry.requires_context}
    anchor_present = any(_has_commercial_energy_anchor(segment.text) for segment in segments)
    usable: list[KeywordHit] = []
    for hit in hits:
        if not (hit.variant_requires_context or hit.entry.requires_context):
            usable.append(hit)
            continue
        adjacent_ids = _adjacent_segment_ids(hit.segment_id, segment_order, segments)
        context_text = " ".join(by_segment[segment_id].text for segment_id in adjacent_ids if segment_id in by_segment)
        if _expanded_phrase_present(hit.entry, context_text):
            usable.append(hit)
            continue
        if any(segment_id in non_acronym_by_segment for segment_id in adjacent_ids):
            usable.append(hit)
            continue
        if len(acronym_tokens) >= 2 and anchor_present:
            usable.append(hit)
            continue
        if normalize_keyword_text(hit.matched_variant) != hit.entry.normalized_token and len(normalize_keyword_text(hit.matched_variant).split()) > 1:
            usable.append(hit)
    return usable


def _adjacent_segment_ids(segment_id: str, segment_order: dict[str, int], segments: list[TranscriptSegment]) -> list[str]:
    index = segment_order.get(segment_id)
    if index is None:
        return [segment_id]
    return [segment.id for segment in segments[max(0, index - 1) : min(len(segments), index + 2)]]


def _expanded_phrase_present(entry: KeywordEntry, text: str) -> bool:
    normalized_text = normalize_keyword_text(text)
    for variant in [entry.phrase, *entry.synonyms]:
        normalized = normalize_keyword_text(variant)
        if normalized and normalized not in AMBIGUOUS_ACRONYMS and len(normalized.split()) > 1:
            if re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", normalized_text):
                return True
    return False


def _has_commercial_energy_anchor(text: str) -> bool:
    terms = set(normalize_keyword_text(text).split())
    return bool(terms & COMMERCIAL_ENERGY_ANCHORS)


def _cap_duplicate_hits(hits: list[KeywordHit]) -> list[KeywordHit]:
    best: dict[tuple[str, str], KeywordHit] = {}
    for hit in hits:
        key = (hit.entry.normalized_token, hit.entry.category)
        existing = best.get(key)
        if existing is None or hit.weight > existing.weight:
            best[key] = hit
    return sorted(best.values(), key=lambda item: (-item.weight, item.entry.phrase))


def _top_categories(hits: list[KeywordHit]) -> list[dict[str, Any]]:
    score_by_category: Counter[str] = Counter()
    count_by_category: Counter[str] = Counter()
    for hit in hits:
        score_by_category[hit.entry.category] += hit.weight
        count_by_category[hit.entry.category] += 1
    return [
        {"category": category, "score": round(score, 3), "count": count_by_category[category]}
        for category, score in score_by_category.most_common(3)
    ]


def _top_intents(hits: list[KeywordHit]) -> list[str]:
    counts: Counter[str] = Counter()
    for hit in hits:
        for tag in hit.entry.intent_tags:
            counts[tag] += 1
    return [tag for tag, _count in counts.most_common(6)]


def _evidence_quote(segments: list[TranscriptSegment], source_ids: list[str]) -> str:
    selected = [segment.text for segment in segments if segment.id in set(source_ids)]
    return compact_text(" ".join(selected), limit=320)


def _confidence(score: float, hits: list[KeywordHit], top_categories: list[dict[str, Any]]) -> EnergyConfidence:
    if not hits:
        return "low"
    non_ambiguous = [hit for hit in hits if not (hit.entry.requires_context or hit.variant_requires_context)]
    top_category_count = int(top_categories[0]["count"]) if top_categories else 0
    max_weight = max(hit.weight for hit in hits)
    direct_phrase = any(hit.entry.normalized_token in STRONG_DIRECT_PHRASES and not hit.variant_requires_context for hit in hits)
    if (score >= 1.6 and max_weight >= 0.9) or (top_category_count >= 3 and len(non_ambiguous) >= 3):
        return "high"
    if (score >= 0.9 and len(hits) >= 2) or direct_phrase or (max_weight >= 0.95 and score >= 0.95):
        return "medium"
    return "low"


def _summary_label(active: bool, top_categories: list[dict[str, Any]]) -> str:
    if not active:
        return "Energy lens inactive"
    categories = [str(item["category"]) for item in top_categories[:2]]
    return "Energy lens" if not categories else f"Energy lens: {' + '.join(categories)}"


def _normalize_min_confidence(value: str) -> EnergyConfidence:
    normalized = str(value or "medium").strip().lower()
    return normalized if normalized in {"low", "medium", "high"} else "medium"  # type: ignore[return-value]


def _confidence_rank(value: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(value, 1)


def _float_or_none(value: str | None) -> float | None:
    try:
        return round(float(str(value or "").strip()), 3)
    except ValueError:
        return None


def _sorted_unique(values: list[str], *, order: list[str] | set[str] | None = None) -> list[str]:
    cleaned = []
    seen = set()
    for value in values:
        item = re.sub(r"\s+", " ", value.strip())
        if not item or item in seen:
            continue
        seen.add(item)
        cleaned.append(item)
    if order is None:
        return sorted(cleaned, key=lambda item: item.lower())
    order_list = list(order)
    return sorted(cleaned, key=lambda item: (order_list.index(item) if item in order_list else 99, item))


def _unique(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
