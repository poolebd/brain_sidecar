from __future__ import annotations

import re


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_for_evidence_match(text: str) -> str:
    clean = text.lower()
    clean = re.sub(r"[\u2010-\u2015]", "-", clean)
    clean = re.sub(r"[^a-z0-9+./&-]+", " ", clean)
    clean = clean.replace("&", " and ")
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def extract_supported_aliases(text: str) -> set[str]:
    clean = normalize_for_evidence_match(text)
    aliases: set[str] = set()

    if re.search(r"\b(s l d|single line|single-line diagram)\b", clean):
        aliases.add("sld")

    if re.search(r"\b(gba|generation agreement)\b", clean):
        aliases.add("generation agreement")
        aliases.add("gba")

    if re.search(r"\b(queue reviewer|q reviewer|cue reviewer)\b", clean):
        aliases.add("reviewer")
        aliases.add("queue reviewer")

    if "demons" in clean and any(
        term in clean
        for term in [
            "power quality",
            "client",
            "gba",
            "epc",
            "generation agreement",
            "agreement",
            "harmonic",
            "sld",
        ]
    ):
        aliases.add("siemens")

    if re.search(r"\b(narc|nerc|nerd side)\b", clean) and (
        "manish" in clean
        or "sme" in clean
        or "slc" in clean
        or "compliance" in clean
        or "expert" in clean
        or "technical" in clean
    ):
        aliases.add("nerc")

    if "sme" in clean or "subject matter expert" in clean:
        aliases.add("expert")
        aliases.add("subject matter expert")

    return aliases


def alias_supported(alias: str, evidence_text: str) -> bool:
    clean_alias = normalize_for_evidence_match(alias)
    if not clean_alias:
        return False
    clean_evidence = normalize_for_evidence_match(evidence_text)
    if _phrase_present(clean_alias, clean_evidence):
        return True
    return clean_alias in extract_supported_aliases(clean_evidence)


def evidence_terms_with_aliases(text: str) -> set[str]:
    clean = normalize_for_evidence_match(text)
    terms = set(_TOKEN_RE.findall(clean))
    for alias in extract_supported_aliases(clean):
        terms.update(_TOKEN_RE.findall(alias))
    return terms


def _phrase_present(phrase: str, text: str) -> bool:
    if not phrase or not text:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None
