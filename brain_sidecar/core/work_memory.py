from __future__ import annotations

import asyncio
import csv
import hashlib
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

from brain_sidecar.config import Settings
from brain_sidecar.core.models import SearchHit, new_id
from brain_sidecar.core.recall import normalize_text, read_document_text
from brain_sidecar.core.storage import Storage


JOB_HISTORY_ROOT = Path("/home/bp/Nextcloud2/Job Hunting")
PAST_WORK_ROOT = Path("/home/bp/Nextcloud2/_library/_shoalstone/past work")
PMP_SUMMARY = JOB_HISTORY_ROOT / "_portfolio" / "Projects" / "PMP_Experience_Summary.csv"

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".csv", ".pdf", ".docx"}
METADATA_ONLY_EXTENSIONS = {".xlsx", ".xlsm", ".xls", ".pptx", ".jpg", ".jpeg", ".png", ".gif", ".bmp"}
EXCLUDED_EXTENSIONS = {
    ".7z",
    ".dat",
    ".dll",
    ".exe",
    ".iso",
    ".m4a",
    ".mov",
    ".mp3",
    ".mp4",
    ".msi",
    ".one",
    ".pkl",
    ".wav",
    ".zip",
}
SENSITIVE_PARTS = {
    "clearance",
    "disability",
    "employee reviews",
    "medical",
    "medical records",
    "personal",
    "poole evals",
    "separation",
}
GUARDRAIL_PARTS = {
    "s&l offer",
    "s and l offer",
    "sargent",
    "employment agreement",
    "code of conduct",
    "conflict of interest",
}
GENERIC_TERMS = {
    "and",
    "assessment",
    "about",
    "all",
    "are",
    "because",
    "but",
    "can",
    "closeout",
    "current",
    "for",
    "from",
    "group",
    "had",
    "has",
    "have",
    "her",
    "his",
    "latest",
    "management",
    "not",
    "one",
    "out",
    "over",
    "phase",
    "planning",
    "project",
    "public",
    "real",
    "replacement",
    "she",
    "smith",
    "status",
    "study",
    "that",
    "their",
    "them",
    "there",
    "they",
    "this",
    "today",
    "the",
    "time",
    "under",
    "upgrade",
    "was",
    "were",
    "what",
    "will",
    "with",
    "you",
    "your",
}
LIVE_REQUIRED_MATCHED_TERMS = 2
LIVE_MIN_RECALL_SCORE = 0.55
LIVE_STRONG_ALIAS_MIN_SCORE = 0.48
MAX_TEXT_BYTES = 2_500_000
MAX_EVIDENCE_PER_PROJECT = 14


@dataclass
class EvidenceDraft:
    source_path: str
    snippet: str
    artifact_type: str
    weight: float


@dataclass
class ProjectDraft:
    key: str
    title: str
    organization: str
    date_range: str
    role: str
    domain: str
    summary: str
    lessons: list[str] = field(default_factory=list)
    triggers: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    source_group: str = "work_history"
    confidence: float = 0.72
    evidence: list[EvidenceDraft] = field(default_factory=list)


@dataclass(frozen=True)
class WorkMemoryIndexReport:
    roots: int
    files_seen: int
    sources_indexed: int
    projects_indexed: int
    evidence_indexed: int
    embeddings_indexed: int
    skipped: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "roots": self.roots,
            "files_seen": self.files_seen,
            "sources_indexed": self.sources_indexed,
            "projects_indexed": self.projects_indexed,
            "evidence_indexed": self.evidence_indexed,
            "embeddings_indexed": self.embeddings_indexed,
            "skipped": self.skipped,
        }


@dataclass(frozen=True)
class WorkMemoryRecallCard:
    project_id: str
    title: str
    organization: str
    date_range: str
    score: float
    confidence: float
    reason: str
    lesson: str
    citations: list[str]
    suggested_say: str = ""
    source_group: str = "work_history"
    card_key: str = ""

    @property
    def text(self) -> str:
        date_part = f", {self.date_range}" if self.date_range else ""
        lesson_part = f" Useful reminder: {self.lesson}" if self.lesson else ""
        return (
            f"This conversation is reminiscent of {self.title} "
            f"({self.organization}{date_part}). Similarity: {self.reason}.{lesson_part}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "title": self.title,
            "organization": self.organization,
            "date_range": self.date_range,
            "score": self.score,
            "confidence": self.confidence,
            "reason": self.reason,
            "lesson": self.lesson,
            "citations": self.citations,
            "suggested_say": self.suggested_say,
            "source_group": self.source_group,
            "card_key": self.card_key,
            "text": self.text,
        }

    def to_search_hit(self) -> SearchHit:
        return SearchHit(
            source_type="work_memory_project",
            source_id=self.project_id,
            text=self.text,
            score=self.score,
            metadata={
                "title": self.title,
                "organization": self.organization,
                "date_range": self.date_range,
                "confidence": self.confidence,
                "reason": self.reason,
                "lesson": self.lesson,
                "citations": self.citations,
                "suggested_say": self.suggested_say,
                "suggested_contribution": self.suggested_say,
                "source_group": self.source_group,
                "card_key": self.card_key,
                "why_now": self.reason,
            },
        )


class WorkMemoryService:
    def __init__(self, storage: Storage, recall_index: Any | None = None, settings: Settings | None = None) -> None:
        self.storage = storage
        self.recall_index = recall_index
        self.settings = settings

    def status(self) -> dict[str, Any]:
        summary = self.storage.work_memory_summary()
        summary["default_roots"] = [str(path) for path in self.default_roots()]
        summary["guardrail"] = (
            "Current employer/client material is guardrail context only and is not used "
            "as recall memory unless explicitly indexed later."
        )
        return summary

    def default_roots(self) -> list[Path]:
        roots = [
            self.job_history_root,
            self.past_work_root,
            self.pas_root,
        ]
        return [root for root in roots if root is not None and root.exists()]

    @property
    def job_history_root(self) -> Path:
        return getattr(self.settings, "work_memory_job_history_root", JOB_HISTORY_ROOT)

    @property
    def past_work_root(self) -> Path:
        return getattr(self.settings, "work_memory_past_work_root", PAST_WORK_ROOT)

    @property
    def pas_root(self) -> Path | None:
        return getattr(self.settings, "work_memory_pas_root", None)

    @property
    def pmp_summary(self) -> Path:
        if self.job_history_root == JOB_HISTORY_ROOT:
            return PMP_SUMMARY
        return self.job_history_root / "_portfolio" / "Projects" / "PMP_Experience_Summary.csv"

    async def reindex(self, roots: list[Path] | None = None, *, embed: bool = True) -> WorkMemoryIndexReport:
        roots = [path.expanduser().resolve() for path in (roots or self.default_roots()) if path.exists()]
        skipped: list[str] = []
        projects: dict[str, ProjectDraft] = {}
        sources_by_path: dict[str, dict[str, Any]] = {}
        files_seen = 0

        self._seed_pmp_projects(projects, sources_by_path, skipped)
        self._seed_star_projects(projects, sources_by_path, skipped)
        self._seed_ta_smith_overview(projects, sources_by_path, skipped)
        self._seed_guardrails(sources_by_path)

        for root in roots:
            for path in iter_candidate_files(root):
                files_seen += 1
                source = describe_source(path)
                sources_by_path.setdefault(str(path), source)
                if source["status"] == "excluded":
                    continue
                self._correlate_source_to_projects(path, source, projects)

        self.storage.clear_work_memory()
        source_id_by_path: dict[str, str] = {}
        for source in sources_by_path.values():
            source_id_by_path[source["path"]] = self.storage.upsert_work_memory_source(**source)

        project_id_by_key: dict[str, str] = {}
        evidence_count = 0
        embeddings_count = 0
        embedding_available = self.recall_index is not None and embed
        for project in sorted(projects.values(), key=lambda item: (item.date_range, item.title)):
            project_id = self.storage.upsert_work_memory_project(
                key=project.key,
                title=project.title,
                organization=project.organization,
                date_range=project.date_range,
                role=project.role,
                domain=project.domain,
                summary=project.summary,
                lessons=dedupe_strings(project.lessons),
                triggers=dedupe_strings(project.triggers + project.aliases),
                source_group=project.source_group,
                confidence=project.confidence,
            )
            project_id_by_key[project.key] = project_id
            for evidence in select_evidence(project.evidence):
                self.storage.add_work_memory_evidence(
                    project_id=project_id,
                    source_id=source_id_by_path.get(evidence.source_path),
                    source_path=evidence.source_path,
                    snippet=evidence.snippet,
                    artifact_type=evidence.artifact_type,
                    weight=evidence.weight,
                )
                evidence_count += 1
            if embedding_available:
                if await self._embed_project(project_id, project):
                    embeddings_count += 1
                else:
                    embedding_available = False
                    skipped.append("Work-memory embeddings skipped after the first unavailable/slow embedding call.")

        return WorkMemoryIndexReport(
            roots=len(roots),
            files_seen=files_seen,
            sources_indexed=len(sources_by_path),
            projects_indexed=len(project_id_by_key),
            evidence_indexed=evidence_count,
            embeddings_indexed=embeddings_count,
            skipped=skipped[:80],
        )

    def search(self, query: str, limit: int = 5, *, manual: bool = False) -> list[WorkMemoryRecallCard]:
        clean = normalize_lookup(query)
        if not clean:
            return []
        query_terms = significant_terms(clean)
        projects = self.storage.work_memory_projects()
        cards: list[WorkMemoryRecallCard] = []
        for project in projects:
            evidence = self.storage.work_memory_evidence(project["id"], limit=6)
            combined = normalize_lookup(
                " ".join(
                    [
                        project["title"],
                        project["organization"],
                        project["domain"],
                        project["summary"],
                        " ".join(project["lessons"]),
                        " ".join(project["triggers"]),
                        " ".join(item["snippet"] for item in evidence),
                    ]
                )
            )
            matched_terms = [term for term in query_terms if term in combined and term not in GENERIC_TERMS]
            alias_hits = [trigger for trigger in project["triggers"] if normalize_lookup(trigger) in clean]
            required_terms = 1 if manual else LIVE_REQUIRED_MATCHED_TERMS
            entity_overlap = live_entity_overlap(clean, combined)
            strong_alias_hit = bool(alias_hits) and bool(entity_overlap)
            metadata_only_ratio = metadata_only_evidence_ratio(evidence)
            if manual:
                if not alias_hits and len(set(matched_terms)) < required_terms:
                    continue
            elif not strong_alias_hit and len(set(matched_terms)) < required_terms:
                continue
            if not manual and not strong_alias_hit and not entity_overlap:
                continue
            if not manual and not strong_alias_hit and metadata_only_ratio >= 0.75:
                continue
            score = score_match(matched_terms, alias_hits, project["confidence"], metadata_only_ratio=metadata_only_ratio)
            if not manual and score < LIVE_MIN_RECALL_SCORE and not (strong_alias_hit and score >= LIVE_STRONG_ALIAS_MIN_SCORE):
                continue
            lesson = project["lessons"][0] if project["lessons"] else project["summary"]
            reason_terms = dedupe_strings(alias_hits + matched_terms)[:7]
            reason = "shared signals around " + ", ".join(reason_terms) if reason_terms else "a similar work pattern"
            source_group = project.get("source_group") or "work_history"
            suggested_say = suggested_contribution(project, lesson)
            cards.append(
                WorkMemoryRecallCard(
                    project_id=project["id"],
                    title=project["title"],
                    organization=project["organization"],
                    date_range=project["date_range"],
                    score=score,
                    confidence=project["confidence"],
                    reason=reason,
                    lesson=lesson,
                    citations=[item["source_path"] for item in evidence[:3]],
                    suggested_say=suggested_say,
                    source_group=source_group,
                    card_key=f"work:{project['id']}",
                )
            )
        return sorted(cards, key=lambda card: card.score, reverse=True)[:limit]

    def record_recall_event(self, session_id: str | None, card: WorkMemoryRecallCard, query: str) -> None:
        self.storage.add_work_memory_recall_event(
            session_id=session_id,
            project_id=card.project_id,
            query=query,
            score=card.score,
            reason=card.reason,
        )

    def _seed_pmp_projects(
        self,
        projects: dict[str, ProjectDraft],
        sources_by_path: dict[str, dict[str, Any]],
        skipped: list[str],
    ) -> None:
        pmp_summary = self.pmp_summary
        if not pmp_summary.exists():
            skipped.append(f"{pmp_summary}: missing")
            return
        sources_by_path[str(pmp_summary)] = describe_source(pmp_summary)
        try:
            with pmp_summary.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
                for row in csv.DictReader(handle):
                    title = clean_title(row.get("Project Title", ""))
                    if not title:
                        continue
                    project = draft_from_pmp_row(row)
                    add_or_merge_project(projects, project)
                    projects[project.key].evidence.append(
                        EvidenceDraft(
                            source_path=str(pmp_summary),
                            snippet=(
                                f"Canonical timeline row: {project.title}; {project.organization}; "
                                f"{project.date_range}; role {project.role}."
                            ),
                            artifact_type="canonical_timeline",
                            weight=1.0,
                        )
                    )
        except Exception as exc:
            skipped.append(f"{pmp_summary}: {exc}")

    def _seed_star_projects(
        self,
        projects: dict[str, ProjectDraft],
        sources_by_path: dict[str, dict[str, Any]],
        skipped: list[str],
    ) -> None:
        for path, seed in star_seed_paths(self.job_history_root).items():
            if not path.exists():
                continue
            sources_by_path[str(path)] = describe_source(path)
            try:
                text = read_small_text(path)
            except Exception as exc:
                skipped.append(f"{path}: {exc}")
                text = ""
            project = replace(
                seed,
                lessons=list(seed.lessons),
                triggers=list(seed.triggers),
                aliases=list(seed.aliases),
                evidence=list(seed.evidence),
            )
            if text:
                project.summary = project.summary or summarize_text(text, max_chars=460)
                project.evidence.append(
                    EvidenceDraft(
                        source_path=str(path),
                        snippet=summarize_text(text, max_chars=720),
                        artifact_type="star_note",
                        weight=1.0,
                    )
                )
            add_or_merge_project(projects, project)

    def _seed_ta_smith_overview(
        self,
        projects: dict[str, ProjectDraft],
        sources_by_path: dict[str, dict[str, Any]],
        skipped: list[str],
    ) -> None:
        path = self.job_history_root / "Standard Interview" / "OPC Notes" / "TA Smith Electrical Overview.txt"
        if not path.exists():
            return
        sources_by_path[str(path)] = describe_source(path)
        try:
            text = read_small_text(path)
        except Exception as exc:
            skipped.append(f"{path}: {exc}")
            return
        snippet = summarize_text(text, max_chars=620)
        for project in projects.values():
            lookup = normalize_lookup(project.title + " " + project.organization)
            if "oglethorpe" in lookup or "smith" in lookup or "opc" in lookup:
                project.evidence.append(
                    EvidenceDraft(
                        source_path=str(path),
                        snippet=snippet,
                        artifact_type="technical_context",
                        weight=0.72,
                    )
                )
                project.triggers.extend(["500 kV", "230 kV", "13.8 kV", "4.16 kV", "480 V", "125 V DC"])

    def _seed_guardrails(self, sources_by_path: dict[str, dict[str, Any]]) -> None:
        offer_root = self.job_history_root / "S&L Offer"
        if not offer_root.exists():
            return
        for path in iter_candidate_files(offer_root):
            source = describe_source(path)
            source["sensitivity"] = "current_employer_guardrail"
            source["status"] = "guardrail"
            source["disabled"] = True
            sources_by_path[str(path)] = source

    def _correlate_source_to_projects(
        self,
        path: Path,
        source: dict[str, Any],
        projects: dict[str, ProjectDraft],
    ) -> None:
        if source["disabled"] and source["sensitivity"] != "historical_metadata":
            return
        haystack = normalize_lookup(" ".join([str(path), path.stem]))
        for project in projects.values():
            weight, alias = alias_match_weight(haystack, project.aliases + project.triggers + [project.title])
            if weight <= 0:
                continue
            project.evidence.append(
                EvidenceDraft(
                    source_path=str(path),
                    snippet=metadata_snippet(path, source, alias),
                    artifact_type=source["status"],
                    weight=weight,
                )
            )

    async def _embed_project(self, project_id: str, project: ProjectDraft) -> bool:
        if self.recall_index is None:
            return False
        text = project_card_text(project)
        metadata = {
            "title": project.title,
            "organization": project.organization,
            "date_range": project.date_range,
            "domain": project.domain,
            "work_memory": True,
        }
        try:
            await asyncio.wait_for(
                self.recall_index.add_text("work_memory_project", project_id, text, metadata),
                timeout=8.0,
            )
            return True
        except Exception:
            return False


def iter_candidate_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def describe_source(path: Path) -> dict[str, Any]:
    source_group = source_group_for_path(path)
    suffix = path.suffix.lower()
    parts = normalize_lookup(" ".join(path.parts))
    stat = safe_stat(path)
    sensitivity = "historical_metadata"
    status = "metadata_only"
    disabled = False

    if any(term in parts for term in SENSITIVE_PARTS):
        sensitivity = "excluded_sensitive"
        status = "excluded"
        disabled = True
    elif any(term in parts for term in GUARDRAIL_PARTS):
        sensitivity = "current_employer_guardrail"
        status = "guardrail"
        disabled = True
    elif suffix in EXCLUDED_EXTENSIONS:
        sensitivity = "unsupported_binary"
        status = "excluded"
        disabled = True
    elif suffix in SUPPORTED_TEXT_EXTENSIONS and stat["size_bytes"] <= MAX_TEXT_BYTES:
        status = "text_supported"
    elif suffix in SUPPORTED_TEXT_EXTENSIONS:
        status = "metadata_only"
    elif suffix in METADATA_ONLY_EXTENSIONS:
        status = "metadata_only"
    else:
        status = "metadata_only"

    return {
        "path": str(path),
        "source_group": source_group,
        "sensitivity": sensitivity,
        "status": status,
        "title": path.stem,
        "content_hash": stat_signature(path, stat),
        "metadata": {
            "suffix": suffix,
            "size_bytes": stat["size_bytes"],
            "mtime": stat["mtime"],
        },
        "disabled": disabled,
    }


def source_group_for_path(path: Path) -> str:
    normalized = normalize_lookup(str(path))
    if "s&l offer" in normalized or "s and l offer" in normalized:
        return "current_role_guardrails"
    if "job hunting" in normalized:
        return "job_history_seed"
    if "shoalstone/past work/navy" in normalized or "shoalstone\\past work\\navy" in normalized:
        return "navy_history"
    if "shoalstone/past work/opc" in normalized or "shoalstone\\past work\\opc" in normalized:
        return "opc_history"
    if "shoalstone/past work/ums group" in normalized or "shoalstone\\past work\\ums group" in normalized:
        return "consulting_history"
    return "work_history"


def safe_stat(path: Path) -> dict[str, float]:
    try:
        stat = path.stat()
        return {"size_bytes": float(stat.st_size), "mtime": float(stat.st_mtime)}
    except OSError:
        return {"size_bytes": 0.0, "mtime": 0.0}


def stat_signature(path: Path, stat: dict[str, float]) -> str:
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8", errors="ignore"))
    digest.update(str(int(stat["size_bytes"])).encode("ascii"))
    digest.update(str(int(stat["mtime"])).encode("ascii"))
    return digest.hexdigest()[:24]


def draft_from_pmp_row(row: dict[str, str]) -> ProjectDraft:
    title = clean_title(row.get("Project Title", ""))
    organization = clean_title(row.get("Organization", ""))
    role = clean_title(row.get("Project Role", "") or row.get("Job Title", ""))
    domain = clean_title(row.get("Functional Area", "") or row.get("Industry", ""))
    date_range = clean_title(row.get("Project Dates", ""))
    enrichment = enrichment_for(title)
    summary = enrichment.get("summary") or (
        f"{title} at {organization}; your role was {role or 'a project contributor'} "
        f"across {domain or 'technical delivery'}."
    )
    return ProjectDraft(
        key=project_key(title),
        title=title,
        organization=organization,
        date_range=date_range,
        role=role,
        domain=domain,
        summary=summary,
        lessons=list(enrichment.get("lessons", [])),
        triggers=list(enrichment.get("triggers", [])),
        aliases=build_aliases(title, organization, enrichment.get("aliases", [])),
        source_group=source_group_for_org(organization),
        confidence=0.92,
    )


def star_seed_paths(job_history_root: Path = JOB_HISTORY_ROOT) -> dict[Path, ProjectDraft]:
    star_root = job_history_root / "Standard Interview" / "STAR PPT" / "ppts"
    seeds = {
        star_root / "stars_500kvbreaker.txt": seed_project(
            "500kV Breaker Replacement - T.A. Smith",
            "Oglethorpe Power",
            "May 2020 - November 2021",
            "Project Manager",
            "Capital Projects",
            "High-consequence interconnect breaker replacement with technical scope, commercial terms, outage planning, acceptance criteria, and commissioning risk.",
            "opc_history",
            ["500kV breaker", "breaker replacement", "outage window", "acceptance criteria", "warranty", "liquidated damages"],
        ),
        star_root / "stars_lightarrestor.txt": seed_project(
            "500kV Lightning Arrester Response - T.A. Smith",
            "Oglethorpe Power",
            "OPC era",
            "Electrical Engineer",
            "Emergent Response",
            "Rapid engineering response after a 500kV lightning arrester failure caused immediate generation risk.",
            "opc_history",
            ["lightning arrester", "emergent response", "600 MW", "failure analysis", "restore generation"],
        ),
        star_root / "gas mains" / "stars.txt": seed_project(
            "PG&E Gas Mains Replacement Cost Model",
            "PG&E / UMS Group",
            "2023 - 2024",
            "Consulting Analytics Lead",
            "Cost Modeling",
            "Predictive cost modeling for gas main replacement variance using operational drivers and scenario analysis.",
            "consulting_history",
            ["PG&E", "gas mains", "cost model", "GIS", "soil", "terrain", "random forest", "cost drivers"],
        ),
        star_root / "ground grid" / "stars.txt": seed_project(
            "Ground Grid Remediation - T.A. Smith",
            "Oglethorpe Power",
            "OPC era",
            "Electrical Engineer",
            "Electrical Safety",
            "Grounding remediation to manage step and touch potential risk without creating avoidable outage impact.",
            "opc_history",
            ["ground grid", "step potential", "touch potential", "soil resistivity", "grounding", "life safety"],
        ),
        star_root / "ogm" / "star.txt": seed_project(
            "Online Generator Monitoring - T.A. Smith",
            "Oglethorpe Power",
            "Mar 2019 - June 2021",
            "Project Manager",
            "Condition Monitoring",
            "Online generator monitoring across hydrogen-cooled generators, tying measurements to failure mechanisms and maintenance decisions.",
            "opc_history",
            ["OGM", "online generator monitoring", "hydrogen cooled", "flux probe", "bus coupler", "shaft voltage", "condition based maintenance"],
        ),
        star_root / "relays" / "stars.txt": seed_project(
            "Transmission Relay Modernization - OPC",
            "Oglethorpe Power",
            "2019 - 2021",
            "Project Manager",
            "Protection and Controls",
            "Modernization of aging 230kV and 500kV relay systems to reduce nuisance trips and improve diagnostics.",
            "opc_history",
            ["relay", "protection", "CT", "PT", "trip matrix", "breaker failure", "nuisance trip", "settings"],
        ),
    }
    return seeds


def seed_project(
    title: str,
    organization: str,
    date_range: str,
    role: str,
    domain: str,
    summary: str,
    source_group: str,
    triggers: list[str],
) -> ProjectDraft:
    enrichment = enrichment_for(title)
    return ProjectDraft(
        key=project_key(title),
        title=title,
        organization=organization,
        date_range=date_range,
        role=role,
        domain=domain,
        summary=summary,
        lessons=dedupe_strings(list(enrichment.get("lessons", []))),
        triggers=dedupe_strings(triggers + list(enrichment.get("triggers", []))),
        aliases=build_aliases(title, organization, list(enrichment.get("aliases", [])) + triggers),
        source_group=source_group,
        confidence=0.9,
    )


def add_or_merge_project(projects: dict[str, ProjectDraft], incoming: ProjectDraft) -> None:
    existing = projects.get(incoming.key)
    if existing is None:
        projects[incoming.key] = incoming
        return
    existing.organization = existing.organization or incoming.organization
    existing.date_range = existing.date_range or incoming.date_range
    existing.role = existing.role or incoming.role
    existing.domain = existing.domain or incoming.domain
    if len(incoming.summary) > len(existing.summary):
        existing.summary = incoming.summary
    existing.lessons = dedupe_strings(existing.lessons + incoming.lessons)
    existing.triggers = dedupe_strings(existing.triggers + incoming.triggers)
    existing.aliases = dedupe_strings(existing.aliases + incoming.aliases)
    existing.confidence = max(existing.confidence, incoming.confidence)
    existing.evidence.extend(incoming.evidence)


def enrichment_for(title: str) -> dict[str, Any]:
    lookup = normalize_lookup(title)
    enrichments = [
        {
            "match": ["online generator monitoring", "ogm"],
            "summary": "Condition-monitoring project for T.A. Smith generators focused on connecting measurements to failure modes and maintenance decisions.",
            "lessons": [
                "Monitoring is only valuable when the measurement maps to a failure mode and an owner knows what decision it changes.",
                "Early vendor/spec choices should be judged by how operators will interpret alarms during real plant conditions.",
            ],
            "triggers": ["generator monitoring", "condition monitoring", "failure mechanisms", "alarm interpretation", "maintenance strategy"],
            "aliases": ["OGM", "online generator monitoring"],
        },
        {
            "match": ["500kv breaker"],
            "summary": "Critical transmission interconnect breaker replacement balancing technical scope, outage constraints, commercial risk, and commissioning quality.",
            "lessons": [
                "For high-consequence equipment, align technical acceptance, outage window, warranty, and risk ownership before execution pressure peaks.",
                "Commercial terms matter when they protect the operating risk you are actually carrying.",
            ],
            "triggers": ["500kV", "breaker", "interconnect", "outage", "commissioning", "acceptance testing"],
            "aliases": ["500 kv breaker", "500kv breaker", "breaker replacement"],
        },
        {
            "match": ["line relay", "relay modernization", "transmission relay"],
            "summary": "Transmission protection modernization across aging relay systems, settings, test reports, and trip logic.",
            "lessons": [
                "Protection work needs a clear trip-path mental model: CT/PT inputs, settings, interlocks, breaker failure, and downstream diagnostics.",
                "Modernization succeeds when test evidence and operating explanations are as strong as the hardware replacement.",
            ],
            "triggers": ["relay", "protection", "settings", "CT", "PT", "trip matrix", "breaker failure"],
            "aliases": ["relay replacement", "line relay", "transmission relay"],
        },
        {
            "match": ["major outage", "2022 outage", "2021 major outage", "2020 major outage", "2019 major outage"],
            "summary": "OPC outage planning and execution work at T.A. Smith with coordination across operations, maintenance, engineering, and vendors.",
            "lessons": [
                "Outage work is won in constraint visibility: prerequisites, handoffs, vendor readiness, and what must be true before the window opens.",
                "The useful question is often not whether work can be done, but what failure would strand the unit inside the outage window.",
            ],
            "triggers": ["outage", "maintenance window", "vendor readiness", "handoff", "critical path"],
            "aliases": ["major outage", "TA Smith outage"],
        },
        {
            "match": ["pdc", "power distribution center", "standby dc", "bus replacement"],
            "summary": "Power Distribution Center and DC system work where reliability, controls power, and execution sequencing were central.",
            "lessons": [
                "Controls power changes deserve special attention because small DC assumptions can become plant-wide reliability problems.",
                "For emergent electrical work, separate the immediate safe-restoration path from the durable design fix.",
            ],
            "triggers": ["PDC", "125 V DC", "station DC", "bus", "controls power", "battery"],
            "aliases": ["PDC", "power distribution center", "standby DC", "PDC bus"],
        },
        {
            "match": ["pg&e", "gas mains", "fleet vehicle", "overheads"],
            "summary": "Utility consulting analytics for PG&E operational cost, fleet readiness, gas mains, and overhead rate questions.",
            "lessons": [
                "When variance is high, the first useful product is often a driver map and planning conversation, not a perfect prediction.",
                "A model is more persuasive when it explains controllable cost drivers in language operators and finance can both use.",
            ],
            "triggers": ["PG&E", "cost drivers", "fleet readiness", "overheads", "gas mains", "scenario analysis"],
            "aliases": ["PGE", "PG&E", "gas mains", "overhead composite", "fleet optimization"],
        },
        {
            "match": ["workforce planning", "saskpower", "workload driven"],
            "summary": "Workload-driven workforce planning for utility operations, linking demand, capacity, productivity, and organizational decisions.",
            "lessons": [
                "Workforce planning gets sharper when workload, productivity, backlog risk, and role constraints are explicit rather than averaged away.",
                "Phase handoffs need continuity in assumptions, not just continuity in spreadsheets.",
            ],
            "triggers": ["workforce planning", "workload", "capacity", "productivity", "backlog", "SaskPower"],
            "aliases": ["SaskPower", "workload driven workforce planning", "WFP"],
        },
        {
            "match": ["american water", "reorganizational", "reorg"],
            "summary": "Enterprise reorganization assessment work focused on operating model, roles, and organizational tradeoffs.",
            "lessons": [
                "Reorg analysis should keep the operating problem visible so the org chart does not become the answer by itself.",
                "Stakeholder language matters because structure changes are experienced as risk before they are experienced as strategy.",
            ],
            "triggers": ["reorg", "operating model", "roles", "enterprise", "stakeholders", "American Water"],
            "aliases": ["American Water", "enterprise reorg", "reorganization"],
        },
        {
            "match": ["damage control", "mob-e", "at/fp", "soh"],
            "summary": "Navy readiness, safety, security, and material assessment work after nuclear training and shipboard leadership roles.",
            "lessons": [
                "Inspection success comes from making ownership, evidence, and weak signals visible before the formal assessment.",
                "High-reliability environments reward procedural clarity and calm escalation when risk is discovered.",
            ],
            "triggers": ["readiness", "inspection", "safety", "security", "material assessment", "Navy"],
            "aliases": ["USN", "Navy", "SOH", "ATFP", "damage control", "MOB-E"],
        },
    ]
    for enrichment in enrichments:
        if any(token in lookup for token in enrichment["match"]):
            return enrichment
    return {
        "summary": "",
        "lessons": [
            "Look for the operating constraint, the owner of the decision, and the evidence that would change the next action."
        ],
        "triggers": [],
        "aliases": [],
    }


def source_group_for_org(organization: str) -> str:
    lookup = normalize_lookup(organization)
    if "navy" in lookup or "uss" in lookup:
        return "navy_history"
    if "oglethorpe" in lookup:
        return "opc_history"
    if "ums" in lookup or "e source" in lookup or "e-source" in lookup or "pg e" in lookup:
        return "consulting_history"
    return "work_history"


def build_aliases(title: str, organization: str, extras: Iterable[str]) -> list[str]:
    aliases = [title, title.replace(" - ", " "), organization]
    title_lookup = normalize_lookup(title)
    if "t a smith" in title_lookup or "ta smith" in title_lookup:
        aliases.extend(["T.A. Smith", "TA Smith"])
    if "500kv" in title_lookup or "500 kv" in title_lookup:
        aliases.extend(["500kV", "500 kV"])
    if "230kv" in title_lookup or "230 kv" in title_lookup:
        aliases.extend(["230kV", "230 kV"])
    if "power distribution center" in title_lookup:
        aliases.append("PDC")
    aliases.extend(extras)
    return dedupe_strings([alias for alias in aliases if alias and len(alias.strip()) >= 2])


def alias_match_weight(haystack: str, aliases: Iterable[str]) -> tuple[float, str]:
    best_weight = 0.0
    best_alias = ""
    for alias in aliases:
        lookup = normalize_lookup(alias)
        if not lookup or len(lookup) < 3:
            continue
        if lookup in haystack:
            weight = min(1.0, 0.55 + len(lookup) / 80)
        else:
            tokens = [token for token in significant_terms(lookup) if token not in GENERIC_TERMS]
            if len(tokens) < 2:
                continue
            matched = [token for token in tokens if token in haystack]
            weight = 0.36 + 0.08 * len(matched) if len(matched) >= min(2, len(tokens)) else 0.0
        if weight > best_weight:
            best_weight = weight
            best_alias = alias
    return best_weight, best_alias


def metadata_snippet(path: Path, source: dict[str, Any], alias: str) -> str:
    size = int(float(source["metadata"].get("size_bytes", 0)))
    alias_part = f" matched on {alias}" if alias else ""
    return f"{path.name}{alias_part}; {source['source_group']} source; {size} bytes; path {path.parent}."


def select_evidence(items: list[EvidenceDraft]) -> list[EvidenceDraft]:
    seen: set[tuple[str, str]] = set()
    unique: list[EvidenceDraft] = []
    for item in sorted(items, key=lambda evidence: evidence.weight, reverse=True):
        key = (item.source_path, item.snippet[:120])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
        if len(unique) >= MAX_EVIDENCE_PER_PROJECT:
            break
    return unique


def read_small_text(path: Path) -> str:
    if path.stat().st_size > MAX_TEXT_BYTES:
        return ""
    if path.suffix.lower() in {".txt", ".md", ".markdown", ".rst", ".csv", ".pdf", ".docx"}:
        return read_document_text(path)
    return ""


def summarize_text(text: str, max_chars: int = 500) -> str:
    clean = normalize_text(text)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rsplit(" ", 1)[0].strip()


def project_card_text(project: ProjectDraft) -> str:
    return normalize_text(
        " ".join(
            [
                project.title,
                project.organization,
                project.date_range,
                project.role,
                project.domain,
                project.summary,
                " ".join(project.lessons),
                " ".join(project.triggers),
            ]
        )
    )


def score_match(
    matched_terms: list[str],
    alias_hits: list[str],
    confidence: float,
    *,
    metadata_only_ratio: float = 0.0,
) -> float:
    score = 0.28 + min(0.38, len(set(matched_terms)) * 0.045) + min(0.24, len(alias_hits) * 0.12)
    score *= 0.82 + min(0.18, confidence * 0.18)
    score *= 1.0 - min(0.28, max(0.0, metadata_only_ratio) * 0.28)
    return round(min(0.98, max(0.05, score)), 4)


def metadata_only_evidence_ratio(evidence: list[dict[str, Any]]) -> float:
    if not evidence:
        return 0.0
    metadata_count = sum(1 for item in evidence if item.get("artifact_type") in {"metadata_only", "guardrail"})
    return metadata_count / max(1, len(evidence))


def suggested_contribution(project: dict[str, Any], lesson: str) -> str:
    title = project.get("title") or "that prior work"
    if lesson:
        return f"I saw a similar pattern on {title}: {lesson}"
    summary = project.get("summary") or "the useful move was making the operating constraint explicit."
    return f"I can connect this to {title}: {summary}"


def significant_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-z0-9][a-z0-9&./+-]{2,}", normalize_lookup(text))
    return dedupe_strings([term for term in terms if term not in GENERIC_TERMS])


def live_entity_overlap(query: str, project_text: str) -> set[str]:
    query_terms = {
        term
        for term in significant_terms(query)
        if term not in GENERIC_TERMS and (len(term) >= 5 or any(char.isdigit() for char in term))
    }
    project_terms = {
        term
        for term in significant_terms(project_text)
        if term not in GENERIC_TERMS and (len(term) >= 5 or any(char.isdigit() for char in term))
    }
    return query_terms & project_terms


def normalize_lookup(text: str) -> str:
    text = text.replace("&", " and ")
    text = text.replace("'", "")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    text = re.sub(r"[^a-zA-Z0-9+./-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()

    # Live ASR tends to spell jargon out phonetically. Normalize the common
    # spoken forms so archival project triggers still match in real time.
    speech_aliases = [
        (r"\bt\.?\s*a\.?\s+smith\b", "ta smith"),
        (r"\bp\s*g\s+(?:and\s+)?e\b", "pg e"),
        (r"\bpg\s+and\s+e\b", "pg e"),
        (r"\bpge\b", "pg e"),
        (r"\bfive\s+hundred\s+(?:k\s*v|kilovolt|kv)\b", "500kv"),
        (r"\b500\s*k\s*v\b", "500kv"),
        (r"\b500\s+kv\b", "500kv"),
        (r"\btwo\s+hundred\s+thirty\s+(?:k\s*v|kilovolt|kv)\b", "230kv"),
        (r"\b230\s*k\s*v\b", "230kv"),
        (r"\b230\s+kv\b", "230kv"),
        (r"\bc\s*t\b", "ct"),
        (r"\bp\s*t\b", "pt"),
    ]
    for pattern, replacement in speech_aliases:
        text = re.sub(pattern, replacement, text)
    return re.sub(r"\s+", " ", text).strip()


def clean_title(text: str) -> str:
    return normalize_text(str(text or "")).replace("Exuction", "Execution").replace("Initiaion", "Initiation")


def project_key(title: str) -> str:
    lookup = normalize_lookup(title)
    key = re.sub(r"[^a-z0-9]+", "_", lookup).strip("_")
    return key or new_id("wmem")


def dedupe_strings(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        clean = normalize_text(str(item or ""))
        if not clean:
            continue
        key = clean.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(clean)
    return output
