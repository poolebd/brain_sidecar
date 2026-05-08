from __future__ import annotations

from dataclasses import dataclass

from brain_sidecar.core.domain_keywords import EnergyConversationFrame, normalize_keyword_text
from brain_sidecar.core.meeting_contract import MeetingContract
from brain_sidecar.core.models import SidecarCard, TranscriptSegment, compact_text
from brain_sidecar.core.sidecar_cards import create_sidecar_card


@dataclass(frozen=True)
class EnergyCardPattern:
    key: str
    title: str
    body: str
    suggested_ask: str
    triggers: tuple[str, ...]
    categories: tuple[str, ...]


ENERGY_CARD_PATTERNS = (
    EnergyCardPattern(
        key="billing-tariff",
        title="Tariff inputs",
        body="Recent transcript points to utility billing or tariff analysis.",
        suggested_ask="Can we get the rate schedule, 12 months of bills, and interval data so we can separate usage, demand charges, and tariff fit?",
        triggers=("utility bill analysis", "tariff analysis", "demand charge management", "time of use rate", "peak demand", "load profile"),
        categories=("operations", "market"),
    ),
    EnergyCardPattern(
        key="audit-benchmarking",
        title="Assessment baseline",
        body="Recent transcript points to audit, benchmarking, or energy-performance baseline work.",
        suggested_ask="What baseline period, facility scope, and EUI or benchmark should we use for the assessment?",
        triggers=("energy audit", "ashrae level 2 audit", "energy star portfolio manager", "energy benchmarking", "energy use intensity", "eui", "energy baseline", "energy performance indicator"),
        categories=("service", "operations", "regulatory"),
    ),
    EnergyCardPattern(
        key="controls-analytics",
        title="Controls evidence",
        body="Recent transcript points to commissioning, controls, or analytics diagnostics.",
        suggested_ask="Do we have trend data, control sequences, and fault history to separate controls issues from equipment issues?",
        triggers=("commissioning", "retro commissioning", "monitoring based commissioning", "emis", "building automation system", "bas", "fault detection and diagnostics", "fdd", "sequence of operations"),
        categories=("operations", "technology"),
    ),
    EnergyCardPattern(
        key="procurement",
        title="Procurement path",
        body="Recent transcript points to energy procurement or renewable-contracting choices.",
        suggested_ask="Are we evaluating physical supply, VPPA, green tariff, or unbundled certificates, and how should RECs/EACs be treated?",
        triggers=("energy procurement", "renewable energy procurement", "power purchase agreement", "ppa", "virtual ppa", "physical ppa", "renewable energy certificate", "energy attribute certificate", "green tariff"),
        categories=("market", "sustainability"),
    ),
    EnergyCardPattern(
        key="flexibility-storage",
        title="Flexibility objective",
        body="Recent transcript points to demand flexibility, load shifting, storage, or distributed resources.",
        suggested_ask="Which loads are flexible, what constraints apply, and what revenue or avoided-cost signal are we optimizing against?",
        triggers=("demand response", "demand flexibility", "load shifting", "peak shaving", "battery energy storage system", "bess", "distributed energy resource", "der", "virtual power plant"),
        categories=("market", "technology"),
    ),
    EnergyCardPattern(
        key="carbon-accounting",
        title="Scope 2 basis",
        body="Recent transcript points to carbon accounting, emissions factors, or decarbonization planning.",
        suggested_ask="Are we using market-based or location-based Scope 2 accounting, and what emissions factors or instruments support the claim?",
        triggers=("scope 1 emissions", "scope 2 emissions", "scope 3 emissions", "carbon accounting", "greenhouse gas inventory", "market based emissions", "location based emissions", "emissions factor", "decarbonization roadmap"),
        categories=("sustainability", "regulatory"),
    ),
    EnergyCardPattern(
        key="finance",
        title="Decision metric",
        body="Recent transcript points to an energy business case, incentive, or project-finance discussion.",
        suggested_ask="What decision metric matters here: simple payback, NPV/IRR, avoided cost, incentives, or a performance guarantee?",
        triggers=("business case", "simple payback", "return on investment", "roi", "net present value", "npv", "internal rate of return", "irr", "capital expenditure", "operating expenditure", "utility incentive", "utility rebate", "energy savings performance contract"),
        categories=("finance",),
    ),
    EnergyCardPattern(
        key="commercial-scoping",
        title="Scope decision",
        body="Recent transcript points to scoping, proposal, or roadmap framing.",
        suggested_ask="What decision should this scope support: screen opportunities, price implementation, support procurement, or build a roadmap?",
        triggers=("request for proposal", "proposal", "scope of work", "feasibility study", "site assessment", "implementation roadmap", "energy strategy"),
        categories=("commercial", "service"),
    ),
)


class EnergyConsultingAgent:
    def cards(
        self,
        session_id: str,
        recent_segments: list[TranscriptSegment],
        frame: EnergyConversationFrame,
        meeting_contract: MeetingContract | None = None,
        *,
        max_cards: int = 1,
    ) -> list[SidecarCard]:
        del meeting_contract
        if not frame.active or frame.confidence not in {"medium", "high"}:
            return []
        candidates = self._candidate_patterns(frame)
        if not candidates:
            return []
        limit = max(1, int(max_cards))
        if frame.confidence != "high":
            limit = 1
        selected: list[tuple[EnergyCardPattern, float]] = []
        used_categories: set[str] = set()
        for pattern, score in candidates:
            pattern_categories = set(pattern.categories)
            if selected and pattern_categories & used_categories:
                continue
            selected.append((pattern, score))
            used_categories |= pattern_categories
            if len(selected) >= limit:
                break
        evidence_ids = frame.evidence_segment_ids or [segment.id for segment in recent_segments[-2:]]
        evidence_quote = frame.evidence_quote or compact_text(" ".join(segment.text for segment in recent_segments[-2:]), limit=320)
        cards: list[SidecarCard] = []
        for pattern, score in selected:
            cards.append(
                create_sidecar_card(
                    session_id=session_id,
                    category="clarification",
                    title=pattern.title,
                    body=pattern.body,
                    suggested_ask=pattern.suggested_ask,
                    why_now=f"{frame.summary_label} is active from current transcript evidence.",
                    priority="normal" if frame.confidence == "medium" else "high",
                    confidence=min(0.92, max(0.72, score / 2.2)),
                    source_segment_ids=evidence_ids,
                    source_type="transcript",
                    card_key=f"energy:{pattern.key}:{':'.join(evidence_ids[-3:])}",
                    ephemeral=True,
                    evidence_quote=evidence_quote,
                )
            )
        return cards

    def _candidate_patterns(self, frame: EnergyConversationFrame) -> list[tuple[EnergyCardPattern, float]]:
        keyword_tokens = {
            normalize_keyword_text(str(item.get("normalized_token") or item.get("phrase") or item.get("matched_variant") or ""))
            for item in frame.top_keywords
        }
        category_scores = {str(item.get("category")): float(item.get("score") or 0.0) for item in frame.top_categories}
        candidates: list[tuple[EnergyCardPattern, float]] = []
        for pattern in ENERGY_CARD_PATTERNS:
            trigger_score = 0.0
            for trigger in pattern.triggers:
                normalized_trigger = normalize_keyword_text(trigger)
                if any(_token_matches(normalized_trigger, token) for token in keyword_tokens):
                    trigger_score += 1.0
            category_score = sum(category_scores.get(category, 0.0) for category in pattern.categories)
            score = trigger_score + category_score
            if trigger_score > 0:
                candidates.append((pattern, score))
        return sorted(candidates, key=lambda item: (-item[1], item[0].key))


def _token_matches(trigger: str, token: str) -> bool:
    return trigger == token or trigger in token or token in trigger
