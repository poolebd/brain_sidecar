from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from brain_sidecar.core.domain_keywords import EnergyConversationFrame
from brain_sidecar.core.meeting_agents import MemoryBoundaryAgent, deterministic_meeting_cards
from brain_sidecar.core.meeting_contract import MeetingContract, contract_prompt_block
from brain_sidecar.core.models import NoteCard, SearchHit, SidecarCard, TranscriptSegment, compact_text, new_id
from brain_sidecar.core.ollama import OllamaClient
from brain_sidecar.core.sidecar_cards import create_sidecar_card, note_card_key

REVIEW_PATH_NAMES = ("greg", "sunil", "kyle")
REVIEW_PATH_CUE_RE = re.compile(
    r"\b("
    r"under review|review from|came for review|for review|reviewer|focal point|"
    r"review\s+(?:all\s+)?(?:four\s+)?documents?|documents?\s+(?:which\s+)?(?:came\s+)?for review"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NoteSynthesisResult:
    notes: list[NoteCard]
    sidecar_cards: list[SidecarCard] = field(default_factory=list)


class NoteSynthesizer:
    def __init__(self, ollama: OllamaClient) -> None:
        self.ollama = ollama

    async def synthesize(
        self,
        session_id: str,
        recent_segments: list[TranscriptSegment],
        recall_hits: list[SearchHit],
        meeting_contract: MeetingContract | None = None,
        energy_frame: EnergyConversationFrame | None = None,
    ) -> NoteSynthesisResult:
        if not recent_segments:
            return NoteSynthesisResult(notes=[])

        transcript = "\n".join(
            f"{source_ref(segment)} [{segment.start_s:.1f}-{segment.end_s:.1f}] "
            f"{speaker_prefix(segment)}{segment.text}"
            for segment in recent_segments
        )
        recall = "\n".join(recall_line(hit) for hit in recall_hits[:6]) or "- No prior context."
        source_ids = valid_source_ids_for_segments(recent_segments)
        system = (
            "You are Brain Sidecar, a local private meeting-intelligence assistant for BP. "
            "Return very few compact JSON cards that help an energy/project consultant during "
            "or immediately after a meeting. Stay grounded only in transcript evidence. "
            "Recall is reminder context only; never turn recall-only entities into current-meeting facts. "
            "Do not invent facts, do not restate the transcript, and do not imply BP promised "
            "something unless speaker_role=user with adequate confidence. If evidence is weak, return no cards."
        )
        contract_block = contract_prompt_block(meeting_contract)
        energy_block = energy_prompt_block(energy_frame)
        user = f"""
{contract_block}
{energy_block}

Transcript:
{transcript}

Relevant recall context for reminders only:
{recall}

Classify only useful, evidence-backed cards:
- action: explicit review, send, reply, confirm, schedule, hours, or follow-up work; use BP-owned only when speaker metadata supports it
- decision: what appears decided or settled
- question: important unresolved technical or coordination question
- risk: assumption or dependency to watch
- clarification: concise question BP should ask to remove ambiguity
- contribution: concise point BP could say from transcript evidence

Prefer fewer cards. Return no card unless it would help an energy/project consultant.

Do not infer finance, HR, policy, maintenance, or generic project-management topics unless explicitly stated.
Do not create CT/PT, relay, protection, breaker, trip-path, interlock, or settings notes unless those concepts are present in transcript evidence.
Do not convert noisy phrases like "contribute", "Brandon", or "taxes" into financial obligations unless the transcript explicitly discusses actual financial obligations.
Do not introduce memory-only entities into a current-meeting note.
Avoid generic meeting notes, transcript echoes, unsupported source IDs, private path exposure, and overconfident advice.

Return JSON only:
{{
  "cards": [
    {{
      "category": "action|decision|question|risk|clarification|contribution|memory|work_memory|web|status|note",
      "title": "short title",
      "body": "one or two grounded sentences",
      "suggested_say": "optional concise sentence BP could say",
      "suggested_ask": "optional concise question BP could ask",
      "why_now": "why this matters right now",
      "priority": "low|normal|high",
      "confidence": 0.0,
      "source_segment_ids": ["segment ids from Transcript only"],
      "evidence_quote": "short direct quote or close excerpt from those transcript segments",
      "owner": "optional owner if directly supported",
      "due_date": "optional due date or timing if directly supported",
      "missing_info": "optional missing detail if directly useful",
      "source_type": "transcript",
      "sources": [],
      "citations": [],
      "card_key": "optional stable dedupe key"
    }}
  ]
}}
"""
        heuristic_cards = heuristic_meeting_cards(session_id, recent_segments)
        try:
            content = await self.ollama.chat(system, user, format_json=True)
            cards = parse_meeting_cards(content, session_id=session_id, recent_segments=recent_segments)
            cards = MemoryBoundaryAgent().filter_current_meeting_cards(cards, recent_segments, recall_hits)
            if heuristic_cards:
                cards = merge_cards(cards, heuristic_cards)
        except Exception:
            cards = heuristic_cards or [fallback_sidecar_card(session_id, recent_segments, source_ids)]
        return NoteSynthesisResult(notes=[note_from_sidecar(card) for card in cards], sidecar_cards=cards)


def fallback_note(session_id: str, recent_segments: list[TranscriptSegment], source_ids: list[str]) -> NoteCard:
    card = fallback_sidecar_card(session_id, recent_segments, source_ids)
    return note_from_sidecar(card)


def fallback_sidecar_card(
    session_id: str,
    recent_segments: list[TranscriptSegment],
    source_ids: list[str],
) -> SidecarCard:
    recent_text = compact_text(" ".join(segment.text.strip() for segment in recent_segments[-4:] if segment.text.strip()), limit=260)
    body = f"Current discussion thread: {recent_text}" if recent_text else "Listening for the next clear technical thread."
    return create_sidecar_card(
        session_id=session_id,
        category="status",
        title="Current thread",
        body=body,
        why_now="Ollama did not return usable structured meeting-intelligence JSON.",
        priority="low",
        confidence=0.42,
        source_segment_ids=source_ids,
        source_type="model_fallback",
        card_key="note:status:current thread",
        ephemeral=True,
        evidence_quote=recent_text,
    )


def parse_meeting_cards(
    content: str,
    *,
    session_id: str,
    recent_segments: list[TranscriptSegment],
) -> list[SidecarCard]:
    payload = _parse_json_object(content)
    source_ids = valid_source_ids_for_segments(recent_segments)
    valid_source_ids = set(valid_source_ids_for_segments(recent_segments))
    recent_text = " ".join(segment.text for segment in recent_segments)
    raw_items = payload.get("cards")
    if raw_items is None:
        raw_items = payload.get("notes", [])
    if not isinstance(raw_items, list):
        return []
    cards: list[SidecarCard] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        source_segment_ids = _validated_source_ids(item, valid_source_ids)
        title = compact_text(item.get("title") or "Sidecar note", limit=140)
        body = compact_text(item.get("body") or "", limit=900)
        suggested_say = item.get("suggested_say")
        suggested_ask = item.get("suggested_ask")
        if not body and not suggested_say and not suggested_ask:
            continue
        if is_echo_card(title, body, suggested_say, suggested_ask, recent_text):
            continue
        category = item.get("category", item.get("kind", "note"))
        source_type = item.get("source_type") or "transcript"
        confidence = item.get("confidence", 0.62)
        priority = item.get("priority", "normal")
        card_key = compact_text(item.get("card_key") or "", limit=180) or note_card_key(category, title)
        cards.append(
            create_sidecar_card(
                session_id=session_id,
                category=category,
                title=title,
                body=body,
                suggested_say=suggested_say,
                suggested_ask=suggested_ask,
                why_now=item.get("why_now") or "Generated from the recent transcript and available context.",
                priority=priority,
                confidence=confidence,
                source_segment_ids=source_segment_ids,
                source_type=source_type,
                sources=item.get("sources") or [],
                citations=item.get("citations") or [],
                card_key=card_key,
                ephemeral=True,
                evidence_quote=item.get("evidence_quote") or "",
                owner=item.get("owner"),
                due_date=item.get("due_date"),
                missing_info=item.get("missing_info"),
            )
        )
    return cards


def heuristic_meeting_cards(session_id: str, recent_segments: list[TranscriptSegment]) -> list[SidecarCard]:
    cards: list[SidecarCard] = heuristic_project_review_cards(session_id, recent_segments)
    cards.extend(deterministic_meeting_cards(session_id, recent_segments))
    return cards[:5]


def energy_prompt_block(frame: EnergyConversationFrame | None) -> str:
    if frame is None or not frame.active:
        return ""
    categories = ", ".join(str(item.get("category")) for item in frame.top_categories[:3]) or "unknown"
    keywords = ", ".join(str(item.get("phrase")) for item in frame.top_keywords[:6]) or "unknown"
    return f"""
Energy consulting lens active.
- Top categories: {categories}
- Top keywords: {keywords}
- Evidence quote: {frame.evidence_quote}
- Use this only to choose better ask/say/do/watch cards. Do not turn keyword matches into facts. Current transcript evidence still controls what can be stated.
"""


def heuristic_project_review_cards(session_id: str, recent_segments: list[TranscriptSegment]) -> list[SidecarCard]:
    combined = " ".join(segment.text for segment in recent_segments)
    clean = combined.lower()
    cards: list[SidecarCard] = []
    if "siemens" in clean and ("document" in clean or "documents" in clean) and "review" in clean:
        selected = select_evidence_segments(recent_segments, ["siemens", "document", "documents", "review", "greg", "sunil"])
        if not (
            any("siemens" in segment.text.lower() for segment in selected)
            and any("document" in segment.text.lower() and "review" in segment.text.lower() for segment in selected)
        ):
            selected = []
    else:
        selected = []
    if selected:
        cards.append(
            heuristic_card(
                session_id,
                selected,
                category="action",
                title="Siemens document review",
                body="Review the Siemens document.",
                evidence_hint="siemens",
                priority="high",
            )
        )
    if "rfi" in clean and ("comment" in clean or "comments" in clean or "scope" in clean or "deviation" in clean):
        selected = select_evidence_segments(recent_segments, ["rfi", "comment", "comments", "scope", "deviation", "deviations"])
        cards.append(
            heuristic_card(
                session_id,
                selected,
                category="action",
                title="RFI log review",
                body="Review the RFI log.",
                evidence_hint="rfi",
                priority="normal",
            )
        )
    if "monday" in clean and ("send" in clean or "comments" in clean or "target" in clean):
        selected = select_evidence_segments(recent_segments, ["monday", "send", "comments", "target"])
        cards.append(
            heuristic_card(
                session_id,
                selected,
                category="action",
                title="Send by Monday",
                body="Send by Monday.",
                evidence_hint="monday",
                priority="high",
            )
        )
    if ("spec" in clean or "scope" in clean) and ("deviation" in clean or "deviations" in clean):
        selected = select_evidence_segments(recent_segments, ["spec", "scope", "deviation", "deviations", "submitted"])
        cards.append(
            heuristic_card(
                session_id,
                selected,
                category="question",
                title="Spec deviation review",
                body="Review the spec deviations.",
                evidence_hint="deviation",
                priority="normal",
            )
        )
    selected = select_review_path_segments(recent_segments)
    names = names_in_segments(selected)
    if len(names) >= 2:
        body_names = ", ".join(name.title() for name in names[:-1])
        body_names = f"{body_names}, or {names[-1].title()}" if len(names) > 2 else " or ".join(name.title() for name in names)
        cards.append(
            heuristic_card(
                session_id,
                selected,
                category="clarification",
                title="Review path",
                body=f"Clarify the review path with {body_names}.",
                evidence_hint=review_path_evidence_hint(selected),
                priority="normal",
            )
        )
    return dedupe_heuristic_cards(cards)


def heuristic_card(
    session_id: str,
    segments: list[TranscriptSegment],
    *,
    category: str,
    title: str,
    body: str,
    evidence_hint: str,
    priority: str,
) -> SidecarCard:
    evidence_segments = segments or []
    source_ids = valid_source_ids_for_segments(evidence_segments)
    evidence_quote = evidence_quote_for(evidence_segments, evidence_hint)
    key = normalize_for_echo(f"heuristic {category} {title}")
    return create_sidecar_card(
        session_id=session_id,
        category=category,
        title=title,
        body=body,
        why_now="Explicit project-review language appeared in the recent transcript.",
        priority=priority,
        confidence=0.7,
        source_segment_ids=source_ids,
        source_type="transcript",
        card_key=f"heuristic:{category}:{key}",
        ephemeral=True,
        evidence_quote=evidence_quote,
    )


def select_evidence_segments(segments: list[TranscriptSegment], terms: list[str]) -> list[TranscriptSegment]:
    selected: list[TranscriptSegment] = []
    for segment in segments:
        text = segment.text.lower()
        if any(term in text for term in terms):
            selected.append(segment)
    if len(selected) == 1 and len(segments) > 1:
        index = segments.index(selected[0])
        neighbors = segments[max(0, index - 1) : min(len(segments), index + 2)]
        selected = dedupe_segments([*selected, *neighbors])
    return selected[:4]


def select_review_path_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    selected: list[TranscriptSegment] = []
    for index, segment in enumerate(segments):
        text = segment.text.lower()
        has_name = any(name in text for name in REVIEW_PATH_NAMES)
        has_cue = bool(REVIEW_PATH_CUE_RE.search(text))
        if has_name and has_cue:
            selected.append(segment)
            continue
        if has_name and nearby_has_review_cue(segments, index):
            selected.append(segment)
            continue
        if has_cue and nearby_has_review_name(segments, index):
            selected.append(segment)
    return dedupe_segments(selected)[:4]


def nearby_has_review_cue(segments: list[TranscriptSegment], index: int) -> bool:
    for neighbor in segments[max(0, index - 1) : min(len(segments), index + 2)]:
        if REVIEW_PATH_CUE_RE.search(neighbor.text):
            return True
    return False


def nearby_has_review_name(segments: list[TranscriptSegment], index: int) -> bool:
    for neighbor in segments[max(0, index - 1) : min(len(segments), index + 2)]:
        text = neighbor.text.lower()
        if any(name in text for name in REVIEW_PATH_NAMES):
            return True
    return False


def names_in_segments(segments: list[TranscriptSegment]) -> list[str]:
    names: list[str] = []
    combined = " ".join(segment.text.lower() for segment in segments)
    for name in REVIEW_PATH_NAMES:
        if name in combined:
            names.append(name)
    return names


def review_path_evidence_hint(segments: list[TranscriptSegment]) -> str:
    for hint in ["under review", "focal point", "reviewer", "for review", "review"]:
        if any(hint in segment.text.lower() for segment in segments):
            return hint
    return "review"


def evidence_quote_for(segments: list[TranscriptSegment], hint: str) -> str:
    for segment in segments:
        if hint.lower() in segment.text.lower():
            return compact_text(segment.text, limit=260)
    return compact_text(" ".join(segment.text for segment in segments[:2]), limit=260)


def dedupe_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    seen: set[str] = set()
    result: list[TranscriptSegment] = []
    for segment in segments:
        if segment.id in seen:
            continue
        seen.add(segment.id)
        result.append(segment)
    return result


def dedupe_heuristic_cards(cards: list[SidecarCard]) -> list[SidecarCard]:
    seen: set[str] = set()
    result: list[SidecarCard] = []
    for card in cards:
        key = card.card_key or f"{card.category}:{card.title}"
        if key in seen or not card.source_segment_ids or not card.evidence_quote:
            continue
        seen.add(key)
        result.append(card)
    return result


def note_from_sidecar(card: SidecarCard) -> NoteCard:
    kind = "context" if card.category in {"status", "memory", "work_memory", "web", "note"} else card.category
    return NoteCard(
        id=new_id("note"),
        session_id=card.session_id,
        kind=kind,
        title=card.title,
        body=card.body,
        source_segment_ids=card.source_segment_ids,
        evidence_quote=card.evidence_quote,
        owner=card.owner,
        due_date=card.due_date,
        missing_info=card.missing_info,
    )


def _parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
        if fenced:
            return json.loads(fenced.group(1))
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def speaker_prefix(segment: TranscriptSegment) -> str:
    if segment.speaker_label:
        confidence = f" {segment.speaker_confidence:.2f}" if segment.speaker_confidence is not None else ""
        low = " low_confidence" if segment.speaker_low_confidence else ""
        return f"speaker={segment.speaker_label} role={segment.speaker_role or 'unknown'} confidence={confidence}{low}: "
    if segment.speaker_role:
        return f"speaker_role={segment.speaker_role}: "
    return "speaker=unknown: "


def recall_line(hit: SearchHit) -> str:
    title = hit.metadata.get("title") if isinstance(hit.metadata, dict) else None
    source = f"{hit.source_type}:{hit.source_id}"
    return f"- {source} score={hit.score:.2f} title={title or ''}: {hit.text[:520]}"


def source_ref(segment: TranscriptSegment) -> str:
    source_ids = segment.source_segment_ids or [segment.id]
    return ",".join(source_ids)


def valid_source_ids_for_segments(segments: list[TranscriptSegment]) -> list[str]:
    seen: set[str] = set()
    source_ids: list[str] = []
    for segment in segments:
        for source_id in segment.source_segment_ids or [segment.id]:
            if source_id in seen:
                continue
            seen.add(source_id)
            source_ids.append(source_id)
    return source_ids


def _validated_source_ids(item: dict, valid_source_ids: set[str]) -> list[str]:
    if "source_segment_ids" not in item:
        return []
    raw = item.get("source_segment_ids")
    if not isinstance(raw, list):
        return []
    return [str(source_id) for source_id in raw if str(source_id) in valid_source_ids]


def is_echo_card(title: object, body: object, suggested_say: object, suggested_ask: object, transcript_text: str) -> bool:
    candidate = normalize_for_echo(" ".join(str(part or "") for part in [title, body, suggested_say, suggested_ask]))
    transcript = normalize_for_echo(transcript_text)
    if not candidate or not transcript:
        return False
    if candidate in transcript and len(candidate) > 80:
        return True
    candidate_tokens = set(candidate.split())
    transcript_tokens = set(transcript.split())
    if len(candidate_tokens) < 8:
        return False
    overlap = len(candidate_tokens & transcript_tokens) / max(1, len(candidate_tokens))
    return overlap >= 0.9


def normalize_for_echo(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def merge_cards(primary: list[SidecarCard], secondary: list[SidecarCard]) -> list[SidecarCard]:
    merged: list[SidecarCard] = []
    seen: set[str] = set()
    for card in [*primary, *secondary]:
        key = card.card_key or f"{card.category}:{card.title.lower()}"
        if key in seen:
            continue
        seen.add(key)
        merged.append(card)
    return merged[:5]
