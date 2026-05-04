from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from brain_sidecar.core.models import NoteCard, SearchHit, SidecarCard, TranscriptSegment, compact_text, new_id
from brain_sidecar.core.ollama import OllamaClient
from brain_sidecar.core.sidecar_cards import create_sidecar_card, note_card_key


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
    ) -> NoteSynthesisResult:
        if not recent_segments:
            return NoteSynthesisResult(notes=[])

        transcript = "\n".join(
            f"{segment.id} [{segment.start_s:.1f}-{segment.end_s:.1f}] "
            f"{speaker_prefix(segment)}{segment.text}"
            for segment in recent_segments
        )
        recall = "\n".join(recall_line(hit) for hit in recall_hits[:6]) or "- No prior context."
        source_ids = [segment.id for segment in recent_segments]
        system = (
            "You are Brain Sidecar, a local private meeting-intelligence assistant for BP. "
            "Return compact JSON cards that help BP contribute in a technical meeting. "
            "Stay grounded only in the transcript and provided recall/work/web context. "
            "Do not invent facts, do not restate the transcript, and do not imply BP promised "
            "something unless speaker_role=user with adequate confidence."
        )
        user = f"""
Transcript:
{transcript}

Relevant recall, work memory, or web context:
{recall}

Classify only useful, actionable cards:
- action: who owes what by when; use BP-owned only when speaker metadata supports it
- decision: what appears decided or settled
- question: important unresolved question
- risk: assumption or failure mode to watch
- clarification: concise question BP should ask to remove ambiguity
- contribution: concise point BP could say from context
- memory/work_memory/web: useful sourced context
- status/note: only when nothing more actionable is justified

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
      "source_type": "transcript|saved_transcript|work_memory|brave_web|local_file|model_fallback",
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
            if heuristic_cards:
                cards = merge_cards(cards, heuristic_cards)
        except Exception:
            cards = heuristic_cards or [fallback_sidecar_card(session_id, recent_segments, source_ids)]
        if not cards:
            cards = [fallback_sidecar_card(session_id, recent_segments, source_ids)]
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
    )


def parse_meeting_cards(
    content: str,
    *,
    session_id: str,
    recent_segments: list[TranscriptSegment],
) -> list[SidecarCard]:
    payload = _parse_json_object(content)
    source_ids = [segment.id for segment in recent_segments]
    valid_source_ids = set(source_ids)
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
        source_segment_ids = _validated_source_ids(item, valid_source_ids, default=source_ids)
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
            )
        )
    return cards


def heuristic_meeting_cards(session_id: str, recent_segments: list[TranscriptSegment]) -> list[SidecarCard]:
    cards: list[SidecarCard] = []
    for segment in recent_segments[-4:]:
        text = segment.text.strip()
        if not text:
            continue
        lower = text.lower()
        commitment = re.search(r"\b(i'll|i will|i can|i'm going to|i am going to|let me)\b", lower)
        if commitment and is_bp_speaker(segment):
            cards.append(
                create_sidecar_card(
                    session_id=session_id,
                    category="action",
                    title="BP follow-up",
                    body=f"BP appears to own this follow-up: {compact_text(text, limit=220)}",
                    suggested_say="I'll own that follow-up and confirm the next step.",
                    why_now="Speaker identity labels this commitment as BP with enough confidence.",
                    priority="high",
                    confidence=max(0.74, float(segment.speaker_confidence or 0.0)),
                    source_segment_ids=[segment.id],
                    source_type="transcript",
                    card_key=f"action:bp:{segment.id}",
                    ephemeral=True,
                )
            )
        elif commitment and segment.speaker_role == "other":
            cards.append(
                create_sidecar_card(
                    session_id=session_id,
                    category="clarification",
                    title="Confirm owner",
                    body="Another speaker used first-person commitment language; do not assign this to BP without confirmation.",
                    suggested_ask="Can we confirm who owns that follow-up?",
                    why_now="Speaker identity indicates the commitment came from someone other than BP.",
                    priority="normal",
                    confidence=0.68,
                    source_segment_ids=[segment.id],
                    source_type="transcript",
                    card_key=f"clarify:owner:{segment.id}",
                    ephemeral=True,
                )
            )
        elif commitment:
            cards.append(
                create_sidecar_card(
                    session_id=session_id,
                    category="clarification",
                    title="Owner unclear",
                    body="A follow-up may have been promised, but the speaker was unknown or low confidence.",
                    suggested_ask="Who should own that follow-up?",
                    why_now="Speaker identity is not confident enough to assign the action to BP.",
                    priority="normal",
                    confidence=0.52,
                    source_segment_ids=[segment.id],
                    source_type="transcript",
                    card_key=f"clarify:unknown-owner:{segment.id}",
                    ephemeral=True,
                )
            )
    risk_text = " ".join(segment.text for segment in recent_segments[-4:]).lower()
    if any(term in risk_text for term in ["risk", "rollback", "failure", "blocked", "deadline", "dependency"]):
        cards.append(
            create_sidecar_card(
                session_id=session_id,
                category="risk",
                title="Watch the risk",
                body="Recent speech points to a risk or dependency that may need an owner or mitigation.",
                suggested_ask="What would make this plan fail, and who owns that mitigation?",
                why_now="Risk language appeared in the latest transcript window.",
                priority="high",
                confidence=0.64,
                source_segment_ids=[segment.id for segment in recent_segments[-4:]],
                source_type="transcript",
                card_key="risk:recent-window",
                ephemeral=True,
            )
        )
    return cards[:3]


def note_from_sidecar(card: SidecarCard) -> NoteCard:
    kind = "context" if card.category in {"status", "memory", "work_memory", "web", "note"} else card.category
    return NoteCard(
        id=new_id("note"),
        session_id=card.session_id,
        kind=kind,
        title=card.title,
        body=card.body,
        source_segment_ids=card.source_segment_ids,
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


def _validated_source_ids(item: dict, valid_source_ids: set[str], *, default: list[str]) -> list[str]:
    if "source_segment_ids" not in item:
        return default
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


def is_bp_speaker(segment: TranscriptSegment) -> bool:
    return (
        segment.speaker_role == "user"
        and not segment.speaker_low_confidence
        and (segment.speaker_confidence is None or segment.speaker_confidence >= 0.82)
    )


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
