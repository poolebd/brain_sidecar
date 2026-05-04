from __future__ import annotations

import json
import re
from dataclasses import dataclass

from brain_sidecar.core.models import NoteCard, SearchHit, TranscriptSegment, new_id
from brain_sidecar.core.ollama import OllamaClient


@dataclass(frozen=True)
class NoteSynthesisResult:
    notes: list[NoteCard]


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
            f"[{segment.start_s:.1f}-{segment.end_s:.1f}] {segment.text}" for segment in recent_segments
        )
        recall = "\n".join(f"- {hit.text[:500]}" for hit in recall_hits[:5]) or "- No prior context."
        source_ids = [segment.id for segment in recent_segments]
        system = (
            "You are a local private note-taking sidecar. Write concise, natural sidecar replies "
            "that sound conversational and useful in the moment. Stay grounded only in the transcript "
            "and recall snippets. Return JSON only; do not invent facts."
        )
        user = f"""
Transcript:
{transcript}

Relevant recall:
{recall}

Return this JSON shape:
{{
  "notes": [
    {{"kind": "topic|action|decision|question|context", "title": "short title", "body": "one to three concise, conversational sentences"}}
  ]
}}
"""
        try:
            content = await self.ollama.chat(system, user, format_json=True)
            payload = _parse_json_object(content)
        except Exception:
            return NoteSynthesisResult(notes=[fallback_note(session_id, recent_segments, source_ids)])
        notes: list[NoteCard] = []
        for item in payload.get("notes", []):
            kind = str(item.get("kind", "topic"))[:32]
            title = str(item.get("title", "Untitled note")).strip()[:140]
            body = str(item.get("body", "")).strip()
            if body:
                notes.append(
                    NoteCard(
                        id=new_id("note"),
                        session_id=session_id,
                        kind=kind,
                        title=title,
                        body=body,
                        source_segment_ids=source_ids,
                    )
                )
        return NoteSynthesisResult(notes=notes)


def fallback_note(session_id: str, recent_segments: list[TranscriptSegment], source_ids: list[str]) -> NoteCard:
    recent_text = " ".join(segment.text.strip() for segment in recent_segments[-4:] if segment.text.strip())
    clipped = recent_text[:320].rsplit(" ", 1)[0].strip() if len(recent_text) > 320 else recent_text
    body = f"I'm hearing: {clipped}" if clipped else "I'm listening for the next clear thread."
    return NoteCard(
        id=new_id("note"),
        session_id=session_id,
        kind="context",
        title="Current thread",
        body=body,
        source_segment_ids=source_ids,
    )


def _parse_json_object(content: str) -> dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
