from __future__ import annotations

import asyncio
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

from brain_sidecar.core.models import TranscriptSegment, new_id


BRAVE_WEB_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


@dataclass(frozen=True)
class WebContextCandidate:
    query: str
    normalized_query: str
    freshness: str | None
    source_segment_ids: list[str]


@dataclass(frozen=True)
class WebContextDecision:
    candidate: WebContextCandidate | None
    skip_reason: str | None = None
    sanitized_query: str = ""
    freshness: str | None = None


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    description: str = ""

    def source_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url}


class WebTriggerDetector:
    """Detect explicit questions where lightweight current web context helps."""

    _QUESTION_RE = re.compile(
        r"\b(what|what's|which|how|should|is|are|do|does|can)\b|[?]",
        re.IGNORECASE,
    )
    _CURRENT_RE = re.compile(
        r"\b(current|currently|latest|new|recent|modern|today|now|best practices?|"
        r"recommended|standard|standards|practice|practices|release|news|changelog|202[4-9])\b",
        re.IGNORECASE,
    )
    _FRESH_MONTH_RE = re.compile(r"\b(latest|news|release|released|recent|today|this month|changelog)\b", re.IGNORECASE)
    _FRESH_YEAR_RE = re.compile(
        r"\b(current|currently|modern|now|best practices?|recommended|standard|standards|practice|practices|tool|tools)\b",
        re.IGNORECASE,
    )
    _TECH_RE = re.compile(
        r"\b(ai|api|asr|auth|browser|cloud|code|coding|compiler|container|cuda|database|db|docker|"
        r"embedding|framework|gpu|index|indexing|kubernetes|library|llm|model|oauth|openapi|package|"
        r"python|react|release|sdk|security|software|sql|standard|standards|tool|tools|typescript|"
        r"vector|web)\b",
        re.IGNORECASE,
    )
    _PERSONAL_MEMORY_RE = re.compile(
        r"\b(what|when|where|who|how)\s+(did|do|were|was|are)\s+(we|i)\b|"
        r"\b(remember|recall|our notes|my notes|that meeting|the meeting)\b",
        re.IGNORECASE,
    )
    _PRIVATE_TAIL_RE = re.compile(
        r"\b(?:for|on|about|inside)\s+(?:our|my|the)\s+"
        r"(?:internal|private|confidential|proprietary|client|customer|company|team|project|account|workspace)"
        r"\b.*$",
        re.IGNORECASE,
    )
    _PRIVATE_WORD_RE = re.compile(
        r"\b(internal|private|confidential|proprietary|client|customer|company|team|workspace|"
        r"our|ours|my|mine|we|us|project)\b",
        re.IGNORECASE,
    )
    _QUESTION_PREFIX_RE = re.compile(
        r"^\s*(hey|okay|ok|so|please|can you|could you|would you|tell me|look up|search for|"
        r"what are|what is|what's|which are|which is|how do|how should|should we|should i|"
        r"is there|are there|do we|does)\b",
        re.IGNORECASE,
    )

    def __init__(self, *, max_query_chars: int = 140) -> None:
        self.max_query_chars = max_query_chars

    def detect(self, recent_segments: list[TranscriptSegment]) -> WebContextCandidate | None:
        return self.decision_for_segments(recent_segments).candidate

    def decision_for_segments(self, recent_segments: list[TranscriptSegment]) -> WebContextDecision:
        if not recent_segments:
            return WebContextDecision(candidate=None, skip_reason="no_recent_transcript")

        window = " ".join(segment.text.strip() for segment in recent_segments[-8:] if segment.text.strip())
        if self._looks_private(window):
            return WebContextDecision(candidate=None, skip_reason="query_looked_private_or_internal")
        question = self._latest_question(window)
        if not question:
            return WebContextDecision(candidate=None, skip_reason="query_lacked_public_current_technical_signals")
        if not self._is_public_current_tech_question(question):
            return WebContextDecision(candidate=None, skip_reason="query_lacked_public_current_technical_signals")

        query = self.build_query(question)
        if len(query.split()) < 3:
            return WebContextDecision(
                candidate=None,
                skip_reason="query_empty_after_sanitization",
                sanitized_query=query,
                freshness=self.freshness_for(question),
            )

        candidate = WebContextCandidate(
            query=query,
            normalized_query=self.normalize_query(query),
            freshness=self.freshness_for(question),
            source_segment_ids=[segment.id for segment in recent_segments[-8:]],
        )
        return WebContextDecision(candidate=candidate, sanitized_query=query, freshness=candidate.freshness)

    def decision_for_manual_query(self, text: str) -> WebContextDecision:
        if self._looks_private(text):
            return WebContextDecision(candidate=None, skip_reason="query_looked_private_or_internal")
        query = self.build_query(text)
        freshness = self.freshness_for(text)
        if len(query.split()) < 2:
            return WebContextDecision(
                candidate=None,
                skip_reason="query_empty_after_sanitization",
                sanitized_query=query,
                freshness=freshness,
            )
        candidate = WebContextCandidate(
            query=query,
            normalized_query=self.normalize_query(query),
            freshness=freshness,
            source_segment_ids=[],
        )
        return WebContextDecision(candidate=candidate, sanitized_query=query, freshness=freshness)

    def build_query(self, text: str) -> str:
        question = self._latest_question(text) or text
        query = re.sub(r"\[[^\]]+\]", " ", question)
        query = query.replace("?", " ")
        query = self._PRIVATE_TAIL_RE.sub(" ", query)
        query = re.sub(r"\bT\.?\s*A\.?\s+Smith\b", " ", query, flags=re.IGNORECASE)
        query = re.sub(r"\b(?:project|account)\s+[A-Z][A-Za-z0-9_.-]+\b", " ", query)
        for _ in range(3):
            query = self._QUESTION_PREFIX_RE.sub(" ", query)
        query = self._PRIVATE_WORD_RE.sub(" ", query)
        query = re.sub(r"[^A-Za-z0-9+.#/\- ]+", " ", query)
        query = re.sub(r"\s+", " ", query).strip(" -/.,")
        return self._cap_query(query)

    def freshness_for(self, text: str) -> str | None:
        if self._FRESH_MONTH_RE.search(text):
            return "pm"
        if self._FRESH_YEAR_RE.search(text):
            return "py"
        return None

    def normalize_query(self, query: str) -> str:
        normalized = re.sub(r"[^a-z0-9+.#]+", " ", query.lower())
        return re.sub(r"\s+", " ", normalized).strip()

    def _latest_question(self, text: str) -> str | None:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return None
        sentences = re.split(r"(?<=[?.!])\s+", compact)
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if self._QUESTION_RE.search(sentence):
                return sentence[-420:]
        return None

    def _is_public_current_tech_question(self, question: str) -> bool:
        if not self._QUESTION_RE.search(question):
            return False
        if self._PERSONAL_MEMORY_RE.search(question) and not self._CURRENT_RE.search(question):
            return False
        return bool(self._CURRENT_RE.search(question) and self._TECH_RE.search(question))

    def _looks_private(self, text: str) -> bool:
        return bool(
            self._PRIVATE_TAIL_RE.search(text)
            or self._PERSONAL_MEMORY_RE.search(text)
            or re.search(r"\bwhat\s+did\s+(we|i)\b", text, re.IGNORECASE)
        )

    def _cap_query(self, query: str) -> str:
        if len(query) <= self.max_query_chars:
            return query
        capped = query[: self.max_query_chars].rsplit(" ", 1)[0].strip()
        return capped or query[: self.max_query_chars].strip()


class BraveSearchClient:
    def __init__(
        self,
        api_key: str,
        *,
        timeout_s: float = 3.0,
        max_results: int = 3,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.timeout_s = timeout_s
        self.max_results = max(1, min(10, max_results))
        self._opener = opener or urllib.request.urlopen
        self.last_error_reason: str | None = None

    async def search(self, query: str, *, freshness: str | None = None) -> list[WebSearchResult]:
        self.last_error_reason = None
        if not self.api_key or not query.strip():
            return []
        return await asyncio.to_thread(self._search_sync, query.strip(), freshness)

    def _search_sync(self, query: str, freshness: str | None) -> list[WebSearchResult]:
        params = {
            "q": query,
            "count": str(self.max_results),
            "safesearch": "moderate",
            "spellcheck": "true",
        }
        if freshness:
            params["freshness"] = freshness
        url = f"{BRAVE_WEB_SEARCH_URL}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": self.api_key,
            },
            method="GET",
        )
        try:
            with self._opener(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout):
            self.last_error_reason = "request_timeout"
            return []
        except (urllib.error.HTTPError, urllib.error.URLError):
            self.last_error_reason = "request_failed"
            return []
        except json.JSONDecodeError:
            self.last_error_reason = "bad_response"
            return []

        web_results = payload.get("web", {}).get("results", [])
        if not isinstance(web_results, list):
            return []

        results: list[WebSearchResult] = []
        for item in web_results[: self.max_results]:
            if not isinstance(item, dict):
                continue
            title = _clean_text(str(item.get("title") or ""))
            url = str(item.get("url") or "").strip()
            description = _clean_text(str(item.get("description") or ""))
            if title and url:
                results.append(WebSearchResult(title=title[:160], url=url, description=description[:260]))
        return results


class WebContextSynthesizer:
    def synthesize(
        self,
        session_id: str,
        candidate: WebContextCandidate,
        results: list[WebSearchResult],
    ) -> dict[str, Any] | None:
        if not results:
            return None

        limited = results[:3]
        snippets = []
        for result in limited:
            if result.description:
                snippets.append(f"{result.title}: {result.description}")
            else:
                snippets.append(result.title)
        body = "I found a few current references: " + "; ".join(snippets)
        return {
            "id": new_id("note"),
            "session_id": session_id,
            "kind": "topic",
            "title": f"Web context: {candidate.query}"[:140],
            "body": body[:760],
            "source_segment_ids": candidate.source_segment_ids,
            "created_at": time.time(),
            "ephemeral": True,
            "source_type": "brave_web",
            "sources": [result.source_dict() for result in limited],
            "citations": [result.url for result in limited],
            "sanitized_query": candidate.query,
            "freshness": candidate.freshness,
            "why_now": "This was a sanitized public/current web lookup; raw transcript was not submitted.",
            "confidence": 0.78,
            "priority": "normal",
            "card_key": f"web:{candidate.normalized_query}",
        }


def _clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", value).strip()
