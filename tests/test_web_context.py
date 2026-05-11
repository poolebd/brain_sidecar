from __future__ import annotations

import urllib.error
from urllib.parse import parse_qs, urlparse

from brain_sidecar.core.models import TranscriptSegment
from brain_sidecar.core.web_context import BraveSearchClient, WebTriggerDetector


def test_trigger_detector_fires_for_current_tech_practice_question() -> None:
    detector = WebTriggerDetector()
    candidate = detector.detect(
        [
            segment("seg_1", "What are current best practices for vector database indexing?"),
        ]
    )

    assert candidate is not None
    assert "current best practices" in candidate.query.lower()
    assert "vector database indexing" in candidate.query.lower()
    assert candidate.freshness == "py"


def test_trigger_detector_ignores_personal_history_question() -> None:
    detector = WebTriggerDetector()

    assert detector.detect([segment("seg_1", "What did we do on the T.A. Smith project?")]) is None
    decision = detector.decision_for_segments([segment("seg_1", "What did we do on the T.A. Smith project?")])
    assert decision.skip_reason == "query_looked_private_or_internal"


def test_query_builder_redacts_private_phrasing_and_caps_length() -> None:
    detector = WebTriggerDetector(max_query_chars=80)
    query = detector.build_query(
        "Can you look up what are current best practices for vector database indexing "
        "for our confidential client ACME internal deployment with proprietary details?"
    )

    assert "vector database indexing" in query.lower()
    assert "client" not in query.lower()
    assert "confidential" not in query.lower()
    assert "acme" not in query.lower()
    assert len(query) <= 80


def test_manual_query_decision_sanitizes_without_raw_private_terms() -> None:
    detector = WebTriggerDetector()
    private = detector.decision_for_manual_query(
        "latest React release for our confidential client ACME internal deployment"
    )
    assert private.skip_reason == "query_looked_private_or_internal"

    public = detector.decision_for_manual_query("latest React release")
    assert public.candidate is not None
    assert public.candidate.query == "latest React release"


def test_brave_client_parses_web_results(event_loop) -> None:
    seen: dict[str, object] = {}

    def opener(request, timeout):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        return FakeResponse(
            b"""
            {
              "web": {
                "results": [
                  {"title": "Vector indexing guide", "url": "https://example.com/vector", "description": "Index tradeoffs."},
                  {"title": "Database release notes", "url": "https://example.com/release", "description": "Fresh features."}
                ]
              }
            }
            """
        )

    client = BraveSearchClient("secret", timeout_s=2.5, max_results=3, opener=opener)
    results = event_loop.run_until_complete(client.search("vector database indexing", freshness="py"))

    params = parse_qs(urlparse(str(seen["url"])).query)
    assert params["q"] == ["vector database indexing"]
    assert params["count"] == ["3"]
    assert params["safesearch"] == ["moderate"]
    assert params["spellcheck"] == ["true"]
    assert params["freshness"] == ["py"]
    assert seen["timeout"] == 2.5
    assert [result.title for result in results] == ["Vector indexing guide", "Database release notes"]


def test_brave_client_handles_timeout_and_rate_limit_as_empty(event_loop) -> None:
    timeout_client = BraveSearchClient("secret", opener=lambda _request, timeout: (_ for _ in ()).throw(TimeoutError()))
    assert event_loop.run_until_complete(timeout_client.search("latest python release")) == []
    assert timeout_client.last_error_reason == "request_timeout"

    def rate_limit(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    rate_limited_client = BraveSearchClient("secret", opener=rate_limit)
    assert event_loop.run_until_complete(rate_limited_client.search("latest python release")) == []
    assert rate_limited_client.last_error_reason == "request_failed"


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def segment(segment_id: str, text: str) -> TranscriptSegment:
    return TranscriptSegment(
        id=segment_id,
        session_id="ses_test",
        start_s=0.0,
        end_s=1.0,
        text=text,
    )
