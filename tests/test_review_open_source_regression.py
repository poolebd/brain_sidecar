from __future__ import annotations

import asyncio
import json
import os
import re
import wave
from pathlib import Path

import pytest

from brain_sidecar.core.review import ReviewMeetingEvaluator
from open_source_review_fixtures import (
    AMI_ES2002A_AUDIO_URL,
    AMI_MANUAL_ANNOTATIONS_URL,
    ensure_ami_es2002a_fixture,
)


pytestmark = pytest.mark.skipif(
    os.getenv("BRAIN_SIDECAR_RUN_OPEN_SOURCE_REGRESSION") != "1",
    reason="Set BRAIN_SIDECAR_RUN_OPEN_SOURCE_REGRESSION=1 to run AMI open-source Review regression.",
)


class AmiRegressionOllama:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        payload = json.loads(user)
        if "post-call review aggregation assistant" in system or "post-call review repair assistant" in system:
            timeline = payload.get("timeline") if isinstance(payload.get("timeline"), list) else []
            timeline_sample = [
                *timeline[:3],
                *timeline[max(0, len(timeline) // 2 - 1):len(timeline) // 2 + 2],
                *timeline[-3:],
            ]
            source_ids = []
            for item in timeline_sample:
                if isinstance(item, dict):
                    source_ids.extend(str(source_id) for source_id in item.get("source_segment_ids") or [])
            source_ids = source_ids[:36] or re.findall(r"\bES2002a_seg_\d+\b", user)[:30]
            review_context = payload.get("review_context") if isinstance(payload.get("review_context"), dict) else {}
            important_terms = [
                str(item)
                for item in payload.get("important_terms_seen") or []
                if str(item).strip()
            ][:8]
            term_clause = f" Important transcript terms covered: {', '.join(important_terms)}." if important_terms else ""
            reference_context = []
            if review_context.get("energy_lens", {}).get("active"):
                reference_context.append(
                    {
                        "kind": "energy_lens",
                        "title": "Energy lens",
                        "body": "Energy context was supplied to the open-source regression.",
                        "source_segment_ids": source_ids[:3],
                    }
                )
            for item in review_context.get("ee_reference_hits") or []:
                reference_context.append(
                    {
                        "kind": "ee_reference",
                        "title": item.get("title"),
                        "body": item.get("body"),
                        "citation": item.get("citation"),
                    }
                )
                break
            for item in review_context.get("web_context_hits") or []:
                reference_context.append(
                    {
                        "kind": "brave_web",
                        "title": item.get("title"),
                        "body": item.get("body"),
                        "citation": item.get("citation"),
                    }
                )
                break
            return json.dumps(
                {
                    "title": "AMI ES2002a design meeting review",
                    "summary": "The open-source AMI meeting review tracks the design discussion across the transcript, prioritizing the main design decisions, action handoffs, and reference/current context instead of a weak single-card heuristic." + term_clause,
                    "key_points": [
                        "The meeting is long enough to exercise whole-call coverage.",
                        "The review summary preserves source ids from the AMI transcript.",
                        "Reference and current-context material is folded into the main brief.",
                    ],
                    "topics": ["AMI ES2002a", "design review", "meeting summary", *important_terms[:6]],
                    "projects": ["Open-source Review regression"],
                    "project_workstreams": [
                        {
                            "project": "Open-source Review regression",
                            "status": "active",
                            "decisions": ["Use whole-call evidence rather than the last transcript window."],
                            "actions": ["Keep this fixture available for regression when summary quality changes."],
                            "risks": ["A weak aggregator can overfit to final-window small talk."],
                            "source_segment_ids": source_ids[:12],
                        }
                    ],
                    "technical_findings": [],
                    "decisions": ["Use whole-call evidence rather than the last transcript window."],
                    "actions": ["Keep this fixture available for regression when summary quality changes."],
                    "unresolved_questions": [],
                    "risks": ["A weak aggregator can overfit to final-window small talk."],
                    "entities": ["AMI"],
                    "lessons": [],
                    "coverage_notes": ["Summary uses AMI manual transcript windows and context inputs."],
                    "reference_context": reference_context,
                    "source_segment_ids": source_ids,
                }
            )
        segment_ids = re.findall(r"\bES2002a_seg_\d+\b", user)
        return json.dumps(
            {
                "window_summary": "AMI design-meeting transcript window.",
                "cards": [
                    {
                        "category": "note",
                        "title": "Design discussion window",
                        "body": "This transcript window contributes to the open-source regression summary.",
                        "priority": "normal",
                        "confidence": 0.72,
                        "source_segment_ids": segment_ids[:2],
                        "evidence_quote": " ".join(str(item.get("text", "")) for item in payload.get("transcript", []))[:180],
                    }
                ],
                "summary_points": ["Design discussion window."],
                "topics": ["design"],
                "projects": ["AMI meeting"],
                "actions": [],
                "decisions": [],
                "unresolved_questions": [],
                "risks": [],
                "entities": ["AMI"],
                "source_segment_ids": segment_ids[:4],
            }
        )


def test_ami_es2002a_open_source_fixture_feeds_summary_regression(tmp_path: Path) -> None:
    fixture = ensure_ami_es2002a_fixture(
        tmp_path / "open-source-fixtures",
        download=os.getenv("BRAIN_SIDECAR_DOWNLOAD_OPEN_SOURCE_FIXTURES") == "1",
    )

    with wave.open(str(fixture.audio_path), "rb") as wav:
        duration_seconds = wav.getnframes() / float(wav.getframerate())

    assert AMI_ES2002A_AUDIO_URL.startswith("https://groups.inf.ed.ac.uk/ami/")
    assert AMI_MANUAL_ANNOTATIONS_URL.startswith("https://groups.inf.ed.ac.uk/ami/")
    assert duration_seconds >= 600
    assert len(fixture.transcript_segments) >= 80

    review_context = {
        "energy_lens": {
            "active": True,
            "summary_label": "Energy lens regression input",
            "confidence": "low",
            "categories": [{"category": "operations", "score": 1.0}],
            "keywords": [{"phrase": "review workflow", "score": 1.0}],
            "source_segment_ids": [fixture.transcript_segments[0].id],
        },
        "ee_reference_hits": [
            {
                "kind": "ee_reference",
                "title": "Open EE reference fixture",
                "body": "Local electrical-engineering references should be available to the Review summary.",
                "citation": "/open/reference/electrical-engineering.md",
            }
        ],
        "web_context_hits": [
            {
                "kind": "brave_web",
                "title": "AMI corpus public documentation",
                "body": "AMI is a public meeting corpus suitable for regression testing.",
                "citation": "https://groups.inf.ed.ac.uk/ami/corpus/",
                "source_segment_ids": [fixture.transcript_segments[1].id],
            }
        ],
        "context_diagnostics": {
            "energy_lens": "included",
            "ee_reference_hits": 1,
            "web_context_hits": 1,
        },
    }

    _cards, summary = asyncio.run(
        ReviewMeetingEvaluator(AmiRegressionOllama()).evaluate(
            "ami-open-source-regression",
            fixture.transcript_segments[:120],
            save_result=False,
            review_context=review_context,
        )
    )

    kinds = {item["kind"] for item in summary["reference_context"]}
    assert {"energy_lens", "ee_reference", "brave_web"} <= kinds
    assert summary["review_standard"] == "energy_consultant_v1"
    assert summary["project_workstreams"][0]["project"] == "Open-source Review regression"
    assert summary["technical_findings"] == []
    assert "Open EE reference fixture" in " ".join(item["title"] for item in summary["reference_context"])
    assert "weak single-card heuristic" in summary["summary"]
    assert summary["portfolio_rollup"]["open_loops"]
    assert summary["review_metrics"]["workstream_count"] >= 1
    assert {"energy_lens", "ee_reference", "brave_web"} <= set(summary["review_metrics"]["context_kinds"])
    assert summary["diagnostics"]["summary_time_span_coverage"] >= 0.66
    assert summary["diagnostics"]["usefulness_status"] == "passed"
