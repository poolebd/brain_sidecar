from __future__ import annotations

import asyncio
import json
import math
import re
import time
import wave
from pathlib import Path

from fastapi.testclient import TestClient

from brain_sidecar.core.audio import AudioCapture
from brain_sidecar.core.asr import TranscribedSpan, TranscriptionResult
from brain_sidecar.core.models import NoteCard, TranscriptSegment
from brain_sidecar.core.notes import NoteSynthesizer
from brain_sidecar.core.review import (
    ReviewMeetingEvaluator,
    ReviewTranscriptSegmenter,
    TranscriptReviewCorrector,
    ReviewWindowExtract,
    _deterministic_review_extracts,
    _review_workstream_candidates,
    _summary_quality_diagnostics,
)
from brain_sidecar.core.sidecar_cards import create_sidecar_card
from brain_sidecar.core.speaker_identity import SpeakerBackendStatus, SpeakerIdentityService, l2_normalize
from brain_sidecar.core.web_context import WebSearchResult
from brain_sidecar.server.app import create_app


class FakeReviewTranscriber:
    backend_name = "faster_whisper"
    model_size = "fake-review-asr"
    last_error = None

    async def load(self) -> None:
        return None

    async def transcribe_pcm16(self, pcm: bytes, start_offset_s: float, *, initial_prompt: str | None = None) -> TranscriptionResult:
        return TranscriptionResult(
            model="fake-review-asr",
            language="en",
            audio_rms=0.04,
            spans=[
                TranscribedSpan(
                    start_s=0.0,
                    end_s=4.0,
                    text="BP will send the RF I log by Monday.",
                ),
                TranscribedSpan(
                    start_s=4.2,
                    end_s=8.0,
                    text="Sunil owns the Siemens document review path.",
                ),
            ],
        )


class ShortLiveCapture(AudioCapture):
    def __init__(self) -> None:
        self.running = True

    async def chunks(self):
        for _ in range(8):
            if not self.running:
                break
            yield b"\0" * 3200
            await asyncio.sleep(0.005)

    async def stop(self) -> None:
        self.running = False


class FakeReviewAsr:
    backend_name = "fake_review_asr"
    model_size = "fake-review-asr"
    last_error = None

    def __init__(self, *, spans: list[TranscribedSpan] | None = None, delay_s: float = 0.0) -> None:
        self.spans = spans
        self.delay_s = delay_s
        self.unload_calls = 0

    async def transcribe_file(
        self,
        path: Path,
        *,
        initial_prompt: str | None = None,
        progress=None,
    ) -> TranscriptionResult:
        if progress:
            await progress("Running fake high-accuracy ASR.", 64)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return TranscriptionResult(
            model="fake-review-asr",
            language="en",
            audio_rms=0.04,
            spans=self.spans or [
                TranscribedSpan(
                    start_s=0.0,
                    end_s=4.0,
                    text="BP will send the RF I log by Monday.",
                ),
                TranscribedSpan(
                    start_s=4.2,
                    end_s=8.0,
                    text="Sunil owns the Siemens document review path.",
                ),
            ],
        )

    async def unload(self) -> None:
        self.unload_calls += 1


class QueueSpeakerBackend:
    model_name = "queue-speaker-test"
    model_version = "test-v1"

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = list(vectors)

    def status(self) -> SpeakerBackendStatus:
        return SpeakerBackendStatus(True, self.model_name, self.model_version, "cpu")

    def embed_pcm16(self, pcm: bytes, sample_rate: int = 16_000) -> list[float]:
        if not self.vectors:
            raise RuntimeError("No queued speaker embedding remains.")
        return l2_normalize(self.vectors.pop(0))


class FakeReviewOllama:
    def __init__(self, *, fail_correction: bool = False) -> None:
        self.fail_correction = fail_correction
        self.meeting_call_count = 0

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        if "conservative transcript correction assistant" in system:
            if self.fail_correction:
                raise RuntimeError("correction unavailable")
            payload = json.loads(user)
            return json.dumps(
                {
                    "segments": [
                        {
                            "id": item["id"],
                            "text": (
                                "BP will send the RFI log by Monday."
                                if "RF I log" in item["text"]
                                else item["text"]
                            ),
                        }
                        for item in payload["segments"]
                    ]
                }
            )

        self.meeting_call_count += 1
        if "post-call review aggregation assistant" in system:
            return json.dumps(
                {
                    "title": "Project onboarding review",
                    "summary": "The review covered the gen-tie project portfolio, immediate RFI follow-up, and the Siemens review owner.",
                    "topics": ["gen-tie", "RFI", "Siemens review"],
                    "decisions": ["Sunil owns the Siemens document review path."],
                    "actions": ["BP will send the RFI log by Monday."],
                    "unresolved_questions": ["Confirm the review path for future Siemens documents."],
                    "entities": ["BP", "Sunil", "Siemens"],
                    "lessons": ["Keep project access and document routing visible in the tracker."],
                    "source_segment_ids": re.findall(r"\bseg_[A-Za-z0-9_]+\b", user)[:2],
                }
            )
        segment_ids = re.findall(r"\bseg_[A-Za-z0-9_]+\b", user)
        first_id = segment_ids[0] if segment_ids else "seg_missing"
        second_id = segment_ids[1] if len(segment_ids) > 1 else first_id
        evidence_text = (
            "BP will send the RF I log by Monday."
            if "RF I log" in user
            else "BP will send the RFI log by Monday."
        )
        return json.dumps(
            {
                "cards": [
                    {
                        "category": "action",
                        "title": "Send RFI log",
                        "body": "The meeting includes an action to send the RFI log by Monday.",
                        "suggested_say": "Let's confirm the RFI log goes out by Monday.",
                        "why_now": "The corrected transcript has an owner and due date.",
                        "priority": "high",
                        "confidence": 0.9,
                        "source_segment_ids": [first_id],
                        "evidence_quote": evidence_text,
                        "due_date": "Monday",
                        "source_type": "transcript",
                    },
                    {
                        "category": "decision",
                        "title": "Review path owner",
                        "body": "Sunil owns the Siemens document review path.",
                        "why_now": "The corrected transcript names the review owner.",
                        "priority": "normal",
                        "confidence": 0.82,
                        "source_segment_ids": [second_id],
                        "evidence_quote": "Sunil owns the Siemens document review path.",
                        "source_type": "transcript",
                    },
                ]
            }
        )


class FakeOrderedCorrectionOllama:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls: list[list[str]] = []

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        payload = json.loads(user)
        items = payload["segments"]
        ids = [item["id"] for item in items]
        self.calls.append(ids)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if ids and ids[0] == "seg_1":
                await asyncio.sleep(0.02)
            else:
                await asyncio.sleep(0.01)
            if any("fail" in item["text"] for item in items):
                raise RuntimeError("batch unavailable")
            return json.dumps(
                {
                    "segments": [
                        {
                            "id": item["id"],
                            "text": item["text"].replace("teh", "the").replace("RF I", "RFI"),
                        }
                        for item in items
                    ]
                }
            )
        finally:
            self.active -= 1


class FakeNoStructuredReviewOllama(FakeReviewOllama):
    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        if "conservative transcript correction assistant" in system:
            return await super().chat(system, user, format_json=format_json)
        return json.dumps({"cards": [], "summary": "", "summary_points": []})


class FakeWeakLateSummaryOllama:
    def __init__(self) -> None:
        self.aggregate_calls = 0
        self.repair_calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        payload = json.loads(user)
        if "post-call review aggregation assistant" in system:
            self.aggregate_calls += 1
            return json.dumps(
                {
                    "title": "Team onboarding",
                    "summary": "New team member should keep the speaker in the loop before bringing people on board or delegating responsibilities.",
                    "topics": ["team communication"],
                    "decisions": [],
                    "actions": [],
                    "unresolved_questions": [],
                    "entities": ["new team member", "speaker"],
                    "lessons": ["Keep BP in the loop."],
                    "source_segment_ids": ["seg_39", "seg_40"],
                }
            )
        if "post-call review repair assistant" in system:
            self.repair_calls += 1
            coverage_ids = ["seg_00", "seg_08", "seg_18", "seg_28", "seg_39"]
            return json.dumps(
                {
                    "title": "Remote-control design review",
                    "summary": "The meeting reviewed an open-source style remote-control design: the team aligned on user requirements, LCD interface behavior, prototype testing, marketing report inputs, and cost constraints, while keeping ownership visible for the next design iteration.",
                    "key_points": [
                        "The user-requirements discussion drives the remote-control feature set.",
                        "The prototype and LCD interface need usability checks before the next design review.",
                        "The marketing report and cost target need updated inputs.",
                    ],
                    "topics": ["remote control", "user requirements", "prototype", "marketing report"],
                    "projects": ["Remote-control design", "Prototype testing"],
                    "project_workstreams": [
                        {
                            "project": "Remote-control design",
                            "status": "active",
                            "actions": ["Update the requirements list and LCD-interface review notes before the next design pass."],
                            "risks": ["Cost and battery-life tradeoffs could constrain the design iteration."],
                            "source_segment_ids": ["seg_00", "seg_20"],
                        },
                        {
                            "project": "Prototype and marketing readiness",
                            "status": "active",
                            "actions": ["Run prototype usability checks and refresh the marketing-report inputs."],
                            "risks": ["Late prototype feedback could weaken the report and schedule."],
                            "source_segment_ids": ["seg_20", "seg_39"],
                        },
                    ],
                    "decisions": ["BP should be kept in the loop before delegation or adding people."],
                    "actions": [
                        "Update the user-requirements list before the next remote-control design review.",
                        "Run prototype usability checks on the LCD interface.",
                        "Add the revised cost target to the marketing report.",
                    ],
                    "unresolved_questions": ["Confirm whether the LCD interface can meet the battery-life and cost target together."],
                    "risks": ["Usability issues or cost drift could affect the prototype schedule."],
                    "entities": ["BP", "Industrial Design", "Marketing"],
                    "lessons": ["Keep delegation visible to BP during the first few months."],
                    "coverage_notes": ["Summary uses early, middle, and late transcript evidence."],
                    "source_segment_ids": ["seg_00", "seg_20", "seg_39"],
                }
            )
        transcript_text = " ".join(str(item.get("text", "")) for item in payload.get("transcript", []))
        source_ids = [str(item.get("id")) for item in payload.get("transcript", []) if item.get("id")]
        project = (
            "User requirements"
            if "requirements" in transcript_text
            else "Prototype testing"
            if "prototype" in transcript_text
            else "Marketing report"
            if "marketing" in transcript_text
            else "Delegation"
        )
        return json.dumps(
            {
                "window_summary": f"{project} workstream discussion with follow-up needed.",
                "priority": "high" if project != "Delegation" else "normal",
                "cards": [
                    {
                        "category": "action",
                        "title": f"{project} follow-up",
                        "body": f"Follow through on the {project} workstream discussed in this transcript window.",
                        "why_now": "The reviewed transcript contains project-specific action language.",
                        "priority": "high",
                        "confidence": 0.86,
                        "source_segment_ids": source_ids[:1],
                        "evidence_quote": transcript_text[:160],
                    }
                ],
                "summary_points": [f"{project} workstream needs follow-up."],
                "topics": [project],
                "projects": [project] if project != "Delegation" else [],
                "actions": [f"Follow through on {project}."] if project != "Delegation" else [],
                "decisions": [],
                "unresolved_questions": [],
                "risks": ["Schedule risk if the prototype feedback is late."] if project in {"Prototype testing", "User requirements"} else [],
                "entities": [project],
                "source_segment_ids": source_ids[:3],
            }
        )


class FakeEchoSummaryOllama:
    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        payload = json.loads(user)
        timeline_text = " ".join(str(item.get("text", "")) for item in payload.get("timeline", []))
        source_ids = re.findall(r"\bseg_[A-Za-z0-9_]+\b", user)[:18]
        return json.dumps(
            {
                "title": "Review summary",
                "summary": timeline_text[:1600],
                "topics": ["remote control", "prototype"],
                "actions": [],
                "decisions": [],
                "unresolved_questions": [],
                "entities": ["design team", "marketing"],
                "source_segment_ids": source_ids,
            }
        )


class FakeEchoThenUsefulSummaryOllama:
    def __init__(self) -> None:
        self.aggregate_calls = 0
        self.repair_calls = 0

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        payload = json.loads(user)
        source_ids = re.findall(r"\bseg_[A-Za-z0-9_]+\b", user)[:36]
        if "post-call review aggregation assistant" in system:
            self.aggregate_calls += 1
            timeline_text = " ".join(str(item.get("text", "")) for item in payload.get("timeline", []))
            return json.dumps(
                {
                    "title": "Remote control transcript",
                    "summary": timeline_text[:1600],
                    "key_points": [timeline_text[:520]],
                    "topics": ["remote control", "prototype", "marketing report"],
                    "project_workstreams": [
                        {
                            "project": "Remote control",
                            "status": "discussed",
                            "source_segment_ids": source_ids[:3],
                        }
                    ],
                    "source_segment_ids": source_ids,
                }
            )
        if "post-call review repair assistant" in system:
            self.repair_calls += 1
            coverage_ids = ["seg_00", "seg_08", "seg_18", "seg_28", "seg_39"]
            return json.dumps(
                {
                    "title": "Remote-control design review",
                    "summary": "The meeting separated remote-control requirements, prototype testing, and marketing-report updates into distinct workstreams with clear follow-up around LCD behavior, usability checks, battery-life assumptions, customer preferences, and revised pricing.",
                    "key_points": [
                        "Remote-control requirements and LCD behavior need review before the next design pass.",
                        "Prototype testing needs usability checks plus battery-life and cost-target confirmation.",
                        "The marketing report needs customer preference, positioning, and pricing updates.",
                    ],
                    "topics": ["remote control", "prototype testing", "marketing report"],
                    "projects": ["Remote-control requirements", "Prototype testing", "Marketing report"],
                    "project_workstreams": [
                        {
                            "project": "Remote-control requirements",
                            "status": "active",
                            "actions": ["Review user requirements, button layout, and LCD display behavior before the next design review."],
                            "source_segment_ids": ["seg_00", "seg_08"],
                        },
                        {
                            "project": "Prototype testing",
                            "status": "active",
                            "actions": ["Run usability checks and confirm battery-life and cost-target assumptions."],
                            "risks": ["Battery-life or cost-target drift could affect the prototype schedule."],
                            "source_segment_ids": ["seg_18", "seg_22"],
                        },
                        {
                            "project": "Marketing report",
                            "status": "active",
                            "actions": ["Add customer preferences, product positioning, and revised pricing assumptions to the report."],
                            "source_segment_ids": ["seg_28", "seg_34"],
                        },
                    ],
                    "technical_findings": [],
                    "actions": [
                        "Review user requirements, button layout, and LCD display behavior before the next design review.",
                        "Run usability checks and confirm battery-life and cost-target assumptions.",
                        "Add customer preferences, product positioning, and revised pricing assumptions to the report.",
                    ],
                    "decisions": [],
                    "unresolved_questions": ["Confirm whether LCD behavior can satisfy battery-life and cost targets together."],
                    "risks": ["Battery-life or cost-target drift could affect the prototype schedule."],
                    "entities": ["BP", "Industrial Design", "Marketing"],
                    "coverage_notes": ["Summary uses early, middle, and late transcript evidence."],
                    "source_segment_ids": coverage_ids,
                }
            )
        return json.dumps({"cards": [], "summary_points": [], "source_segment_ids": source_ids[:3]})


class FakeAlwaysWeakSummaryOllama(FakeEchoThenUsefulSummaryOllama):
    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        if "post-call review repair assistant" in system:
            self.repair_calls += 1
        elif "post-call review aggregation assistant" in system:
            self.aggregate_calls += 1
        payload = json.loads(user)
        source_ids = re.findall(r"\bseg_[A-Za-z0-9_]+\b", user)[:36]
        timeline_text = " ".join(str(item.get("text", "")) for item in payload.get("timeline", []))
        return json.dumps(
            {
                "title": "Meeting topics",
                "summary": "The meeting covered remote control, prototype, marketing report, customer preference, pricing, design, requirements, and delegation.",
                "key_points": [timeline_text[:520] or "The meeting covered remote control, prototype, and marketing report."],
                "topics": ["remote control", "prototype", "marketing report"],
                "project_workstreams": [{"project": "Remote control", "status": "discussed", "source_segment_ids": source_ids[:3]}],
                "technical_findings": [
                    {
                        "topic": "EE Index references",
                        "findings": ["EE Index and Energy Lens references should guide the review."],
                        "reference_context": ["EE Index", "Energy Lens"],
                        "source_segment_ids": source_ids[:3],
                    }
                ],
                "source_segment_ids": source_ids,
            }
        )


class FakeForcedContextThenCleanSummaryOllama(FakeAlwaysWeakSummaryOllama):
    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        payload = json.loads(user)
        source_ids = re.findall(r"\bseg_[A-Za-z0-9_]+\b", user)[:36]
        if "post-call review aggregation assistant" in system:
            self.aggregate_calls += 1
            return json.dumps(
                {
                    "title": "Remote-control review",
                    "summary": "The meeting covered remote-control requirements and forced EE Index references into the interpretation.",
                    "topics": ["remote control", "EE Index"],
                    "technical_findings": [
                        {
                            "topic": "EE Index references",
                            "findings": ["EE Index and Energy Lens references should guide this remote-control review."],
                            "reference_context": ["EE Index", "Energy Lens"],
                            "source_segment_ids": source_ids[:3],
                        }
                    ],
                    "source_segment_ids": source_ids,
                }
            )
        if "post-call review repair assistant" in system:
            self.repair_calls += 1
            coverage_ids = ["seg_00", "seg_18", "seg_39"]
            return json.dumps(
                {
                    "title": "Remote-control requirements review",
                    "summary": "The meeting focused on remote-control requirements, prototype testing, and marketing-report updates without turning unrelated external references into technical findings.",
                    "key_points": ["Requirements, prototype testing, and marketing-report updates are the useful review structure."],
                    "topics": ["remote control", "prototype testing", "marketing report"],
                    "projects": ["Remote-control requirements", "Prototype testing", "Marketing report"],
                    "project_workstreams": [
                        {
                            "project": "Remote-control requirements",
                            "status": "active",
                            "actions": ["Turn the requirements and interface discussion into next-pass design updates."],
                            "source_segment_ids": ["seg_00", "seg_08", "seg_18"],
                        },
                        {
                            "project": "Prototype testing",
                            "status": "active",
                            "actions": ["Run prototype usability checks and confirm the battery-life and cost-target assumptions."],
                            "risks": ["Late prototype feedback could affect the design pass."],
                            "source_segment_ids": ["seg_18", "seg_22"],
                        },
                        {
                            "project": "Marketing report",
                            "status": "active",
                            "actions": ["Refresh the marketing report with customer preference, positioning, and revised pricing inputs."],
                            "source_segment_ids": ["seg_28", "seg_34"],
                        },
                    ],
                    "technical_findings": [],
                    "actions": [
                        "Turn the requirements and interface discussion into next-pass design updates.",
                        "Run prototype usability checks and confirm the battery-life and cost-target assumptions.",
                        "Refresh the marketing report with customer preference, positioning, and revised pricing inputs.",
                    ],
                    "decisions": [],
                    "unresolved_questions": [],
                    "risks": [],
                    "entities": ["BP"],
                    "source_segment_ids": coverage_ids,
                }
            )
        return json.dumps({"cards": [], "source_segment_ids": source_ids[:3]})


class FakeCurrentWeakSummaryOllama:
    def __init__(self) -> None:
        self.aggregate_calls = 0
        self.repair_calls = 0

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        if "post-call review repair assistant" in system:
            self.repair_calls += 1
        elif "post-call review aggregation assistant" in system:
            self.aggregate_calls += 1
        return json.dumps(
            {
                "title": "Gen-tie Project Updates and Next Steps",
                "summary": "This meeting focused on several projects, including ongoing gen-tie work, and addressed utility requirements, site assessments, and permitting. Discussions covered actions needed to clarify requirements, schedule site visits, and confirm landowner easements.",
                "key_points": [
                    "Clarify utility requirements for the gen-tie project.",
                    "Schedule a site visit with S&L to validate pole locations and landowner impacts.",
                    "The Shipy Project is preparing to hand over its design to grid.",
                ],
                "projects": ["gen-tie project", "Shipy Project", "Chickade Project"],
                "project_workstreams": [
                    {
                        "project": "Gen-tie Project",
                        "status": "Active",
                        "actions": ["Schedule Site Visit", "Contact Utility"],
                        "owners": ["Unknown"],
                        "source_segment_ids": ["seg_00", "seg_20"],
                    },
                    {
                        "project": "Shipy Project",
                        "status": "active",
                        "actions": [{"description": "Grant access to gen-tie routing design and chickade project files.", "owner": "Unknown speaker"}],
                        "owners": ["Unknown speaker"],
                        "source_segment_ids": ["seg_05", "seg_25"],
                    },
                ],
                "technical_findings": [
                    {
                        "topic": "voltage, substation, and transformer",
                        "question": "and so a lot of that stuff is on the high voltage side they'll ask us for gen-tie scope which is like bringing their BESS collection substation gen-tie line",
                        "findings": [
                            "and so a lot of that stuff is on the high voltage side they'll ask us for gen-tie scope which is like bringing their BESS collection substation gen-tie line"
                        ],
                        "reference_context": ["Pacific Gas and Electric Reviews | pge.com @ PissedConsumer"],
                        "source_segment_ids": ["seg_00", "seg_20"],
                    }
                ],
                "actions": [
                    "BP to review transformer cut sheets.",
                    {"description": "Grant access to gen-tie routing design and chickade project files.", "owner": "Unknown speaker"},
                    "Schedule Site Visit",
                ],
                "decisions": [],
                "unresolved_questions": ["What containment options will EPC power offer?"],
                "risks": ["Landowner disputes regarding easements"],
                "source_segment_ids": ["seg_00", "seg_20", "seg_30"],
            }
        )


class FakeContextSummaryOllama(FakeReviewOllama):
    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        if "post-call review aggregation assistant" not in system:
            return await super().chat(system, user, format_json=format_json)
        payload = json.loads(user)
        context = payload.get("review_context") if isinstance(payload.get("review_context"), dict) else {}
        source_ids = re.findall(r"\bseg_[A-Za-z0-9_]+\b", user)[:12]
        reference_context = []
        energy = context.get("energy_lens") if isinstance(context.get("energy_lens"), dict) else {}
        if energy.get("active"):
            reference_context.append(
                {
                    "kind": "energy_lens",
                    "title": energy.get("summary_label") or "Energy lens",
                    "body": "Utility tariff and demand-charge language was treated as energy-consulting context.",
                    "source_segment_ids": energy.get("source_segment_ids") or source_ids[:2],
                }
            )
        for item in context.get("ee_reference_hits") or []:
            reference_context.append(
                {
                    "kind": "ee_reference",
                    "title": item.get("title"),
                    "body": item.get("body"),
                    "citation": item.get("citation"),
                    "source_segment_ids": source_ids[:2],
                }
            )
            break
        for item in context.get("web_context_hits") or []:
            reference_context.append(
                {
                    "kind": "brave_web",
                    "title": item.get("title"),
                    "body": item.get("body"),
                    "citation": item.get("citation"),
                    "source_segment_ids": item.get("source_segment_ids") or source_ids[:2],
                }
            )
            break
        return json.dumps(
            {
                "title": "Transformer protection and tariff review",
                "summary": "The review covered transformer protection settings, utility tariff analysis, and the current public standards question that affects the protection-setting recommendation.",
                "key_points": [
                    "Transformer protection settings need to be checked against the EE reference material.",
                    "Utility tariff and demand-charge assumptions are part of the meeting context.",
                    "Current public IEEE-style guidance was requested for the protection-setting question.",
                ],
                "topics": ["transformer protection", "utility tariff", "current standards"],
                "projects": ["Transformer protection review"],
                "project_workstreams": [
                    {
                        "project": "Transformer protection review",
                        "status": "decision_needed",
                        "actions": ["BP should review transformer protection settings against reference and current public guidance."],
                        "risks": ["Protection-setting assumptions may be weak without current public context."],
                        "open_questions": ["Which current transformer protection practices should shape the recommendation?"],
                        "source_segment_ids": source_ids[:3],
                    }
                ],
                "technical_findings": [
                    {
                        "topic": "Transformer protection settings",
                        "question": "Which current transformer protection practices should shape the recommendation?",
                        "assumptions": ["Utility tariff and demand-charge assumptions are part of the meeting context."],
                        "methods": ["Compare the recommendation against EE Index material and current public guidance."],
                        "findings": ["Transformer protection settings need to be checked against the EE reference material."],
                        "recommendations": ["BP should review transformer protection settings against reference and current public guidance."],
                        "risks": ["Protection-setting assumptions may be weak without current public context."],
                        "reference_context": [item.get("title") for item in reference_context if item.get("title")],
                        "confidence": "medium",
                        "source_segment_ids": source_ids[:3],
                    }
                ],
                "decisions": [],
                "actions": ["BP should review transformer protection settings against reference and current public guidance."],
                "unresolved_questions": ["Which current transformer protection practices should shape the recommendation?"],
                "risks": ["Protection-setting assumptions may be weak without current public context."],
                "entities": ["BP", "utility"],
                "lessons": [],
                "coverage_notes": ["Summary uses energy lens, EE index, Brave web context, and transcript evidence."],
                "reference_context": reference_context,
                "source_segment_ids": source_ids,
            }
        )


class FakeEnergyConsultantStandardOllama(FakeReviewOllama):
    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        if "post-call review aggregation assistant" not in system:
            return await super().chat(system, user, format_json=format_json)
        source_ids = re.findall(r"\bseg_[A-Za-z0-9_]+\b", user)[:10]
        return json.dumps(
            {
                "review_standard": "energy_consultant_v1",
                "title": "Portfolio and protection review",
                "summary": "The meeting separated Apollo RFI follow-up from Delta transformer protection analysis, preserving owners, technical assumptions, and next actions for each workstream.",
                "key_points": [
                    "Apollo has an RFI handoff due Monday.",
                    "Delta needs transformer protection settings checked against EE references and current practices.",
                ],
                "topics": ["Apollo RFI", "Delta transformer protection"],
                "projects": ["Apollo", "Delta transformer protection"],
                "project_workstreams": [
                    {
                        "project": "Apollo",
                        "status": "active",
                        "actions": ["BP will send the RFI package to Greg by Monday."],
                        "risks": ["Apollo pricing is blocked until Greg receives the RFI package."],
                        "owners": ["BP"],
                        "source_segment_ids": source_ids[:2],
                    },
                    {
                        "project": "Delta transformer protection",
                        "status": "decision_needed",
                        "actions": ["Compare transformer protection settings against the EE Index and current public practice."],
                        "open_questions": ["Which relay curve and CT ratio assumptions are final?"],
                        "owners": ["Sunil"],
                        "source_segment_ids": source_ids[2:5],
                    },
                ],
                "technical_findings": [
                    {
                        "topic": "Transformer protection settings",
                        "question": "Which relay curve and CT ratio assumptions should shape the recommendation?",
                        "assumptions": ["Relay curves, CT ratios, and breaker failure settings are the relevant basis."],
                        "methods": ["Compare settings against EE Index references and current public guidance."],
                        "findings": ["Transformer protection is unresolved until the relay and CT assumptions are confirmed."],
                        "recommendations": ["Confirm relay curve and CT ratio assumptions before issuing the recommendation."],
                        "risks": ["Wrong assumptions could create a weak protection-setting recommendation."],
                        "data_gaps": ["Final relay curve and CT ratio assumptions."],
                        "reference_context": ["EE Index", "Current public web context"],
                        "confidence": "medium",
                        "source_segment_ids": source_ids[2:5],
                    }
                ],
                "decisions": [],
                "actions": [
                    "BP will send the RFI package to Greg by Monday.",
                    "Compare transformer protection settings against the EE Index and current public practice.",
                ],
                "unresolved_questions": ["Which relay curve and CT ratio assumptions are final?"],
                "risks": [
                    "Apollo pricing is blocked until Greg receives the RFI package.",
                    "Wrong assumptions could create a weak protection-setting recommendation.",
                ],
                "entities": ["BP", "Greg", "Sunil"],
                "lessons": [],
                "source_segment_ids": source_ids,
            }
        )


class FakeThinAbstractThenUsefulOllama:
    def __init__(self) -> None:
        self.aggregate_calls = 0
        self.repair_calls = 0

    async def chat(self, system: str, user: str, *, format_json: bool = False) -> str:
        source_ids = re.findall(r"\bseg_[A-Za-z0-9_]+\b", user)[:12]
        if "post-call review aggregation assistant" in system:
            self.aggregate_calls += 1
            return json.dumps(
                {
                    "review_standard": "energy_consultant_v1",
                    "title": "General design support review",
                    "summary": "The meeting covered design, assumptions, handoff, support scope, and follow-up needs.",
                    "key_points": ["Design, support, and handoff items need follow-up."],
                    "topics": ["design", "support", "handoff"],
                    "projects": ["Collector routing", "Requirements handoff", "Transformer containment"],
                    "project_workstreams": [
                        {
                            "project": "Collector routing",
                            "status": "active",
                            "actions": ["Follow up on the design handoff."],
                            "source_segment_ids": ["seg_1"],
                        },
                        {
                            "project": "Requirements handoff",
                            "status": "active",
                            "actions": ["Review the support scope."],
                            "source_segment_ids": ["seg_3"],
                        },
                        {
                            "project": "Transformer containment",
                            "status": "active",
                            "actions": ["Confirm the assumptions."],
                            "source_segment_ids": ["seg_5"],
                        },
                    ],
                    "actions": ["Follow up on the design handoff.", "Review the support scope.", "Confirm the assumptions."],
                    "source_segment_ids": source_ids or ["seg_1", "seg_3", "seg_5"],
                }
            )
        if "post-call review repair assistant" in system:
            self.repair_calls += 1
            return json.dumps(
                {
                    "review_standard": "energy_consultant_v1",
                    "title": "Collector, requirements, and containment review",
                    "summary": "The review separates collector routing, requirements handoff, and transformer containment into distinct workstreams: BP needs north-feeder routing confirmation, Maya needs owner-load assumptions for the EPC package, and BP should request the SPCC and oil-containment basis.",
                    "key_points": [
                        "Collector routing is waiting on the north-feeder routing basis and easement assumption.",
                        "Requirements handoff is blocked until owner-load assumptions and the missing requirement list are reviewed.",
                        "Transformer containment needs SPCC basis and secondary containment volume detail from EPC.",
                    ],
                    "topics": ["collector routing", "requirements handoff", "transformer containment"],
                    "projects": ["Collector routing", "Requirements handoff", "Transformer containment"],
                    "project_workstreams": [
                        {
                            "project": "Collector routing",
                            "status": "decision_needed",
                            "actions": ["BP needs to confirm the north feeder routing basis before the design handoff."],
                            "risks": ["The easement assumption is unclear."],
                            "source_segment_ids": ["seg_1", "seg_2"],
                        },
                        {
                            "project": "Requirements handoff",
                            "status": "blocked",
                            "actions": ["Maya needs the owner load assumptions before the EPC package can move."],
                            "risks": ["The support scope is blocked until the missing requirement list is reviewed."],
                            "source_segment_ids": ["seg_3", "seg_4"],
                        },
                        {
                            "project": "Transformer containment",
                            "status": "active",
                            "actions": ["BP should request the SPCC basis and oil containment detail from EPC."],
                            "risks": ["The containment design has a data gap around secondary containment volume."],
                            "source_segment_ids": ["seg_5", "seg_6"],
                        },
                    ],
                    "actions": [
                        "BP needs to confirm the north feeder routing basis before the design handoff.",
                        "Maya needs the owner load assumptions before the EPC package can move.",
                        "BP should request the SPCC basis and oil containment detail from EPC.",
                    ],
                    "unresolved_questions": ["Confirm the easement assumption and secondary containment volume."],
                    "risks": ["Requirements handoff and containment volume remain open."],
                    "coverage_notes": ["Summary uses separate source ids for each abstract workstream."],
                    "source_segment_ids": ["seg_1", "seg_3", "seg_5"],
                }
            )
        return json.dumps({"cards": [], "summary_points": [], "source_segment_ids": source_ids[:3]})


class FakeReviewWebSearch:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def search(self, query: str, *, freshness: str | None = None) -> list[WebSearchResult]:
        self.queries.append(query)
        return [
            WebSearchResult(
                title="Current transformer protection practice",
                url="https://example.org/current-transformer-protection",
                description="Public guidance on current transformer protection coordination and relay settings.",
            )
        ]


def test_review_job_requires_approval_before_persisting_meeting_output(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        install_fake_review_runtime(client)

        response = client.post(
            "/api/review/jobs",
            data={"save_result": "true", "title": "Uploaded review"},
            files={"file": ("call.wav", b"fake audio", "audio/wav")},
        )

        assert response.status_code == 200
        payload = wait_for_review(client, response.json()["job_id"])
        assert payload["status"] == "completed_awaiting_validation"
        assert payload["raw_audio_retained"] is True
        assert payload["temporary_audio_retained"] is True
        assert payload["session_id"] is None
        assert payload["asr_backend"] == "fake_review_asr"
        assert payload["asr_model"] == "fake-review-asr"
        assert payload["progress_percent"] == 100
        assert all("progress" in step for step in payload["steps"])
        assert payload["clean_segments"][0]["text"] == "BP will send the RFI log by Monday."
        assert any(card["category"] == "action" for card in payload["meeting_cards"])
        assert client.get("/api/sessions").json()["sessions"] == []

        approved = client.post(f"/api/review/jobs/{payload['job_id']}/approve").json()
        assert approved["status"] == "approved"
        assert approved["session_id"]
        assert approved["temporary_audio_retained"] is False
        assert approved["audio_deleted_at"] is not None

        detail = client.get(f"/api/sessions/{approved['session_id']}").json()
        assert detail["transcript_segments"][0]["text"] == "BP will send the RFI log by Monday."
        assert detail["note_cards"][0]["source_segment_ids"]
        assert detail["summary"]["summary"] == (
            "The review covered the gen-tie project portfolio, immediate RFI follow-up, and the Siemens review owner."
        )
        assert "BP will send the RFI log by Monday." in detail["summary"]["actions"]

        job = client.app.state.review_service.jobs[payload["job_id"]]
        assert job["temporary_audio_retained"] is False
        assert client.app.state.review_service.review_asr.unload_calls >= 1


def test_review_labels_bp_speech_before_meeting_evaluation_and_persists_on_approval(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_SPEAKER_ENROLLMENT_MINIMUM_SECONDS", "15")
    monkeypatch.setenv("BRAIN_SIDECAR_SPEAKER_ENROLLMENT_TARGET_SECONDS", "20")
    with TestClient(create_app()) as client:
        install_fake_review_runtime(client)
        manager = client.app.state.manager
        manager.speaker_identity = SpeakerIdentityService(
            manager.storage,
            manager.settings,
            backend=QueueSpeakerBackend([[1.0, 0.0, 0.0]] * 8 + [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        )
        enrollment = manager.speaker_identity.start_enrollment()
        training_pcm = tone_pcm(seconds=8.0)
        for _ in range(8):
            manager.speaker_identity.add_enrollment_sample(enrollment["id"], training_pcm)
        manager.speaker_identity.finalize_enrollment(enrollment["id"])

        def condition_audio(source_path: Path, conditioned_path: Path) -> None:
            write_wav(conditioned_path, pcm=tone_pcm(seconds=9.0))

        client.app.state.review_service._condition_audio = condition_audio
        response = client.post(
            "/api/review/jobs",
            data={"title": "Speaker review"},
            files={"file": ("call.wav", b"fake audio", "audio/wav")},
        )

        payload = wait_for_review(client, response.json()["job_id"])
        assert payload["status"] == "completed_awaiting_validation"
        assert payload["clean_segments"][0]["speaker_label"] == "BP"
        assert payload["clean_segments"][0]["speaker_role"] == "user"
        assert payload["clean_segments"][1]["speaker_label"] == "Other speaker"
        assert payload["clean_segments"][1]["speaker_role"] == "other"
        assert payload["diagnostics"]["speaker_identity"]["bp_segment_count"] == 1
        assert payload["diagnostics"]["speaker_identity"]["other_segment_count"] == 1

        approved = client.post(f"/api/review/jobs/{payload['job_id']}/approve").json()
        detail = client.get(f"/api/sessions/{approved['session_id']}").json()
        assert detail["transcript_segments"][0]["speaker_label"] == "BP"
        assert detail["transcript_segments"][1]["speaker_label"] == "Other speaker"
        recent_labels = manager.storage.recent_diarization_segments(limit=4)
        assert any(row["display_speaker_label"] == "BP" for row in recent_labels)
        assert any(row["display_speaker_label"] == "Other speaker" for row in recent_labels)


def test_live_stop_queues_review_without_creating_session(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        install_fake_review_runtime(client)
        manager = client.app.state.manager
        manager.has_active_capture = lambda: True
        manager.transcriber = FakeReviewTranscriber()
        manager._build_capture = lambda **kwargs: ShortLiveCapture()

        start = client.post("/api/live/start", json={"title": "Live handoff"})
        assert start.status_code == 200
        live_id = start.json()["live_id"]
        time.sleep(0.08)
        stop = client.post(f"/api/live/{live_id}/stop")

        assert stop.status_code == 200
        payload = stop.json()
        assert payload["status"] == "queued"
        assert payload["review_job_id"]
        assert payload["temporary_audio_retained"] is True
        assert client.get("/api/sessions").json()["sessions"] == []
        job_response = client.get(f"/api/review/jobs/{payload['review_job_id']}")
        assert job_response.status_code == 200
        job = job_response.json()
        assert job["source"] == "live"
        assert job["session_id"] is None

        canceled = client.post(f"/api/review/jobs/{payload['review_job_id']}/cancel").json()
        assert canceled["status"] == "canceled"
        assert canceled["temporary_audio_retained"] is False


def test_review_latest_cancel_and_unload(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        asr = FakeReviewAsr(delay_s=5.0)
        install_fake_review_runtime(client, asr=asr)
        client.app.state.manager.has_active_capture = lambda: True

        response = client.post(
            "/api/review/jobs",
            data={"save_result": "true", "title": "Cancel me"},
            files={"file": ("cancel.wav", b"fake audio", "audio/wav")},
        )

        assert response.status_code == 200
        job_id = response.json()["job_id"]
        cancel_response = client.post(f"/api/review/jobs/{job_id}/cancel")
        assert cancel_response.status_code == 200
        payload = cancel_response.json()
        assert payload["status"] == "canceled"
        assert payload["active"] is False
        assert payload["temporary_audio_retained"] is False
        latest = client.get("/api/review/jobs/latest").json()
        assert latest["job_id"] == job_id
        assert latest["status"] == "canceled"
        assert asr.unload_calls == 0


def test_review_evaluates_long_transcript_in_windows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    spans = [
        TranscribedSpan(
            start_s=float(index * 2),
            end_s=float(index * 2 + 1),
            text=(
                "BP will send the RFI log by Monday."
                if index % 2 == 0
                else "Sunil owns the Siemens document review path."
            ),
        )
        for index in range(52)
    ]
    ollama = FakeReviewOllama()
    with TestClient(create_app()) as client:
        install_fake_review_runtime(client, asr=FakeReviewAsr(spans=spans), ollama=ollama)

        response = client.post(
            "/api/review/jobs",
            data={"save_result": "false"},
            files={"file": ("long.wav", b"fake audio", "audio/wav")},
        )

        payload = wait_for_review(client, response.json()["job_id"])
        assert payload["status"] == "completed_awaiting_validation"
        assert payload["raw_segment_count"] == 52
        assert ollama.meeting_call_count >= 3


def test_review_summary_folds_energy_ee_and_brave_context(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("BRAIN_SIDECAR_WEB_CONTEXT_ENABLED", "true")
    monkeypatch.setenv("BRAIN_SIDECAR_BRAVE_SEARCH_API_KEY", "test-key")
    spans = [
        TranscribedSpan(
            start_s=0.0,
            end_s=5.0,
            text="We need utility bill analysis, tariff analysis, and demand charge review before pricing this site.",
        ),
        TranscribedSpan(
            start_s=5.2,
            end_s=10.0,
            text="BP should compare the transformer protection recommendation against the electrical reference index.",
        ),
        TranscribedSpan(
            start_s=10.2,
            end_s=15.0,
            text="What are the current IEEE standard practices for transformer protection settings?",
        ),
    ]
    with TestClient(create_app()) as client:
        ollama = FakeContextSummaryOllama()
        web_search = FakeReviewWebSearch()
        install_fake_review_runtime(client, asr=FakeReviewAsr(spans=spans), ollama=ollama)
        manager = client.app.state.manager
        manager.web_search = web_search
        manager.storage.upsert_document_chunk(
            Path("/open/reference/electrical-engineering/transformer-protection.md"),
            0,
            "Transformer protection settings should coordinate relay curves, CT ratios, breaker failure, and utility protection practices.",
            {},
        )

        response = client.post(
            "/api/review/jobs",
            data={"save_result": "false", "title": "Open transformer review"},
            files={"file": ("open-transformer.wav", b"fake audio", "audio/wav")},
        )

        payload = wait_for_review(client, response.json()["job_id"])
        assert payload["status"] == "completed_awaiting_validation"
        summary = payload["summary"]
        assert "transformer protection" in summary["summary"]
        assert summary["review_standard"] == "energy_consultant_v1"
        assert summary["project_workstreams"][0]["project"] == "Transformer protection review"
        assert summary["technical_findings"][0]["topic"] == "Transformer protection settings"
        assert "EE Index material" in summary["technical_findings"][0]["methods"][0]
        kinds = {item["kind"] for item in summary["reference_context"]}
        assert {"energy_lens", "ee_reference", "brave_web"} <= kinds
        assert summary["context_diagnostics"]["energy_lens"] == "included"
        assert summary["context_diagnostics"]["ee_reference_hits"] >= 1
        assert summary["context_diagnostics"]["web_context_hits"] >= 1
        assert web_search.queries
        assert client.get("/api/sessions").json()["sessions"] == []


def test_energy_consultant_standard_separates_projects_and_technical_findings() -> None:
    segments = [
        review_segment("seg_1", 0.0, "For Apollo, BP will send the RFI package to Greg by Monday."),
        review_segment("seg_2", 5.0, "Apollo pricing is blocked until Greg receives the RFI package."),
        review_segment("seg_3", 10.0, "For Delta, Sunil needs transformer protection settings compared against the electrical reference index."),
        review_segment("seg_4", 15.0, "The relay curves, CT ratios, and breaker failure settings are the assumptions for the Delta recommendation."),
        review_segment("seg_5", 20.0, "What are the current public practices for transformer protection settings?"),
    ]

    _cards, summary = asyncio.run(
        ReviewMeetingEvaluator(FakeEnergyConsultantStandardOllama()).evaluate(
            "energy-consultant-standard",
            segments,
            save_result=False,
            review_context={
                "energy_lens": {"active": True, "summary_label": "Energy lens: operations"},
                "ee_reference_hits": [{"kind": "ee_reference", "title": "EE Index transformer protection", "body": "Relay coordination reference."}],
                "web_context_hits": [{"kind": "brave_web", "title": "Current public protection practice", "body": "Public guidance."}],
                "context_diagnostics": {"energy_lens": "included", "ee_reference_hits": 1, "web_context_hits": 1},
            },
        )
    )

    assert summary["review_standard"] == "energy_consultant_v1"
    assert [item["project"] for item in summary["project_workstreams"]] == ["Apollo", "Delta transformer protection"]
    assert summary["project_workstreams"][0]["actions"] == ["BP will send the RFI package to Greg by Monday."]
    assert "Apollo pricing" in summary["project_workstreams"][0]["risks"][0]
    assert summary["technical_findings"][0]["topic"] == "Transformer protection settings"
    assert "relay curve" in summary["technical_findings"][0]["question"].lower()
    assert "EE Index" in summary["technical_findings"][0]["reference_context"]
    assert summary["portfolio_rollup"]["bp_next_actions"] == ["BP will send the RFI package to Greg by Monday."]
    assert summary["portfolio_rollup"]["risk_posture"] == "blocked"
    assert summary["review_metrics"]["workstream_count"] == 2
    assert summary["review_metrics"]["technical_finding_count"] == 1
    assert {"energy_lens", "ee_reference", "brave_web"} <= set(summary["review_metrics"]["context_kinds"])
    assert summary["diagnostics"]["summary_quality_flags"] == []


def test_review_usefulness_gate_requires_bp_next_action_when_bp_commits() -> None:
    segments = [
        review_segment("seg_1", 0.0, "For Apollo, BP will send the RFI package to Greg by Monday."),
        review_segment("seg_2", 5.0, "Maya needs the owner load assumptions before the EPC package can move."),
        review_segment("seg_3", 10.0, "Apollo pricing is blocked until Greg receives the RFI package."),
    ]
    extracts = _deterministic_review_extracts(segments)
    summary = {
        "title": "Apollo and EPC handoff",
        "summary": "The meeting covered the Apollo RFI path and EPC owner-load assumptions.",
        "topics": ["Apollo", "EPC owner-load assumptions"],
        "projects": ["Apollo", "EPC package"],
        "project_workstreams": [
            {
                "project": "Apollo",
                "status": "blocked",
                "risks": ["Apollo pricing is blocked until Greg receives the RFI package."],
                "source_segment_ids": ["seg_2", "seg_3"],
            },
            {
                "project": "EPC package",
                "status": "active",
                "actions": ["Maya needs the owner load assumptions before the EPC package can move."],
                "source_segment_ids": ["seg_2"],
            },
        ],
        "actions": ["Maya needs the owner load assumptions before the EPC package can move."],
        "source_segment_ids": ["seg_2", "seg_3"],
    }

    diagnostics = _summary_quality_diagnostics(summary, extracts, segments)

    assert "missing_bp_next_action" in diagnostics["usefulness_flags"]
    assert diagnostics["usefulness_status"] == "needs_repair"


def test_review_usefulness_gate_preserves_abstract_multi_project_workstreams() -> None:
    segments = [
        review_segment("seg_1", 0.0, "Collector routing segment 1: BP needs to confirm the north feeder routing basis before the design handoff."),
        review_segment("seg_2", 5.0, "Collector routing segment 2: The easement assumption is unclear and should be checked against the one-line."),
        review_segment("seg_3", 10.0, "Requirements handoff segment 1: Maya needs the owner load assumptions before the EPC package can move."),
        review_segment("seg_4", 15.0, "Requirements handoff segment 2: The customer support scope is blocked until the missing requirement list is reviewed."),
        review_segment("seg_5", 20.0, "Transformer containment segment 1: BP should request the SPCC basis and oil containment detail from EPC."),
        review_segment("seg_6", 25.0, "Transformer containment segment 2: The containment design has a data gap around secondary containment volume."),
    ]
    extracts = _deterministic_review_extracts(segments)
    candidates = _review_workstream_candidates(segments, extracts, [])
    candidate_names = {candidate["project"] for candidate in candidates}

    assert {"Collector routing", "Requirements handoff", "Transformer containment"} <= candidate_names

    collapsed_summary = {
        "title": "Project review",
        "summary": "The meeting covered design, assumptions, handoff, and support scope.",
        "topics": ["design", "assumptions", "support"],
        "projects": ["General project"],
        "project_workstreams": [
            {
                "project": "General project",
                "status": "active",
                "actions": ["Follow up on the design, handoff, and containment items."],
                "source_segment_ids": ["seg_1"],
            }
        ],
        "actions": ["Follow up on the design, handoff, and containment items."],
        "source_segment_ids": ["seg_1"],
    }
    collapsed_diagnostics = _summary_quality_diagnostics(
        collapsed_summary,
        extracts,
        segments,
        workstream_candidates=candidates,
    )

    assert "collapsed_multiple_workstreams" in collapsed_diagnostics["summary_quality_flags"]

    separated_summary = {
        "title": "Collector, requirements, and containment review",
        "summary": "The meeting separated collector routing, requirements handoff, and transformer containment into different follow-up paths with concrete data-basis questions.",
        "topics": ["collector routing", "requirements handoff", "transformer containment"],
        "projects": ["Collector routing", "Requirements handoff", "Transformer containment"],
        "project_workstreams": [
            {
                "project": "Collector routing",
                "status": "decision_needed",
                "actions": ["BP needs to confirm the north feeder routing basis before the design handoff."],
                "risks": ["The easement assumption is unclear."],
                "source_segment_ids": ["seg_1", "seg_2"],
            },
            {
                "project": "Requirements handoff",
                "status": "blocked",
                "actions": ["Maya needs the owner load assumptions before the EPC package can move."],
                "risks": ["The support scope is blocked until the missing requirement list is reviewed."],
                "source_segment_ids": ["seg_3", "seg_4"],
            },
            {
                "project": "Transformer containment",
                "status": "active",
                "actions": ["BP should request the SPCC basis and oil containment detail from EPC."],
                "risks": ["The containment design has a data gap around secondary containment volume."],
                "source_segment_ids": ["seg_5", "seg_6"],
            },
        ],
        "actions": [
            "BP needs to confirm the north feeder routing basis before the design handoff.",
            "Maya needs the owner load assumptions before the EPC package can move.",
            "BP should request the SPCC basis and oil containment detail from EPC.",
        ],
        "source_segment_ids": ["seg_1", "seg_3", "seg_5"],
    }
    separated_diagnostics = _summary_quality_diagnostics(
        separated_summary,
        extracts,
        segments,
        workstream_candidates=candidates,
    )

    assert "collapsed_multiple_workstreams" not in separated_diagnostics["summary_quality_flags"]
    assert "missing_candidate_workstream" not in separated_diagnostics["summary_quality_flags"]
    assert "abstract_payload_too_thin" not in separated_diagnostics["summary_quality_flags"]
    assert "vague_abstract_workstream" not in separated_diagnostics["usefulness_flags"]


def test_review_usefulness_gate_flags_thin_abstract_payloads() -> None:
    segments = [
        review_segment("seg_1", 0.0, "Collector routing segment 1: BP needs to confirm the north feeder routing basis before the design handoff."),
        review_segment("seg_2", 5.0, "Collector routing segment 2: The easement assumption is unclear and should be checked against the one-line."),
        review_segment("seg_3", 10.0, "Requirements handoff segment 1: Maya needs the owner load assumptions before the EPC package can move."),
        review_segment("seg_4", 15.0, "Requirements handoff segment 2: The customer support scope is blocked until the missing requirement list is reviewed."),
        review_segment("seg_5", 20.0, "Transformer containment segment 1: BP should request the SPCC basis and oil containment detail from EPC."),
        review_segment("seg_6", 25.0, "Transformer containment segment 2: The containment design has a data gap around secondary containment volume."),
    ]
    extracts = _deterministic_review_extracts(segments)
    candidates = _review_workstream_candidates(segments, extracts, [])
    thin_summary = {
        "title": "Design support review",
        "summary": "The meeting covered design handoff, support scope, and assumptions.",
        "topics": ["design", "support", "assumptions"],
        "projects": ["Collector routing", "Requirements handoff", "Transformer containment"],
        "project_workstreams": [
            {
                "project": "Collector routing",
                "status": "active",
                "actions": ["Follow up on the design handoff."],
                "source_segment_ids": ["seg_1"],
            },
            {
                "project": "Requirements handoff",
                "status": "active",
                "actions": ["Review the support scope."],
                "source_segment_ids": ["seg_3"],
            },
            {
                "project": "Transformer containment",
                "status": "active",
                "actions": ["Confirm the assumptions."],
                "source_segment_ids": ["seg_5"],
            },
        ],
        "actions": ["Follow up on the design handoff.", "Review the support scope.", "Confirm the assumptions."],
        "source_segment_ids": ["seg_1", "seg_3", "seg_5"],
    }

    diagnostics = _summary_quality_diagnostics(
        thin_summary,
        extracts,
        segments,
        workstream_candidates=candidates,
    )

    assert "abstract_payload_too_thin" in diagnostics["summary_quality_flags"]
    assert "vague_abstract_workstream" in diagnostics["usefulness_flags"]
    assert diagnostics["usefulness_status"] == "needs_repair"


def test_review_usefulness_gate_flags_thin_technical_findings() -> None:
    segments = [
        review_segment("seg_1", 0.0, "BP needs transformer protection settings compared against the electrical reference index."),
        review_segment("seg_2", 5.0, "Relay curves, CT ratios, and breaker failure settings are the basis for the Delta recommendation."),
        review_segment("seg_3", 10.0, "Confirm which current IEEE practice applies before issuing the recommendation."),
    ]
    extracts = _deterministic_review_extracts(segments)
    summary = {
        "title": "Transformer protection review",
        "summary": "The meeting reviewed transformer protection settings for the Delta recommendation.",
        "topics": ["transformer protection"],
        "projects": ["Delta transformer protection"],
        "project_workstreams": [
            {
                "project": "Delta transformer protection",
                "status": "decision_needed",
                "actions": ["Confirm which current IEEE practice applies before issuing the recommendation."],
                "source_segment_ids": ["seg_1", "seg_2", "seg_3"],
            }
        ],
        "technical_findings": [
            {
                "topic": "Transformer protection settings",
                "findings": ["Transformer protection needs review."],
                "source_segment_ids": ["seg_1", "seg_2"],
            }
        ],
        "actions": ["Confirm which current IEEE practice applies before issuing the recommendation."],
        "source_segment_ids": ["seg_1", "seg_2", "seg_3"],
    }

    diagnostics = _summary_quality_diagnostics(summary, extracts, segments)

    assert "technical_finding_too_thin" in diagnostics["summary_quality_flags"]
    assert diagnostics["usefulness_status"] == "needs_repair"


def test_review_usefulness_gate_flags_low_value_reference_context() -> None:
    segments = [
        review_segment("seg_1", 0.0, "BP needs transformer protection settings compared against the electrical reference index."),
        review_segment("seg_2", 5.0, "Relay curves, CT ratios, and breaker failure settings are the assumptions for the recommendation."),
        review_segment("seg_3", 10.0, "Confirm current public protection practices before issuing the recommendation."),
    ]
    extracts = _deterministic_review_extracts(segments)
    review_context = {
        "energy_lens": {"active": True, "summary_label": "Energy lens: technical analysis"},
        "ee_reference_hits": [{"kind": "ee_reference", "title": "EE Index transformer protection", "body": "Relay coordination reference."}],
        "web_context_hits": [{"kind": "brave_web", "title": "Current public protection practice", "body": "Public guidance."}],
        "context_diagnostics": {"energy_lens": "included", "ee_reference_hits": 1, "web_context_hits": 1},
    }
    summary = {
        "title": "Transformer protection review",
        "summary": "The meeting reviewed transformer protection settings for the Delta recommendation.",
        "topics": ["transformer protection"],
        "projects": ["Delta transformer protection"],
        "project_workstreams": [
            {
                "project": "Delta transformer protection",
                "status": "decision_needed",
                "actions": ["Confirm current public protection practices before issuing the recommendation."],
                "source_segment_ids": ["seg_1", "seg_2", "seg_3"],
            }
        ],
        "technical_findings": [
            {
                "topic": "Transformer protection settings",
                "assumptions": ["Relay curves, CT ratios, and breaker failure settings are the assumptions."],
                "methods": ["Compare the recommendation against the transcript assumptions."],
                "findings": ["Transformer protection remains unresolved until those assumptions are confirmed."],
                "source_segment_ids": ["seg_1", "seg_2", "seg_3"],
            }
        ],
        "coverage_notes": ["Summary uses transcript evidence."],
        "reference_context": [
            {"kind": "ee_reference", "title": "EE Index transformer protection", "body": "Relay coordination reference."},
            {"kind": "brave_web", "title": "Current public protection practice", "body": "Public guidance."},
        ],
        "actions": ["Confirm current public protection practices before issuing the recommendation."],
        "source_segment_ids": ["seg_1", "seg_2", "seg_3"],
    }

    diagnostics = _summary_quality_diagnostics(
        summary,
        extracts,
        segments,
        review_context=review_context,
    )

    assert "reference_context_low_value" in diagnostics["summary_quality_flags"]
    assert diagnostics["usefulness_status"] == "needs_repair"


def test_review_usefulness_gate_repairs_thin_abstract_multi_project_summary() -> None:
    segments = [
        review_segment("seg_1", 0.0, "Collector routing segment 1: BP needs to confirm the north feeder routing basis before the design handoff."),
        review_segment("seg_2", 5.0, "Collector routing segment 2: The easement assumption is unclear and should be checked against the one-line."),
        review_segment("seg_3", 10.0, "Requirements handoff segment 1: Maya needs the owner load assumptions before the EPC package can move."),
        review_segment("seg_4", 15.0, "Requirements handoff segment 2: The customer support scope is blocked until the missing requirement list is reviewed."),
        review_segment("seg_5", 20.0, "Transformer containment segment 1: BP should request the SPCC basis and oil containment detail from EPC."),
        review_segment("seg_6", 25.0, "Transformer containment segment 2: The containment design has a data gap around secondary containment volume."),
    ]
    extracts = _deterministic_review_extracts(segments)
    ollama = FakeThinAbstractThenUsefulOllama()

    summary = asyncio.run(
        ReviewMeetingEvaluator(ollama)._aggregate_summary(
            "thin-abstract-review",
            segments,
            extracts,
            cards=[],
        )
    )

    assert ollama.aggregate_calls == 1
    assert ollama.repair_calls == 1
    assert summary["diagnostics"]["usefulness_status"] == "repaired"
    assert "abstract_payload_too_thin" not in summary["diagnostics"]["summary_quality_flags"]
    assert [item["project"] for item in summary["project_workstreams"]] == [
        "Collector routing",
        "Requirements handoff",
        "Transformer containment",
    ]


def test_fallback_review_summary_keeps_primary_points_compact() -> None:
    long_segments = [
        review_segment(
            f"seg_{index:02d}",
            float(index * 5),
            (
                "Remote-control requirements discussion with repeated wording about design, prototype, "
                "marketing report, customer preference, price target, and implementation details that "
                "would be too long to show as a primary Review key point without compaction. "
            ) * 2,
        )
        for index in range(18)
    ]

    _cards, summary = asyncio.run(
        ReviewMeetingEvaluator(FakeNoStructuredReviewOllama()).evaluate(
            "fallback-compact",
            long_segments,
            save_result=False,
        )
    )

    assert summary["review_standard"] == "energy_consultant_v1"
    assert summary["key_points"]
    assert max(len(point) for point in summary["key_points"]) <= 180
    assert summary["project_workstreams"]


def test_review_summary_rejects_late_generic_summary_for_long_meeting() -> None:
    ollama = FakeWeakLateSummaryOllama()
    segments = open_source_design_review_segments()

    cards, summary = asyncio.run(
        ReviewMeetingEvaluator(ollama).evaluate("review-test", segments, save_result=False)
    )

    assert cards
    assert ollama.aggregate_calls == 1
    assert ollama.repair_calls == 1
    assert "remote-control design" in summary["summary"]
    assert "prototype" in summary["summary"]
    assert "marketing report" in summary["summary"]
    assert "new team member should keep the speaker" not in summary["summary"].lower()
    assert any("prototype" in action.lower() or "requirements" in action.lower() for action in summary["actions"])
    assert set(summary["diagnostics"]["summary_time_buckets"]) == {"early", "late", "middle"}
    assert summary["diagnostics"]["summary_quality_flags"] == []


def test_review_usefulness_gate_repairs_transcript_echo_and_weak_workstreams() -> None:
    ollama = FakeEchoThenUsefulSummaryOllama()
    segments = open_source_design_review_segments()

    _cards, summary = asyncio.run(
        ReviewMeetingEvaluator(ollama).evaluate_existing_transcript("review-usefulness", segments, save_result=False)
    )

    assert ollama.aggregate_calls == 1
    assert ollama.repair_calls == 1
    assert summary["diagnostics"]["usefulness_status"] == "repaired"
    assert summary["diagnostics"]["usefulness_score"] >= 0.72
    assert "remote-control requirements" in summary["summary"].lower()
    assert "marketing-report updates" in summary["summary"].lower()
    assert "transcript_echo_key_points" not in summary["diagnostics"]["usefulness_flags"]
    assert any(item.get("actions") for item in summary["project_workstreams"])


def test_review_usefulness_gate_marks_failed_repair_for_manual_validation() -> None:
    ollama = FakeAlwaysWeakSummaryOllama()
    segments = open_source_design_review_segments()

    _cards, summary = asyncio.run(
        ReviewMeetingEvaluator(ollama).evaluate_existing_transcript("review-low-usefulness", segments, save_result=False)
    )

    assert ollama.aggregate_calls == 1
    assert ollama.repair_calls == 1
    assert summary["diagnostics"]["usefulness_status"] == "low_usefulness"
    assert summary["diagnostics"]["usefulness_score"] < 0.72
    assert "manual validation" in " ".join(summary["coverage_notes"]).lower()
    assert "did not pass the usefulness gate" in summary["summary"]


def test_review_usefulness_gate_removes_forced_ee_context_from_nontechnical_review() -> None:
    ollama = FakeForcedContextThenCleanSummaryOllama()
    segments = open_source_design_review_segments()
    review_context = {
        "energy_lens": {"active": True, "summary_label": "Energy lens regression input"},
        "ee_reference_hits": [{"kind": "ee_reference", "title": "EE Index transformer protection", "body": "Relay coordination reference."}],
        "web_context_hits": [{"kind": "brave_web", "title": "Current public design review context", "body": "Public context."}],
        "context_diagnostics": {"energy_lens": "included", "ee_reference_hits": 1, "web_context_hits": 1},
    }

    _cards, summary = asyncio.run(
        ReviewMeetingEvaluator(ollama).evaluate_existing_transcript(
            "review-forced-context",
            segments,
            save_result=False,
            review_context=review_context,
        )
    )

    assert ollama.repair_calls == 1
    assert summary["technical_findings"] == []
    assert summary["diagnostics"]["usefulness_status"] == "repaired"
    assert "unsupported_technical_findings" not in summary["diagnostics"]["usefulness_flags"]
    assert any(item["kind"] == "ee_reference" for item in summary["reference_context"])


def test_review_summary_falls_back_to_card_brief_for_current_weak_output() -> None:
    ollama = FakeCurrentWeakSummaryOllama()
    segments = current_weak_review_regression_segments()
    cards = current_weak_review_regression_cards()
    extracts = [
        ReviewWindowExtract(
            cards=cards,
            window_summary="Current weak Review regression fixture with useful cards and weak primary summary.",
            projects=["gen-tie", "PGE", "EPC skid", "transformer losses", "load flow"],
            actions=[card.body for card in cards if card.category == "action"],
            unresolved_questions=[card.body for card in cards if card.category == "question"],
            risks=["Breaker-failure coordination and transformer containment remain unresolved."],
            source_segment_ids=[source_id for card in cards for source_id in card.source_segment_ids],
        )
    ]
    review_context = {
        "energy_lens": {"active": True, "summary_label": "Energy lens: technology + technical + commercial"},
        "ee_reference_hits": [{"kind": "ee_reference", "title": "DOE Electrical Science", "body": "Transformer and breaker reference material."}],
        "web_context_hits": [
            {"kind": "brave_web", "title": "Pacific Gas and Electric Reviews | pge.com @ PissedConsumer", "body": "Consumer reviews."},
            {"kind": "brave_web", "title": "Navigating the PGE Reviews - Oreate AI Blog", "body": "Blog summary."},
        ],
        "context_diagnostics": {"energy_lens": "included", "ee_reference_hits": 1, "web_context_hits": 2},
    }

    summary = asyncio.run(
        ReviewMeetingEvaluator(ollama)._aggregate_summary(
            "current-weak-review",
            segments,
            extracts,
            cards=cards,
            review_context=review_context,
        )
    )

    assert ollama.aggregate_calls == 1
    assert ollama.repair_calls == 1
    assert summary["diagnostics"]["usefulness_status"] == "passed"
    assert set(summary["diagnostics"]["summary_time_buckets"]) == {"early", "middle", "late"}
    assert len(summary["project_workstreams"]) >= 4
    assert summary["portfolio_rollup"]["open_loops"]
    assert summary["review_metrics"]["workstream_count"] >= 4
    assert summary["review_metrics"]["technical_finding_count"] >= 1
    assert any("Westwood" in item["project"] or "relay" in item["project"].lower() for item in summary["project_workstreams"])
    assert any("transformer" in item["project"].lower() for item in summary["project_workstreams"])
    assert all(isinstance(item, str) and "description" not in item for item in summary["actions"])
    assert "Unknown" not in json.dumps(summary["project_workstreams"])
    assert "PissedConsumer" not in json.dumps(summary["reference_context"])
    assert "Oreate" not in json.dumps(summary["reference_context"])
    assert "and so a lot of that stuff" not in json.dumps(summary["technical_findings"])


def test_regenerate_session_review_replaces_weak_saved_summary(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        ollama = FakeWeakLateSummaryOllama()
        install_fake_review_runtime(client, ollama=ollama)
        manager = client.app.state.manager
        record = manager.storage.create_session("Weak saved review")
        manager.storage.set_session_status(record.id, "stopped", ended_at=time.time(), save_transcript=True)
        for segment in open_source_design_review_segments(session_id=record.id):
            manager.storage.upsert_transcript_segment(segment)
        manager.storage.add_note(
            NoteCard(
                id="old-note",
                session_id=record.id,
                kind="context",
                title="Weak old note",
                body="New team member should keep BP in the loop.",
                source_segment_ids=["seg_39"],
            )
        )
        manager.storage.upsert_session_memory_summary(
            session_id=record.id,
            title="Weak summary",
            summary="New team member should keep the speaker in the loop.",
            topics=["team communication"],
            decisions=[],
            actions=[],
            unresolved_questions=[],
            entities=["speaker"],
            lessons=[],
            source_segment_ids=["seg_39"],
        )

        response = client.post(f"/api/sessions/{record.id}/review/regenerate")

        assert response.status_code == 200
        payload = response.json()
        assert payload["summary"]["title"] == "Remote-control design review"
        assert "remote-control design" in payload["summary"]["summary"]
        assert "marketing report" in payload["summary"]["summary"]
        assert payload["summary"]["projects"]
        assert payload["summary"]["actions"]
        assert all(note["id"] != "old-note" for note in payload["note_cards"])
        assert client.get("/api/review/jobs").json()["jobs"] == []


def test_review_temporary_mode_does_not_persist_session(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        install_fake_review_runtime(client)

        response = client.post(
            "/api/review/jobs",
            data={"save_result": "false"},
            files={"file": ("temporary.wav", b"fake audio", "audio/wav")},
        )

        assert response.status_code == 200
        payload = wait_for_review(client, response.json()["job_id"])
        assert payload["status"] == "completed_awaiting_validation"
        assert payload["session_id"] is None
        assert payload["clean_segments"]
        assert client.get("/api/sessions").json()["sessions"] == []
        discarded = client.post(f"/api/review/jobs/{payload['job_id']}/discard").json()
        assert discarded["status"] == "discarded"
        assert discarded["temporary_audio_retained"] is False


def test_review_falls_back_to_raw_asr_when_correction_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        install_fake_review_runtime(client, ollama=FakeReviewOllama(fail_correction=True))

        response = client.post(
            "/api/review/jobs",
            data={"save_result": "false"},
            files={"file": ("fallback.wav", b"fake audio", "audio/wav")},
        )

        payload = wait_for_review(client, response.json()["job_id"])
        assert payload["status"] == "completed_awaiting_validation"
        assert payload["clean_segments"][0]["text"] == "BP will send the RF I log by Monday."
        assert payload["meeting_cards"]


def test_review_marks_incomplete_when_structured_evaluation_has_no_cards(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        install_fake_review_runtime(client, ollama=FakeNoStructuredReviewOllama())

        response = client.post(
            "/api/review/jobs",
            data={"save_result": "false"},
            files={"file": ("no-cards.wav", b"fake audio", "audio/wav")},
        )

        payload = wait_for_review(client, response.json()["job_id"])
        assert payload["status"] == "completed_awaiting_validation"
        assert payload["meeting_cards"][0]["category"] == "status"
        assert payload["meeting_cards"][0]["title"] == "Review incomplete"
        assert "RFI" in payload["summary"]["summary"]
        assert "Siemens" in payload["summary"]["summary"]
        assert payload["summary"]["coverage_notes"] == ["Fallback summary built from ranked transcript windows."]


def test_transcript_review_correction_keeps_order_and_falls_back_by_batch() -> None:
    ollama = FakeOrderedCorrectionOllama()
    segments = [
        review_segment("seg_1", 0.0, "BP has teh RF I log."),
        review_segment("seg_2", 3.0, "Sunil owns teh Siemens path."),
        review_segment("seg_3", 6.0, "fail this batch without edits."),
        review_segment("seg_4", 9.0, "Leave this batch original too."),
    ]

    corrected = asyncio.run(
        TranscriptReviewCorrector(ollama, batch_size=2, concurrency=2).correct(segments)
    )

    assert [segment.id for segment in corrected] == ["seg_1", "seg_2", "seg_3", "seg_4"]
    assert corrected[0].text == "BP has the RFI log."
    assert corrected[1].text == "Sunil owns the Siemens path."
    assert corrected[2].text == "fail this batch without edits."
    assert corrected[3].text == "Leave this batch original too."
    assert ollama.max_active == 2
    assert ollama.calls == [["seg_1", "seg_2"], ["seg_3", "seg_4"]]


def test_review_segmenter_splits_long_overlapping_asr_chunks() -> None:
    first_text = " ".join([f"alpha{i}" for i in range(50)] + ["shared", "handoff", "phrase"])
    second_text = " ".join(["shared", "handoff", "phrase"] + [f"beta{i}" for i in range(50)])
    segments = [
        review_segment("raw_1", 0.0, first_text, end_s=45.0),
        review_segment("raw_2", 42.0, second_text, end_s=87.0),
    ]

    result = ReviewTranscriptSegmenter().segment(segments, overlap_seconds=3.0)

    assert len(result) >= 4
    assert all(segment.end_s > segment.start_s for segment in result)
    assert all(len(segment.text.split()) <= 45 for segment in result)
    assert not result[len(result) // 2].text.startswith("shared handoff phrase")


def test_transcript_review_correction_batches_by_segment_and_word_budget() -> None:
    ollama = FakeOrderedCorrectionOllama()
    segments = [
        review_segment(f"seg_{index}", float(index), " ".join(["teh"] * 160))
        for index in range(9)
    ]

    corrected = asyncio.run(
        TranscriptReviewCorrector(ollama, batch_size=12, concurrency=3).correct(segments)
    )

    assert len(corrected) == 9
    assert all(len(call) <= 5 for call in ollama.calls)
    assert [segment.id for segment in corrected] == [f"seg_{index}" for index in range(9)]


def test_review_rejects_unsupported_audio_extension(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BRAIN_SIDECAR_DATA_DIR", str(tmp_path / "data"))
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/review/jobs",
            data={"save_result": "true"},
            files={"file": ("notes.txt", b"not audio", "text/plain")},
        )

    assert response.status_code == 400
    assert "Unsupported audio extension" in response.json()["detail"]


def install_fake_review_runtime(
    client: TestClient,
    *,
    asr: FakeReviewAsr | None = None,
    ollama: FakeReviewOllama | None = None,
) -> None:
    service = client.app.state.review_service
    manager = client.app.state.manager
    manager.transcriber = FakeReviewTranscriber()
    manager.ollama = ollama or FakeReviewOllama()
    manager.notes = NoteSynthesizer(manager.ollama)
    service.manager = manager
    service.review_asr = asr or FakeReviewAsr()

    def condition_audio(source_path: Path, conditioned_path: Path) -> None:
        write_wav(conditioned_path)

    service._condition_audio = condition_audio


def wait_for_review(client: TestClient, job_id: str) -> dict:
    for _ in range(80):
        response = client.get(f"/api/review/jobs/{job_id}")
        payload = response.json()
        if payload["status"] in {"completed_awaiting_validation", "approved", "error"}:
            return payload
        time.sleep(0.05)
    raise AssertionError("review job did not finish")


def open_source_design_review_segments(*, session_id: str = "review-test") -> list[TranscriptSegment]:
    texts = []
    for index in range(14):
        texts.append(f"Remote-control requirements segment {index}: review user requirements, button layout, and LCD display behavior before the next design review.")
    for index in range(14):
        texts.append(f"Prototype testing segment {index}: run usability checks, confirm battery-life assumptions, and update the cost target for the prototype.")
    for index in range(10):
        texts.append(f"Marketing report segment {index}: add customer preferences, product positioning, and revised pricing assumptions to the marketing report.")
    for index in range(4):
        texts.append(f"Closing delegation segment {index}: keep BP in the loop before bringing people on board or delegating responsibilities.")
    return [
        review_segment(f"seg_{index:02d}", float(index * 5), text, session_id=session_id)
        for index, text in enumerate(texts)
    ]


def current_weak_review_regression_segments(*, session_id: str = "review-test") -> list[TranscriptSegment]:
    texts: list[str] = []
    for index in range(30):
        texts.append(
            "and so a lot of that stuff is on the high voltage side they'll ask us for gen-tie scope "
            "bringing their BESS collection substation gen-tie line to the PGE substation and tracking client questions."
        )
    for index in range(30):
        texts.append(
            "Westwood and PGE need relay coordination review because breaker failure responses and deadline close block comments "
            "did not resolve the technical concern, so schedule a working meeting."
        )
    for index in range(30):
        texts.append(
            "EPC skid order form, transformer oil containment, transformer losses at 225 MVA, load flow, short circuit studies, "
            "and client drawing review need follow-up before the next project meeting."
        )
    return [
        review_segment(f"seg_{index:02d}", float(index * 8), text, session_id=session_id)
        for index, text in enumerate(texts)
    ]


def current_weak_review_regression_cards(session_id: str = "review-test") -> list:
    specs = [
        (
            "note",
            "Gen-tie Project Details",
            "The project involved drawing a high-line from the Lark project site to the PGE Portland General Electric substation and creating preliminary underground line design drawings.",
            ["seg_02", "seg_08"],
        ),
        (
            "note",
            "CIA Task Authorizations",
            "Phase A focuses on coordination with PGE. Phase B authorization is pending while Phase B-style questions are being handled as a workaround.",
            ["seg_18", "seg_25"],
        ),
        (
            "action",
            "Schedule Meeting with Westwood and PGE",
            "A meeting is needed to resolve Westwood's responses on breaker failure coordination and deadline close block comments.",
            ["seg_34", "seg_41"],
        ),
        (
            "note",
            "Strategy for PGE/Westwood Comment Resolution",
            "The team should walk through PGE comments with Westwood and prepare a revised response for PGE.",
            ["seg_42", "seg_47"],
        ),
        (
            "action",
            "Review EPC Power Skid Order Form",
            "The team needs a technical review of the EPC skid order form, including transformer winding configuration and oil containment requirements.",
            ["seg_62", "seg_66"],
        ),
        (
            "action",
            "Confirm Secondary Containment with Civil Engineer",
            "Nate Compton should review the documents to confirm whether secondary containment is required for transformer oil volume.",
            ["seg_68", "seg_72"],
        ),
        (
            "question",
            "Transformer Loss Estimation at 225 MVA",
            "The team needs to estimate transformer losses at the 225 MVA limit for BESS vendor round-trip efficiency guarantees.",
            ["seg_76", "seg_80"],
        ),
        (
            "note",
            "Client Drawing Review - Load Flow Priority",
            "A high-level client drawing review is needed, with load flow as the immediate priority before the next meeting.",
            ["seg_84", "seg_88"],
        ),
    ]
    return [
        create_sidecar_card(
            session_id=session_id,
            category=category,
            title=title,
            body=body,
            why_now="Evidence-backed card from the current weak Review regression fixture.",
            source_segment_ids=source_ids,
            source_type="transcript",
            evidence_quote=body,
            ephemeral=True,
        )
        for category, title, body, source_ids in specs
    ]


def review_segment(
    segment_id: str,
    start_s: float,
    text: str,
    *,
    end_s: float | None = None,
    session_id: str = "review-test",
) -> TranscriptSegment:
    return TranscriptSegment(
        id=segment_id,
        session_id=session_id,
        start_s=start_s,
        end_s=end_s if end_s is not None else start_s + 2.0,
        text=text,
        source_segment_ids=[segment_id],
    )


def tone_pcm(*, seconds: float, sample_rate: int = 16_000, frequency: float = 220.0, amplitude: float = 0.25) -> bytes:
    frames = int(seconds * sample_rate)
    samples = bytearray()
    for index in range(frames):
        value = int(amplitude * 32767 * math.sin(2.0 * math.pi * frequency * (index / sample_rate)))
        samples.extend(value.to_bytes(2, byteorder="little", signed=True))
    return bytes(samples)


def write_wav(path: Path, *, pcm: bytes | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(pcm if pcm is not None else b"\0" * 16_000 * 2)
