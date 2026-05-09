from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from brain_sidecar.core.models import compact_text


MeetingMode = Literal["quiet", "balanced", "assertive"]

DEFAULT_MEETING_GOAL = (
    "Offload meeting obligations, questions, risks, follow-ups, useful context, "
    "and post-call synthesis for BP."
)

DEFAULT_MEETING_REMINDERS = [
    "Prefer silence over noisy, weak, or unsupported cards.",
    "Ground current-meeting cards only in the current transcript with valid source IDs and evidence.",
    "Use saved memory, technical references, and web context as reminder context only, never as current-meeting facts.",
    "Assign BP ownership conservatively and only when speaker_role is user with high confidence or BP is explicitly named.",
    "Focus outputs on ask, say, do, watch, and remember items.",
    "Prepare a post-call brief with actions, decisions, questions, risks, follow-up language, references, and evidence.",
]


@dataclass(frozen=True)
class ContractReminder:
    text: str
    enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "enabled": self.enabled}


@dataclass(frozen=True)
class MeetingContract:
    goal: str = DEFAULT_MEETING_GOAL
    mode: MeetingMode = "quiet"
    reminders: list[str] = field(default_factory=lambda: list(DEFAULT_MEETING_REMINDERS))

    def to_dict(self) -> dict[str, object]:
        return {
            "goal": self.goal,
            "mode": self.mode,
            "reminders": list(self.reminders),
        }


def default_meeting_contract() -> MeetingContract:
    return MeetingContract()


def normalize_meeting_contract(value: object | None = None) -> MeetingContract:
    if value is None:
        return default_meeting_contract()
    if isinstance(value, MeetingContract):
        payload = value.to_dict()
    elif isinstance(value, dict):
        payload = value
    elif hasattr(value, "model_dump"):
        payload = value.model_dump()
    elif hasattr(value, "dict"):
        payload = value.dict()
    else:
        payload = {}

    goal = compact_text(payload.get("goal") or DEFAULT_MEETING_GOAL, limit=420)
    mode = str(payload.get("mode") or "quiet").strip().lower()
    if mode not in {"quiet", "balanced", "assertive"}:
        mode = "quiet"
    reminders = _normalize_reminders(payload.get("reminders"))
    return MeetingContract(
        goal=goal or DEFAULT_MEETING_GOAL,
        mode=mode,  # type: ignore[arg-type]
        reminders=reminders or list(DEFAULT_MEETING_REMINDERS),
    )


def contract_prompt_block(contract: MeetingContract | None) -> str:
    normalized = normalize_meeting_contract(contract)
    reminders = "\n".join(f"- {item}" for item in normalized.reminders)
    return (
        "Meeting-only Dross contract:\n"
        f"- Mode: {normalized.mode}\n"
        f"- Goal: {normalized.goal}\n"
        "- Contract reminders:\n"
        f"{reminders}\n"
        "- Prefer silence over unsupported output. Memory is reminder context only, not evidence.\n"
    )


def _normalize_reminders(value: object) -> list[str]:
    if value is None:
        return []
    raw_items: list[object]
    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = [value]

    reminders: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        if isinstance(item, ContractReminder):
            if not item.enabled:
                continue
            text = item.text
        elif isinstance(item, dict):
            if item.get("enabled") is False:
                continue
            text = str(item.get("text") or item.get("label") or item.get("name") or "")
        else:
            text = str(item or "")
        reminder = compact_text(text, limit=220)
        key = reminder.lower()
        if not reminder or key in seen:
            continue
        reminders.append(reminder)
        seen.add(key)
        if len(reminders) >= 8:
            break
    return reminders
