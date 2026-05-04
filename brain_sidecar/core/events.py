from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SidecarEvent:
    type: str
    session_id: str | None
    payload: dict[str, Any]
    at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "session_id": self.session_id,
            "at": self.at,
            "payload": self.payload,
        }


EVENT_AUDIO_STATUS = "audio_status"
EVENT_TRANSCRIPT_PARTIAL = "transcript_partial"
EVENT_TRANSCRIPT_FINAL = "transcript_final"
EVENT_NOTE_UPDATE = "note_update"
EVENT_RECALL_HIT = "recall_hit"
EVENT_SIDECAR_CARD = "sidecar_card"
EVENT_GPU_STATUS = "gpu_status"
EVENT_SPEAKER_PROFILE_UPDATE = "speaker_profile_update"
EVENT_ERROR = "error"
