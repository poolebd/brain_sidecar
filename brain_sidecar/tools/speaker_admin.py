from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from brain_sidecar.config import load_settings
from brain_sidecar.core.speaker_identity import SpeakerIdentityService
from brain_sidecar.core.storage import Storage


BAD_LEGACY_CORRECTIONS = [
    (
        "Brain, Sidecar, learn my voice for local transcription.",
        "Brain Sidecar, learn my voice for local transposition.",
    ),
    (
        "The quick brown fox checks the microphone and the GPU.",
        "The quick brown box checks the microphone and the GPU.",
    ),
    (
        "If the room is noisy, prefer the words I commonly correct.",
        "The room is noisy. Prefer the words are commonly correct.",
    ),
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Speaker identity profile admin tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate-profile", help="Print speaker identity status and metrics as JSON.")
    evaluate.add_argument("--profile-id", default="self_bp")

    reset = subparsers.add_parser("reset-profile", help="Delete learned speaker embeddings for the self profile.")
    reset.add_argument("--profile-id", default="self_bp")
    reset.add_argument("--backup", action="store_true")

    quarantine = subparsers.add_parser(
        "quarantine-legacy-asr-corrections",
        help="Quarantine known bad legacy ASR phrase corrections and deactivate legacy voice guidance.",
    )
    quarantine.add_argument("--backup", action="store_true")

    args = parser.parse_args(argv)
    storage = _storage()
    service = SpeakerIdentityService(storage, load_settings())

    if args.command == "evaluate-profile":
        print(json.dumps(service.status(), indent=2, sort_keys=True))
        return

    if args.command == "reset-profile":
        backup = _backup_db(storage.db_path) if args.backup else None
        payload = service.reset_profile()
        payload["backup_path"] = str(backup) if backup else None
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if args.command == "quarantine-legacy-asr-corrections":
        backup = _backup_db(storage.db_path) if args.backup else None
        quarantined = storage.quarantine_voice_corrections(
            BAD_LEGACY_CORRECTIONS,
            reason="known bad legacy ASR correction after speaker identity migration",
        )
        try:
            storage.set_voice_profile_active(False)
        except Exception:
            pass
        print(
            json.dumps(
                {
                    "quarantined_corrections": quarantined,
                    "legacy_voice_profile_active": False,
                    "backup_path": str(backup) if backup else None,
                    "speaker_identity_status": service.status()["enrollment_status"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return


def _storage() -> Storage:
    settings = load_settings()
    storage = Storage(settings.data_dir)
    storage.connect()
    return storage


def _backup_db(db_path: Path) -> Path:
    if not db_path.exists():
        raise RuntimeError(f"Database does not exist: {db_path}")
    backup = db_path.with_suffix(f".sqlite3.{time.strftime('%Y%m%d-%H%M%S')}.bak")
    shutil.copy2(db_path, backup)
    return backup


if __name__ == "__main__":
    main()
