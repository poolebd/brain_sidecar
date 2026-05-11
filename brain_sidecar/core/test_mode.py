from __future__ import annotations

import json
import re
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from brain_sidecar.config import Settings
from brain_sidecar.core.models import new_id


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".aac", ".mp4", ".ogg", ".flac", ".webm"}


@dataclass(frozen=True)
class PreparedTestAudio:
    run_id: str
    source_path: Path
    fixture_wav: Path
    duration_seconds: float
    artifact_dir: Path
    report_path: Path
    expected_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_path": str(self.source_path),
            "fixture_wav": str(self.fixture_wav),
            "duration_seconds": self.duration_seconds,
            "artifact_dir": str(self.artifact_dir),
            "report_path": str(self.report_path),
            "expected_terms": self.expected_terms,
        }


class TestModeService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def run_dir(self) -> Path:
        return self.settings.test_audio_run_dir.expanduser()

    def prepare_audio(
        self,
        source_path: Path,
        *,
        max_seconds: float | None = None,
        expected_terms: list[str] | None = None,
    ) -> PreparedTestAudio:
        source = source_path.expanduser().resolve()
        if not source.exists() or not source.is_file():
            raise ValueError(f"Source audio file does not exist: {source}")
        if source.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise ValueError(f"Unsupported audio extension {source.suffix!r}. Supported: {supported}")
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError("max_seconds must be greater than zero.")

        run_id = new_id("testrun")
        artifact_dir = self.run_dir / run_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        fixture_wav = artifact_dir / "input.wav"
        report_path = artifact_dir / "report.json"

        self._convert_to_fixture_wav(source, fixture_wav, max_seconds=max_seconds)
        duration_seconds = _wav_duration_seconds(fixture_wav)
        prepared = PreparedTestAudio(
            run_id=run_id,
            source_path=source,
            fixture_wav=fixture_wav,
            duration_seconds=duration_seconds,
            artifact_dir=artifact_dir,
            report_path=report_path,
            expected_terms=_clean_expected_terms(expected_terms or []),
        )
        (artifact_dir / "prepared.json").write_text(
            json.dumps(prepared.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return prepared

    def write_report(self, run_id: str, report: dict[str, Any]) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("Invalid test run id.")
        artifact_dir = self.run_dir / run_id
        if not artifact_dir.exists() or not artifact_dir.is_dir():
            raise ValueError(f"Test run does not exist: {run_id}")
        report_path = artifact_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return {"run_id": run_id, "report_path": str(report_path)}

    def _convert_to_fixture_wav(
        self,
        source_path: Path,
        fixture_wav: Path,
        *,
        max_seconds: float | None,
    ) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source_path),
        ]
        if max_seconds is not None:
            command.extend(["-t", f"{max_seconds:.3f}"])
        command.extend(
            [
                "-ac",
                "1",
                "-ar",
                str(self.settings.audio_sample_rate),
                "-sample_fmt",
                "s16",
                str(fixture_wav),
            ]
        )
        try:
            result = subprocess.run(command, capture_output=True, check=False, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required for recorded audio test mode.") from exc
        if result.returncode != 0:
            message = result.stderr.strip() or "ffmpeg failed to convert the audio file."
            raise RuntimeError(message)


def _clean_expected_terms(terms: list[str]) -> list[str]:
    cleaned = []
    seen = set()
    for term in terms:
        value = re.sub(r"\s+", " ", str(term)).strip()
        key = value.lower()
        if value and key not in seen:
            cleaned.append(value[:120])
            seen.add(key)
    return cleaned[:64]


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())
