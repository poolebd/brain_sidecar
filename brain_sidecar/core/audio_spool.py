from __future__ import annotations

import os
import time
import wave
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PATH = object()


@dataclass
class AudioSpoolResult:
    path: Path | None
    bytes_written: int
    duration_seconds: float
    finalized_at: float | None


class AudioSpoolWriter:
    """Local temporary WAV writer used only for Review handoff."""

    def __init__(self, final_path: Path, *, sample_rate: int) -> None:
        self.final_path = final_path
        self.partial_path = final_path.with_suffix(final_path.suffix + ".partial")
        self.sample_rate = sample_rate
        self.bytes_written = 0
        self.finalized_at: float | None = None
        self._wav: wave.Wave_write | None = None
        self._closed = False

    def start(self) -> None:
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path.unlink(missing_ok=True)
        self.final_path.unlink(missing_ok=True)
        self._wav = wave.open(str(self.partial_path), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.sample_rate)

    def write_pcm16(self, pcm: bytes) -> None:
        if not pcm or self._closed:
            return
        if self._wav is None:
            self.start()
        assert self._wav is not None
        self._wav.writeframes(pcm)
        self.bytes_written += len(pcm)

    def finalize(self) -> AudioSpoolResult:
        if self._closed:
            return self.result()
        self._closed = True
        if self._wav is not None:
            self._wav.close()
            self._wav = None
        if self.bytes_written > 0 and self.partial_path.exists():
            os.replace(self.partial_path, self.final_path)
            self.finalized_at = time.time()
            return self.result()
        self.partial_path.unlink(missing_ok=True)
        self.final_path.unlink(missing_ok=True)
        self.finalized_at = time.time()
        return self.result(path=None)

    def abort(self) -> None:
        self._closed = True
        if self._wav is not None:
            self._wav.close()
            self._wav = None
        self.partial_path.unlink(missing_ok=True)
        if self.bytes_written == 0:
            self.final_path.unlink(missing_ok=True)

    def result(self, *, path: Path | None | object = _DEFAULT_PATH) -> AudioSpoolResult:
        result_path = self.final_path if path is _DEFAULT_PATH else path
        if not isinstance(result_path, Path):
            result_path = None
        duration_seconds = self.bytes_written / float(max(1, self.sample_rate * 2))
        return AudioSpoolResult(
            path=result_path if self.bytes_written > 0 else None,
            bytes_written=self.bytes_written,
            duration_seconds=duration_seconds,
            finalized_at=self.finalized_at,
        )


def delete_audio_file(path: Path | None) -> None:
    if path is None:
        return
    path.unlink(missing_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.unlink(missing_ok=True)
    conditioned = path.parent / "conditioned.wav"
    conditioned.unlink(missing_ok=True)
    try:
        if path.parent.exists() and not any(path.parent.iterdir()):
            path.parent.rmdir()
    except OSError:
        return
