from __future__ import annotations

import asyncio
import subprocess
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator

from brain_sidecar.core.devices import DeviceInfo

MIN_INPUT_GAIN_DB = -12.0
MAX_INPUT_GAIN_DB = 12.0


class AudioCapture(ABC):
    @abstractmethod
    async def chunks(self) -> AsyncIterator[bytes]:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError


class FFmpegAudioCapture(AudioCapture):
    def __init__(
        self,
        device: DeviceInfo,
        sample_rate: int = 16_000,
        chunk_ms: int = 500,
        input_gain_db: float = 0.0,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.input_gain_db = max(MIN_INPUT_GAIN_DB, min(MAX_INPUT_GAIN_DB, float(input_gain_db)))
        self._process: subprocess.Popen[bytes] | None = None
        self._running = False

    def _args(self) -> list[str]:
        input_format = "pulse" if self.device.driver == "pulse" else "alsa"
        args = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            input_format,
            "-i",
            self.device.ffmpeg_input,
        ]
        if abs(self.input_gain_db) >= 0.05:
            args.extend(["-af", f"volume={self.input_gain_db:.1f}dB"])
        args.extend([
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ])
        return args

    async def chunks(self) -> AsyncIterator[bytes]:
        bytes_per_chunk = int(self.sample_rate * 2 * (self.chunk_ms / 1000))
        self._process = subprocess.Popen(
            self._args(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._running = True
        assert self._process.stdout is not None
        while self._running:
            data = await asyncio.to_thread(self._process.stdout.read, bytes_per_chunk)
            if not data:
                break
            yield data

    async def stop(self) -> None:
        self._running = False
        process = self._process
        if process and process.poll() is None:
            process.terminate()
            try:
                await asyncio.to_thread(process.wait, 2)
            except subprocess.TimeoutExpired:
                process.kill()


class FixtureWavAudioCapture(AudioCapture):
    """Test/preview source that streams a WAV file like a microphone."""

    def __init__(self, path: Path, sample_rate: int = 16_000, chunk_ms: int = 500) -> None:
        self.path = path
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self._running = True

    async def chunks(self) -> AsyncIterator[bytes]:
        frames_per_chunk = int(self.sample_rate * (self.chunk_ms / 1000))
        with wave.open(str(self.path), "rb") as wav:
            if wav.getnchannels() != 1 or wav.getframerate() != self.sample_rate or wav.getsampwidth() != 2:
                raise RuntimeError("Fixture WAV must be mono 16 kHz PCM16.")
            while self._running:
                data = wav.readframes(frames_per_chunk)
                if not data:
                    break
                yield data
                await asyncio.sleep(self.chunk_ms / 1000)

    async def stop(self) -> None:
        self._running = False
