from __future__ import annotations

import asyncio
import subprocess
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncIterator

from brain_sidecar.core.devices import DeviceInfo


class AudioCapture(ABC):
    @abstractmethod
    async def chunks(self) -> AsyncIterator[bytes]:
        raise NotImplementedError

    @abstractmethod
    async def stop(self) -> None:
        raise NotImplementedError


class BrowserAudioCapture(AudioCapture):
    """Live PCM source fed by a browser WebSocket."""

    def __init__(self, *, max_chunks: int = 24) -> None:
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=max(1, max_chunks))
        self._running = True

    async def chunks(self) -> AsyncIterator[bytes]:
        while self._running:
            data = await self._queue.get()
            try:
                if data is None:
                    break
                yield data
            finally:
                self._queue.task_done()

    async def feed(self, data: bytes) -> bool:
        if not self._running or not data:
            return False
        if self._queue.full():
            try:
                stale = self._queue.get_nowait()
                self._queue.task_done()
                if stale is None:
                    await self._queue.put(None)
                    return False
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(data)
        return True

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._queue.full():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(None)


class FFmpegAudioCapture(AudioCapture):
    def __init__(self, device: DeviceInfo, sample_rate: int = 16_000, chunk_ms: int = 500) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self._process: subprocess.Popen[bytes] | None = None
        self._running = False

    def _args(self) -> list[str]:
        input_format = "pulse" if self.device.driver == "pulse" else "alsa"
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            input_format,
            "-i",
            self.device.ffmpeg_input,
            "-ac",
            "1",
            "-ar",
            str(self.sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ]

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
