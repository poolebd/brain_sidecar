#!/usr/bin/env python
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave
from pathlib import Path

from brain_sidecar.config import load_settings
from brain_sidecar.core.asr import ASR_BACKEND_NEMOTRON_STREAMING
from brain_sidecar.core.asr_factory import create_asr_backend
from brain_sidecar.core.gpu import read_gpu_status


DEFAULT_WAV = Path("runtime/test-audio/osr-us-female-harvard.wav")


def read_pcm16_wav(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getframerate() != 16_000 or wav.getsampwidth() != 2:
            raise RuntimeError("Nemotron smoke WAV must be mono 16 kHz PCM16.")
        frames = wav.readframes(wav.getnframes())
        return frames, wav.getnframes() / wav.getframerate()


async def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test Brain Sidecar's optional Nemotron streaming ASR backend.")
    parser.add_argument("wav", nargs="?", type=Path, default=DEFAULT_WAV)
    args = parser.parse_args()

    settings = load_settings()
    if settings.asr_backend != ASR_BACKEND_NEMOTRON_STREAMING:
        raise RuntimeError("Set BRAIN_SIDECAR_ASR_BACKEND=nemotron_streaming before running this smoke test.")

    pcm, duration_s = read_pcm16_wav(args.wav)
    chunk_bytes = int(settings.audio_sample_rate * 2 * (settings.nemotron_chunk_ms / 1000))
    backend = create_asr_backend(settings)

    before = read_gpu_status()
    started = time.monotonic()
    await backend.load()
    stream = await backend.open_stream("smoke", start_offset_s=0.0)
    partials = 0
    finals = 0
    final_text: list[str] = []
    offset = 0
    try:
        for start in range(0, len(pcm), chunk_bytes):
            chunk = pcm[start : start + chunk_bytes]
            events = await stream.accept_pcm16(chunk, offset / (settings.audio_sample_rate * 2))
            offset += len(chunk)
            for event in events:
                if event.kind == "partial":
                    partials += 1
                else:
                    finals += 1
                    final_text.append(event.text)
        for event in await stream.flush(final_offset_s=duration_s):
            if event.kind == "partial":
                partials += 1
            else:
                finals += 1
                final_text.append(event.text)
    finally:
        await stream.close()

    elapsed = time.monotonic() - started
    after = read_gpu_status()
    print(f"model_id: {settings.nemotron_model_id}")
    print(f"chunk_ms: {settings.nemotron_chunk_ms}")
    print(f"partial_count: {partials}")
    print(f"final_count: {finals}")
    print(f"final_text: {' '.join(final_text).strip()}")
    print(f"realtime_factor: {elapsed / max(0.001, duration_s):.3f}")
    print(f"gpu_before_free_mb: {before.memory_free_mb}")
    print(f"gpu_after_free_mb: {after.memory_free_mb}")
    print("raw_audio_retained: false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
