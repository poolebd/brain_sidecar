from __future__ import annotations

import asyncio
import wave
from pathlib import Path

from brain_sidecar.core.audio import FFmpegAudioCapture, FixtureWavAudioCapture
from brain_sidecar.core.devices import DeviceInfo


def test_ffmpeg_capture_applies_input_gain_filter() -> None:
    capture = FFmpegAudioCapture(
        DeviceInfo(
            id="alsa:plughw:2,0",
            label="Server mic",
            driver="alsa",
            ffmpeg_input="plughw:2,0",
        ),
        input_gain_db=6,
    )

    args = capture._args()

    assert args[args.index("-af") + 1] == "volume=6.0dB"


def test_ffmpeg_capture_omits_zero_gain_filter() -> None:
    capture = FFmpegAudioCapture(
        DeviceInfo(
            id="alsa:plughw:2,0",
            label="Server mic",
            driver="alsa",
            ffmpeg_input="plughw:2,0",
        ),
        input_gain_db=0,
    )

    assert "-af" not in capture._args()


def test_ffmpeg_capture_clamps_excessive_input_gain() -> None:
    capture = FFmpegAudioCapture(
        DeviceInfo(
            id="alsa:plughw:2,0",
            label="Server mic",
            driver="alsa",
            ffmpeg_input="plughw:2,0",
        ),
        input_gain_db=24,
    )

    args = capture._args()

    assert args[args.index("-af") + 1] == "volume=12.0dB"


def test_fixture_wav_capture_can_pause_and_resume(event_loop, tmp_path: Path) -> None:
    fixture = tmp_path / "fixture.wav"
    _write_wav(fixture, sample_rate=16_000, seconds=1)
    capture = FixtureWavAudioCapture(fixture, sample_rate=16_000, chunk_ms=50)

    async def scenario() -> None:
        assert await capture.pause() is True
        chunks = capture.chunks()
        first_chunk = asyncio.create_task(anext(chunks))
        await asyncio.sleep(0.06)
        assert first_chunk.done() is False

        assert await capture.resume() is True
        data = await asyncio.wait_for(first_chunk, timeout=0.2)
        assert data
        await capture.stop()
        await chunks.aclose()

    event_loop.run_until_complete(scenario())


def _write_wav(path: Path, *, sample_rate: int, seconds: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\0" * sample_rate * 2 * seconds)
