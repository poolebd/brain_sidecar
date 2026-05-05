from __future__ import annotations

from brain_sidecar.core.audio import FFmpegAudioCapture
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
