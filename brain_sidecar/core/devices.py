from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    label: str
    driver: str
    ffmpeg_input: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "driver": self.driver,
            "ffmpeg_input": self.ffmpeg_input,
        }


def _run(args: list[str]) -> str:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=3)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "") + (result.stderr or "")


def list_audio_devices() -> list[DeviceInfo]:
    devices: dict[str, DeviceInfo] = {}

    for line in _run(["arecord", "-l"]).splitlines():
        match = re.search(r"card (\d+): ([^,]+).*device (\d+): (.+)", line)
        if not match:
            continue
        card, card_name, device, device_name = match.groups()
        ffmpeg_input = f"plughw:{card},{device}"
        device_id = f"alsa:{ffmpeg_input}"
        devices[device_id] = DeviceInfo(
            id=device_id,
            label=f"ALSA {card_name.strip()} / {device_name.strip()}",
            driver="alsa",
            ffmpeg_input=ffmpeg_input,
        )

    return sorted(devices.values(), key=_device_sort_key)


def _device_sort_key(device: DeviceInfo) -> tuple[int, str]:
    label = device.label.lower()
    return (0 if "usb" in label else 1, label)


def find_device(device_id: str | None) -> DeviceInfo | None:
    devices = list_audio_devices()
    if device_id is None:
        return devices[0] if devices else None
    return next((device for device in devices if device.id == device_id), None)
