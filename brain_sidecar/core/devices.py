from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeviceInfo:
    id: str
    label: str
    driver: str
    ffmpeg_input: str
    hardware_id: str = ""
    preferred: bool = False

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "id": self.id,
            "label": self.label,
            "driver": self.driver,
            "ffmpeg_input": self.ffmpeg_input,
            "hardware_id": self.hardware_id,
            "preferred": self.preferred,
        }


def _run(args: list[str]) -> str:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=3)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "") + (result.stderr or "")


def list_audio_devices(
    *,
    preferred_device_id: str | None = None,
    preferred_device_match: str | None = None,
    preferred_device_label: str | None = None,
) -> list[DeviceInfo]:
    devices: dict[str, DeviceInfo] = {}

    for line in _run(["arecord", "-l"]).splitlines():
        match = re.search(r"card (\d+): ([^,]+).*device (\d+): (.+)", line)
        if not match:
            continue
        card, card_name, device, device_name = match.groups()
        ffmpeg_input = f"plughw:{card},{device}"
        device_id = f"alsa:{ffmpeg_input}"
        hardware_id = _alsa_card_hardware_id(card)
        base_label = f"ALSA {card_name.strip()} / {device_name.strip()}"
        preferred = _matches_preference(
            device_id=device_id,
            label=base_label,
            ffmpeg_input=ffmpeg_input,
            hardware_id=hardware_id,
            preferred_device_id=preferred_device_id,
            preferred_device_match=preferred_device_match,
        )
        devices[device_id] = DeviceInfo(
            id=device_id,
            label=_display_label(base_label, preferred=preferred, preferred_device_label=preferred_device_label),
            driver="alsa",
            ffmpeg_input=ffmpeg_input,
            hardware_id=hardware_id,
            preferred=preferred,
        )

    return sorted(devices.values(), key=_device_sort_key)


def _device_sort_key(device: DeviceInfo) -> tuple[int, str]:
    label = device.label.lower()
    if device.preferred:
        return (0, label)
    return (1 if "usb" in label or device.hardware_id else 2, label)


def find_device(
    device_id: str | None,
    *,
    preferred_device_id: str | None = None,
    preferred_device_match: str | None = None,
    preferred_device_label: str | None = None,
) -> DeviceInfo | None:
    devices = list_audio_devices(
        preferred_device_id=preferred_device_id,
        preferred_device_match=preferred_device_match,
        preferred_device_label=preferred_device_label,
    )
    if device_id is None:
        if preferred_device_id or preferred_device_match:
            return next((device for device in devices if device.preferred), None)
        return devices[0] if devices else None
    return next((device for device in devices if device.id == device_id), None)


def _alsa_card_hardware_id(card: str) -> str:
    return _read_text(Path(f"/proc/asound/card{card}/usbid")).strip()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _matches_preference(
    *,
    device_id: str,
    label: str,
    ffmpeg_input: str,
    hardware_id: str,
    preferred_device_id: str | None,
    preferred_device_match: str | None,
) -> bool:
    if preferred_device_id and device_id == preferred_device_id.strip():
        return True
    needle = (preferred_device_match or "").strip().lower()
    if not needle:
        return False
    haystack = " ".join([device_id, label, ffmpeg_input, hardware_id]).lower()
    return needle in haystack


def _display_label(base_label: str, *, preferred: bool, preferred_device_label: str | None) -> str:
    clean_label = (preferred_device_label or "").strip()
    if preferred and clean_label:
        return f"{clean_label} ({base_label})"
    return base_label
