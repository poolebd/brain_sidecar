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
    healthy: bool = True
    in_use: bool = False
    score: int = 0
    selection_reason: str = ""

    def to_dict(self) -> dict[str, str | bool | int]:
        return {
            "id": self.id,
            "label": self.label,
            "driver": self.driver,
            "ffmpeg_input": self.ffmpeg_input,
            "hardware_id": self.hardware_id,
            "healthy": self.healthy,
            "in_use": self.in_use,
            "score": self.score,
            "selection_reason": self.selection_reason,
        }


def _run(args: list[str]) -> str:
    try:
        result = subprocess.run(args, check=False, capture_output=True, text=True, timeout=3)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return (result.stdout or "") + (result.stderr or "")


def list_audio_devices(*, probe: bool = False) -> list[DeviceInfo]:
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
        score, reason = _score_capture_device(base_label, hardware_id, ffmpeg_input)
        healthy = True
        in_use = False
        if probe:
            probe_result = _probe_capture_result(ffmpeg_input)
            healthy = probe_result["healthy"]
            in_use = probe_result["in_use"]
        if not healthy:
            reason = f"{reason}; probe failed"
        elif in_use:
            reason = f"{reason}; USB mic is in use by active capture"
        devices[device_id] = DeviceInfo(
            id=device_id,
            label=base_label,
            driver="alsa",
            ffmpeg_input=ffmpeg_input,
            hardware_id=hardware_id,
            healthy=healthy,
            in_use=in_use,
            score=score,
            selection_reason=reason,
        )

    return sorted(devices.values(), key=_device_sort_key)


def _device_sort_key(device: DeviceInfo) -> tuple[int, int, str]:
    label = device.label.lower()
    return (0 if device.healthy else 1, -device.score, label)


def find_device(device_id: str | None, *, probe: bool = True) -> DeviceInfo | None:
    devices = list_audio_devices(probe=probe)
    if device_id is None:
        return next((device for device in devices if device.healthy and not device.in_use), None)
    return next((device for device in devices if device.id == device_id and device.healthy and not device.in_use), None)


def _alsa_card_hardware_id(card: str) -> str:
    return _read_text(Path(f"/proc/asound/card{card}/usbid")).strip()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _score_capture_device(label: str, hardware_id: str, ffmpeg_input: str) -> tuple[int, str]:
    haystack = " ".join([label, hardware_id, ffmpeg_input]).lower()
    score = 10
    reasons: list[str] = []
    if hardware_id:
        score += 45
        reasons.append("USB hardware id")
    if "usb" in haystack:
        score += 30
        reasons.append("USB capture")
    if any(token in haystack for token in ("mic", "microphone", "audio")):
        score += 10
        reasons.append("capture input")
    if any(token in haystack for token in ("webcam", "camera")):
        score -= 6
        reasons.append("camera microphone")
    if any(token in haystack for token in ("pch", "analog", "hdmi", "monitor", "loopback", "output")):
        score -= 35
        reasons.append("lower priority built-in/output device")
    reason = ", ".join(reasons) if reasons else "available ALSA capture device"
    return score, reason


def _probe_capture(ffmpeg_input: str) -> bool:
    return _probe_capture_result(ffmpeg_input)["healthy"]


def _probe_capture_result(ffmpeg_input: str) -> dict[str, bool]:
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "alsa",
                "-i",
                ffmpeg_input,
                "-t",
                "0.2",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {"healthy": False, "in_use": False}
    if result.returncode == 0:
        return {"healthy": True, "in_use": False}
    output = f"{result.stdout or ''}\n{result.stderr or ''}".lower()
    if any(marker in output for marker in ("device or resource busy", "resource busy", "device busy", "busy")):
        return {"healthy": True, "in_use": True}
    return {"healthy": False, "in_use": False}
