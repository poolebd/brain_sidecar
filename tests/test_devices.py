from __future__ import annotations

import brain_sidecar.core.devices as devices


def test_list_audio_devices_uses_alsa_plughw_and_prefers_usb(monkeypatch) -> None:
    output = "\n".join(
        [
            "card 0: PCH [HDA Intel PCH], device 0: ALC287 Analog [ALC287 Analog]",
            "card 2: Device [USB PnP Audio Device], device 0: USB Audio [USB Audio]",
        ]
    )
    monkeypatch.setattr(devices, "_run", lambda args: output if args == ["arecord", "-l"] else "")

    listed = devices.list_audio_devices()

    assert [device.ffmpeg_input for device in listed] == ["plughw:2,0", "plughw:0,0"]
    assert listed[0].id == "alsa:plughw:2,0"
    assert "USB PnP Audio Device" in listed[0].label


def test_list_audio_devices_ignores_pulse_pipewire_rows(monkeypatch) -> None:
    def fake_run(args: list[str]) -> str:
        if args == ["arecord", "-l"]:
            return "card 2: Device [USB PnP Audio Device], device 0: USB Audio [USB Audio]"
        if args[:1] == ["pactl"]:
            return "1\talsa_input.usb-broken\tPipeWire\tfloat32le 2ch 48000Hz"
        return ""

    monkeypatch.setattr(devices, "_run", fake_run)

    listed = devices.list_audio_devices()

    assert len(listed) == 1
    assert listed[0].driver == "alsa"
