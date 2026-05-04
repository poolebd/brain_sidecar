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
    assert listed[0].to_dict()["preferred"] is False


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


def test_list_audio_devices_prefers_configured_usb_match_and_label(monkeypatch) -> None:
    output = "\n".join(
        [
            "card 0: Webcam [Other USB Camera], device 0: USB Audio [USB Audio]",
            "card 2: Device [USB PnP Audio Device], device 0: USB Audio [USB Audio]",
        ]
    )

    def fake_read(path):
        if "card2" in str(path):
            return "0c76:161e\n"
        if "card0" in str(path):
            return "9999:0001\n"
        return ""

    monkeypatch.setattr(devices, "_run", lambda args: output if args == ["arecord", "-l"] else "")
    monkeypatch.setattr(devices, "_read_text", fake_read)

    listed = devices.list_audio_devices(
        preferred_device_match="0c76:161e",
        preferred_device_label="Anker PowerConf C200 microphone",
    )

    assert listed[0].id == "alsa:plughw:2,0"
    assert listed[0].hardware_id == "0c76:161e"
    assert listed[0].preferred is True
    assert listed[0].label.startswith("Anker PowerConf C200 microphone")
    assert listed[0].to_dict()["hardware_id"] == "0c76:161e"


def test_find_device_does_not_fall_back_when_preferred_mic_is_missing(monkeypatch) -> None:
    output = "card 2: Device [USB PnP Audio Device], device 0: USB Audio [USB Audio]"
    monkeypatch.setattr(devices, "_run", lambda args: output if args == ["arecord", "-l"] else "")
    monkeypatch.setattr(devices, "_read_text", lambda path: "0c76:161e\n")

    found = devices.find_device(None, preferred_device_match="291a:3369")

    assert found is None
    assert devices.find_device("alsa:plughw:2,0", preferred_device_match="291a:3369") is not None
