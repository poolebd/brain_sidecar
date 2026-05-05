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
    monkeypatch.setattr(devices, "_read_text", lambda path: "0c76:161e\n" if "card2" in str(path) else "")

    listed = devices.list_audio_devices()

    assert [device.ffmpeg_input for device in listed] == ["plughw:2,0", "plughw:0,0"]
    assert listed[0].id == "alsa:plughw:2,0"
    assert "USB PnP Audio Device" in listed[0].label
    assert listed[0].to_dict()["healthy"] is True
    assert listed[0].score > listed[1].score


def test_list_audio_devices_ignores_pulse_pipewire_rows(monkeypatch) -> None:
    def fake_run(args: list[str]) -> str:
        if args == ["arecord", "-l"]:
            return "card 2: Device [USB PnP Audio Device], device 0: USB Audio [USB Audio]"
        if args[:1] == ["pactl"]:
            return "1\talsa_input.usb-broken\tPipeWire\tfloat32le 2ch 48000Hz"
        return ""

    monkeypatch.setattr(devices, "_run", fake_run)
    monkeypatch.setattr(devices, "_read_text", lambda path: "0c76:161e\n")

    listed = devices.list_audio_devices()

    assert len(listed) == 1
    assert listed[0].driver == "alsa"


def test_list_audio_devices_marks_probe_failures_and_keeps_best_healthy_first(monkeypatch) -> None:
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
    monkeypatch.setattr(
        devices,
        "_probe_capture_result",
        lambda ffmpeg_input: {"healthy": ffmpeg_input == "plughw:2,0", "in_use": False},
    )

    listed = devices.list_audio_devices(probe=True)

    assert listed[0].id == "alsa:plughw:2,0"
    assert listed[0].hardware_id == "0c76:161e"
    assert listed[0].healthy is True
    assert "USB capture" in listed[0].selection_reason
    assert listed[0].to_dict()["hardware_id"] == "0c76:161e"
    assert listed[1].healthy is False
    assert "probe failed" in listed[1].selection_reason


def test_find_device_autoselects_best_healthy_server_mic(monkeypatch) -> None:
    output = "\n".join(
        [
            "card 0: PCH [HDA Intel PCH], device 0: ALC287 Analog [ALC287 Analog]",
            "card 2: Device [USB PnP Audio Device], device 0: USB Audio [USB Audio]",
        ]
    )
    monkeypatch.setattr(devices, "_run", lambda args: output if args == ["arecord", "-l"] else "")
    monkeypatch.setattr(devices, "_read_text", lambda path: "0c76:161e\n")
    monkeypatch.setattr(
        devices,
        "_probe_capture_result",
        lambda ffmpeg_input: {"healthy": ffmpeg_input != "plughw:2,0", "in_use": False},
    )

    found = devices.find_device(None, probe=True)

    assert found is not None
    assert found.id == "alsa:plughw:0,0"
    assert devices.find_device("alsa:plughw:2,0", probe=True) is None
    assert devices.find_device("alsa:plughw:2,0", probe=False) is not None


def test_list_audio_devices_reports_busy_usb_mic_as_in_use(monkeypatch) -> None:
    output = "card 2: Device [USB PnP Audio Device], device 0: USB Audio [USB Audio]"
    monkeypatch.setattr(devices, "_run", lambda args: output if args == ["arecord", "-l"] else "")
    monkeypatch.setattr(devices, "_read_text", lambda path: "0c76:161e\n")
    monkeypatch.setattr(
        devices,
        "_probe_capture_result",
        lambda ffmpeg_input: {"healthy": True, "in_use": True},
    )

    listed = devices.list_audio_devices(probe=True)

    assert listed[0].healthy is True
    assert listed[0].in_use is True
    assert listed[0].to_dict()["in_use"] is True
    assert "in use by active capture" in listed[0].selection_reason
    assert devices.find_device(None, probe=True) is None
