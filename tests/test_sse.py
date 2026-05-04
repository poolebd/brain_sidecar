from brain_sidecar.core.events import SidecarEvent
from brain_sidecar.server.app import SSE_HEARTBEAT, encode_sse_event


def test_sse_event_encoding_includes_id_type_and_json_payload() -> None:
    event = SidecarEvent(type="audio_status", session_id="ses_1", payload={"status": "listening"})

    encoded = encode_sse_event(event)

    assert encoded.startswith(f"id: {event.id}\n")
    assert "\nevent: audio_status\n" in encoded
    assert '"status": "listening"' in encoded
    assert encoded.endswith("\n\n")


def test_sse_heartbeat_format_is_comment_frame() -> None:
    assert SSE_HEARTBEAT == ": heartbeat\n\n"
