from brain_sidecar.core.event_bus import EventBus
from brain_sidecar.core.events import SidecarEvent


async def _collect_one(bus: EventBus, session_id: str) -> SidecarEvent:
    async for event in bus.subscribe(session_id):
        return event
    raise AssertionError("subscription ended")


def test_event_bus_fans_out_to_session(event_loop) -> None:
    bus = EventBus()
    task = event_loop.create_task(_collect_one(bus, "ses_1"))
    event_loop.run_until_complete(
        bus.publish(SidecarEvent(type="transcript_final", session_id="ses_1", payload={"text": "hi"}))
    )
    event = event_loop.run_until_complete(task)

    assert event.payload["text"] == "hi"
