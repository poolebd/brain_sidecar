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


def test_event_bus_replays_after_last_event_id(event_loop) -> None:
    bus = EventBus()
    first = SidecarEvent(type="transcript_final", session_id="ses_1", payload={"text": "first"})
    second = SidecarEvent(type="transcript_final", session_id="ses_1", payload={"text": "second"})
    event_loop.run_until_complete(bus.publish(first))
    event_loop.run_until_complete(bus.publish(second))

    task = event_loop.create_task(_collect_one_after(bus, "ses_1", first.id))
    event = event_loop.run_until_complete(task)

    assert event.id == second.id


def test_event_bus_counts_overflow(event_loop) -> None:
    bus = EventBus(queue_size=1)
    queue = event_loop.run_until_complete(bus.subscribe_queue("ses_1"))
    event_loop.run_until_complete(bus.publish(SidecarEvent(type="one", session_id="ses_1", payload={})))
    event_loop.run_until_complete(bus.publish(SidecarEvent(type="two", session_id="ses_1", payload={})))

    assert queue.qsize() == 1
    assert bus.drop_count("ses_1") == 1
    event_loop.run_until_complete(bus.unsubscribe_queue(queue, "ses_1"))


async def _collect_one_after(bus: EventBus, session_id: str, last_id: str) -> SidecarEvent:
    async for event in bus.subscribe(session_id, replay_after_id=last_id):
        return event
    raise AssertionError("subscription ended")
