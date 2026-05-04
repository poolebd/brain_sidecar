from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from typing import AsyncIterator

from brain_sidecar.core.events import SidecarEvent


class EventBus:
    """Small fan-out bus shared by API clients and future GUI adapters."""

    def __init__(self, queue_size: int = 200, replay_size: int = 300) -> None:
        self._queue_size = queue_size
        self._replay_size = replay_size
        self._lock = asyncio.Lock()
        self._subscribers: dict[str, set[asyncio.Queue[SidecarEvent]]] = defaultdict(set)
        self._drop_counts: dict[str, int] = defaultdict(int)
        self._replay: dict[str, deque[SidecarEvent]] = defaultdict(lambda: deque(maxlen=self._replay_size))

    async def publish(self, event: SidecarEvent) -> None:
        async with self._lock:
            self._remember_locked(event)
            queues = list(self._subscribers.get("*", set()))
            if event.session_id:
                queues.extend(self._subscribers.get(event.session_id, set()))

        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                key = event.session_id or "*"
                self._drop_counts[key] += 1

    async def subscribe(
        self,
        session_id: str | None = None,
        *,
        replay_after_id: str | None = None,
    ) -> AsyncIterator[SidecarEvent]:
        key = session_id or "*"
        queue = await self.subscribe_queue(session_id, replay_after_id=replay_after_id)
        try:
            while True:
                yield await queue.get()
        finally:
            await self.unsubscribe_queue(queue, session_id)

    async def subscribe_queue(
        self,
        session_id: str | None = None,
        *,
        replay_after_id: str | None = None,
    ) -> asyncio.Queue[SidecarEvent]:
        key = session_id or "*"
        queue: asyncio.Queue[SidecarEvent] = asyncio.Queue(maxsize=self._queue_size)
        async with self._lock:
            self._subscribers[key].add(queue)
            replay = self._replay_after_locked(key, replay_after_id)
        for event in replay:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self._drop_counts[key] += 1
                break
        return queue

    async def unsubscribe_queue(self, queue: asyncio.Queue[SidecarEvent], session_id: str | None = None) -> None:
        key = session_id or "*"
        async with self._lock:
            self._subscribers[key].discard(queue)
            if not self._subscribers[key]:
                self._subscribers.pop(key, None)

    def drop_count(self, session_id: str | None = None) -> int:
        return int(self._drop_counts.get(session_id or "*", 0))

    def replay_events(self, session_id: str | None = None, *, after_id: str | None = None) -> list[SidecarEvent]:
        key = session_id or "*"
        return self._replay_after_locked(key, after_id)

    def _remember_locked(self, event: SidecarEvent) -> None:
        self._replay["*"].append(event)
        if event.session_id:
            self._replay[event.session_id].append(event)

    def _replay_after_locked(self, key: str, after_id: str | None) -> list[SidecarEvent]:
        events = list(self._replay.get(key, []))
        if not after_id:
            return []
        for index, event in enumerate(events):
            if event.id == after_id:
                return events[index + 1 :]
        return events
