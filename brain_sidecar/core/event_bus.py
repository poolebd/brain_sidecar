from __future__ import annotations

import asyncio
from collections import defaultdict
from contextlib import suppress
from typing import AsyncIterator

from brain_sidecar.core.events import SidecarEvent


class EventBus:
    """Small fan-out bus shared by API clients and future GUI adapters."""

    def __init__(self, queue_size: int = 200) -> None:
        self._queue_size = queue_size
        self._lock = asyncio.Lock()
        self._subscribers: dict[str, set[asyncio.Queue[SidecarEvent]]] = defaultdict(set)

    async def publish(self, event: SidecarEvent) -> None:
        async with self._lock:
            queues = list(self._subscribers.get("*", set()))
            if event.session_id:
                queues.extend(self._subscribers.get(event.session_id, set()))

        for queue in queues:
            with suppress(asyncio.QueueFull):
                queue.put_nowait(event)

    async def subscribe(self, session_id: str | None = None) -> AsyncIterator[SidecarEvent]:
        key = session_id or "*"
        queue: asyncio.Queue[SidecarEvent] = asyncio.Queue(maxsize=self._queue_size)
        async with self._lock:
            self._subscribers[key].add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            async with self._lock:
                self._subscribers[key].discard(queue)
                if not self._subscribers[key]:
                    self._subscribers.pop(key, None)
