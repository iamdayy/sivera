import asyncio
import json
from uuid import UUID


class EventHub:
    def __init__(self, max_clients: int) -> None:
        self.max_clients = max_clients
        self.subscribers: dict[UUID, set[asyncio.Queue]] = {}
        self.closed = False

    @property
    def client_count(self) -> int:
        return sum(len(queues) for queues in self.subscribers.values())

    def subscribe(self, camera_id: UUID) -> asyncio.Queue:
        if self.closed:
            raise ValueError("prediction_stream_unavailable")
        if self.client_count >= self.max_clients:
            raise ValueError("prediction_stream_capacity")
        queue = asyncio.Queue(maxsize=1)
        self.subscribers.setdefault(camera_id, set()).add(queue)
        return queue

    def unsubscribe(self, camera_id: UUID, queue: asyncio.Queue) -> None:
        queues = self.subscribers.get(camera_id)
        if queues is not None:
            queues.discard(queue)
            if not queues:
                del self.subscribers[camera_id]

    @staticmethod
    def offer(queue: asyncio.Queue, value: str | None) -> None:
        if queue.full():
            queue.get_nowait()
        queue.put_nowait(value)

    async def publish(self, camera_id: UUID, event: dict) -> None:
        if self.closed or camera_id not in self.subscribers:
            return
        payload = await asyncio.to_thread(json.dumps, event, allow_nan=False, separators=(",", ":"))
        for queue in tuple(self.subscribers.get(camera_id, ())):
            self.offer(queue, payload)

    def close(self) -> None:
        self.closed = True
        for queues in self.subscribers.values():
            for queue in queues:
                self.offer(queue, None)
        self.subscribers.clear()
