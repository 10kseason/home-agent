import asyncio, hashlib, time
from typing import AsyncIterable, Callable, Dict, List, Optional
from loguru import logger
try:
    from .schemas import Event
except ModuleNotFoundError:
    from .types import Event

class DedupStore:
    def __init__(self, window_seconds: int = 3600):
        self.window = window_seconds
        self.store: Dict[str, float] = {}

    def _key(self, e: Event) -> str:
        # content hash by type + text-ish payload
        payload_str = str({k: e.payload.get(k) for k in sorted(e.payload.keys())})
        return hashlib.sha1((e.type + payload_str).encode("utf-8")).hexdigest()

    def seen_recently(self, e: Event) -> bool:
        key = self._key(e)
        now = time.time()
        # purge occasionally
        for k, ts in list(self.store.items()):
            if now - ts > self.window:
                self.store.pop(k, None)
        if key in self.store and now - self.store[key] <= self.window:
            return True
        self.store[key] = now
        return False

class EventBus:
    def __init__(self, max_queue_size: int = 1000, dedup_window: int = 3600):
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        self.dedup = DedupStore(window_seconds=dedup_window)
        self.subscribers: Dict[str, List[Callable[[Event], None]]] = {}

    async def publish(self, e: Event):
        if self.dedup.seen_recently(e):
            logger.debug(f"Deduped event: {e.type}")
            return
        priority = e.priority
        await self.queue.put((priority, time.time(), e))

    def subscribe(self, event_prefix: str, handler: Callable[[Event], None]):
        self.subscribers.setdefault(event_prefix, []).append(handler)

    async def run(self):
        logger.info("EventBus loop started.")
        while True:
            _, _, e = await self.queue.get()
            handled = False
            for prefix, handlers in self.subscribers.items():
                if e.type.startswith(prefix):
                    for h in handlers:
                        try:
                            await h(e)  # handler is async
                            handled = True
                        except Exception as ex:
                            logger.exception(f"Handler error for {e.type}: {ex}")
            if not handled:
                logger.warning(f"No subscriber for event: {e.type}")
