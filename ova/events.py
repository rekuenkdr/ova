"""Generic server→client event bus.

Thread-safe singleton. Tools call publish() from sync threadpool workers;
SSE subscribers are async queues drained by the API endpoint.
"""

import asyncio
import threading
import time

from .utils import get_logger

logger = get_logger("events")


class EventBus:
    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Store the running event loop (call from async startup)."""
        self._loop = loop

    def publish(self, event_type: str, data: dict | None = None):
        """Thread-safe — safe to call from tool functions in threadpool."""
        # Sanitize event type: reject newlines (SSE field delimiter injection)
        if not isinstance(event_type, str) or '\n' in event_type or '\r' in event_type:
            logger.warning(f"Rejected event with invalid type (contains newline)")
            return
        event_type = event_type[:64]  # Length cap
        event = {"type": event_type, "data": data or {}, "ts": time.time()}
        with self._lock:
            dead = []
            for q in self._subscribers:
                try:
                    if self._loop and self._loop.is_running():
                        self._loop.call_soon_threadsafe(q.put_nowait, event)
                except asyncio.QueueFull:
                    pass  # Drop event for slow consumers rather than OOM
                except Exception:
                    dead.append(q)
            for q in dead:
                self._subscribers.remove(q)

    _MAX_SUBSCRIBERS = 5

    def subscribe(self) -> asyncio.Queue | None:
        """Create a new subscriber queue for SSE streaming.

        Returns None if the maximum subscriber count has been reached.
        """
        with self._lock:
            if len(self._subscribers) >= self._MAX_SUBSCRIBERS:
                return None
            q: asyncio.Queue = asyncio.Queue(maxsize=100)
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove a subscriber queue."""
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass


event_bus = EventBus()
