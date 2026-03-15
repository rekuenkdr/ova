"""In-memory timer tool with active expiration notifications."""
import atexit
import time
import threading

from ._base import publish_event

TOOL_ENABLED_DEFAULT = True

_MAX_TIMERS = 20
_timers: dict[str, dict] = {}
_lock = threading.Lock()


def _on_expire(label: str, seconds: int):
    """Callback fired by threading.Timer when a timer expires."""
    with _lock:
        _timers.pop(label, None)
    publish_event("timer_expired", {"label": label, "seconds": seconds})


def set_timer(label: str, seconds: int) -> str:
    """Set a countdown timer.

    Args:
        label: A short name for the timer (e.g. "pasta", "break").
        seconds: Duration in seconds (1-3600).

    Returns:
        str: Confirmation message with the timer details.
    """
    label = str(label).strip()[:64]
    if not label:
        return "Error: timer label cannot be empty."
    seconds = max(1, min(int(seconds), 3600))
    t = threading.Timer(seconds, _on_expire, args=(label, seconds))
    t.daemon = True
    with _lock:
        old = _timers.get(label)
        if old and "timer" in old:
            old["timer"].cancel()
        elif len(_timers) >= _MAX_TIMERS:
            return f"Error: maximum {_MAX_TIMERS} concurrent timers reached. Cancel or wait for one to finish."
        _timers[label] = {"start": time.time(), "seconds": seconds, "timer": t}
    t.start()
    return f"Timer '{label}' set for {seconds} seconds."


def check_timers() -> str:
    """Check the status of all active timers.

    Args:
        (none)

    Returns:
        str: A summary of all timers and their remaining time.
    """
    with _lock:
        if not _timers:
            return "No active timers."
        lines = []
        expired = []
        for label, info in _timers.items():
            elapsed = time.time() - info["start"]
            remaining = info["seconds"] - elapsed
            if remaining <= 0:
                lines.append(f"- {label}: DONE (finished {abs(remaining):.0f}s ago)")
                expired.append(label)
            else:
                mins, secs = divmod(int(remaining), 60)
                if mins:
                    lines.append(f"- {label}: {mins}m {secs}s remaining")
                else:
                    lines.append(f"- {label}: {secs}s remaining")
        for label in expired:
            del _timers[label]
    return "\n".join(lines)


def _cleanup():
    """Cancel all active timers on process exit."""
    with _lock:
        for info in _timers.values():
            if "timer" in info:
                info["timer"].cancel()


atexit.register(_cleanup)
