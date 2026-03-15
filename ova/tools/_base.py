"""Shared utilities for tool implementations."""
import os


def get_pipeline_language() -> str:
    """Return the configured pipeline language code."""
    return os.getenv("OVA_LANGUAGE", "es")


def get_pipeline_timezone() -> str:
    """Return the configured timezone (defaults to UTC)."""
    return os.getenv("OVA_TIMEZONE", "UTC")


def publish_event(event_type: str, data: dict | None = None):
    """Publish an event to all connected frontend clients."""
    from ..events import event_bus
    event_bus.publish(event_type, data)
