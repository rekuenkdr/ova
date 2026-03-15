"""Prosody tag parsing for natural TTS pauses.

Parses [pause:X] / [p:X] tags from LLM output text, splitting it into
text segments and pause segments for TTS synthesis.
"""
import os
import re
from dataclasses import dataclass
from typing import Union

MAX_PAUSE_DURATION = float(os.getenv("OVA_MAX_PAUSE_DURATION", "3.0"))

_PAUSE_PATTERN = re.compile(r'\[\s*(?:pause|p)\s*:\s*(\d+(?:\.\d+)?)\s*\]', re.IGNORECASE)


@dataclass
class TextSegment:
    text: str


@dataclass
class PauseSegment:
    duration_sec: float


ProsodySegment = Union[TextSegment, PauseSegment]


def parse_prosody(text: str, max_pause: float = MAX_PAUSE_DURATION) -> list[ProsodySegment]:
    """Split text on [pause:X] / [p:X] tags into segments.

    Args:
        text: Input text potentially containing pause tags.
        max_pause: Maximum pause duration in seconds (clamped).

    Returns:
        List of TextSegment and PauseSegment in order.
        If no tags found, returns [TextSegment(text=text)].
    """
    segments: list[ProsodySegment] = []
    last_end = 0

    for match in _PAUSE_PATTERN.finditer(text):
        # Text before this pause tag
        before = text[last_end:match.start()].strip()
        if before:
            segments.append(TextSegment(text=before))

        duration = min(float(match.group(1)), max_pause)
        if duration > 0:
            segments.append(PauseSegment(duration_sec=duration))

        last_end = match.end()

    # Text after the last tag (or entire text if no tags)
    after = text[last_end:].strip()
    if after:
        segments.append(TextSegment(text=after))

    # No tags found → return original text as single segment
    if not segments:
        stripped = text.strip()
        if stripped:
            return [TextSegment(text=stripped)]
        return []

    return segments


def strip_prosody_tags(text: str) -> str:
    """Remove all [pause:X] / [p:X] tags from text.

    Used to clean LLM responses before storing in conversation context.
    """
    return _PAUSE_PATTERN.sub('', text).strip()
