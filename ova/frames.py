"""Lightweight frame abstractions for pipeline control signals and data flow.

Frames make control signals (interrupt, end-of-stream) explicit and typed,
improving observability and enabling future extensibility without a full
pipeline rewrite.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioFrame:
    """A chunk of audio data flowing through the pipeline."""
    audio: np.ndarray
    sample_rate: int
    chunk_index: int = 0
    is_first: bool = False
    is_last: bool = False


@dataclass
class TextFrame:
    """A text segment (partial or complete) from ASR or LLM."""
    text: str
    is_partial: bool = False


@dataclass
class ControlFrame:
    """A control signal flowing through the pipeline."""
    action: str  # "interrupt", "end_of_stream", "start_of_stream"


@dataclass
class InterruptFrame(ControlFrame):
    """Signals that the current pipeline operation should stop (e.g., barge-in)."""
    action: str = field(default="interrupt")
    reason: str = "barge_in"
