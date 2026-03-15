"""Core abstractions for LLM providers.

Defines the protocol and data classes that all LLM providers must implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable


class LLMError(Exception):
    """LLM provider error with HTTP status code."""

    def __init__(self, message: str, status_code: int = 500):
        self.status_code = status_code
        super().__init__(message)


@dataclass
class ToolCall:
    """Normalized tool call from any LLM provider.

    Attributes:
        id: Tool call ID (None for Ollama, string for OpenAI-compatible).
        name: Function name to call.
        arguments: Parsed keyword arguments dict.
    """
    id: str | None
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized LLM response from any provider.

    Attributes:
        content: Text content of the response (may be empty if tool_calls present).
        tool_calls: List of tool calls, or None if no tools were invoked.
        raw_message: Provider-specific message object for appending to context.
            Ollama: the response.message object.
            OpenAI: a dict with role/content/tool_calls for context tracking.
    """
    content: str | None
    tool_calls: list[ToolCall] | None = None
    raw_message: Any = None


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM providers must implement."""

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list | None = None,
        stream: bool = False,
        options: dict | None = None,
    ) -> LLMResponse | Iterator[str]:
        """Send messages to the LLM.

        Args:
            messages: Conversation messages list.
            tools: Optional list of tool definitions (callables or dicts).
            stream: If True, returns an Iterator[str] of tokens.
            options: Provider-specific options (e.g. num_predict, max_tokens).

        Returns:
            LLMResponse when stream=False, Iterator[str] when stream=True.
        """
        ...

    def format_tools(self, tools: list) -> list:
        """Convert tool definitions to provider-specific format.

        Args:
            tools: Mix of callables and OllamaTool objects from the registry.

        Returns:
            List in the format expected by this provider's API.
        """
        ...

    def build_tool_result_message(self, tool_call_id: str | None, name: str, result: str) -> dict:
        """Build a tool result message for appending to context.

        Args:
            tool_call_id: The tool call ID (required for OpenAI, None for Ollama).
            name: The tool function name.
            result: The string result from the tool.

        Returns:
            Message dict in provider-specific format.
        """
        ...

    def build_user_message(self, text: str, image_base64: str | None = None) -> dict:
        """Build a user message, optionally with an image.

        Args:
            text: User message text.
            image_base64: Optional base64-encoded image data.

        Returns:
            Message dict in provider-specific format.
        """
        ...

    def warmup(self) -> None:
        """Warm up the LLM (e.g. load into VRAM)."""
        ...

    def unload(self) -> None:
        """Unload the LLM from memory (e.g. Ollama keep_alive=0)."""
        ...
