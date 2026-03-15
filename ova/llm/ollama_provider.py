"""Ollama LLM provider.

Wraps the existing Ollama chat interface, extracting the code previously
embedded in pipeline.py into a clean provider implementation.
"""

from __future__ import annotations

from typing import Any, Iterator

from ..utils import get_logger
from .base import LLMError, LLMResponse, ToolCall

logger = get_logger("llm")


class OllamaProvider:
    """LLM provider using Ollama's native Python client."""

    def __init__(self, model: str):
        self.model = model

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list | None = None,
        stream: bool = False,
        options: dict | None = None,
    ) -> LLMResponse | Iterator[str]:
        from ollama import chat as ollama_chat

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "think": False,
            "stream": stream,
            "options": options or {},
        }

        if tools:
            kwargs["tools"] = tools

        if stream:
            # Streaming: keep_alive=-1 keeps model loaded
            kwargs["keep_alive"] = -1
            return self._stream(kwargs)

        try:
            response = ollama_chat(**kwargs)
        except Exception as e:
            raise self._handle_error(e) from e

        # Parse tool calls if present
        tool_calls = None
        if response.message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=None,  # Ollama doesn't use tool call IDs
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in response.message.tool_calls
            ]

        return LLMResponse(
            content=response.message.content,
            tool_calls=tool_calls,
            raw_message=response.message,  # Ollama message object for context
        )

    def _stream(self, kwargs: dict) -> Iterator[str]:
        from ollama import chat as ollama_chat

        try:
            stream = ollama_chat(**kwargs)
        except Exception as e:
            raise self._handle_error(e) from e

        try:
            for chunk in stream:
                yield chunk.message.content or ""
        finally:
            if hasattr(stream, 'close'):
                stream.close()

    def format_tools(self, tools: list) -> list:
        # Ollama accepts both callables and OllamaTool objects natively
        return tools

    def build_tool_result_message(self, tool_call_id: str | None, name: str, result: str) -> dict:
        return {"role": "tool", "tool_name": name, "content": result}

    def build_user_message(self, text: str, image_base64: str | None = None) -> dict:
        if image_base64:
            return {"role": "user", "content": text, "images": [image_base64]}
        return {"role": "user", "content": text}

    def _handle_error(self, e: Exception) -> LLMError:
        """Convert Ollama client exceptions to LLMError."""
        from ollama import ResponseError

        if isinstance(e, ConnectionError):
            return LLMError("Cannot connect to Ollama server (is 'ollama serve' running?)", status_code=502)
        if isinstance(e, ResponseError):
            msg = str(e)
            if "not found" in msg:
                return LLMError(f"Ollama model not found: {self.model}", status_code=404)
            return LLMError(f"Ollama error: {msg}", status_code=getattr(e, "status_code", 500))
        return LLMError(f"Ollama error: {e}", status_code=500)

    def warmup(self) -> None:
        logger.info(f"Warming up LLM ({self.model})...")
        try:
            self.chat(
                messages=[
                    {"role": "system", "content": "Respond with exactly one word."},
                    {"role": "user", "content": "Hi"},
                ],
            )
            logger.info("LLM warmup complete")
        except LLMError as e:
            logger.warning(f"LLM warmup failed (non-fatal): {e}")
        except Exception as e:
            logger.warning(f"LLM warmup failed (non-fatal): {e}")

    def unload(self) -> None:
        from ollama import chat as ollama_chat

        try:
            ollama_chat(model=self.model, messages=[], keep_alive=0)
        except Exception:
            pass
