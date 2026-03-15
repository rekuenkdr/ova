"""OpenAI-compatible LLM provider.

Wraps any server that exposes the OpenAI /v1/chat/completions API,
including TensorRT-LLM, vLLM, and the official OpenAI API.
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable, Iterator

from ..utils import get_logger
from .base import LLMError, LLMResponse, ToolCall

logger = get_logger("llm")


def _parse_google_docstring(func: Callable) -> tuple[str, dict[str, dict]]:
    """Parse a Google-style docstring into description and parameter info.

    Args:
        func: The function to parse.

    Returns:
        Tuple of (description, {param_name: {"type": ..., "description": ...}}).
    """
    doc = inspect.getdoc(func) or ""
    lines = doc.split("\n")

    # Extract description (everything before Args:)
    desc_lines = []
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("Args:"):
            break
        desc_lines.append(lines[i])
        i += 1
    description = " ".join(line.strip() for line in desc_lines).strip()

    # Parse Args: section
    params: dict[str, dict] = {}
    if i < len(lines) and lines[i].strip().startswith("Args:"):
        i += 1
        current_param = None
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith("Returns:") or line.startswith("Raises:"):
                break
            # Check for new param: "name (type): description" or "name: description"
            match = re.match(r"(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)", line)
            if match:
                current_param = match.group(1)
                param_type = match.group(2) or "string"
                param_desc = match.group(3).strip()
                params[current_param] = {
                    "type": _python_type_to_json(param_type),
                    "description": param_desc,
                }
            elif current_param:
                # Continuation line for current param
                params[current_param]["description"] += " " + line
            i += 1

    return description, params


def _python_type_to_json(type_str: str) -> str:
    """Map Python type annotations to JSON Schema types."""
    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
    }
    return type_map.get(type_str.strip().lower(), "string")


def _callable_to_openai_tool(func: Callable) -> dict:
    """Convert a Python callable with Google-style docstring to OpenAI tool format.

    Args:
        func: The function to convert.

    Returns:
        OpenAI function tool definition dict.
    """
    description, doc_params = _parse_google_docstring(func)

    sig = inspect.signature(func)
    properties = {}
    required = []

    for name, param in sig.parameters.items():
        prop: dict[str, Any] = {}
        if name in doc_params:
            prop["type"] = doc_params[name]["type"]
            prop["description"] = doc_params[name]["description"]
        else:
            # Infer type from annotation
            ann = param.annotation
            if ann is not inspect.Parameter.empty:
                prop["type"] = _python_type_to_json(ann.__name__ if hasattr(ann, '__name__') else str(ann))
            else:
                prop["type"] = "string"
            prop["description"] = ""

        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required if required else None,
            },
        },
    }


def _ollama_tool_to_openai(tool) -> dict:
    """Convert an OllamaTool object to OpenAI tool format.

    Args:
        tool: An ollama.Tool instance.

    Returns:
        OpenAI function tool definition dict.
    """
    params_obj = tool.function.parameters
    properties = {}
    if params_obj and params_obj.properties:
        for name, prop in params_obj.properties.items():
            entry: dict[str, Any] = {"type": prop.type or "string"}
            if prop.description:
                entry["description"] = prop.description
            if prop.enum:
                entry["enum"] = prop.enum
            properties[name] = entry

    return {
        "type": "function",
        "function": {
            "name": tool.function.name,
            "description": tool.function.description or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": params_obj.required if params_obj and params_obj.required else None,
            },
        },
    }


class OpenAIProvider:
    """LLM provider for OpenAI-compatible APIs (TensorRT-LLM, vLLM, etc.)."""

    def __init__(self, model: str, base_url: str, api_key: str = "not-needed"):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, max_retries=0)
        return self._client

    def _handle_api_error(self, e: Exception) -> LLMError:
        """Convert OpenAI SDK exceptions to LLMError with clean messages."""
        import openai

        if isinstance(e, openai.AuthenticationError):
            return LLMError("LLM authentication failed: invalid or missing API key", status_code=401)
        if isinstance(e, openai.NotFoundError):
            return LLMError(f"LLM model or endpoint not found: {self.model} at {self.base_url}", status_code=404)
        if isinstance(e, openai.RateLimitError):
            return LLMError("LLM rate limit exceeded", status_code=429)
        if isinstance(e, openai.APIStatusError):
            return LLMError(f"LLM API error {e.status_code}", status_code=e.status_code)
        if isinstance(e, openai.APIConnectionError):
            return LLMError(f"Cannot connect to LLM server at {self.base_url}", status_code=502)
        return LLMError(f"LLM error: {e}", status_code=500)

    def chat(
        self,
        messages: list[dict],
        *,
        tools: list | None = None,
        stream: bool = False,
        options: dict | None = None,
    ) -> LLMResponse | Iterator[str]:
        import openai

        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }

        # Map Ollama options to OpenAI parameters
        if options:
            if "num_predict" in options:
                kwargs["max_tokens"] = options["num_predict"]
            if "temperature" in options:
                kwargs["temperature"] = options["temperature"]

        if tools:
            kwargs["tools"] = tools

        if stream:
            return self._stream(client, kwargs)

        try:
            response = client.chat.completions.create(**kwargs)
        except (openai.APIStatusError, openai.APIConnectionError) as e:
            raise self._handle_api_error(e) from e

        choice = response.choices[0]
        message = choice.message

        # Parse tool calls
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                )
                for tc in message.tool_calls
            ]

        # Build raw_message for context tracking (OpenAI format)
        raw = {"role": "assistant", "content": message.content}
        if message.tool_calls:
            raw["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            raw_message=raw,
        )

    def _stream(self, client, kwargs: dict) -> Iterator[str]:
        import openai

        try:
            stream = client.chat.completions.create(**kwargs)
        except (openai.APIStatusError, openai.APIConnectionError) as e:
            raise self._handle_api_error(e) from e

        try:
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            if hasattr(stream, 'close'):
                stream.close()

    def format_tools(self, tools: list) -> list:
        """Convert mixed callables/OllamaTool objects to OpenAI format."""
        from ollama import Tool as OllamaTool

        result = []
        for tool in tools:
            if callable(tool) and not isinstance(tool, OllamaTool):
                result.append(_callable_to_openai_tool(tool))
            elif isinstance(tool, OllamaTool):
                result.append(_ollama_tool_to_openai(tool))
            elif isinstance(tool, dict):
                # Already in OpenAI format
                result.append(tool)
        return result

    def build_tool_result_message(self, tool_call_id: str | None, name: str, result: str) -> dict:
        return {"role": "tool", "tool_call_id": tool_call_id or name, "content": result}

    def build_user_message(self, text: str, image_base64: str | None = None) -> dict:
        if image_base64:
            return {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                ],
            }
        return {"role": "user", "content": text}

    def _is_local(self) -> bool:
        """Check if base_url points to a local server."""
        from urllib.parse import urlparse
        host = urlparse(self.base_url).hostname or ""
        return host in ("localhost", "127.0.0.1", "0.0.0.0")

    def warmup(self) -> None:
        if not self._is_local():
            logger.info(f"Skipping LLM warmup (remote API: {self.base_url})")
            return
        logger.info(f"Warming up LLM ({self.model} @ {self.base_url})...")
        try:
            self.chat(
                messages=[
                    {"role": "system", "content": "Respond with exactly one word."},
                    {"role": "user", "content": "Hi"},
                ],
                options={"num_predict": 10},
            )
            logger.info("LLM warmup complete")
        except LLMError as e:
            logger.warning(f"LLM warmup failed (non-fatal): {e}")
        except Exception as e:
            logger.warning(f"LLM warmup failed (non-fatal): {e}")

    def unload(self) -> None:
        # External server manages its own memory — nothing to do
        pass
