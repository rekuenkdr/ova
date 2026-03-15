"""Factory for creating LLM providers based on environment configuration.

Reads OVA_LLM_PROVIDER to determine which provider to instantiate.
Uses lazy imports to avoid pulling in unused dependencies.
"""

from __future__ import annotations

import os

from ..utils import get_logger
from .base import LLMProvider

logger = get_logger("llm")


def create_llm_provider(model: str) -> LLMProvider:
    """Create an LLM provider based on environment configuration.

    Args:
        model: The model name/identifier to use.

    Returns:
        An LLMProvider instance (OllamaProvider or OpenAIProvider).

    Raises:
        ValueError: If OVA_LLM_PROVIDER has an unknown value.
    """
    provider_name = os.getenv("OVA_LLM_PROVIDER", "ollama").lower()

    if provider_name == "ollama":
        from .ollama_provider import OllamaProvider
        logger.info(f"Using Ollama provider (model={model})")
        return OllamaProvider(model=model)

    elif provider_name == "openai":
        from .openai_provider import OpenAIProvider
        base_url = os.getenv("OVA_LLM_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("OVA_LLM_API_KEY", "not-needed")
        logger.info(f"Using OpenAI-compatible provider (model={model}, base_url={base_url})")
        return OpenAIProvider(model=model, base_url=base_url, api_key=api_key)

    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider_name}'. "
            f"Set OVA_LLM_PROVIDER to 'ollama' or 'openai'."
        )
