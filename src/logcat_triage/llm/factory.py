"""Build the configured provider from :class:`Settings`."""

from __future__ import annotations

from logcat_triage.config import Settings
from logcat_triage.llm.base import LLMError, LLMProvider
from logcat_triage.llm.ollama import OllamaProvider


def get_provider(settings: Settings | None = None) -> LLMProvider:
    """Return the provider selected by ``LLM_PROVIDER`` (default: Ollama ``qwen3:8b``)."""
    settings = settings or Settings()
    model = settings.resolved_model()
    if model is None:
        raise LLMError(f"Set LLM_MODEL for provider '{settings.llm_provider}'")
    timeout = settings.llm_timeout
    retries = settings.llm_max_retries
    tokens = settings.llm_max_tokens
    temperature = settings.llm_temperature
    if settings.llm_provider == "anthropic":
        from logcat_triage.llm.anthropic import AnthropicProvider

        key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else None
        return AnthropicProvider(
            model,
            key,
            timeout=timeout,
            max_retries=retries,
            max_tokens=tokens,
            temperature=temperature,
        )
    if settings.llm_provider == "openai":
        from logcat_triage.llm.openai import OpenAIProvider

        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else None
        return OpenAIProvider(
            model,
            key,
            timeout=timeout,
            max_retries=retries,
            max_tokens=tokens,
            temperature=temperature,
        )
    return OllamaProvider(
        model,
        settings.ollama_host,
        timeout=timeout,
        max_retries=retries,
        max_tokens=tokens,
        temperature=temperature,
    )
