"""LLM provider abstraction: Ollama by default, Anthropic and OpenAI as optional extras."""

from logcat_triage.llm.base import Completion, LLMError, LLMProvider
from logcat_triage.llm.factory import get_provider

__all__ = ["Completion", "LLMError", "LLMProvider", "get_provider"]
