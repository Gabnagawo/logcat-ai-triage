"""OpenAI provider. The SDK is an optional extra: ``uv sync --extra openai``."""

from __future__ import annotations

import time
from typing import Any

from logcat_triage.llm.base import Completion, LLMError


class OpenAIProvider:
    """Chat Completions via the official SDK, imported lazily."""

    name = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None,
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        max_tokens: int = 900,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set")
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise LLMError("Install the extra: uv sync --extra openai") from exc
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = openai.OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def complete(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None
    ) -> Completion:
        kwargs: dict[str, Any] = {}
        if json_schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "report", "schema": json_schema},
            }
        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **kwargs,
            )
        except Exception as exc:  # the SDK raises many error types
            raise LLMError(f"OpenAI call failed: {exc}") from exc
        usage = response.usage
        return Completion(
            text=(response.choices[0].message.content or "").strip(),
            latency_s=time.perf_counter() - started,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )
