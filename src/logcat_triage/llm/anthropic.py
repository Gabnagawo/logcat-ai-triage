"""Anthropic provider. The SDK is an optional extra: ``uv sync --extra anthropic``."""

from __future__ import annotations

import json
import time
from typing import Any

from logcat_triage.llm.base import Completion, LLMError


class AnthropicProvider:
    """Claude via the official SDK, imported lazily so the base install stays small."""

    name = "anthropic"

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
            raise LLMError("ANTHROPIC_API_KEY is not set")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on installed extras
            raise LLMError("Install the extra: uv sync --extra anthropic") from exc
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=max_retries
        )

    def complete(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None
    ) -> Completion:
        if json_schema is not None:
            user = f"{user}\n\nReturn only JSON matching this schema:\n{json.dumps(json_schema)}"
        started = time.perf_counter()
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # the SDK raises many error types
            raise LLMError(f"Anthropic call failed: {exc}") from exc
        text = "".join(getattr(block, "text", "") for block in message.content)
        return Completion(
            text=text.strip(),
            latency_s=time.perf_counter() - started,
            prompt_tokens=message.usage.input_tokens,
            completion_tokens=message.usage.output_tokens,
        )
