"""Provider protocol and shared result types."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class LLMError(RuntimeError):
    """Raised when a provider cannot produce a completion (after retries)."""


@dataclass(frozen=True)
class Completion:
    """Text returned by a provider plus the numbers we benchmark."""

    text: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMProvider(Protocol):
    """Anything that turns a system + user prompt into text."""

    name: str
    model: str

    def complete(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None
    ) -> Completion:
        """Return one completion. ``json_schema`` asks for structured JSON output."""
        ...


def strip_reasoning(text: str) -> str:
    """Drop ``<think>...</think>`` blocks that reasoning models may emit."""
    return _THINK.sub("", text).strip()
