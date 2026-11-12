"""Deterministic provider for tests and offline demos."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from logcat_triage.llm.base import Completion, LLMError

Responder = Callable[[str, str], str]


class FakeProvider:
    """Returns canned responses in order (or from a callable) and records every prompt."""

    name = "fake"

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        responder: Responder | None = None,
        model: str = "fake-model",
        fail_times: int = 0,
    ) -> None:
        self.model = model
        self._responses = list(responses or [])
        self._responder = responder
        self._fail_times = fail_times
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def complete(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None
    ) -> Completion:
        self.calls.append((system, user, json_schema))
        if self._fail_times > 0:
            self._fail_times -= 1
            raise LLMError("fake failure")
        if self._responder is not None:
            text = self._responder(system, user)
        elif self._responses:
            text = self._responses.pop(0)
        else:
            raise LLMError("FakeProvider has no more responses")
        return Completion(
            text=text,
            latency_s=0.0,
            prompt_tokens=len(user) // 4,
            completion_tokens=len(text) // 4,
        )
