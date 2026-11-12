"""Ollama chat provider (default). Talks to ``/api/chat`` over HTTP with httpx."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import httpx

from logcat_triage.llm.base import Completion, LLMError, strip_reasoning
from logcat_triage.llm.retry import call_with_retries


class RetryableHTTPError(RuntimeError):
    """A 429/5xx response worth retrying."""


def check_response(response: httpx.Response, service: str) -> None:
    """Raise :class:`RetryableHTTPError` for 429/5xx and :class:`LLMError` for other 4xx."""
    if response.status_code == 429 or response.status_code >= 500:
        raise RetryableHTTPError(f"HTTP {response.status_code}: {response.text[:200]}")
    if response.status_code >= 400:
        raise LLMError(f"{service} returned HTTP {response.status_code}: {response.text[:200]}")


class OllamaProvider:
    """Chat completions from a local Ollama server, bounded by timeout and retries."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        max_tokens: int = 900,
        temperature: float = 0.2,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client or httpx.Client()
        self._sleep = sleep

    def complete(
        self, system: str, user: str, *, json_schema: dict[str, Any] | None = None
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": 8192,
            },
        }
        if json_schema is not None:
            payload["format"] = json_schema

        def _call() -> dict[str, Any]:
            response = self._client.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
            check_response(response, "Ollama")
            data: dict[str, Any] = response.json()
            return data

        started = time.perf_counter()
        try:
            data = call_with_retries(
                _call,
                retries=self.max_retries,
                retry_on=(httpx.TransportError, RetryableHTTPError),
                sleep=self._sleep,
            )
        except (httpx.TransportError, RetryableHTTPError) as exc:
            raise LLMError(
                f"Ollama call failed after {self.max_retries + 1} attempts: {exc}"
            ) from exc
        latency = time.perf_counter() - started
        text = strip_reasoning(str(data.get("message", {}).get("content", "")))
        return Completion(
            text=text,
            latency_s=latency,
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
