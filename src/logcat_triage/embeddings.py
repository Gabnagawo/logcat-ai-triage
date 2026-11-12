"""Turn failure signatures into vectors: Ollama embeddings by default, TF-IDF as fallback."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Literal, Protocol

import httpx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from logcat_triage.config import Settings
from logcat_triage.llm.base import LLMError
from logcat_triage.llm.ollama import RetryableHTTPError, check_response
from logcat_triage.llm.retry import call_with_retries

log = logging.getLogger(__name__)

EmbedderName = Literal["ollama", "tfidf"]
# nomic-embed-text expects a task prefix; "clustering" is the documented one for this use.
_NOMIC_PREFIX = "clustering: "


class EmbeddingError(RuntimeError):
    """Raised when the embedding backend is unavailable."""


class Embedder(Protocol):
    """Maps texts to an L2-normalized ``(n, d)`` matrix."""

    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class TfidfEmbedder:
    """Word uni/bi-gram TF-IDF over normalized signatures. No network, fully deterministic."""

    name = "tfidf"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        vectorizer = TfidfVectorizer(
            token_pattern=r"<\w+>|[A-Za-z_$][\w$]*",
            lowercase=False,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        dense: np.ndarray = np.asarray(matrix.todense(), dtype=np.float64)
        return dense


class OllamaEmbedder:
    """Embeddings from Ollama's ``/api/embed`` endpoint, batched and retried."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        batch_size: int = 32,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.name = f"ollama:{model}"
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self._client = client or httpx.Client()
        self._sleep = sleep

    def _prefix(self, text: str) -> str:
        return f"{_NOMIC_PREFIX}{text}" if "nomic" in self.model else text

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [self._prefix(t) for t in batch],
            "truncate": True,
        }

        def _call() -> list[list[float]]:
            response = self._client.post(
                f"{self.host}/api/embed", json=payload, timeout=self.timeout
            )
            check_response(response, "Ollama")
            vectors: list[list[float]] = response.json()["embeddings"]
            return vectors

        try:
            return call_with_retries(
                _call,
                retries=self.max_retries,
                retry_on=(httpx.TransportError, RetryableHTTPError),
                sleep=self._sleep,
            )
        except (httpx.TransportError, RetryableHTTPError, LLMError, KeyError) as exc:
            raise EmbeddingError(f"Ollama embeddings unavailable: {exc}") from exc

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            rows.extend(self._embed_batch(texts[start : start + self.batch_size]))
        matrix: np.ndarray = normalize(np.asarray(rows, dtype=np.float64))
        return matrix


def build_embedder(name: EmbedderName, settings: Settings | None = None) -> Embedder:
    """Create the named embedder from settings."""
    settings = settings or Settings()
    if name == "tfidf":
        return TfidfEmbedder()
    return OllamaEmbedder(
        settings.embed_model,
        settings.ollama_host,
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
    )


def embed_with_fallback(
    texts: list[str], embedder: Embedder, *, allow_fallback: bool = True
) -> tuple[np.ndarray, str, str | None]:
    """Embed ``texts``; if Ollama is unavailable and fallback is allowed, use TF-IDF.

    Returns ``(vectors, embedder_name_used, fallback_reason)``.
    """
    try:
        return embedder.embed(texts), embedder.name, None
    except EmbeddingError as exc:
        if not allow_fallback:
            raise
        log.warning("%s; falling back to TF-IDF", exc)
        fallback = TfidfEmbedder()
        return fallback.embed(texts), fallback.name, str(exc)
