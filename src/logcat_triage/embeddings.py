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
    """Embeddings from Ollama's ``/api/embed`` endpoint, with timeouts and retries.

    ``batch_size`` defaults to 1. On the shared Ollama 0.40 server used for the results,
    32-input batches sent while another client was generating returned some all-zero rows and
    some rows that differed between two runs by a cosine distance of up to 0.78, while
    single-input requests were repeatable (docs/results/embedding-stability-*.{txt,json}).
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        host: str = "http://localhost:11434",
        *,
        timeout: float = 120.0,
        max_retries: int = 2,
        batch_size: int = 1,
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

    def _request(self, batch: list[str]) -> np.ndarray:
        """One ``/api/embed`` call with transport retries; validates shape and finiteness."""
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [self._prefix(t) for t in batch],
            "truncate": True,
        }

        def _call() -> np.ndarray:
            response = self._client.post(
                f"{self.host}/api/embed", json=payload, timeout=self.timeout
            )
            check_response(response, "Ollama")
            array = np.asarray(response.json()["embeddings"], dtype=np.float64)
            if array.ndim != 2 or array.shape[0] != len(batch) or not np.isfinite(array).all():
                raise RetryableHTTPError("Ollama returned malformed or non-finite embeddings")
            return array

        try:
            return call_with_retries(
                _call,
                retries=self.max_retries,
                retry_on=(httpx.TransportError, RetryableHTTPError),
                sleep=self._sleep,
            )
        except (httpx.TransportError, RetryableHTTPError, LLMError, KeyError) as exc:
            raise EmbeddingError(f"Ollama embeddings unavailable: {exc}") from exc

    def _embed_batch(self, batch: list[str]) -> np.ndarray:
        # Observed on a busy shared Ollama (0.40.x): HTTP 200 where a random subset of rows
        # are all zeros. Zero vectors break cosine distance, so re-request just those rows.
        vectors = self._request(batch)
        for attempt in range(1, self.max_retries + 2):
            bad = np.flatnonzero(np.linalg.norm(vectors, axis=1) == 0)
            if bad.size == 0:
                return vectors
            log.warning("re-embedding %d zero vector(s), attempt %d", bad.size, attempt)
            self._sleep(float(attempt))
            vectors[bad] = self._request([batch[i] for i in bad])
        bad = np.flatnonzero(np.linalg.norm(vectors, axis=1) == 0)
        if bad.size:
            raise EmbeddingError(f"Ollama returned zero vectors for {bad.size} input(s)")
        return vectors

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0))
        parts = [
            self._embed_batch(texts[start : start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]
        matrix: np.ndarray = normalize(np.vstack(parts))
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
