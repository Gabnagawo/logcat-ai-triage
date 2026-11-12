"""Smoke tests against a real Ollama. Run with: uv run pytest -m live --no-cov"""

from __future__ import annotations

import pytest

from logcat_triage.config import Settings
from logcat_triage.embeddings import OllamaEmbedder
from logcat_triage.llm import get_provider
from logcat_triage.pipeline import triage_sources

pytestmark = pytest.mark.live


def test_ollama_embeddings_live() -> None:
    settings = Settings()
    vectors = OllamaEmbedder(settings.embed_model, settings.ollama_host).embed(["a", "b"])
    assert vectors.shape[0] == 2


def test_ollama_report_live(mixed_lines) -> None:
    settings = Settings()
    result = triage_sources(
        [("mixed.logcat", mixed_lines)],
        OllamaEmbedder(settings.embed_model, settings.ollama_host),
        get_provider(settings),
    )
    assert result.reports
    assert result.reports[0].generated_by.startswith("ollama:")
