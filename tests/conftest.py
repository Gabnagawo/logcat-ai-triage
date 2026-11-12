from __future__ import annotations

import json
from pathlib import Path

import pytest

from logcat_triage.llm.fake import FakeProvider
from logcat_triage.parser import parse_lines

FIXTURES = Path(__file__).parent / "fixtures"

GOOD_DRAFT = {
    "title": "Notes editor: NullPointerException when opening a deleted note",
    "steps_to_reproduce": ["Open the notes list", "Open a note that was deleted remotely"],
    "severity": "critical",
    "severity_rationale": "Crash in a core flow, seen 2 times.",
    "stack_summary": "java.lang.NullPointerException\n* NoteEditorFragment.bindNote",
    "root_cause_hypothesis": "The note lookup returns null and bindNote does not handle it.",
}


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own LLM settings out of unit tests."""
    for name in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "OLLAMA_HOST",
        "EMBED_MODEL",
        "LLM_TIMEOUT",
        "LLM_MAX_RETRIES",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def mixed_log_text() -> str:
    return (FIXTURES / "mixed.logcat").read_text(encoding="utf-8")


@pytest.fixture
def mixed_lines(mixed_log_text: str):
    return list(parse_lines(mixed_log_text.splitlines()))


@pytest.fixture
def good_draft_json() -> str:
    return json.dumps(GOOD_DRAFT)


@pytest.fixture
def fake_provider(good_draft_json: str) -> FakeProvider:
    return FakeProvider(responder=lambda _system, _user: good_draft_json)
