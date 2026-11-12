from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from logcat_triage.embeddings import TfidfEmbedder
from logcat_triage.llm.fake import FakeProvider
from logcat_triage.models import BugReport, FailureKind
from logcat_triage.pipeline import TriageOptions, triage_sources
from logcat_triage.reports import (
    build_prompt,
    heuristic_draft,
    heuristic_steps,
    latency_stats,
    parse_draft,
)
from tests.conftest import GOOD_DRAFT


def _triage(mixed_lines, provider, **kwargs):
    return triage_sources(
        [("mixed.logcat", mixed_lines)], TfidfEmbedder(), provider, TriageOptions(**kwargs)
    )


def test_parse_draft_tolerates_fences_and_cleans_fields() -> None:
    raw = "```json\n" + json.dumps({**GOOD_DRAFT, "severity": " High "}) + "\n```"
    draft = parse_draft(raw)
    assert draft.severity == "high"
    assert "* " not in draft.stack_summary
    assert draft.stack_summary.splitlines()[1] == "NoteEditorFragment.bindNote"


def test_parse_draft_truncates_long_title_and_steps() -> None:
    data = {**GOOD_DRAFT, "title": "x" * 300, "steps_to_reproduce": [f"s{i}" for i in range(15)]}
    draft = parse_draft(json.dumps(data))
    assert len(draft.title) == 160
    assert len(draft.steps_to_reproduce) == 10


@pytest.mark.parametrize("raw", ["no json here", '{"title": "only"}', '{"severity": "urgent"}'])
def test_parse_draft_rejects_invalid(raw: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        parse_draft(raw)


def test_llm_reports_have_deterministic_fields(mixed_lines, fake_provider) -> None:
    result = _triage(mixed_lines, fake_provider)
    assert result.summary.clusters == 5
    assert result.summary.reports_generated == 5
    report = result.reports[0]
    assert report.generated_by == "fake:fake-model"
    assert report.title == GOOD_DRAFT["title"]
    assert report.occurrence_count == result.clusters[0].size
    assert report.affected_packages == result.clusters[0].packages
    assert report.steps_label.startswith("inferred from log context")
    assert result.summary.llm == "fake:fake-model"
    assert result.summary.llm_latency is not None
    assert result.summary.llm_latency["fallbacks"] == 0


def test_prompt_contains_evidence_and_rules(mixed_lines, fake_provider) -> None:
    _triage(mixed_lines, fake_provider, max_reports=5)
    prompts = [user for _, user, _ in fake_provider.calls]
    crash_prompt = next(p for p in prompts if "NullPointerException" in p)
    assert "Occurrences: 1 across 1 log file(s)" in crash_prompt
    assert "* org.example.notes.editor.NoteEditorFragment.bindNote(NoteEditorFragment.kt)" in (
        crash_prompt
    )
    assert "openNote id=42" in crash_prompt
    assert "inferred ONLY from the log lines" in crash_prompt
    schema = fake_provider.calls[0][2]
    assert schema is not None
    assert "severity" in schema["properties"]


def test_invalid_json_is_retried_once(mixed_lines, good_draft_json) -> None:
    provider = FakeProvider(["not json", good_draft_json])
    result = _triage(mixed_lines, provider, max_reports=1)
    assert result.reports[0].generated_by == "fake:fake-model"
    assert len(provider.calls) == 2


def test_two_invalid_answers_fall_back_to_heuristic(mixed_lines) -> None:
    provider = FakeProvider(["nope", "still nope"])
    report = _triage(mixed_lines, provider, max_reports=1).reports[0]
    assert report.generated_by == "heuristic"
    assert report.fallback_reason
    assert "invalid JSON" in report.fallback_reason


def test_provider_error_falls_back_to_heuristic(mixed_lines) -> None:
    provider = FakeProvider(fail_times=10)
    result = _triage(mixed_lines, provider, max_reports=2)
    assert {r.generated_by for r in result.reports} == {"heuristic"}
    assert result.reports[0].fallback_reason == "fake failure"
    assert result.summary.llm_latency is None


def test_no_provider_and_max_reports(mixed_lines) -> None:
    result = _triage(mixed_lines, None, max_reports=2)
    assert len(result.reports) == 2
    assert result.summary.llm is None
    assert all(r.generated_by == "heuristic" for r in result.reports)


def test_heuristic_titles_and_severity(mixed_lines) -> None:
    result = _triage(mixed_lines, None)
    by_kind = {r.kind: r for r in result.reports}
    crash = by_kind[FailureKind.CRASH]
    assert crash.title == ("org.example.notes: NullPointerException in NoteEditorFragment.bindNote")
    assert crash.severity == "high"  # single occurrence
    native = next(r for r in result.reports if r.exception_type == "SIGSEGV")
    assert native.title == "org.example.player: SIGSEGV in decode_frame (libplayercore.so)"
    assert native.severity == "critical"
    assert by_kind[FailureKind.LOGGED_EXCEPTION].severity == "medium"
    assert by_kind[FailureKind.ANR].severity == "high"


def test_heuristic_steps_filter_noise() -> None:
    steps = heuristic_steps(
        [
            "I/Zygote: Process: zygote64 forked child",
            "I/ActivityTaskManager: START u0 {flg=0x1 cmp=org.example.notes/.EditorActivity} from",
            "I/ActivityTaskManager: Displayed org.example.notes/.EditorActivity: +300ms",
            "D/NoteEditor: openNote id=42",
        ]
    )
    assert steps == [
        "Open org.example.notes/.EditorActivity",
        "Reach the state logged as 'NoteEditor: openNote id=42'",
    ]
    assert heuristic_steps([]) == ["Use the app until the failure appears in logcat"]


def test_build_prompt_without_frames_or_context(mixed_lines) -> None:
    result = _triage(mixed_lines, None)
    anr_cluster = next(c for c in result.clusters if c.kind is FailureKind.ANR)
    events = {e.id: e for e in result.events}
    rep = events[anr_cluster.representative_id].model_copy(update={"context": [], "frames": []})
    prompt = build_prompt(anr_cluster, rep, sources=1)
    assert "Stack frames" not in prompt
    assert "Log lines before" not in prompt
    draft = heuristic_draft(anr_cluster, rep)
    assert draft.steps_to_reproduce == ["Use the app until the failure appears in logcat"]


def test_latency_stats() -> None:
    def report(latency: float | None, by: str) -> BugReport:
        return BugReport(
            **GOOD_DRAFT,
            cluster_id=1,
            kind=FailureKind.CRASH,
            occurrence_count=1,
            affected_packages=[],
            generated_by=by,
            latency_s=latency,
        )

    stats = latency_stats(
        [report(10.0, "ollama:q"), report(20.0, "ollama:q"), report(None, "heuristic")]
    )
    assert stats == {
        "reports": 2.0,
        "mean": 15.0,
        "median": 15.0,
        "min": 10.0,
        "max": 20.0,
        "total": 30.0,
        "fallbacks": 1.0,
    }
    assert latency_stats([report(None, "heuristic")]) is None
    odd = latency_stats([report(1.0, "x"), report(2.0, "x"), report(9.0, "x")])
    assert odd is not None
    assert odd["median"] == 2.0


def test_empty_input_produces_empty_result() -> None:
    result = triage_sources([("empty", [])], TfidfEmbedder(), None)
    assert result.clusters == []
    assert result.reports == []
    assert result.summary.embedder == "tfidf"
