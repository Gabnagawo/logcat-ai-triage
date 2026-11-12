"""Pydantic models shared by every stage of the pipeline."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class FailureKind(StrEnum):
    """The four failure shapes the extractor understands."""

    CRASH = "crash"
    ANR = "anr"
    NATIVE_CRASH = "native_crash"
    LOGGED_EXCEPTION = "logged_exception"


class LogLine(BaseModel):
    """One parsed logcat line. Unparseable lines keep ``level``/``tag`` as ``None``."""

    number: int = Field(description="1-based line number inside the source file")
    timestamp: str | None = None
    pid: int | None = None
    tid: int | None = None
    level: str | None = None
    tag: str | None = None
    message: str
    raw: str


class FailureEvent(BaseModel):
    """A single failure occurrence extracted from a log."""

    id: str
    kind: FailureKind
    source: str
    start_line: int
    timestamp: str | None = None
    pid: int | None = None
    package: str | None = None
    thread: str | None = None
    exception_type: str | None = None
    message: str = ""
    frames: list[str] = Field(default_factory=list)
    caused_by: list[str] = Field(default_factory=list)
    context: list[str] = Field(
        default_factory=list, description="Log lines that preceded the failure (for steps)"
    )
    raw: list[str] = Field(default_factory=list)
    signature: str = Field(default="", description="Normalized text used for clustering")


class Cluster(BaseModel):
    """A group of events judged to be the same underlying failure."""

    id: int
    kind: FailureKind
    size: int
    event_ids: list[str]
    representative_id: str
    exception_type: str | None = None
    packages: list[str]
    first_seen: str | None = None
    last_seen: str | None = None


Severity = Literal["critical", "high", "medium", "low"]


class ReportDraft(BaseModel):
    """The part of a bug report the LLM is asked to write (strict JSON schema)."""

    title: str = Field(max_length=160)
    steps_to_reproduce: list[str] = Field(min_length=1, max_length=10)
    severity: Severity
    severity_rationale: str
    stack_summary: str
    root_cause_hypothesis: str


class BugReport(ReportDraft):
    """A Jira-ready report: LLM draft plus fields computed deterministically."""

    cluster_id: int
    kind: FailureKind
    exception_type: str | None = None
    occurrence_count: int
    affected_packages: list[str]
    steps_label: str = "inferred from log context, not verified on a device"
    first_seen: str | None = None
    last_seen: str | None = None
    sample_event_ids: list[str] = Field(default_factory=list)
    generated_by: str = Field(description="'<provider>:<model>' or 'heuristic'")
    latency_s: float | None = None
    completion_tokens: int | None = None
    fallback_reason: str | None = None


class TriageSummary(BaseModel):
    """Run-level metadata included in every output format."""

    generated_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    sources: list[str]
    total_lines: int
    events_by_kind: dict[str, int]
    embedder: str
    embedder_fallback: str | None = None
    distance_threshold: float
    clusters: int
    reports_generated: int
    llm: str | None = None
    llm_latency: dict[str, float] | None = Field(
        default=None, description="Report-writing latency stats in seconds (LLM reports only)"
    )


class TriageResult(BaseModel):
    """Everything a triage run produced."""

    summary: TriageSummary
    clusters: list[Cluster]
    reports: list[BugReport]
    events: list[FailureEvent]
