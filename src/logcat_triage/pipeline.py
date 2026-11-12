"""End-to-end triage: parse -> extract -> normalize -> embed -> cluster -> report."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from logcat_triage.cluster import build_clusters, cluster_events, default_threshold
from logcat_triage.embeddings import Embedder, embed_with_fallback
from logcat_triage.extract import extract_events
from logcat_triage.llm.base import LLMProvider
from logcat_triage.models import FailureEvent, LogLine, TriageResult, TriageSummary
from logcat_triage.normalize import normalize_events
from logcat_triage.parser import read_log
from logcat_triage.reports import latency_stats, write_reports

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TriageOptions:
    """Knobs for one triage run."""

    threshold: float | None = None
    allow_fallback: bool = True
    max_reports: int = 10


def triage_sources(
    sources: Sequence[tuple[str, list[LogLine]]],
    embedder: Embedder,
    provider: LLMProvider | None,
    options: TriageOptions | None = None,
) -> TriageResult:
    """Triage already-parsed logs. ``sources`` is a list of ``(name, lines)`` pairs."""
    options = options or TriageOptions()
    events: list[FailureEvent] = []
    for name, lines in sources:
        events.extend(extract_events(lines, name))
    normalize_events(events)
    log.info("extracted %d events from %d source(s)", len(events), len(sources))

    if events:
        vectors, used, fallback = embed_with_fallback(
            [e.signature for e in events], embedder, allow_fallback=options.allow_fallback
        )
    else:
        used, fallback = embedder.name, None
    threshold = options.threshold if options.threshold is not None else default_threshold(used)
    clusters = (
        build_clusters(events, cluster_events(events, vectors, threshold), vectors)
        if events
        else []
    )
    reports = write_reports(clusters, events, provider, max_reports=options.max_reports)
    summary = TriageSummary(
        sources=[name for name, _ in sources],
        total_lines=sum(len(lines) for _, lines in sources),
        events_by_kind=dict(sorted(Counter(e.kind.value for e in events).items())),
        embedder=used,
        embedder_fallback=fallback,
        distance_threshold=threshold,
        clusters=len(clusters),
        reports_generated=len(reports),
        llm=f"{provider.name}:{provider.model}" if provider else None,
        llm_latency=latency_stats(reports) if provider else None,
    )
    return TriageResult(summary=summary, clusters=clusters, reports=reports, events=events)


def triage_files(
    paths: Sequence[Path],
    embedder: Embedder,
    provider: LLMProvider | None,
    options: TriageOptions | None = None,
) -> TriageResult:
    """Triage logcat files from disk."""
    return triage_sources([(p.name, read_log(p)) for p in paths], embedder, provider, options)
