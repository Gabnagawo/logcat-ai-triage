"""Turn each cluster into a Jira-ready bug report.

The LLM writes only the judgment fields (title, steps, severity, summary, hypothesis).
Occurrence counts, packages and timestamps are computed from the data, so they cannot be
hallucinated. If the LLM is unavailable or returns invalid JSON, a deterministic heuristic
report is produced instead and marked as such.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence

from pydantic import ValidationError

from logcat_triage.extract import guess_package
from logcat_triage.llm.base import LLMError, LLMProvider
from logcat_triage.models import BugReport, Cluster, FailureEvent, FailureKind, ReportDraft
from logcat_triage.normalize import normalize_frame, normalize_message

log = logging.getLogger(__name__)

MAX_PROMPT_FRAMES = 15
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = (
    "You are a senior Android QA engineer. You write concise, factual Jira bug reports "
    "from logcat evidence. Never state facts the evidence does not support; label guesses "
    "as hypotheses. Reply with a single JSON object and nothing else."
)

_RULES = """Rules:
- title: at most 100 characters, "<area>: <symptom>" style, name the exception or ANR.
- steps_to_reproduce: 2 to 6 short imperative steps inferred ONLY from the log lines above
  (activities started, user actions, background work). Do not invent UI details that the
  log does not show. The reader knows these steps are inferred.
- severity: one of critical, high, medium, low.
  critical = crash, ANR or native crash in a core user flow, or data loss.
  high = crash in a secondary flow, or a frequent failure with clear user impact.
  medium = handled exception with user-visible impact (e.g. data not saved).
  low = handled exception without visible impact.
- severity_rationale: one sentence that cites the failure kind and the occurrence count.
- stack_summary: 2 to 4 lines: the exception line and the app frames that matter.
- root_cause_hypothesis: 1 to 3 sentences, phrased as a hypothesis."""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_APP_MARK = re.compile(r"^\s*(?:\d+\.)?\s*\*\s+")


def _is_app_frame(frame: str, packages: Sequence[str]) -> bool:
    return any(pkg in frame for pkg in packages)


def build_prompt(cluster: Cluster, representative: FailureEvent, sources: int) -> str:
    """Render the user prompt for one cluster."""
    frames = [normalize_frame(f) for f in representative.frames[:MAX_PROMPT_FRAMES]]
    lines = [
        "Write a Jira bug report for this failure cluster.",
        "",
        f"Failure kind: {cluster.kind.value}",
        f"Exception / signal: {cluster.exception_type or 'n/a'}",
        f"Message (normalized): {normalize_message(representative.message) or 'n/a'}",
        f"Occurrences: {cluster.size} across {sources} log file(s)",
        f"Affected packages: {', '.join(cluster.packages) or 'unknown'}",
        f"Thread: {representative.thread or 'unknown'}",
    ]
    if representative.caused_by:
        lines.append(
            "Caused by: " + " | ".join(normalize_message(c) for c in representative.caused_by)
        )
    if frames:
        lines.append("Stack frames (normalized, app frames marked with *):")
        for index, frame in enumerate(frames, start=1):
            mark = "*" if _is_app_frame(frame, cluster.packages) else " "
            lines.append(f"  {index:>2}.{mark} {frame}")
    if representative.context:
        lines.append("Log lines before the failure (oldest first):")
        lines.extend(f"  {line}" for line in representative.context)
    lines += ["", _RULES]
    return "\n".join(lines)


def parse_draft(text: str) -> ReportDraft:
    """Parse model output into a :class:`ReportDraft`, tolerating code fences and extra text."""
    match = _JSON_OBJECT.search(text)
    if match is None:
        raise ValueError("no JSON object in model output")
    data = json.loads(match.group(0))
    if isinstance(data.get("title"), str):
        data["title"] = data["title"].strip()[:160]
    if isinstance(data.get("steps_to_reproduce"), list):
        data["steps_to_reproduce"] = [str(s) for s in data["steps_to_reproduce"]][:10]
    if isinstance(data.get("severity"), str):
        data["severity"] = data["severity"].strip().lower()
    if isinstance(data.get("stack_summary"), str):
        # Models sometimes copy the "*" app-frame markers from the prompt.
        data["stack_summary"] = "\n".join(
            _APP_MARK.sub("", line) for line in data["stack_summary"].splitlines()
        )
    return ReportDraft.model_validate(data)


_NATIVE_LOCATION = re.compile(r"([\w.+-]+\.so) \(([^()+]+)")
_START = re.compile(r"START u\d+ \{.*?cmp=(?P<cmp>[\w./$]+)")
_NOISE_TAGS = frozenset(
    {
        "ActivityManager",
        "Choreographer",
        "OkHttp",
        "OpenGLRenderer",
        "ProfileInstaller",
        "System",
        "System.err",
        "WM-WorkerWrapper",
        "Zygote",
        "art",
        "chatty",
    }
)


def _frame_location(frame: str) -> str | None:
    """``a.b.Cls.method(Cls.kt)`` -> ``Cls.method``; native ``lib.so (fn)`` -> ``fn (lib.so)``."""
    if native := _NATIVE_LOCATION.search(frame):
        return f"{native.group(2).strip()} ({native.group(1)})"
    parts = frame.split("(")[0].rsplit(".", 2)
    return ".".join(parts[-2:]) if len(parts) >= 2 else None


def heuristic_steps(context: Sequence[str]) -> list[str]:
    """Turn context lines into rough steps: activity starts and app-specific log lines."""
    steps: list[str] = []
    for line in context:
        level_tag, _, message = line.partition(": ")
        tag = level_tag.partition("/")[2]
        if start := _START.search(message):
            steps.append(f"Open {start['cmp']}")
        elif tag not in _NOISE_TAGS and tag != "ActivityTaskManager":
            steps.append(f"Reach the state logged as '{tag}: {message}'")
    return steps[-5:] or ["Use the app until the failure appears in logcat"]


def heuristic_draft(cluster: Cluster, representative: FailureEvent) -> ReportDraft:
    """A deterministic report used when no LLM is available."""
    short_type = (cluster.exception_type or cluster.kind.value).rsplit(".", 1)[-1]
    app_frames = [
        normalize_frame(f) for f in representative.frames if _is_app_frame(f, cluster.packages)
    ]
    area = (
        cluster.packages[0] if cluster.packages else (guess_package(representative.frames) or "app")
    )
    title = f"{area}: {short_type}"
    if app_frames and (where := _frame_location(app_frames[0])):
        title += f" in {where}"
    steps = heuristic_steps(representative.context)
    severity = {
        FailureKind.CRASH: "critical" if cluster.size >= 3 else "high",
        FailureKind.ANR: "high",
        FailureKind.NATIVE_CRASH: "critical",
        FailureKind.LOGGED_EXCEPTION: "medium",
    }[cluster.kind]
    header = f"{cluster.exception_type}: {normalize_message(representative.message)}".strip(": ")
    return ReportDraft(
        title=title[:160],
        steps_to_reproduce=steps,
        severity=severity,  # type: ignore[arg-type]
        severity_rationale=f"{cluster.kind.value} observed {cluster.size} time(s); rule-based.",
        stack_summary="\n".join([header, *app_frames[:3]]),
        root_cause_hypothesis="Not generated: the LLM was unavailable for this cluster.",
    )


def _finish(
    draft: ReportDraft,
    cluster: Cluster,
    *,
    generated_by: str,
    latency_s: float | None = None,
    completion_tokens: int | None = None,
    fallback_reason: str | None = None,
) -> BugReport:
    return BugReport(
        **draft.model_dump(),
        cluster_id=cluster.id,
        kind=cluster.kind,
        exception_type=cluster.exception_type,
        occurrence_count=cluster.size,
        affected_packages=cluster.packages,
        first_seen=cluster.first_seen,
        last_seen=cluster.last_seen,
        sample_event_ids=cluster.event_ids[:5],
        generated_by=generated_by,
        latency_s=latency_s,
        completion_tokens=completion_tokens,
        fallback_reason=fallback_reason,
    )


def write_report(
    cluster: Cluster,
    events_by_id: dict[str, FailureEvent],
    provider: LLMProvider | None,
) -> BugReport:
    """Write one report; falls back to the heuristic draft on any LLM failure."""
    representative = events_by_id[cluster.representative_id]
    if provider is None:
        return _finish(heuristic_draft(cluster, representative), cluster, generated_by="heuristic")
    sources = len({events_by_id[i].source for i in cluster.event_ids})
    prompt = build_prompt(cluster, representative, sources)
    schema = ReportDraft.model_json_schema()
    total_latency = 0.0
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            completion = provider.complete(SYSTEM_PROMPT, prompt, json_schema=schema)
        except LLMError as exc:
            last_error = str(exc)
            break  # transport retries already happened inside the provider
        total_latency += completion.latency_s
        try:
            draft = parse_draft(completion.text)
        except (ValueError, ValidationError) as exc:
            last_error = f"invalid JSON on attempt {attempt}: {str(exc)[:200]}"
            log.warning("cluster %d: %s", cluster.id, last_error)
            continue
        return _finish(
            draft,
            cluster,
            generated_by=f"{provider.name}:{provider.model}",
            latency_s=round(total_latency, 3),
            completion_tokens=completion.completion_tokens,
        )
    return _finish(
        heuristic_draft(cluster, representative),
        cluster,
        generated_by="heuristic",
        latency_s=round(total_latency, 3) if total_latency else None,
        fallback_reason=last_error,
    )


def write_reports(
    clusters: Sequence[Cluster],
    events: Sequence[FailureEvent],
    provider: LLMProvider | None,
    *,
    max_reports: int = 10,
) -> list[BugReport]:
    """Write reports for the ``max_reports`` largest clusters, one LLM call at a time."""
    events_by_id = {e.id: e for e in events}
    return [write_report(c, events_by_id, provider) for c in list(clusters)[:max_reports]]


def latency_stats(reports: Sequence[BugReport]) -> dict[str, float] | None:
    """Summarize LLM latency over reports that were actually written by an LLM."""
    values = sorted(
        r.latency_s for r in reports if r.latency_s is not None and r.generated_by != "heuristic"
    )
    if not values:
        return None
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {
        "reports": float(len(values)),
        "mean": round(sum(values) / len(values), 2),
        "median": round(median, 2),
        "min": round(values[0], 2),
        "max": round(values[-1], 2),
        "total": round(sum(values), 2),
        "fallbacks": float(sum(1 for r in reports if r.generated_by == "heuristic")),
    }
