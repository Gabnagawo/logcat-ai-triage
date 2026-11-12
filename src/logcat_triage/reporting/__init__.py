"""Output writers: Markdown, single-file HTML and JSON."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from logcat_triage.models import TriageResult
from logcat_triage.reporting.html import render_html
from logcat_triage.reporting.markdown import render_markdown

FORMATS = ("md", "html", "json")


def render_json(result: TriageResult) -> str:
    """Serialize the full result (summary, clusters, reports and events)."""
    return result.model_dump_json(indent=2) + "\n"


def write_outputs(
    result: TriageResult, out_dir: Path, formats: Iterable[str] = FORMATS, stem: str = "triage"
) -> list[Path]:
    """Write the requested formats to ``out_dir/<stem>.<ext>`` and return the paths."""
    renderers = {"md": render_markdown, "html": render_html, "json": render_json}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        if fmt not in renderers:
            raise ValueError(f"unknown format '{fmt}', expected one of {', '.join(FORMATS)}")
        path = out_dir / f"{stem}.{fmt}"
        path.write_text(renderers[fmt](result), encoding="utf-8", newline="\n")
        written.append(path)
    return written


__all__ = ["FORMATS", "render_html", "render_json", "render_markdown", "write_outputs"]
