"""``logcat-triage`` command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from logcat_triage import __version__
from logcat_triage.adb import AdbClient, AdbError, FakeTransport, SubprocessTransport, Transport
from logcat_triage.cluster import default_threshold
from logcat_triage.config import Settings
from logcat_triage.embeddings import EmbedderName, EmbeddingError, build_embedder
from logcat_triage.evaluate import evaluate, sweep
from logcat_triage.llm import LLMError, LLMProvider, get_provider
from logcat_triage.parser import parse_lines
from logcat_triage.pipeline import TriageOptions, triage_files, triage_sources
from logcat_triage.reporting import FORMATS, write_outputs
from logcat_triage.synth.generator import generate_dataset

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Triage Android logcat: extract failures, cluster them, draft Jira-ready bug reports.",
)

EmbedderOpt = Annotated[
    str, typer.Option("--embedder", "-e", help="ollama (default, falls back to tfidf) or tfidf")
]


def _embedder_name(value: str) -> EmbedderName:
    if value not in ("ollama", "tfidf"):
        raise typer.BadParameter("embedder must be 'ollama' or 'tfidf'")
    return "ollama" if value == "ollama" else "tfidf"


def _transport(fake_device: Path | None) -> Transport:
    return FakeTransport.from_file(fake_device) if fake_device else SubprocessTransport()


def _collect_paths(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(
                sorted(p for p in path.iterdir() if p.suffix in (".logcat", ".log", ".txt"))
            )
        elif path.is_file():
            files.append(path)
        else:
            raise typer.BadParameter(f"not found: {path}")
    if not files:
        raise typer.BadParameter("no logcat files found")
    return files


@app.callback()
def main(
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging")] = False,
) -> None:
    """Configure logging for every command."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def triage(
    paths: Annotated[list[Path] | None, typer.Argument(help="Logcat files or folders")] = None,
    adb: Annotated[bool, typer.Option("--adb", help="Collect with 'adb logcat -d'")] = False,
    serial: Annotated[str | None, typer.Option(help="Device serial for --adb")] = None,
    fake_device: Annotated[
        Path | None, typer.Option(help="With --adb: replay this file through the fake transport")
    ] = None,
    embedder: EmbedderOpt = "ollama",
    no_fallback: Annotated[
        bool, typer.Option("--no-fallback", help="Fail instead of falling back to TF-IDF")
    ] = False,
    threshold: Annotated[
        float | None, typer.Option(help="Cosine distance threshold (default: tuned per embedder)")
    ] = None,
    no_llm: Annotated[
        bool, typer.Option("--no-llm", help="Skip the LLM; write heuristic reports")
    ] = False,
    max_reports: Annotated[
        int, typer.Option(min=0, help="LLM reports for the N largest clusters")
    ] = 10,
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o")] = Path("out"),
    fmt: Annotated[
        str, typer.Option("--format", "-f", help="Comma-separated: md,html,json")
    ] = "md,html,json",
) -> None:
    """Extract, cluster and report failures from logcat files or a device."""
    settings = Settings()
    formats = [f.strip() for f in fmt.split(",") if f.strip()]
    if bad := [f for f in formats if f not in FORMATS]:
        raise typer.BadParameter(f"unknown format(s): {', '.join(bad)}")
    provider: LLMProvider | None = None
    if not no_llm:
        try:
            provider = get_provider(settings)
        except LLMError as exc:
            typer.secho(f"LLM unavailable ({exc}); writing heuristic reports", fg="yellow")
    embed = build_embedder(_embedder_name(embedder), settings)
    options = TriageOptions(
        threshold=threshold, allow_fallback=not no_fallback, max_reports=max_reports
    )
    try:
        if adb:
            client = AdbClient(_transport(fake_device), serial=serial)
            text = client.dump_logcat()
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "collected.logcat").write_text(text, encoding="utf-8", newline="\n")
            result = triage_sources(
                [("adb-logcat", list(parse_lines(text.splitlines())))], embed, provider, options
            )
        else:
            result = triage_files(_collect_paths(paths or []), embed, provider, options)
    except (AdbError, EmbeddingError) as exc:
        typer.secho(f"error: {exc}", fg="red", err=True)
        raise typer.Exit(2) from exc

    written = write_outputs(result, out_dir, formats)
    s = result.summary
    typer.echo(
        f"{sum(s.events_by_kind.values())} failures {s.events_by_kind} -> {s.clusters} clusters "
        f"({s.embedder}, threshold {s.distance_threshold}) -> {s.reports_generated} reports"
    )
    if s.embedder_fallback:
        typer.secho(f"embedder fallback: {s.embedder_fallback}", fg="yellow")
    if s.llm_latency:
        lat = s.llm_latency
        typer.echo(
            f"LLM {s.llm}: {int(lat['reports'])} reports, mean {lat['mean']}s, "
            f"median {lat['median']}s, max {lat['max']}s, fallbacks {int(lat['fallbacks'])}"
        )
    for report in result.reports:
        typer.echo(
            f"  BUG-{report.cluster_id} [{report.severity}] x{report.occurrence_count} {report.title}"
        )
    for path in written:
        typer.echo(f"wrote {path}")


@app.command()
def collect(
    out: Annotated[Path, typer.Option("--out", "-o", help="Where to save the dump")],
    serial: Annotated[str | None, typer.Option(help="Device serial")] = None,
    tail: Annotated[int | None, typer.Option(help="Only the last N lines")] = None,
    fake_device: Annotated[
        Path | None, typer.Option(help="Replay this file through the fake transport")
    ] = None,
) -> None:
    """Dump the device's logcat buffers (read-only) to a file."""
    try:
        client = AdbClient(_transport(fake_device), serial=serial)
        devices = client.list_devices()
        if not devices:
            raise AdbError("no device in 'device' state")
        text = client.dump_logcat(tail=tail)
    except AdbError as exc:
        typer.secho(f"error: {exc}", fg="red", err=True)
        raise typer.Exit(2) from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    typer.echo(f"wrote {len(text.splitlines())} lines from {serial or devices[0]} to {out}")


@app.command()
def synth(
    out_dir: Annotated[Path, typer.Option("--out-dir", "-o")] = Path("samples"),
    seeds: Annotated[list[int] | None, typer.Option("--seed", help="Repeatable")] = None,
    failures: Annotated[int, typer.Option(min=1, help="Failures per file")] = 40,
) -> None:
    """Generate seeded synthetic logcat files plus ground-truth labels."""
    paths = generate_dataset(out_dir, seeds or [1, 2, 3], failures=failures)
    for path in paths:
        typer.echo(f"wrote {path} (+ labels)")


def _embedders(names: list[str] | None) -> list[EmbedderName]:
    return [_embedder_name(n) for n in (names or ["tfidf", "ollama"])]


@app.command("evaluate")
def evaluate_cmd(
    paths: Annotated[list[Path], typer.Argument(help="Labeled logcat files or folders")],
    embedders: Annotated[
        list[str] | None, typer.Option("--embedder", "-e", help="Repeatable; default both")
    ] = None,
    threshold: Annotated[float | None, typer.Option(help="Override the tuned threshold")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write raw JSON here")] = None,
) -> None:
    """Score clustering against ground-truth labels (ARI, NMI, purity, cluster count)."""

    files = _collect_paths(paths)
    settings = Settings()
    results = []
    for name in _embedders(embedders):
        embed = build_embedder(name, settings)
        try:
            thr = threshold if threshold is not None else default_threshold(embed.name)
            result = evaluate(files, embed, thr)
        except EmbeddingError as exc:
            typer.secho(f"error: {exc}", fg="red", err=True)
            raise typer.Exit(2) from exc
        results.append(result)
        c, x = result.clustering, result.extraction
        typer.echo(
            f"{result.embedder:<26} thr={result.threshold:<5} ARI={c.ari:.3f} NMI={c.nmi:.3f} "
            f"purity={c.purity:.3f} clusters={c.clusters_found}/{c.true_families} "
            f"extraction P={x.precision:.3f} R={x.recall:.3f} ({x.matched}/{x.labeled}) "
            f"embed={result.embed_seconds}s"
        )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = [r.model_dump() for r in results]
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        typer.echo(f"wrote {out}")


@app.command("sweep")
def sweep_cmd(
    paths: Annotated[list[Path], typer.Argument(help="Labeled logcat files or folders")],
    thresholds: Annotated[str, typer.Option(help="Comma-separated distances")],
    embedders: Annotated[
        list[str] | None, typer.Option("--embedder", "-e", help="Repeatable; default both")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
) -> None:
    """Score several thresholds (use a development seed, not the evaluation set)."""
    files = _collect_paths(paths)
    values = [float(t) for t in thresholds.split(",") if t.strip()]
    rows = []
    for name in _embedders(embedders):
        for r in sweep(files, build_embedder(name, Settings()), values):
            c = r.clustering
            typer.echo(
                f"{r.embedder:<26} thr={r.threshold:<6} ARI={c.ari:.3f} NMI={c.nmi:.3f} "
                f"purity={c.purity:.3f} clusters={c.clusters_found}/{c.true_families}"
            )
            rows.append(r.model_dump(exclude={"confusion", "embed_seconds"}))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8", newline="\n")
        typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
