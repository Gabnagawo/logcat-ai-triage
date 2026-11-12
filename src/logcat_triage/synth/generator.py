"""Seeded generator for realistic synthetic logcat files with ground-truth labels."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from logcat_triage.synth.context import APPS, MAIN_ACTIVITY, Label, SynthContext
from logcat_triage.synth.families import FAMILIES

# Relative frequency of each family: real crash dashboards are long-tailed.
DEFAULT_WEIGHTS: dict[str, float] = {
    "npe_note_editor": 5,
    "ise_player_fragment": 4,
    "network_main_weather": 3,
    "sqlite_notes_logged": 3,
    "anr_player_main": 2,
    "oom_weather_bitmap": 2,
    "native_sigsegv_player": 1,
    "npe_notes_list": 3,
    "ise_player_released": 2,
    "anr_notes_service": 1,
}

LABELS_SUFFIX = ".labels.json"


@dataclass
class SynthLog:
    """One generated logcat file plus its ground truth."""

    name: str
    seed: int
    lines: list[str]
    labels: list[Label]

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        """Write ``<name>.logcat`` and ``<name>.labels.json`` into ``out_dir``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"{self.name}.logcat"
        labels_path = out_dir / f"{self.name}{LABELS_SUFFIX}"
        log_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8", newline="\n")
        payload = {
            "generator": "logcat_triage.synth",
            "seed": self.seed,
            "log": log_path.name,
            "families": sorted({label.family for label in self.labels}),
            "failures": [asdict(label) for label in self.labels],
        }
        labels_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        return log_path, labels_path


def generate_log(
    seed: int,
    failures: int = 40,
    noise_per_step: tuple[int, int] = (3, 12),
    name: str | None = None,
) -> SynthLog:
    """Generate one file with ``failures`` injected failures drawn from all families."""
    rng = random.Random(seed)
    start = datetime(2026, 1, 1) + timedelta(
        days=rng.randint(0, 300), seconds=rng.randint(0, 86000)
    )
    ctx = SynthContext(rng=rng, clock=start)
    ctx.lines.append("--------- beginning of main")
    ctx.noise(rng.randint(10, 30))

    names = list(DEFAULT_WEIGHTS)
    weights = [DEFAULT_WEIGHTS[n] for n in names]
    # Guarantee every family appears at least once, then sample the rest by weight.
    schedule = names + rng.choices(names, weights=weights, k=max(0, failures - len(names)))
    rng.shuffle(schedule)
    for family in schedule[:failures]:
        if rng.random() < 0.4:
            app = rng.choice(APPS)
            ctx.start_activity(app, MAIN_ACTIVITY[app])
        ctx.noise(rng.randint(*noise_per_step))
        FAMILIES[family](ctx)
        ctx.tick(200, 5000)
        ctx.noise(rng.randint(*noise_per_step))
    return SynthLog(name=name or f"synthetic-{seed}", seed=seed, lines=ctx.lines, labels=ctx.labels)


def generate_dataset(out_dir: Path, seeds: list[int], failures: int = 40) -> list[Path]:
    """Generate one file per seed; returns the written ``.logcat`` paths."""
    paths = []
    for seed in seeds:
        log_path, _ = generate_log(seed, failures=failures, name=f"device-{seed:03d}").write(
            out_dir
        )
        paths.append(log_path)
    return paths


def load_labels(path: Path) -> list[Label]:
    """Load a ``*.labels.json`` file written by :meth:`SynthLog.write`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Label(**item) for item in data["failures"]]


def labels_path_for(log_path: Path) -> Path:
    """``samples/device-001.logcat`` -> ``samples/device-001.labels.json``."""
    return log_path.with_name(log_path.stem + LABELS_SUFFIX)
