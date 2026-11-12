"""Check whether Ollama returns the same embedding for the same text across runs/batch sizes.

Usage: uv run python scripts/embedding_stability.py samples -o docs/results/embedding-stability.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from logcat_triage.embeddings import OllamaEmbedder
from logcat_triage.extract import extract_events
from logcat_triage.normalize import normalize_events
from logcat_triage.parser import read_log


def _signatures(folder: Path) -> list[str]:
    events = []
    for path in sorted(folder.glob("*.logcat")):
        events += normalize_events(extract_events(read_log(path), path.name))
    return [e.signature for e in events]


def _compare(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    distance = 1.0 - (a * b).sum(axis=1)
    return {
        "max_cosine_distance": round(float(distance.max()), 4),
        "mean_cosine_distance": round(float(distance.mean()), 4),
        "rows_over_0.01": int((distance > 0.01).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    texts = _signatures(args.folder)
    runs = {
        "batch32_a": OllamaEmbedder(batch_size=32).embed(texts),
        "batch32_b": OllamaEmbedder(batch_size=32).embed(texts),
        "batch1_a": OllamaEmbedder(batch_size=1).embed(texts),
        "batch1_b": OllamaEmbedder(batch_size=1).embed(texts),
    }
    result = {
        "texts": len(texts),
        "batch32_vs_batch32": _compare(runs["batch32_a"], runs["batch32_b"]),
        "batch1_vs_batch1": _compare(runs["batch1_a"], runs["batch1_b"]),
        "batch32_vs_batch1": _compare(runs["batch32_a"], runs["batch1_a"]),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
