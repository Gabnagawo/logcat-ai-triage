"""Score extraction and clustering against the synthetic ground truth."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
)

from logcat_triage.cluster import cluster_events
from logcat_triage.embeddings import Embedder, embed_with_fallback
from logcat_triage.extract import extract_events
from logcat_triage.models import FailureEvent
from logcat_triage.normalize import normalize_events
from logcat_triage.parser import read_log
from logcat_triage.synth.generator import labels_path_for, load_labels


class ClusteringScores(BaseModel):
    """External clustering metrics plus cluster counts."""

    ari: float
    nmi: float
    purity: float
    homogeneity: float
    completeness: float
    clusters_found: int
    true_families: int


class ExtractionScores(BaseModel):
    """How many labeled failures the extractor found, and how many extra events it reported."""

    labeled: int
    extracted: int
    matched: int
    precision: float
    recall: float


class EvalResult(BaseModel):
    """One evaluation run of one embedder at one threshold."""

    embedder: str
    threshold: float
    files: list[str]
    extraction: ExtractionScores
    clustering: ClusteringScores
    embed_seconds: float
    confusion: dict[str, dict[str, int]]


def purity_score(truth: Sequence[str], predicted: Sequence[int]) -> float:
    """Fraction of items whose cluster's majority family matches their own family."""
    if not truth:
        return 0.0
    by_cluster: dict[int, Counter[str]] = {}
    for family, cluster in zip(truth, predicted, strict=True):
        by_cluster.setdefault(cluster, Counter())[family] += 1
    majority = sum(counter.most_common(1)[0][1] for counter in by_cluster.values())
    return majority / len(truth)


def score_clustering(truth: Sequence[str], predicted: Sequence[int]) -> ClusteringScores:
    """Compute ARI, NMI, purity, homogeneity, completeness and cluster counts."""
    return ClusteringScores(
        ari=round(float(adjusted_rand_score(truth, predicted)), 4),
        nmi=round(float(normalized_mutual_info_score(truth, predicted)), 4),
        purity=round(purity_score(truth, predicted), 4),
        homogeneity=round(float(homogeneity_score(truth, predicted)), 4),
        completeness=round(float(completeness_score(truth, predicted)), 4),
        clusters_found=len(set(predicted)),
        true_families=len(set(truth)),
    )


def load_labeled_events(paths: Sequence[Path]) -> tuple[list[FailureEvent], list[str], int, int]:
    """Extract events from each labeled log and join them to the ground truth by line.

    Returns ``(matched_events, families, labeled_total, extracted_total)``.
    """
    matched: list[FailureEvent] = []
    families: list[str] = []
    labeled_total = 0
    extracted_total = 0
    for path in paths:
        truth = {label.line: label.family for label in load_labels(labels_path_for(path))}
        events = normalize_events(extract_events(read_log(path), path.name))
        labeled_total += len(truth)
        extracted_total += len(events)
        for event in events:
            if event.start_line in truth:
                matched.append(event)
                families.append(truth[event.start_line])
    return matched, families, labeled_total, extracted_total


def evaluate(paths: Sequence[Path], embedder: Embedder, threshold: float) -> EvalResult:
    """Run extraction + embedding + clustering on labeled logs and score the result."""
    events, families, labeled, extracted = load_labeled_events(paths)
    started = time.perf_counter()
    vectors, used, _ = embed_with_fallback(
        [e.signature for e in events], embedder, allow_fallback=False
    )
    embed_seconds = time.perf_counter() - started
    predicted = [int(x) for x in cluster_events(events, vectors, threshold)]
    confusion: dict[str, dict[str, int]] = {}
    for family, cluster in zip(families, predicted, strict=True):
        row = confusion.setdefault(family, {})
        row[str(cluster)] = row.get(str(cluster), 0) + 1
    matched = len(events)
    return EvalResult(
        embedder=used,
        threshold=threshold,
        files=[p.name for p in paths],
        extraction=ExtractionScores(
            labeled=labeled,
            extracted=extracted,
            matched=matched,
            precision=round(matched / extracted, 4) if extracted else 0.0,
            recall=round(matched / labeled, 4) if labeled else 0.0,
        ),
        clustering=score_clustering(families, predicted),
        embed_seconds=round(embed_seconds, 3),
        confusion=dict(sorted(confusion.items())),
    )


def sweep(
    paths: Sequence[Path], embedder: Embedder, thresholds: Sequence[float]
) -> list[EvalResult]:
    """Evaluate several thresholds while embedding only once."""
    events, families, labeled, extracted = load_labeled_events(paths)
    vectors, used, _ = embed_with_fallback(
        [e.signature for e in events], embedder, allow_fallback=False
    )
    results = []
    for threshold in thresholds:
        predicted = [int(x) for x in cluster_events(events, vectors, threshold)]
        results.append(
            EvalResult(
                embedder=used,
                threshold=threshold,
                files=[p.name for p in paths],
                extraction=ExtractionScores(
                    labeled=labeled,
                    extracted=extracted,
                    matched=len(events),
                    precision=round(len(events) / extracted, 4) if extracted else 0.0,
                    recall=round(len(events) / labeled, 4) if labeled else 0.0,
                ),
                clustering=score_clustering(families, predicted),
                embed_seconds=0.0,
                confusion={},
            )
        )
    return results
