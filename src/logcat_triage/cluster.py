"""Group similar failures with agglomerative clustering on cosine distance.

The number of clusters is not known up front, so clustering uses a distance threshold
instead of ``n_clusters``. Events of different kinds (crash vs ANR vs native) are never
merged: they are clustered separately and the results concatenated.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from logcat_triage.models import Cluster, FailureEvent

# Chosen on a held-out development seed (seeds 900-902, see docs/results/threshold-sweep.json),
# never on the committed samples that the README results are computed on.
DEFAULT_THRESHOLDS: dict[str, float] = {"tfidf": 0.45, "ollama": 0.02}


def default_threshold(embedder_name: str) -> float:
    """Return the tuned threshold for ``tfidf`` or any ``ollama:<model>`` embedder."""
    key = "tfidf" if embedder_name == "tfidf" else "ollama"
    return DEFAULT_THRESHOLDS[key]


def cluster_vectors(vectors: np.ndarray, threshold: float) -> np.ndarray:
    """Return a label per row. Average linkage, cosine distance, threshold-based cut."""
    count = vectors.shape[0]
    if count == 0:
        return np.zeros(0, dtype=int)
    if count == 1:
        return np.zeros(1, dtype=int)
    model = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold,
    )
    labels: np.ndarray = model.fit_predict(vectors)
    return labels


def cluster_events(events: list[FailureEvent], vectors: np.ndarray, threshold: float) -> np.ndarray:
    """Cluster within each failure kind; returns one global label per event."""
    labels = np.full(len(events), -1, dtype=int)
    by_kind: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        by_kind[event.kind.value].append(index)
    offset = 0
    for kind in sorted(by_kind):
        indices = by_kind[kind]
        local = cluster_vectors(vectors[indices], threshold)
        labels[indices] = local + offset
        offset += int(local.max()) + 1 if len(local) else 0
    return labels


def _medoid(vectors: np.ndarray) -> int:
    """Index of the row with the highest mean cosine similarity to the others."""
    similarity = vectors @ vectors.T
    return int(np.argmax(similarity.mean(axis=1)))


def build_clusters(
    events: list[FailureEvent], labels: np.ndarray, vectors: np.ndarray
) -> list[Cluster]:
    """Materialize clusters, largest first, with ids renumbered from 1."""
    members: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        members[int(label)].append(index)
    ordered = sorted(
        members.values(),
        key=lambda idx: (-len(idx), events[idx[0]].source, events[idx[0]].start_line),
    )
    clusters: list[Cluster] = []
    for cluster_id, indices in enumerate(ordered, start=1):
        group = [events[i] for i in indices]
        representative = group[_medoid(vectors[indices])]
        stamps = sorted(e.timestamp for e in group if e.timestamp)
        clusters.append(
            Cluster(
                id=cluster_id,
                kind=representative.kind,
                size=len(group),
                event_ids=[e.id for e in group],
                representative_id=representative.id,
                exception_type=representative.exception_type,
                packages=sorted({e.package for e in group if e.package}),
                first_seen=stamps[0] if stamps else None,
                last_seen=stamps[-1] if stamps else None,
            )
        )
    return clusters
