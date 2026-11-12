from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from logcat_triage.cluster import build_clusters, cluster_events, cluster_vectors, default_threshold
from logcat_triage.embeddings import TfidfEmbedder
from logcat_triage.evaluate import evaluate, purity_score, score_clustering, sweep
from logcat_triage.extract import extract_events
from logcat_triage.models import FailureEvent, FailureKind
from logcat_triage.normalize import normalize_events
from logcat_triage.parser import parse_lines
from logcat_triage.synth.families import FAMILIES
from logcat_triage.synth.generator import (
    DEFAULT_WEIGHTS,
    generate_dataset,
    generate_log,
    labels_path_for,
    load_labels,
)


def test_generator_is_deterministic() -> None:
    a = generate_log(7, failures=15)
    b = generate_log(7, failures=15)
    c = generate_log(8, failures=15)
    assert a.lines == b.lines
    assert a.lines != c.lines


def test_every_family_appears_and_labels_point_at_failure_lines() -> None:
    log = generate_log(3, failures=len(FAMILIES))
    assert {label.family for label in log.labels} == set(FAMILIES) == set(DEFAULT_WEIGHTS)
    markers = ("FATAL EXCEPTION", "ANR in", "Fatal signal", "Exception:")
    for label in log.labels:
        assert any(marker in log.lines[label.line - 1] for marker in markers)


def test_extractor_recovers_every_labeled_failure() -> None:
    log = generate_log(11, failures=60)
    events = extract_events(list(parse_lines(log.lines)), "s")
    assert sorted(e.start_line for e in events) == sorted(lab.line for lab in log.labels)


def test_write_and_load_labels(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path, [5], failures=12)
    labels_path = labels_path_for(paths[0])
    assert labels_path.name == "device-005.labels.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    assert payload["seed"] == 5
    assert len(load_labels(labels_path)) == 12


def test_cluster_vectors_edge_cases() -> None:
    assert cluster_vectors(np.zeros((0, 3)), 0.5).tolist() == []
    assert cluster_vectors(np.ones((1, 3)), 0.5).tolist() == [0]
    vectors = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    labels = cluster_vectors(vectors, 0.1)
    assert labels[0] == labels[1] != labels[2]


def _event(i: int, kind: FailureKind, pkg: str, ts: str) -> FailureEvent:
    return FailureEvent(id=f"f:{i}", kind=kind, source="f", start_line=i, package=pkg, timestamp=ts)


def test_kinds_are_never_merged_and_clusters_are_ordered() -> None:
    events = [
        _event(1, FailureKind.CRASH, "org.example.a", "01-01 00:00:01.000"),
        _event(2, FailureKind.ANR, "org.example.a", "01-01 00:00:02.000"),
        _event(3, FailureKind.CRASH, "org.example.b", "01-01 00:00:03.000"),
    ]
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    labels = cluster_events(events, vectors, 0.5)
    assert labels[0] == labels[2] != labels[1]
    clusters = build_clusters(events, labels, vectors)
    assert [c.size for c in clusters] == [2, 1]
    assert clusters[0].id == 1
    assert clusters[0].packages == ["org.example.a", "org.example.b"]
    assert clusters[0].first_seen == "01-01 00:00:01.000"
    assert clusters[0].last_seen == "01-01 00:00:03.000"


def test_default_threshold() -> None:
    assert default_threshold("tfidf") == pytest.approx(0.45)
    assert default_threshold("ollama:nomic-embed-text") == pytest.approx(0.02)


def test_tfidf_embedder_is_normalized() -> None:
    vectors = TfidfEmbedder().embed(["crash a.B.c", "crash a.B.d", "anr x"])
    assert vectors.shape[0] == 3
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    assert TfidfEmbedder().embed([]).shape == (0, 0)


def test_purity_and_scores() -> None:
    truth = ["a", "a", "b", "b"]
    assert purity_score(truth, [0, 0, 1, 1]) == 1.0
    assert purity_score(truth, [0, 0, 0, 0]) == 0.5
    assert purity_score([], []) == 0.0
    perfect = score_clustering(truth, [5, 5, 9, 9])
    assert (perfect.ari, perfect.nmi, perfect.clusters_found, perfect.true_families) == (
        1.0,
        1.0,
        2,
        2,
    )


def test_evaluate_and_sweep_on_generated_data(tmp_path: Path) -> None:
    paths = generate_dataset(tmp_path, [21, 22], failures=25)
    result = evaluate(paths, TfidfEmbedder(), 0.45)
    assert result.extraction.recall == 1.0
    assert result.extraction.precision == 1.0
    assert result.clustering.true_families == len(FAMILIES)
    assert result.clustering.ari > 0.8
    assert sum(sum(row.values()) for row in result.confusion.values()) == 50
    rows = sweep(paths, TfidfEmbedder(), [0.01, 0.45])
    assert rows[0].clustering.clusters_found >= rows[1].clustering.clusters_found


def test_normalized_signatures_of_one_family_are_close() -> None:
    log = generate_log(4, failures=30)
    events = normalize_events(extract_events(list(parse_lines(log.lines)), "s"))
    truth = {lab.line: lab.family for lab in log.labels}
    npe = [e.signature for e in events if truth[e.start_line] == "npe_note_editor"]
    assert len(npe) >= 2
    first_lines = {s.splitlines()[3] for s in npe}  # normalized message line
    assert len(first_lines) == 1
