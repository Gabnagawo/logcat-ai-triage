# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-12

### Added
- `logcat-triage` CLI with `triage`, `collect`, `synth`, `evaluate`, `sweep` and `version`.
- Logcat parser for `threadtime` (with or without year) and `brief` formats.
- Extraction of Java crashes (`FATAL EXCEPTION`), ANRs (`ANR in`), native crashes
  (`Fatal signal` lines merged with tombstone backtraces) and logged exceptions with
  `Caused by` chains, tolerant of interleaved output from other processes.
- Normalization of stack traces: line numbers, PIDs/TIDs, timestamps, addresses, install-path
  hashes, build ids and numeric message parameters are removed.
- Clustering with scikit-learn agglomerative clustering (cosine, average linkage, distance
  threshold) per failure kind, on Ollama `nomic-embed-text` embeddings or TF-IDF (automatic
  fallback when Ollama is unavailable, or `--embedder tfidf`).
- Jira-ready bug reports per cluster written by an LLM through a JSON schema (title, inferred
  steps, severity with rationale, stack summary, root-cause hypothesis), with counts and
  packages computed from the data and a rule-based fallback report.
- Provider abstraction: Ollama (default, `qwen3:8b`), Anthropic and OpenAI as lazy optional
  extras, and a fake provider for tests. Timeouts and bounded retries on every call.
- Markdown, single-file HTML and JSON outputs.
- Allowlisted, read-only ADB wrapper (`adb devices`, `adb logcat -d`) using argument lists with
  `shell=False`, plus a fake transport for tests and demos.
- Seeded synthetic logcat generator: 10 bugs from 7 failure families in 3 fictional apps, with
  ground-truth labels; three generated samples committed under `samples/`.
- Ground-truth evaluation (ARI, NMI, purity, homogeneity, completeness, cluster count,
  extraction precision/recall) and threshold sweeps; measured results in `docs/results/`.
- CI (ruff, mypy, pytest with coverage gate, offline demo, hadolint), Dockerfile and Makefile.

### Fixed (before release, from the pre-publication review)
- The adb serial check needs an alphanumeric first character and a full match, so a
  serial such as `-e` or one ending in a newline is rejected.
- `validate_args(["-s", serial])` raises `AdbError` instead of `IndexError`.
- Raw test and lint output saved in `docs/results/pytest.txt` and `lint.txt`.

[0.1.0]: https://github.com/Gabnagawo/logcat-ai-triage/releases/tag/v0.1.0
