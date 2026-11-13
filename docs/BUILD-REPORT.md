# Build report: logcat-ai-triage v0.1.0

Built 2026-11-12 on Windows 11 (Python 3.12.10, uv 0.12.10, Ollama 0.40.16 shared with other
sessions). Local commits only: no remote, no push. Annotated tag `v0.1.0`.

## Folder tree

```text
logcat-ai-triage/
├── .github/workflows/ci.yml     ruff, ruff format, mypy, pytest + coverage gate, offline demo, hadolint
├── Dockerfile, .dockerignore    python:3.12-slim + uv, non-root user
├── Makefile                     install, lint, test, demo, demo-offline, eval, eval-large, sweep, stability
├── pyproject.toml, uv.lock      hatchling, extras: anthropic, openai
├── .env.example, .gitignore, .python-version, LICENSE (MIT), CHANGELOG.md, README.md
├── docs/
│   ├── REPO-META.md             GitHub description + topics
│   ├── BUILD-REPORT.md          this file
│   └── results/                 raw output of every number in the README
│       ├── clustering-eval.{json,txt}          samples/ (3 files, 120 failures)
│       ├── clustering-eval-10files.{json,txt}  seeds 100-109 (10 files, 400 failures)
│       ├── threshold-sweep.{json,txt}          development seeds 900-902
│       ├── embedding-stability-idle.json, embedding-stability-under-load.txt
│       ├── demo/                               final demo (md, html, json, stdout)
│       ├── demo-run1-embedding-fallback/       contended run, TF-IDF fallback
│       └── demo-run2-batched-embeddings-under-load/  contended run, corrupted batched embeddings
├── samples/                     device-00{1,2,3}.logcat + .labels.json (seeds 1-3)
├── scripts/embedding_stability.py
├── src/logcat_triage/
│   ├── cli.py                   typer app: triage, collect, synth, evaluate, sweep, version
│   ├── adb.py                   allowlisted adb (devices, logcat -d), shell=False, FakeTransport
│   ├── parser.py, extract.py, normalize.py
│   ├── embeddings.py            Ollama nomic-embed-text + TF-IDF fallback
│   ├── cluster.py, evaluate.py  agglomerative clustering, ARI/NMI/purity
│   ├── reports.py, pipeline.py, models.py, config.py
│   ├── llm/                     base, ollama (default), anthropic, openai (lazy extras), fake, retry, factory
│   ├── reporting/               markdown, html (single file), json
│   └── synth/                   seeded generator, 10 bugs / 7 families, noise
└── tests/                       10 test modules + fixtures/mixed.logcat
```

## Tests and quality gates

| Check | Result |
|---|---|
| `uv run ruff check .` | All checks passed |
| `uv run ruff format --check .` | 43 files already formatted |
| `uv run mypy` (strict) | no issues in 29 source files |
| `uv run pytest` | 116 passed, 2 live deselected, coverage 98.91% (gate 80%); raw output in `docs/results/pytest.txt` |
| `uv run pytest -m live --no-cov` | 2 passed against the local Ollama |

## Demo

Command: `uv run logcat-triage triage samples -o docs/results/demo` (`make demo`).

120 failures extracted from 4,445 lines (84 crashes, 15 ANRs, 15 logged exceptions, 6 native
crashes), 12 clusters with Ollama `nomic-embed-text` at threshold 0.02, 10 Jira-ready reports
written by `qwen3:8b`, 0 fallbacks, 3 min 21 s end to end.

## Measured results

Clustering against ground truth (10 bugs); thresholds tuned on seeds 900-902 only:

| Set | Embedder | ARI | NMI | Purity | Clusters / true |
|---|---|---|---|---|---|
| samples (120) | TF-IDF 0.45 | 0.994 | 0.993 | 1.000 | 11 / 10 |
| samples (120) | Ollama 0.02 | 0.966 | 0.979 | 1.000 | 12 / 10 |
| seeds 100-109 (400) | TF-IDF 0.45 | 0.996 | 0.994 | 1.000 | 11 / 10 |
| seeds 100-109 (400) | Ollama 0.02 | 0.980 | 0.983 | 1.000 | 13 / 10 |

Extraction precision and recall: 1.000 / 1.000 on both sets.

Report latency with `qwen3:8b` (10 sequential calls): final run mean 18.7 s, median 16.9 s,
min 14.1 s, max 37.5 s; earlier contended run mean 82.1 s, median 74.7 s, max 175.2 s (one
120 s timeout, then a successful retry). 10/10 schema-valid answers on the first try in both runs.

## Findings during the build

- The shared Ollama returned HTTP 200 with all-zero and inconsistent embeddings for batched
  `/api/embed` requests while another client was generating. Fixes: one input per request,
  re-request of zero rows, TF-IDF fallback. Evidence kept in `docs/results/`.
- Overriding `num_ctx` made Ollama reload `qwen3:8b` on every call while other sessions used a
  different context size; the override was removed.
- `qwen3:8b` rated all 10 clusters `high`, including the handled SQLite exception.

## Known limitations

Synthetic data only; native crashes without a tombstone and the wrapped SQLite error split into
extra clusters; global thresholds; ANR traces not collected; inferred steps can be wrong; LLM
severity is not calibrated; corrupted non-zero embeddings from a busy server are not detectable.

## Not done

- Dockerfile not built (Docker daemon off) and not linted locally (running hadolint through
  `uvx` was blocked by the sandbox); hadolint runs in CI instead.
- No real Android device: the subprocess ADB transport is tested only with a mocked
  `subprocess.run`.
- CI has not run (no remote, by design).
