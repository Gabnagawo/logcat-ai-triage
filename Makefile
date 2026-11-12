.PHONY: install lint format typecheck test test-live demo demo-offline eval sweep samples docker-build clean

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check . --fix
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest

test-live:
	uv run pytest -m live --no-cov

# Full pipeline on the committed samples: Ollama embeddings + qwen3:8b bug reports.
demo:
	uv run logcat-triage triage samples -o out/demo

# Same pipeline with no network: TF-IDF clustering and rule-based reports.
demo-offline:
	uv run logcat-triage triage samples --embedder tfidf --no-llm -o out/demo-offline

# Clustering quality on the committed samples for both embedders.
eval:
	uv run logcat-triage evaluate samples -e tfidf -e ollama -o docs/results/clustering-eval.json

# Threshold selection on held-out development seeds (never on samples/).
sweep:
	uv run logcat-triage synth --out-dir data/dev --seed 900 --seed 901 --seed 902
	uv run logcat-triage sweep data/dev -e tfidf -e ollama \
		--thresholds 0.005,0.01,0.015,0.02,0.025,0.03,0.05,0.1,0.2,0.3,0.4,0.45,0.5,0.6,0.7 \
		-o docs/results/threshold-sweep.json

samples:
	uv run logcat-triage synth --out-dir samples --seed 1 --seed 2 --seed 3

docker-build:
	docker build -t logcat-ai-triage:0.1.0 .

clean:
	rm -rf out data .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml
