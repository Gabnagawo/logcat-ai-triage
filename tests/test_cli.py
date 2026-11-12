from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from logcat_triage import __version__, cli
from logcat_triage.llm.fake import FakeProvider
from tests.conftest import FIXTURES, GOOD_DRAFT

runner = CliRunner()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    out = tmp_path / "data"
    result = runner.invoke(
        cli.app,
        ["synth", "--out-dir", str(out), "--seed", "31", "--seed", "32", "--failures", "20"],
    )
    assert result.exit_code == 0, result.output
    return out


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every Ollama HTTP call fail fast, as if the server were down."""

    def refuse(self: httpx.Client, *args: object, **kwargs: object) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx.Client, "post", refuse)
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")


def test_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.output.strip() == __version__


def test_triage_tfidf_no_llm(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        cli.app, ["triage", str(dataset), "-e", "tfidf", "--no-llm", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert "40 failures" in result.output
    assert (out / "triage.md").exists()
    assert (out / "triage.html").exists()
    data = json.loads((out / "triage.json").read_text(encoding="utf-8"))
    assert data["summary"]["embedder"] == "tfidf"
    assert data["summary"]["llm"] is None


def test_triage_with_fake_llm(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeProvider(responder=lambda _s, _u: json.dumps(GOOD_DRAFT))
    monkeypatch.setattr(cli, "get_provider", lambda _settings: fake)
    out = tmp_path / "out"
    result = runner.invoke(
        cli.app,
        ["triage", str(dataset / "device-031.logcat"), "-e", "tfidf", "--max-reports", "2",
         "-o", str(out), "-f", "md"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "LLM fake:fake-model: 2 reports" in result.output
    assert sorted(p.name for p in out.iterdir()) == ["triage.md"]


def test_triage_falls_back_when_ollama_is_down(
    dataset: Path, tmp_path: Path, offline: None
) -> None:
    out = tmp_path / "out"
    result = runner.invoke(cli.app, ["triage", str(dataset), "--max-reports", "1", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "embedder fallback" in result.output
    data = json.loads((out / "triage.json").read_text(encoding="utf-8"))
    assert data["summary"]["embedder"] == "tfidf"
    assert data["reports"][0]["generated_by"] == "heuristic"


def test_triage_no_fallback_fails(dataset: Path, tmp_path: Path, offline: None) -> None:
    result = runner.invoke(
        cli.app, ["triage", str(dataset), "--no-fallback", "--no-llm", "-o", str(tmp_path)]
    )
    assert result.exit_code == 2


def test_triage_from_fake_device(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = runner.invoke(
        cli.app,
        ["triage", "--adb", "--fake-device", str(FIXTURES / "mixed.logcat"), "-e", "tfidf",
         "--no-llm", "-o", str(out)],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert (out / "collected.logcat").exists()
    assert "5 failures" in result.output


def test_triage_rejects_bad_input(tmp_path: Path) -> None:
    assert runner.invoke(cli.app, ["triage", str(tmp_path / "missing")]).exit_code != 0
    assert runner.invoke(cli.app, ["triage", str(tmp_path)]).exit_code != 0  # empty folder
    bad_format = runner.invoke(
        cli.app, ["triage", str(FIXTURES), "--no-llm", "-e", "tfidf", "-f", "pdf"]
    )
    assert bad_format.exit_code != 0
    bad_embedder = runner.invoke(cli.app, ["triage", str(FIXTURES), "--no-llm", "-e", "bert"])
    assert bad_embedder.exit_code != 0


def test_triage_unavailable_llm_writes_heuristic_reports(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")  # no LLM_MODEL -> provider cannot be built
    result = runner.invoke(
        cli.app, ["triage", str(dataset), "-e", "tfidf", "--max-reports", "1", "-o", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "LLM unavailable" in result.output


def test_collect_with_fake_device(tmp_path: Path) -> None:
    out = tmp_path / "dump.logcat"
    result = runner.invoke(
        cli.app, ["collect", "-o", str(out), "--fake-device", str(FIXTURES / "mixed.logcat")]
    )
    assert result.exit_code == 0, result.output
    assert "from emulator-5554" in result.output
    assert out.read_text(encoding="utf-8").startswith("--------- beginning of main")


def test_collect_reports_adb_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        cli.app,
        ["collect", "-o", str(tmp_path / "x"), "--serial", "bad;serial", "--fake-device",
         str(FIXTURES / "mixed.logcat")],
    )  # fmt: skip
    assert result.exit_code == 2


def test_evaluate_and_sweep(dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "eval.json"
    result = runner.invoke(cli.app, ["evaluate", str(dataset), "-e", "tfidf", "-o", str(out)])
    assert result.exit_code == 0, result.output
    assert "ARI=" in result.output
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert rows[0]["embedder"] == "tfidf"
    assert rows[0]["threshold"] == 0.45

    sweep_out = tmp_path / "sweep.json"
    result = runner.invoke(
        cli.app,
        ["sweep", str(dataset), "-e", "tfidf", "--thresholds", "0.1,0.5", "-o", str(sweep_out)],
    )
    assert result.exit_code == 0, result.output
    assert len(json.loads(sweep_out.read_text(encoding="utf-8"))) == 2


def test_evaluate_fails_cleanly_when_ollama_is_down(dataset: Path, offline: None) -> None:
    result = runner.invoke(cli.app, ["evaluate", str(dataset), "-e", "ollama"])
    assert result.exit_code == 2
