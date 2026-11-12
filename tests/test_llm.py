from __future__ import annotations

import json
import sys
import types
from typing import Any

import httpx
import numpy as np
import pytest

from logcat_triage.config import Settings
from logcat_triage.embeddings import (
    EmbeddingError,
    OllamaEmbedder,
    TfidfEmbedder,
    build_embedder,
    embed_with_fallback,
)
from logcat_triage.llm import LLMError, get_provider
from logcat_triage.llm.base import strip_reasoning
from logcat_triage.llm.fake import FakeProvider
from logcat_triage.llm.ollama import OllamaProvider
from logcat_triage.llm.retry import call_with_retries


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _no_sleep(_: float) -> None:
    return None


def test_ollama_chat_success_sends_bounded_request() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {"content": '<think>hmm</think>{"ok": true}'},
                "prompt_eval_count": 120,
                "eval_count": 40,
            },
        )

    provider = OllamaProvider("qwen3:8b", client=_client(handler), max_tokens=321)
    completion = provider.complete("sys", "user", json_schema={"type": "object"})
    assert completion.text == '{"ok": true}'
    assert (completion.prompt_tokens, completion.completion_tokens) == (120, 40)
    assert seen["think"] is False
    assert seen["stream"] is False
    assert seen["options"]["num_predict"] == 321
    assert seen["format"] == {"type": "object"}


def test_ollama_retries_server_errors_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"message": {"content": "done"}})

    provider = OllamaProvider("m", client=_client(handler), max_retries=2, sleep=_no_sleep)
    assert provider.complete("s", "u").text == "done"
    assert calls["n"] == 3


def test_ollama_gives_up_after_retries() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    provider = OllamaProvider("m", client=_client(handler), max_retries=1, sleep=_no_sleep)
    with pytest.raises(LLMError, match="after 2 attempts"):
        provider.complete("s", "u")


def test_ollama_client_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="model not found")

    provider = OllamaProvider("m", client=_client(handler), sleep=_no_sleep)
    with pytest.raises(LLMError, match="404"):
        provider.complete("s", "u")
    assert calls["n"] == 1


def test_call_with_retries_reraises() -> None:
    sleeps: list[float] = []

    def boom() -> None:
        raise ValueError("x")

    with pytest.raises(ValueError, match="x"):
        call_with_retries(
            boom, retries=2, retry_on=(ValueError,), backoff_s=0.5, sleep=sleeps.append
        )
    assert sleeps == [0.5, 1.0]


def test_strip_reasoning() -> None:
    assert strip_reasoning("<think>\na\n</think>\n answer ") == "answer"


def test_fake_provider_modes() -> None:
    fake = FakeProvider(["one"], fail_times=1)
    with pytest.raises(LLMError):
        fake.complete("s", "u")
    assert fake.complete("s", "u").text == "one"
    with pytest.raises(LLMError, match="no more"):
        fake.complete("s", "u")
    assert len(fake.calls) == 3


def test_factory_defaults_to_ollama_qwen() -> None:
    provider = get_provider(Settings(_env_file=None))
    assert (provider.name, provider.model) == ("ollama", "qwen3:8b")


def test_factory_openai_requires_model() -> None:
    with pytest.raises(LLMError, match="LLM_MODEL"):
        get_provider(Settings(_env_file=None, llm_provider="openai"))


def test_factory_hosted_providers_require_keys() -> None:
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        get_provider(Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key=None))
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        get_provider(
            Settings(_env_file=None, llm_provider="openai", llm_model="m", openai_api_key=None)
        )


def test_anthropic_provider_with_stub_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Messages:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            block = types.SimpleNamespace(text='{"a": 1}')
            usage = types.SimpleNamespace(input_tokens=10, output_tokens=5)
            return types.SimpleNamespace(content=[block], usage=usage)

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            captured["init"] = kwargs
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Anthropic))
    provider = get_provider(
        Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="k")
    )
    completion = provider.complete("sys", "user", json_schema={"type": "object"})
    assert provider.model == "claude-sonnet-5-5"
    assert completion.text == '{"a": 1}'
    assert completion.completion_tokens == 5
    assert "Return only JSON" in captured["messages"][0]["content"]
    assert captured["init"]["max_retries"] == 2


def test_anthropic_errors_become_llm_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class Messages:
        def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("rate limited")

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Anthropic))
    provider = get_provider(
        Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="k")
    )
    with pytest.raises(LLMError, match="rate limited"):
        provider.complete("s", "u")


def test_openai_provider_with_stub_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class Completions:
        def create(self, **kwargs: Any) -> Any:
            captured.update(kwargs)
            message = types.SimpleNamespace(content="hello")
            usage = types.SimpleNamespace(prompt_tokens=3, completion_tokens=1)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)], usage=usage
            )

    class OpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = types.SimpleNamespace(completions=Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=OpenAI))
    provider = get_provider(
        Settings(_env_file=None, llm_provider="openai", llm_model="gpt-x", openai_api_key="k")
    )
    completion = provider.complete("s", "u", json_schema={"type": "object"})
    assert completion.text == "hello"
    assert captured["response_format"]["type"] == "json_schema"

    class Failing:
        def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("nope")

    provider._client.chat.completions = Failing()  # type: ignore[attr-defined]
    with pytest.raises(LLMError, match="nope"):
        provider.complete("s", "u")


def test_ollama_embedder_batches_prefixes_and_normalizes() -> None:
    batches: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        batches.append(body["input"])
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0] for _ in body["input"]]})

    embedder = OllamaEmbedder(client=_client(handler), batch_size=2)
    vectors = embedder.embed(["a", "b", "c"])
    assert vectors.shape == (3, 2)
    assert np.allclose(vectors[0], [0.6, 0.8])
    assert [len(b) for b in batches] == [2, 1]
    assert batches[0][0] == "clustering: a"
    assert embedder.embed([]).shape == (0, 0)


def test_ollama_embedder_failure_and_fallback() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    embedder = OllamaEmbedder(client=_client(handler), max_retries=0, sleep=_no_sleep)
    with pytest.raises(EmbeddingError):
        embedder.embed(["x"])
    vectors, used, reason = embed_with_fallback(["x a", "y b"], embedder)
    assert used == "tfidf"
    assert reason
    assert "unavailable" in reason
    assert vectors.shape[0] == 2
    with pytest.raises(EmbeddingError):
        embed_with_fallback(["x"], embedder, allow_fallback=False)


def test_ollama_embedder_retries_zero_vectors_then_fails() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        body = json.loads(request.content)
        if calls["n"] == 1:
            return httpx.Response(200, json={"embeddings": [[0.0, 0.0] for _ in body["input"]]})
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0] for _ in body["input"]]})

    embedder = OllamaEmbedder(client=_client(handler), max_retries=1, sleep=_no_sleep)
    assert embedder.embed(["a"]).tolist() == [[1.0, 0.0]]
    assert calls["n"] == 2

    def always_zero(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[0.0, 0.0]]})

    broken = OllamaEmbedder(client=_client(always_zero), max_retries=1, sleep=_no_sleep)
    with pytest.raises(EmbeddingError, match="zero"):
        broken.embed(["a"])


def test_ollama_embedder_re_requests_only_zero_rows() -> None:
    inputs: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        inputs.append(body["input"])
        if len(inputs) == 1:
            return httpx.Response(200, json={"embeddings": [[0.0, 0.0], [1.0, 0.0]]})
        return httpx.Response(200, json={"embeddings": [[0.0, 2.0]]})

    embedder = OllamaEmbedder(client=_client(handler), batch_size=2, sleep=_no_sleep)
    assert embedder.embed(["a", "b"]).tolist() == [[0.0, 1.0], [1.0, 0.0]]
    assert inputs[1] == ["clustering: a"]


def test_ollama_embedder_rejects_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})  # one row for two inputs

    embedder = OllamaEmbedder(client=_client(handler), batch_size=2, max_retries=0, sleep=_no_sleep)
    with pytest.raises(EmbeddingError, match="malformed"):
        embedder.embed(["a", "b"])


def test_build_embedder() -> None:
    settings = Settings(_env_file=None, embed_model="nomic-embed-text")
    assert isinstance(build_embedder("tfidf", settings), TfidfEmbedder)
    ollama = build_embedder("ollama", settings)
    assert ollama.name == "ollama:nomic-embed-text"
