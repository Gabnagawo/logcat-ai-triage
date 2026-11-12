"""Runtime configuration read from environment variables (and an optional ``.env``)."""

from __future__ import annotations

from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["ollama", "anthropic", "openai"]

DEFAULT_MODELS: dict[str, str | None] = {
    "ollama": "qwen3:8b",
    "anthropic": "claude-sonnet-5-5",
    "openai": None,  # deliberately unset: pass LLM_MODEL
}


class Settings(BaseSettings):
    """All knobs that affect LLM and embedding calls."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: ProviderName = "ollama"
    llm_model: str | None = None
    ollama_host: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    llm_timeout: float = 120.0
    llm_max_retries: int = 2
    llm_max_tokens: int = 900
    llm_temperature: float = 0.2
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    def resolved_model(self) -> str | None:
        """The model to use: ``LLM_MODEL`` if set, else the provider default."""
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]
