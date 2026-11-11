"""Runtime settings, read from the environment and the repo-root `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/rag_arena/config.py -> repo root. Only used for defaults; Docker images set
# RAG_ARENA_DATA_DIR and RAG_ARENA_RESULTS_DIR explicitly.
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_ARENA_",
        env_file=(REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql://rag:rag@localhost:5433/rag_arena",
        validation_alias=AliasChoices("RAG_ARENA_DATABASE_URL", "DATABASE_URL"),
    )
    data_dir: Path = REPO_ROOT / "data"
    results_dir: Path = REPO_ROOT / "results"
    model_cache_dir: Path | None = None

    gemini_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )

    corpus_ref: str = "0.162.4"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"
    generator_model: str = "gemini-3.5-flash-lite"
    judge_model: str = "claude-opus-5-5"
    eval_author_model: str = "claude-opus-5-5"

    # Client-side limits. The Gemini free tier allows 15 requests a minute per model; stay
    # just under it.
    gemini_rpm: int = 14
    gemini_concurrency: int = 4

    cors_origins: list[str] = ["http://localhost:3300"]
    ask_rate_limit_per_minute: int = 20

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def eval_path(self) -> Path:
        return self.data_dir / "eval" / "qa.jsonl"

    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key and self.gemini_api_key.get_secret_value())

    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key and self.anthropic_api_key.get_secret_value())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
