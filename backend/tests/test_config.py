from pathlib import Path

import pytest

from rag_arena.config import Settings


def test_database_url_accepts_plain_env_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/x")
    monkeypatch.delenv("RAG_ARENA_DATABASE_URL", raising=False)
    assert Settings(_env_file=None).database_url == "postgresql://u:p@db:5432/x"


def test_missing_keys_are_reported_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert not s.has_gemini()
    assert not s.has_anthropic()


def test_data_paths_derive_from_data_dir(tmp_path: Path) -> None:
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.raw_dir == tmp_path / "raw"
    assert s.eval_path == tmp_path / "eval" / "qa.jsonl"
