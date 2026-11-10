from __future__ import annotations

from pathlib import Path

import pytest

from rag_arena.corpus.loader import Document, load_corpus

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_root() -> Path:
    return FIXTURES / "fastapi_mini"


@pytest.fixture
def mini_docs(mini_root: Path) -> list[Document]:
    return load_corpus(mini_root)
