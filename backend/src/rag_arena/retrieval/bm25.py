"""BM25 over chunk texts with bm25s, cached on disk per chunk set."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import bm25s
import Stemmer

_STEMMER = Stemmer.Stemmer("english")


def _prepare(text: str) -> str:
    # Split snake_case identifiers so "response_model" matches "response model" and back.
    return re.sub(r"_+", " ", text)


def tokenize(texts: Sequence[str]) -> list[list[str]]:
    """Lowercase, split on non-word characters and underscores, drop stopwords (NLTK list), stem."""
    out = bm25s.tokenize(
        [_prepare(t) for t in texts],
        stopwords="english_plus",
        stemmer=_STEMMER,
        return_ids=False,
        show_progress=False,
    )
    return [list(toks) for toks in out]


class BM25Index:
    def __init__(self, ids: list[str], retriever: Any) -> None:
        self.ids = ids
        self._retriever = retriever

    @classmethod
    def build(cls, ids: Sequence[str], texts: Sequence[str]) -> BM25Index:
        if len(ids) != len(texts):
            raise ValueError("one id per text")
        retriever = bm25s.BM25()  # Lucene variant, k1=1.5, b=0.75
        retriever.index(tokenize(texts), show_progress=False)
        return cls(list(ids), retriever)

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Top-k (chunk id, score), best first; chunks with a zero score are dropped."""
        tokens = tokenize([query])
        if not tokens[0] or not self.ids:
            return []
        k = min(k, len(self.ids))
        docs, scores = self._retriever.retrieve(tokens, k=k, show_progress=False)
        return [
            (self.ids[int(i)], float(s))
            for i, s in zip(docs[0], scores[0], strict=True)
            if float(s) > 0
        ]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._retriever.save(str(directory), show_progress=False)
        (directory / "ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        retriever = bm25s.BM25.load(str(directory), show_progress=False)
        ids = json.loads((directory / "ids.json").read_text(encoding="utf-8"))
        return cls(ids, retriever)
