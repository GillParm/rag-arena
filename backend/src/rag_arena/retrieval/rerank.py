"""Cross-encoder reranking with fastembed."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, texts: Sequence[str]) -> list[float]: ...


class FastReranker:
    def __init__(self, model_name: str, cache_dir: Path | None = None, batch_size: int = 16):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextCrossEncoder(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        scores = self._model.rerank(query, list(texts), batch_size=self.batch_size)
        return [float(s) for s in scores]


class OverlapReranker:
    """Scores by query-term overlap. Deterministic stand-in for tests."""

    model_name = "overlap"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        terms = set(query.lower().split())
        return [float(len(terms & set(t.lower().split()))) for t in texts]
