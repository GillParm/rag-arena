"""Per-chunk-set search primitives that the pipelines compose."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import numpy as np

from rag_arena.db import repo
from rag_arena.embedding import Embedder, Vector
from rag_arena.retrieval.bm25 import BM25Index

DenseFn = Callable[[Vector, int], list[tuple[str, float]]]


@dataclass
class SearchBackend:
    set_id: str
    bm25: BM25Index
    chunks: dict[str, repo.ChunkRow]
    embedder: Embedder
    dense: DenseFn

    def bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.bm25.search(query, k)

    def embed_queries(self, queries: Sequence[str]) -> Vector:
        return self.embedder.embed_queries(queries)

    def dense_search(self, vector: Vector, k: int) -> list[tuple[str, float]]:
        return self.dense(vector, k)


def pg_dense(
    conn_factory: Callable[[], AbstractContextManager[repo.Conn]], set_id: str, dim: int
) -> DenseFn:
    """Dense search against the set's pgvector partition (HNSW, cosine)."""

    def search(vector: Vector, k: int) -> list[tuple[str, float]]:
        with conn_factory() as conn:
            return repo.dense_search(conn, set_id, dim, vector, k)

    return search


def numpy_dense(ids: Sequence[str], matrix: np.ndarray[Any, Any]) -> DenseFn:
    """Exact in-memory cosine search over normalised vectors. Used by offline tests."""
    id_list = list(ids)

    def search(vector: Vector, k: int) -> list[tuple[str, float]]:
        scores = matrix @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [(id_list[i], float(scores[i])) for i in order]

    return search
