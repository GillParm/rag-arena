"""Text embedders: fastembed for real runs, a hashing embedder for offline tests."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

Vector = np.ndarray[Any, np.dtype[np.float32]]

# BGE v1.5 retrieval models expect this instruction on short queries (not on passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed_passages(self, texts: Sequence[str]) -> Vector: ...

    def embed_queries(self, texts: Sequence[str]) -> Vector: ...


def _normalize(m: Vector) -> Vector:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Vector = (m / norms).astype(np.float32)
    return out


class FastEmbedder:
    """Local ONNX embeddings via fastembed (CPU, no API calls)."""

    def __init__(
        self, model_name: str, cache_dir: Path | None = None, batch_size: int = 32
    ) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(
            model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None
        )
        self.batch_size = batch_size
        self.dim = int(self._model.embedding_size)
        self.query_prefix = BGE_QUERY_PREFIX if "bge-" in model_name.lower() else ""

    def embed_passages(self, texts: Sequence[str]) -> Vector:
        vecs = list(self._model.embed(list(texts), batch_size=self.batch_size))
        return _normalize(np.asarray(vecs, dtype=np.float32))

    def embed_queries(self, texts: Sequence[str]) -> Vector:
        prefixed = [self.query_prefix + t for t in texts]
        vecs = list(self._model.embed(prefixed, batch_size=self.batch_size))
        return _normalize(np.asarray(vecs, dtype=np.float32))


_TOKEN = re.compile(r"[a-z0-9_]+")


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. Good enough for tests; never for results."""

    def __init__(self, dim: int = 384) -> None:
        self.model_name = f"hash-{dim}"
        self.dim = dim

    def _embed(self, text: str) -> Vector:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in _TOKEN.findall(text.lower()):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_passages(self, texts: Sequence[str]) -> Vector:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._embed(t) for t in texts]))

    def embed_queries(self, texts: Sequence[str]) -> Vector:
        return self.embed_passages(texts)
