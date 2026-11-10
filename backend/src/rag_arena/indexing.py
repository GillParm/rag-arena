"""Build a chunk set: chunk every document, embed, load into Postgres, index, build BM25."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_arena.chunking.base import ChunkerSpec, build_chunks, make_chunker
from rag_arena.corpus.loader import Document
from rag_arena.db import repo
from rag_arena.embedding import Embedder
from rag_arena.retrieval.bm25 import BM25Index


@dataclass(frozen=True)
class IndexStats:
    set_id: str
    chunks: int
    mean_chars: float
    seconds: float


def bm25_dir(cache_dir: Path, set_id: str) -> Path:
    return cache_dir / "bm25" / set_id


def _fingerprint(info: repo.ChunkSetInfo) -> dict[str, object]:
    return {"set_id": info.id, "chunk_count": info.chunk_count, "params": info.params}


def build_bm25(conn: repo.Conn, set_id: str, cache_dir: Path) -> BM25Index:
    """Build BM25 from the chunk rows in Postgres and cache it on disk."""
    info = repo.get_chunk_set(conn, set_id)
    if info is None:
        raise LookupError(f"chunk set {set_id!r} is not indexed")
    rows = repo.fetch_chunks(conn, set_id)
    index = BM25Index.build([r.id for r in rows], [r.index_text for r in rows])
    directory = bm25_dir(cache_dir, set_id)
    index.save(directory)
    (directory / "meta.json").write_text(json.dumps(_fingerprint(info)), encoding="utf-8")
    return index


def load_bm25(conn: repo.Conn, set_id: str, cache_dir: Path) -> BM25Index:
    """Load the cached BM25 index, rebuilding it if the chunk set changed since it was saved."""
    info = repo.get_chunk_set(conn, set_id)
    if info is None:
        raise LookupError(f"chunk set {set_id!r} is not indexed")
    directory = bm25_dir(cache_dir, set_id)
    meta = directory / "meta.json"
    if meta.exists() and json.loads(meta.read_text(encoding="utf-8")) == _fingerprint(info):
        return BM25Index.load(directory)
    return build_bm25(conn, set_id, cache_dir)


def build_chunk_set(
    conn: repo.Conn,
    docs: Sequence[Document],
    spec: ChunkerSpec,
    embedder: Embedder,
    cache_dir: Path,
    log: Callable[[str], None] = lambda _: None,
) -> IndexStats:
    t0 = time.perf_counter()
    chunker = make_chunker(spec)
    chunks = [c for d in docs for c in build_chunks(chunker, d)]
    log(f"{spec.set_id}: {len(chunks)} chunks, embedding with {embedder.model_name}")
    vectors = embedder.embed_passages([c.index_text for c in chunks])
    with conn.transaction():
        repo.upsert_documents(conn, docs)
        repo.create_chunk_set(
            conn,
            spec.set_id,
            chunker=spec.name,
            params=chunker.params(),
            embedding_model=embedder.model_name,
            embedding_dim=embedder.dim,
        )
        repo.insert_chunks(conn, spec.set_id, chunks, vectors)
        repo.create_vector_index(conn, spec.set_id, embedder.dim)
    build_bm25(conn, spec.set_id, cache_dir)
    mean = sum(len(c.text) for c in chunks) / max(len(chunks), 1)
    return IndexStats(spec.set_id, len(chunks), mean, time.perf_counter() - t0)
