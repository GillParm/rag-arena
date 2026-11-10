"""Integration tests against a real Postgres with pgvector. Run with `pytest -m db`.

They use their own database (RAG_ARENA_TEST_DATABASE_URL, default rag_arena_test on the
compose service) and skip when it cannot be reached.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from rag_arena.chunking.base import ChunkerSpec
from rag_arena.config import Settings
from rag_arena.corpus.loader import Document
from rag_arena.db import repo
from rag_arena.db.connection import connect, connection
from rag_arena.db.migrate import migrate, reset
from rag_arena.embedding import HashEmbedder
from rag_arena.indexing import bm25_dir, build_chunk_set, load_bm25
from rag_arena.retrieval.pipelines import run_pipeline
from rag_arena.runtime import Runtime

pytestmark = pytest.mark.db

TEST_URL = os.environ.get(
    "RAG_ARENA_TEST_DATABASE_URL", "postgresql://rag:rag@localhost:5433/rag_arena_test"
)


def _ensure_database(url: str) -> None:
    info = psycopg.conninfo.conninfo_to_dict(url)
    name = str(info["dbname"])
    admin = psycopg.conninfo.make_conninfo(url, dbname="postgres")
    with psycopg.connect(admin, autocommit=True, connect_timeout=3) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone()
        if not exists:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


@pytest.fixture(scope="module")
def db_url() -> Iterator[str]:
    try:
        _ensure_database(TEST_URL)
    except psycopg.OperationalError as exc:
        pytest.skip(f"no test database: {exc}")
    reset(TEST_URL)
    migrate(TEST_URL)
    yield TEST_URL
    reset(TEST_URL)


@pytest.fixture
def conn(db_url: str) -> Iterator[repo.Conn]:
    with connection(db_url) as c:
        yield c


def test_migrations_are_idempotent(db_url: str) -> None:
    assert migrate(db_url) == []
    with connection(db_url) as c:
        names = [r[0] for r in c.execute("SELECT name FROM schema_migrations")]
        assert names == ["001_init.sql"]
        ext = c.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'").fetchone()
        assert ext is not None


def test_build_chunk_set_loads_rows_index_and_bm25(
    conn: repo.Conn, mini_docs: list[Document], tmp_path: Path
) -> None:
    embedder = HashEmbedder(384)
    stats = build_chunk_set(conn, mini_docs, ChunkerSpec("heading"), embedder, tmp_path)
    assert stats.chunks > 0
    info = repo.get_chunk_set(conn, "heading")
    assert info is not None
    assert info.chunk_count == stats.chunks
    assert info.embedding_dim == 384
    rows = repo.fetch_chunks(conn, "heading")
    assert len(rows) == stats.chunks
    doc_text = {d.id: d.text for d in mini_docs}
    assert all(r.text == doc_text[r.document_id][r.start : r.end] for r in rows)
    indexes = [
        r[0]
        for r in conn.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'chunks_heading'")
    ]
    assert any("hnsw" in i and "vector(384)" in i for i in indexes)
    assert (bm25_dir(tmp_path, "heading") / "ids.json").exists()


def test_dense_search_finds_the_matching_chunk(
    conn: repo.Conn, mini_docs: list[Document], tmp_path: Path
) -> None:
    embedder = HashEmbedder(384)
    build_chunk_set(conn, mini_docs, ChunkerSpec("fixed"), embedder, tmp_path)
    rows = repo.fetch_chunks(conn, "fixed")
    target = rows[-1]
    vec = embedder.embed_queries([target.index_text])[0]
    hits = repo.dense_search(conn, "fixed", 384, vec, 3)
    assert hits[0][0] == target.id
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)
    fetched = repo.get_chunks(conn, [target.id, "fixed:missing:9"])
    assert list(fetched) == [target.id]


def test_rebuilding_a_set_replaces_it_and_bm25_follows(
    conn: repo.Conn, mini_docs: list[Document], tmp_path: Path
) -> None:
    embedder = HashEmbedder(384)
    build_chunk_set(conn, mini_docs, ChunkerSpec("recursive"), embedder, tmp_path)
    first = len(repo.fetch_chunks(conn, "recursive"))
    spec = ChunkerSpec("recursive", {"size": 120})
    build_chunk_set(conn, mini_docs, spec, embedder, tmp_path)
    assert len(repo.fetch_chunks(conn, "recursive")) == first
    assert len(repo.fetch_chunks(conn, spec.set_id)) > first
    index = load_bm25(conn, spec.set_id, tmp_path)
    assert len(index.ids) == len(repo.fetch_chunks(conn, spec.set_id))
    repo.drop_chunk_set(conn, spec.set_id)
    assert repo.get_chunk_set(conn, spec.set_id) is None


def test_invalid_set_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="invalid chunk set id"):
        repo.partition_name("x; DROP TABLE documents")


async def test_runtime_runs_pipelines_against_postgres(
    db_url: str, mini_docs: list[Document], tmp_path: Path
) -> None:
    embedder = HashEmbedder(384)
    settings = Settings(_env_file=None, data_dir=tmp_path, database_url=db_url)
    with connection(db_url) as c:
        build_chunk_set(c, mini_docs, ChunkerSpec("heading"), embedder, settings.cache_dir)
    shared = connect(db_url)
    try:
        rt = Runtime(settings, lambda: nullcontext(shared), embedder=embedder)
        backend = rt.backend("heading")
        result = await run_pipeline("hybrid", "fastapi dev command live server", backend)
        assert result.hits
        top = backend.chunks[result.chunk_ids[0]]
        assert "Run it" in top.heading_path
    finally:
        shared.close()
