"""Explicit SQL for documents, chunk sets and chunks."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from rag_arena.chunking.base import Chunk
from rag_arena.corpus.loader import Document

Conn = psycopg.Connection[Any]

_SET_ID = re.compile(r"^[a-z][a-z0-9_-]{0,62}$")


@dataclass(frozen=True)
class ChunkRow:
    id: str
    chunk_set: str
    document_id: str
    ordinal: int
    start: int
    end: int
    text: str
    index_text: str
    heading_path: list[str]
    url: str
    doc_title: str


@dataclass(frozen=True)
class ChunkSetInfo:
    id: str
    chunker: str
    params: dict[str, Any]
    embedding_model: str
    embedding_dim: int
    chunk_count: int


def partition_name(set_id: str) -> str:
    if not _SET_ID.match(set_id):
        raise ValueError(f"invalid chunk set id: {set_id!r}")
    return "chunks_" + set_id.replace("-", "_")


def upsert_documents(conn: Conn, docs: Sequence[Document]) -> None:
    rows = [
        (
            d.id,
            d.path,
            d.title,
            d.url,
            d.text,
            d.sha256,
            json.dumps([s.__dict__ | {"path": list(s.path)} for s in d.sections]),
        )
        for d in docs
    ]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO documents (id, path, title, url, text, sha256, sections)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                path = EXCLUDED.path, title = EXCLUDED.title, url = EXCLUDED.url,
                text = EXCLUDED.text, sha256 = EXCLUDED.sha256, sections = EXCLUDED.sections
            """,
            rows,
        )


def get_document(conn: Conn, doc_id: str) -> dict[str, Any] | None:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, path, title, url, text FROM documents WHERE id = %s", (doc_id,))
        return cur.fetchone()


def drop_chunk_set(conn: Conn, set_id: str) -> None:
    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(partition_name(set_id))))
    conn.execute("DELETE FROM chunk_sets WHERE id = %s", (set_id,))


def create_chunk_set(
    conn: Conn,
    set_id: str,
    *,
    chunker: str,
    params: dict[str, Any],
    embedding_model: str,
    embedding_dim: int,
) -> None:
    """Replace the chunk set: a fresh row in chunk_sets and an empty partition."""
    drop_chunk_set(conn, set_id)
    conn.execute(
        """
        INSERT INTO chunk_sets (id, chunker, params, embedding_model, embedding_dim)
        VALUES (%s, %s, %s::jsonb, %s, %s)
        """,
        (set_id, chunker, json.dumps(params), embedding_model, embedding_dim),
    )
    conn.execute(
        sql.SQL("CREATE TABLE {} PARTITION OF chunks FOR VALUES IN ({})").format(
            sql.Identifier(partition_name(set_id)), sql.Literal(set_id)
        )
    )


def insert_chunks(
    conn: Conn, set_id: str, chunks: Sequence[Chunk], embeddings: np.ndarray[Any, Any]
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("one embedding per chunk")
    with (
        conn.cursor() as cur,
        cur.copy(
            """
        COPY chunks (chunk_set, id, document_id, ordinal, start_char, end_char,
                     text, index_text, heading_path, url, embedding)
        FROM STDIN
        """
        ) as copy,
    ):
        for chunk, vec in zip(chunks, embeddings, strict=True):
            copy.write_row(
                (
                    set_id,
                    chunk.chunk_id(set_id),
                    chunk.doc_id,
                    chunk.ordinal,
                    chunk.start,
                    chunk.end,
                    chunk.text,
                    chunk.index_text,
                    list(chunk.heading_path),
                    chunk.url,
                    vec.astype(np.float32),
                )
            )
    conn.execute("UPDATE chunk_sets SET chunk_count = %s WHERE id = %s", (len(chunks), set_id))


def create_vector_index(conn: Conn, set_id: str, dim: int, *, m: int = 16, ef: int = 64) -> None:
    """HNSW index on the set's partition. Build after loading: it is faster than incremental."""
    part = partition_name(set_id)
    conn.execute(
        sql.SQL(
            "CREATE INDEX {} ON {} USING hnsw ((embedding::vector({})) vector_cosine_ops)"
            " WITH (m = {}, ef_construction = {})"
        ).format(
            sql.Identifier(f"{part}_hnsw"),
            sql.Identifier(part),
            sql.Literal(int(dim)),
            sql.Literal(int(m)),
            sql.Literal(int(ef)),
        )
    )
    conn.execute(sql.SQL("ANALYZE {}").format(sql.Identifier(part)))


def list_chunk_sets(conn: Conn) -> list[ChunkSetInfo]:
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, chunker, params, embedding_model, embedding_dim, chunk_count"
            " FROM chunk_sets ORDER BY id"
        )
        return [ChunkSetInfo(**row) for row in cur.fetchall()]


def get_chunk_set(conn: Conn, set_id: str) -> ChunkSetInfo | None:
    return next((s for s in list_chunk_sets(conn) if s.id == set_id), None)


_CHUNK_COLUMNS = sql.SQL(
    "c.id, c.chunk_set, c.document_id, c.ordinal, c.start_char AS start, c.end_char AS end,"
    " c.text, c.index_text, c.heading_path, c.url, d.title AS doc_title"
)


def fetch_chunks(conn: Conn, set_id: str) -> list[ChunkRow]:
    """All chunks of a set in a stable order (document, ordinal)."""
    query = sql.SQL(
        "SELECT {} FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE c.chunk_set = %s ORDER BY c.document_id, c.ordinal"
    ).format(_CHUNK_COLUMNS)
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (set_id,))
        return [ChunkRow(**row) for row in cur.fetchall()]


def get_chunks(conn: Conn, ids: Sequence[str]) -> dict[str, ChunkRow]:
    if not ids:
        return {}
    query = sql.SQL(
        "SELECT {} FROM chunks c JOIN documents d ON d.id = c.document_id"
        " WHERE c.chunk_set = ANY(%s) AND c.id = ANY(%s)"
    ).format(_CHUNK_COLUMNS)
    sets = sorted({i.split(":", 1)[0] for i in ids})
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(query, (sets, list(ids)))
        return {row["id"]: ChunkRow(**row) for row in cur.fetchall()}


def dense_search(
    conn: Conn,
    set_id: str,
    dim: int,
    query_vec: np.ndarray[Any, Any],
    k: int,
    *,
    ef_search: int = 100,
) -> list[tuple[str, float]]:
    """Nearest chunks by cosine distance. Returns (chunk id, cosine similarity)."""
    conn.execute(sql.SQL("SET hnsw.ef_search = {}").format(sql.Literal(int(ef_search))))
    query = sql.SQL(
        "SELECT id, 1 - (embedding::vector({dim}) <=> %(q)s::vector({dim})) AS score"
        " FROM {part} ORDER BY embedding::vector({dim}) <=> %(q)s::vector({dim}) LIMIT %(k)s"
    ).format(dim=sql.Literal(int(dim)), part=sql.Identifier(partition_name(set_id)))
    rows = conn.execute(query, {"q": query_vec.astype(np.float32), "k": k}).fetchall()
    return [(str(r[0]), float(r[1])) for r in rows]
