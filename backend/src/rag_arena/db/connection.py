"""Connections with the pgvector type registered.

Connections run in autocommit mode; code that needs atomic writes opens an explicit
`with conn.transaction():` block, so a long job commits each step as it finishes.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool


def connect(url: str, *, vector: bool = True) -> psycopg.Connection[Any]:
    conn = psycopg.connect(url, autocommit=True)
    if vector:
        register_vector(conn)
    return conn


@contextmanager
def connection(url: str, *, vector: bool = True) -> Iterator[psycopg.Connection[Any]]:
    conn = connect(url, vector=vector)
    try:
        yield conn
    finally:
        conn.close()


def make_pool(url: str, *, min_size: int = 1, max_size: int = 8) -> ConnectionPool:
    def configure(conn: psycopg.Connection[Any]) -> None:
        register_vector(conn)

    return ConnectionPool(
        url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"autocommit": True},
        configure=configure,
        open=True,
    )
