"""Apply numbered .sql migrations once each, in order."""

from __future__ import annotations

from importlib import resources

from rag_arena.db.connection import connection

LOCK_ID = 7_303_300  # arbitrary advisory lock key so concurrent migrators queue up


def migration_files() -> list[tuple[str, str]]:
    files = resources.files("rag_arena.db") / "migrations"
    out = [
        (f.name, f.read_text(encoding="utf-8")) for f in files.iterdir() if f.name.endswith(".sql")
    ]
    return sorted(out)


def migrate(url: str) -> list[str]:
    """Apply pending migrations. Returns the names applied in this call."""
    applied_now: list[str] = []
    with connection(url, vector=False) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (LOCK_ID,))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done = {row[0] for row in conn.execute("SELECT name FROM schema_migrations")}
            for name, sql in migration_files():
                if name in done:
                    continue
                with conn.transaction():
                    # bytes, not str: the file is a trusted multi-statement script, not a literal
                    conn.execute(sql.encode())
                    conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (name,))
                applied_now.append(name)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_ID,))
    return applied_now


def reset(url: str) -> None:
    """Drop everything this project created. Used by tests and `db reset`."""
    with connection(url, vector=False) as conn:
        conn.execute(
            "DROP TABLE IF EXISTS chunks, chunk_sets, documents, schema_migrations CASCADE"
        )
