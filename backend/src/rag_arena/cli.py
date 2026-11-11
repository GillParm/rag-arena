"""Command-line entry point: `rag-arena --help`."""

from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext

import typer

from rag_arena import __version__
from rag_arena.bench.cli import bench_app
from rag_arena.chunking.base import ChunkerSpec
from rag_arena.config import Settings, get_settings
from rag_arena.corpus.download import corpus_root, fetch_corpus
from rag_arena.corpus.loader import Document, load_corpus
from rag_arena.corpus.manifest import build_manifest, diff_manifest, read_manifest, write_manifest
from rag_arena.db import repo
from rag_arena.db.connection import connection
from rag_arena.db.migrate import migrate, reset
from rag_arena.embedding import FastEmbedder
from rag_arena.evalset.build import build_eval_set
from rag_arena.evalset.review import review
from rag_arena.evalset.schema import read_items
from rag_arena.indexing import build_chunk_set
from rag_arena.llm.claude import ClaudeJSON, SpendLedger
from rag_arena.retrieval.pipelines import PIPELINES, run_pipeline
from rag_arena.runtime import Runtime

app = typer.Typer(
    help="Benchmark and compare RAG retrieval pipelines on one corpus.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def doctor() -> None:
    """Show which settings and API keys are configured (never the key values)."""
    s = get_settings()
    typer.echo(f"data dir:        {s.data_dir}")
    typer.echo(f"results dir:     {s.results_dir}")
    typer.echo(f"database:        {s.database_url.rsplit('@', 1)[-1]}")
    typer.echo(f"gemini key:      {'set' if s.has_gemini() else 'missing'}")
    typer.echo(f"anthropic key:   {'set' if s.has_anthropic() else 'missing'}")


corpus_app = typer.Typer(help="Download and inspect the FastAPI docs corpus.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")


def _load_docs(s: Settings) -> list[Document]:
    root = corpus_root(s.raw_dir, s.corpus_ref)
    if not (root / "LICENSE").exists():
        raise typer.BadParameter(
            f"no corpus at {root}; run `rag-arena corpus download` first", param_hint="corpus"
        )
    return load_corpus(root)


@corpus_app.command("download")
def corpus_download(
    force: bool = typer.Option(False, help="Re-download even if the archive is cached."),
) -> None:
    """Fetch the pinned FastAPI release, check its licence and write data/manifest.json."""
    s = get_settings()
    result = fetch_corpus(s.raw_dir, s.corpus_ref, force=force)
    typer.echo(f"ref {result.ref} (commit {result.commit}), {result.files_extracted} files")
    typer.echo(f"licence: MIT, {result.licence_holder}")
    docs = load_corpus(result.root)
    manifest_path = s.data_dir / "manifest.json"
    if manifest_path.exists():
        diff = diff_manifest(read_manifest(manifest_path), docs)
        if not diff.ok:
            typer.echo(f"warning: corpus differs from the committed manifest: {diff}", err=True)
    manifest = build_manifest(
        docs, ref=result.ref, commit=result.commit, licence=result.licence_holder
    )
    write_manifest(manifest_path, manifest)
    typer.echo(f"{len(docs)} documents, {manifest['total_chars']:,} characters")


@corpus_app.command("verify")
def corpus_verify() -> None:
    """Check the downloaded corpus against the committed manifest."""
    s = get_settings()
    diff = diff_manifest(read_manifest(s.data_dir / "manifest.json"), _load_docs(s))
    if not diff.ok:
        typer.echo(f"mismatch: {diff}", err=True)
        raise typer.Exit(1)
    typer.echo("corpus matches data/manifest.json")


db_app = typer.Typer(help="Database migrations.", no_args_is_help=True)
app.add_typer(db_app, name="db")


@db_app.command("migrate")
def db_migrate() -> None:
    """Apply pending SQL migrations."""
    applied = migrate(get_settings().database_url)
    typer.echo(f"applied: {', '.join(applied)}" if applied else "schema is up to date")


@db_app.command("reset")
def db_reset(yes: bool = typer.Option(False, "--yes", help="Confirm dropping all tables.")) -> None:
    """Drop all rag-arena tables (the indexes must be rebuilt afterwards)."""
    if not yes:
        raise typer.BadParameter("pass --yes to drop all tables")
    reset(get_settings().database_url)
    typer.echo("dropped")


index_app = typer.Typer(help="Build and inspect chunk-set indexes.", no_args_is_help=True)
app.add_typer(index_app, name="index")

DEFAULT_CHUNKERS = ["fixed", "recursive", "heading"]


def _embedder(s: Settings) -> FastEmbedder:
    return FastEmbedder(s.embedding_model, cache_dir=s.model_cache_dir or s.cache_dir / "models")


@index_app.command("build")
def index_build(
    chunker: list[str] = typer.Option(
        DEFAULT_CHUNKERS, "--chunker", "-c", help="Chunker spec, e.g. heading or fixed:size=800."
    ),
) -> None:
    """Chunk, embed and index the corpus for each chunker (replaces existing sets)."""
    s = get_settings()
    docs = _load_docs(s)
    migrate(s.database_url)
    embedder = _embedder(s)
    with connection(s.database_url) as conn:
        for value in chunker:
            stats = build_chunk_set(
                conn, docs, ChunkerSpec.parse(value), embedder, s.cache_dir, log=typer.echo
            )
            typer.echo(
                f"{stats.set_id}: {stats.chunks} chunks, mean {stats.mean_chars:.0f} chars,"
                f" {stats.seconds:.1f}s"
            )


@index_app.command("status")
def index_status() -> None:
    """List the indexed chunk sets."""
    with connection(get_settings().database_url) as conn:
        sets = repo.list_chunk_sets(conn)
    if not sets:
        typer.echo("no chunk sets; run `rag-arena index build`")
    for cs in sets:
        typer.echo(
            f"{cs.id:<12} {cs.chunk_count:>5} chunks  {cs.embedding_model} ({cs.embedding_dim}d)"
        )


@contextmanager
def _runtime(s: Settings) -> Iterator[Runtime]:
    with connection(s.database_url) as conn:
        yield Runtime(s, lambda: nullcontext(conn))


@app.command()
def search(
    question: str,
    pipeline: str = typer.Option("hybrid", "--pipeline", "-p", help=f"One of {PIPELINES}."),
    chunker: str = typer.Option("heading", "--chunker", "-c", help="Chunk set id."),
    k: int = typer.Option(5, help="Results to show."),
) -> None:
    """Run one retrieval pipeline and print the ranked chunks."""
    s = get_settings()
    with _runtime(s) as rt:
        backend = rt.backend(chunker)
        result = asyncio.run(
            run_pipeline(
                pipeline,
                question,
                backend,
                reranker=rt.reranker if pipeline == "hybrid_rerank" else None,
                rewriter=rt.rewriter if pipeline == "hybrid_rewrite" else None,
            )
        )
    for rank, hit in enumerate(result.hits[:k], start=1):
        row = backend.chunks[hit.chunk_id]
        typer.echo(f"{rank:>2}. {hit.score:8.4f}  {hit.chunk_id}  ({' > '.join(row.heading_path)})")
    if result.rewrites:
        typer.echo("rewrites: " + " | ".join(result.rewrites))
    timings = ", ".join(f"{k} {v:.1f}ms" for k, v in result.timings.items())
    typer.echo(f"{result.latency_ms:.1f} ms ({timings})")


eval_app = typer.Typer(help="Build, inspect and review the eval set.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


def _ledger(s: Settings) -> SpendLedger:
    return SpendLedger(s.results_dir / "spend" / "anthropic.jsonl")


def _claude(s: Settings, model: str) -> ClaudeJSON:
    if not s.anthropic_api_key or not s.has_anthropic():
        raise typer.BadParameter("ANTHROPIC_API_KEY is not set", param_hint="env")
    return ClaudeJSON(s.anthropic_api_key.get_secret_value(), model, _ledger(s))


@eval_app.command("build")
def eval_build(
    candidates: int = typer.Option(150, help="Sections to write candidate questions from."),
    target: int = typer.Option(100, help="Questions to keep."),
    mode: str = typer.Option("batch", help="batch (half price, slower) or sync."),
) -> None:
    """Write candidate questions with Claude, filter them and save data/eval/qa.jsonl."""
    if mode not in ("batch", "sync"):
        raise typer.BadParameter("mode must be batch or sync")
    s = get_settings()
    docs = _load_docs(s)
    claude = _claude(s, s.eval_author_model)
    items = build_eval_set(
        docs,
        claude,
        _embedder(s),
        s.eval_path.parent,
        corpus_ref=s.corpus_ref,
        n_candidates=candidates,
        target=target,
        mode="batch" if mode == "batch" else "sync",
        log=typer.echo,
    )
    typer.echo(f"{len(items)} questions; Anthropic spend so far ${_ledger(s).total_usd():.2f}")


@eval_app.command("stats")
def eval_stats() -> None:
    """Summarise the eval set by status, question type and page."""
    items = read_items(get_settings().eval_path)
    by_status = Counter(i.status for i in items)
    by_type = Counter(i.question_type for i in items)
    typer.echo(f"{len(items)} entries; {sum(i.usable for i in items)} usable")
    typer.echo("status: " + ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())))
    typer.echo("type:   " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))
    typer.echo(f"pages:  {len({i.doc_id for i in items})}")
    typer.echo(f"human reviewed: {sum(i.human_reviewed for i in items)}")


@eval_app.command("review")
def eval_review(
    all_entries: bool = typer.Option(False, "--all", help="Include entries already reviewed."),
) -> None:
    """Approve, edit or reject entries one by one; progress is saved after each decision."""
    s = get_settings()
    doc_texts: dict[str, str] = {}
    root = corpus_root(s.raw_dir, s.corpus_ref)
    if (root / "LICENSE").exists():
        doc_texts = {d.id: d.text for d in load_corpus(root)}
    counts = review(
        s.eval_path,
        ask=lambda prompt, default: str(
            typer.prompt(prompt, default=default, show_default=bool(default))
        ),
        say=typer.echo,
        doc_texts=doc_texts,
        include_reviewed=all_entries,
    )
    typer.echo(", ".join(f"{k} {v}" for k, v in counts.items()))


@app.command()
def spend() -> None:
    """Total Anthropic spend recorded in results/spend/anthropic.jsonl, by purpose."""
    rows = _ledger(get_settings()).rows()
    by_purpose: dict[str, list[float]] = {}
    for r in rows:
        by_purpose.setdefault(r["purpose"], []).append(float(r["cost_usd"]))
    for purpose, costs in sorted(by_purpose.items()):
        typer.echo(f"{purpose:<16} {len(costs):>5} calls  ${sum(costs):8.4f}")
    typer.echo(
        f"{'total':<16} {len(rows):>5} calls  ${sum(sum(c) for c in by_purpose.values()):8.4f}"
    )


app.add_typer(bench_app, name="bench")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8300, envvar="PORT"),
    reload: bool = typer.Option(False),
) -> None:
    """Run the HTTP API."""
    import uvicorn

    uvicorn.run("rag_arena.api.app:create_app", factory=True, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
