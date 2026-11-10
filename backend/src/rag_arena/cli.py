"""Command-line entry point: `rag-arena --help`."""

from __future__ import annotations

import typer

from rag_arena import __version__
from rag_arena.config import Settings, get_settings
from rag_arena.corpus.download import corpus_root, fetch_corpus
from rag_arena.corpus.loader import Document, load_corpus
from rag_arena.corpus.manifest import build_manifest, diff_manifest, read_manifest, write_manifest

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


if __name__ == "__main__":
    app()
