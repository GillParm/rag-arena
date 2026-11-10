"""Command-line entry point: `rag-arena --help`."""

from __future__ import annotations

import typer

from rag_arena import __version__
from rag_arena.config import get_settings

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


if __name__ == "__main__":
    app()
