"""`rag-arena bench run` and `rag-arena bench report`."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
from contextlib import nullcontext
from typing import Any

import typer

from rag_arena.bench.report import build_summary, write_outputs
from rag_arena.bench.runner import (
    BenchConfig,
    PipelineSpec,
    Stores,
    choose_headline,
    default_specs,
    stage_generation,
    stage_judge,
    stage_retrieval,
    write_config,
)
from rag_arena.config import REPO_ROOT, Settings, get_settings
from rag_arena.db.connection import connection
from rag_arena.evalset.schema import QAItem, read_items
from rag_arena.judge import load_rubric
from rag_arena.llm.claude import ClaudeJSON, SpendLedger
from rag_arena.retrieval.rerank import FastReranker, Reranker
from rag_arena.runtime import Runtime

bench_app = typer.Typer(help="Run the benchmark and build the report.", no_args_is_help=True)

RERANKER_VARIANTS = [
    PipelineSpec("hybrid_rerank_minilm", "hybrid_rerank", "Xenova/ms-marco-MiniLM-L-6-v2"),
    PipelineSpec("hybrid_rerank_jina", "hybrid_rerank", "jinaai/jina-reranker-v1-turbo-en"),
]


def _items(s: Settings, limit: int | None) -> list[QAItem]:
    items = [i for i in read_items(s.eval_path) if i.usable]
    return items[:limit] if limit else items


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _meta(s: Settings, items: list[QAItem]) -> dict[str, Any]:
    manifest_path = s.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    _, rubric_sha = load_rubric()
    ledger = SpendLedger(s.results_dir / "spend" / "anthropic.jsonl")
    spend: dict[str, float] = {}
    for row in ledger.rows():
        spend[row["purpose"]] = spend.get(row["purpose"], 0.0) + float(row["cost_usd"])
    return {
        "corpus": {
            "ref": manifest.get("ref"),
            "commit": manifest.get("commit"),
            "documents": manifest.get("documents"),
        },
        "eval_set_sha256": hashlib.sha256(s.eval_path.read_bytes()).hexdigest(),
        "human_reviewed": sum(i.human_reviewed for i in items),
        "generator_model": s.generator_model,
        "judge_model": s.judge_model,
        "embedding_model": s.embedding_model,
        "reranker_model": s.reranker_model,
        "rubric_sha256": rubric_sha,
        "code_commit": _git_commit(),
        "anthropic_spend_usd": {k: round(v, 4) for k, v in sorted(spend.items())},
    }


@bench_app.command("run")
def bench_run(
    run_name: str = typer.Option("main", help="Directory under results/runs/."),
    chunker: list[str] = typer.Option(
        ["fixed", "recursive", "heading"], "--chunker", "-c", help="Chunk sets to compare."
    ),
    headline: str | None = typer.Option(None, help="Skip the selection rule and use this set."),
    limit: int | None = typer.Option(None, help="Only the first N questions (smoke runs)."),
    generate: bool = typer.Option(True, help="Generate answers on the headline chunk set."),
    judge: bool = typer.Option(True, help="Judge the generated answers with Claude."),
    judge_mode: str = typer.Option("batch", help="batch (half price) or sync."),
    reranker_variants: bool = typer.Option(
        False, help="Also run hybrid_rerank with two smaller cross-encoders (retrieval only)."
    ),
) -> None:
    """Run the benchmark: retrieval everywhere, then generation and judging on one chunk set.

    Safe to re-run: finished work is read back from results/runs/<name>/.
    """
    s = get_settings()
    items = _items(s, limit)
    specs = default_specs() + (RERANKER_VARIANTS if reranker_variants else [])
    cfg = BenchConfig(
        run_dir=s.results_dir / "runs" / run_name,
        chunk_sets=chunker,
        specs=specs,
        headline=headline,
        generate=generate,
        judge=judge,
        judge_mode="sync" if judge_mode == "sync" else "batch",
    )
    stores = Stores(cfg.run_dir)
    rerankers: dict[str, Reranker] = {}

    with connection(s.database_url) as conn:
        rt = Runtime(s, lambda: nullcontext(conn))

        def reranker_for(model: str | None) -> Reranker:
            if model is None or model == s.reranker_model:
                return rt.reranker
            if model not in rerankers:
                rerankers[model] = FastReranker(model, cache_dir=rt.model_dir)
            return rerankers[model]

        async def all_stages() -> tuple[str, dict[str, dict[str, float]]]:
            # One event loop for every stage: the Gemini client and its rate limiter are
            # bound to the loop they first run on.
            await stage_retrieval(
                items, cfg, stores, rt.backend, reranker_for, lambda: rt.rewriter, typer.echo
            )
            chosen, rule = choose_headline(items, stores, cfg.chunk_sets)
            head = headline or chosen
            typer.echo(f"headline chunk set: {head} (rule: {json.dumps(rule)})")
            base_specs = default_specs()
            complete = True
            if generate:
                complete = await stage_generation(
                    items, base_specs, head, stores, rt.backend(head), rt.llm, typer.echo
                )
            if judge and complete:
                if not s.anthropic_api_key or not s.has_anthropic():
                    raise typer.BadParameter("ANTHROPIC_API_KEY is not set", param_hint="env")
                claude = ClaudeJSON(
                    s.anthropic_api_key.get_secret_value(),
                    s.judge_model,
                    SpendLedger(s.results_dir / "spend" / "anthropic.jsonl"),
                )
                await asyncio.to_thread(
                    stage_judge,
                    items,
                    base_specs,
                    head,
                    stores,
                    rt.backend(head),
                    claude,
                    cfg.judge_mode,
                    typer.echo,
                )
            return head, rule

        head, rule = asyncio.run(all_stages())
    write_config(cfg, {"headline": head, "headline_rule_scores": rule, "questions": len(items)})
    _report(s, run_name, items, specs, cfg.chunk_sets, head, rule)


def _report(
    s: Settings,
    run_name: str,
    items: list[QAItem],
    specs: list[PipelineSpec],
    chunk_sets: list[str],
    head: str,
    rule: dict[str, dict[str, float]],
) -> None:
    run_dir = s.results_dir / "runs" / run_name
    summary, rows = build_summary(
        items,
        Stores(run_dir),
        specs=specs,
        chunk_sets=chunk_sets,
        headline=head,
        headline_rule=rule,
        meta={"run": run_name, **_meta(s, items)},
    )
    write_outputs(summary, rows, run_dir)
    typer.echo(f"wrote {run_dir / 'summary.json'}")
    typer.echo((run_dir / "summary.md").read_text())


@bench_app.command("report")
def bench_report(run_name: str = typer.Option("main")) -> None:
    """Rebuild the summary from a run's raw records (no model calls)."""
    s = get_settings()
    run_dir = s.results_dir / "runs" / run_name
    config = json.loads((run_dir / "config.json").read_text())
    specs = [PipelineSpec(**p) for p in config["pipelines"]]
    items = _items(s, config.get("questions"))
    _report(
        s,
        run_name,
        items,
        specs,
        config["chunk_sets"],
        config["headline"],
        config["headline_rule_scores"],
    )
