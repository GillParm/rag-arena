"""Benchmark stages. Each writes append-only records keyed by (pipeline, chunk set, question)
so an interrupted run resumes where it stopped and never repeats a paid call."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from rag_arena.bench.store import RecordStore
from rag_arena.evalset.schema import QAItem
from rag_arena.generation import CONTEXT_CHUNKS, format_passages, generate_answer
from rag_arena.judge import judge_custom_id, judge_request, load_rubric
from rag_arena.llm.base import Usage
from rag_arena.llm.claude import ClaudeJSON, ClaudeResult, ResultCache
from rag_arena.llm.gemini import DailyQuotaExhaustedError
from rag_arena.llm.ratelimit import RetryGiveUpError
from rag_arena.metrics import Span, first_relevant_rank, mean, mrr_at_k, recall_at_k
from rag_arena.retrieval.backend import SearchBackend
from rag_arena.retrieval.pipelines import PIPELINES, PipelineConfig, run_pipeline
from rag_arena.retrieval.rerank import Reranker
from rag_arena.retrieval.rewrite import Rewriter, RewriteResult

Log = Callable[[str], None]


@dataclass(frozen=True)
class PipelineSpec:
    """A pipeline as benchmarked: its id in the results, the base pipeline it runs, and for
    reranker variants, which cross-encoder to use."""

    id: str
    base: str
    reranker_model: str | None = None


def default_specs() -> list[PipelineSpec]:
    return [PipelineSpec(p, p) for p in PIPELINES]


@dataclass
class BenchConfig:
    run_dir: Path
    chunk_sets: list[str]
    specs: list[PipelineSpec] = field(default_factory=default_specs)
    pipeline_cfg: PipelineConfig = field(default_factory=PipelineConfig)
    headline: str | None = None  # None: pick by the pre-registered rule
    generate: bool = True
    judge: bool = True
    judge_mode: Literal["batch", "sync"] = "batch"


class Stores:
    def __init__(self, run_dir: Path) -> None:
        self.retrieval = RecordStore(run_dir / "retrieval.jsonl", ("pipeline", "chunk_set", "qid"))
        self.rewrites = RecordStore(run_dir / "rewrites.jsonl", ("question",))
        self.generation = RecordStore(
            run_dir / "generation.jsonl", ("pipeline", "chunk_set", "qid")
        )
        self.judge = ResultCache(run_dir / "judge.jsonl")


class CachedRewriter:
    """Rewrites are per question, so every chunk set reuses the same rewrites (and the
    latency measured when they were first produced)."""

    def __init__(self, inner: Rewriter, store: RecordStore) -> None:
        self.inner = inner
        self.store = store

    async def rewrite(self, question: str) -> RewriteResult:
        cached = self.store.get((question,))
        if cached is not None:
            return RewriteResult(
                rewrites=cached["rewrites"],
                usage=Usage(**cached["usage"]),
                latency_ms=cached["latency_ms"],
                model=cached["model"],
            )
        result = await self.inner.rewrite(question)
        self.store.append(
            {
                "question": question,
                "rewrites": result.rewrites,
                "usage": result.usage.to_dict(),
                "latency_ms": result.latency_ms,
                "model": result.model,
            }
        )
        return result


def gold_span(item: QAItem) -> Span:
    return Span(item.doc_id, item.evidence_start, item.evidence_end)


def record_spans(rec: dict[str, Any]) -> list[Span]:
    return [Span(d, s, e) for d, s, e in rec["spans"]]


async def prefetch_rewrites(items: Sequence[QAItem], rewriter: CachedRewriter, log: Log) -> None:
    """Write missing rewrites concurrently (bounded by the client's rate limiter) before the
    timed retrieval loop. Each rewrite keeps the latency of its own successful model call."""
    todo = [it for it in items if not rewriter.store.has((it.question,))]
    if not todo:
        return
    log(f"writing query rewrites for {len(todo)} questions")

    async def one(it: QAItem) -> None:
        try:
            await rewriter.rewrite(it.question)
        except (DailyQuotaExhaustedError, RetryGiveUpError) as exc:
            log(f"rewrite failed for {it.id}: {exc}")

    await asyncio.gather(*(one(it) for it in todo))


async def stage_retrieval(
    items: Sequence[QAItem],
    cfg: BenchConfig,
    stores: Stores,
    backend_for: Callable[[str], SearchBackend],
    reranker_for: Callable[[str | None], Reranker],
    rewriter: Callable[[], Rewriter],
    log: Log,
) -> None:
    """Run every pipeline spec on every question for every chunk set, one query at a time so
    latency is measured without contention."""
    for set_id in cfg.chunk_sets:
        backend = backend_for(set_id)
        for spec in cfg.specs:
            todo = [it for it in items if not stores.retrieval.has((spec.id, set_id, it.id))]
            if not todo:
                continue
            reranker = reranker_for(spec.reranker_model) if spec.base == "hybrid_rerank" else None
            rw = None
            if spec.base == "hybrid_rewrite":
                rw = CachedRewriter(rewriter(), stores.rewrites)
                await prefetch_rewrites(todo, rw, log)
            # Warm-up query (not recorded) so model loading never lands in a latency sample.
            warm_base = "hybrid" if spec.base == "hybrid_rewrite" else spec.base
            await run_pipeline(
                warm_base, "warm up", backend, cfg=cfg.pipeline_cfg, reranker=reranker
            )
            log(f"retrieval {spec.id} on {set_id}: {len(todo)} questions")
            for it in todo:
                try:
                    res = await run_pipeline(
                        spec.base,
                        it.question,
                        backend,
                        cfg=cfg.pipeline_cfg,
                        reranker=reranker,
                        rewriter=rw,
                    )
                except (DailyQuotaExhaustedError, RetryGiveUpError) as exc:
                    log(f"stopping retrieval for {spec.id}: {exc}")
                    break
                rec = {"qid": it.id, **res.to_dict(), "pipeline": spec.id, "base": spec.base}
                rec["spans"] = [
                    [
                        backend.chunks[h.chunk_id].document_id,
                        backend.chunks[h.chunk_id].start,
                        backend.chunks[h.chunk_id].end,
                    ]
                    for h in res.hits
                ]
                stores.retrieval.append(rec)


def choose_headline(
    items: Sequence[QAItem], stores: Stores, chunk_sets: Sequence[str]
) -> tuple[str, dict[str, dict[str, float]]]:
    """Pre-registered rule: the chunk set with the highest mean recall@5 across the five base
    pipelines; ties go to the higher mean MRR@10."""
    table: dict[str, dict[str, float]] = {}
    for set_id in chunk_sets:
        r5s, mrrs = [], []
        for p in PIPELINES:
            ranks = []
            for it in items:
                rec = stores.retrieval.get((p, set_id, it.id))
                if rec is None:
                    raise RuntimeError(f"missing retrieval record {(p, set_id, it.id)}")
                ranks.append(first_relevant_rank(record_spans(rec), gold_span(it)))
            r5s.append(recall_at_k(ranks, 5))
            mrrs.append(mrr_at_k(ranks, 10))
        table[set_id] = {"mean_recall@5": mean(r5s), "mean_mrr@10": mean(mrrs)}
    best = max(chunk_sets, key=lambda s: (table[s]["mean_recall@5"], table[s]["mean_mrr@10"]))
    return best, table


async def stage_generation(
    items: Sequence[QAItem],
    specs: Sequence[PipelineSpec],
    chunk_set: str,
    stores: Stores,
    backend: SearchBackend,
    llm: Any,
    log: Log,
) -> bool:
    """Generate answers from each pipeline's top chunks. Returns False if it had to stop early
    (daily quota), in which case re-running resumes."""
    todo = [
        (spec, it)
        for spec in specs
        for it in items
        if not stores.generation.has((spec.id, chunk_set, it.id))
    ]
    log(f"generation on {chunk_set}: {len(todo)} answers to write")
    stop = asyncio.Event()
    done = 0

    async def one(spec: PipelineSpec, it: QAItem) -> None:
        nonlocal done
        if stop.is_set():
            return
        rec = stores.retrieval.get((spec.id, chunk_set, it.id))
        if rec is None:
            return
        chunks = [backend.chunks[cid] for cid, _ in rec["hits"][:CONTEXT_CHUNKS]]
        try:
            answer = await generate_answer(llm, it.question, chunks)
        except DailyQuotaExhaustedError as exc:
            if not stop.is_set():
                log(f"daily quota exhausted; re-run later to resume ({exc})")
            stop.set()
            return
        except RetryGiveUpError as exc:
            log(f"giving up on {spec.id}/{it.id} for now: {exc}")
            return
        stores.generation.append(
            {"qid": it.id, "pipeline": spec.id, "chunk_set": chunk_set, **answer.to_dict()}
        )
        done += 1
        if done % 25 == 0:
            log(f"  {done}/{len(todo)} answers")

    await asyncio.gather(*(one(spec, it) for spec, it in todo))
    return not stop.is_set()


def stage_judge(
    items: Sequence[QAItem],
    specs: Sequence[PipelineSpec],
    chunk_set: str,
    stores: Stores,
    backend: SearchBackend,
    claude: ClaudeJSON,
    mode: Literal["batch", "sync"],
    log: Log,
) -> dict[str, ClaudeResult]:
    rubric, _ = load_rubric()
    reqs = []
    for spec in specs:
        for it in items:
            gen = stores.generation.get((spec.id, chunk_set, it.id))
            if gen is None:
                continue
            chunks = [backend.chunks[cid] for cid in gen["context_ids"]]
            reqs.append(
                judge_request(
                    custom_id=judge_custom_id(spec.id, chunk_set, it.id),
                    rubric=rubric,
                    question=it.question,
                    reference=it.answer,
                    evidence=it.evidence,
                    passages=format_passages(chunks),
                    answer=gen["text"],
                )
            )
    log(f"judging {len(reqs)} answers ({mode})")
    if mode == "batch":
        return claude.run_batch(reqs, purpose="judge", cache=stores.judge, log=log)
    return claude.run_sync(reqs, purpose="judge", cache=stores.judge)


def write_config(cfg: BenchConfig, extra: dict[str, Any]) -> None:
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.run_dir / "config.json"
    data = {
        "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        "chunk_sets": cfg.chunk_sets,
        "pipelines": [asdict(s) for s in cfg.specs],
        "pipeline_config": asdict(cfg.pipeline_cfg),
        **extra,
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
