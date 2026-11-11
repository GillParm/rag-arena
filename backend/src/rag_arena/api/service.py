"""The logic behind the API, independent of HTTP so it can be tested directly."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Protocol

from rag_arena.api.schemas import (
    AnswerOut,
    AskRequest,
    AskResponse,
    ChunkOut,
    ChunkSetInfoOut,
    ConfigResponse,
    LatencyOut,
    PipelineInfo,
    PipelineResult,
)
from rag_arena.db.repo import ChunkSetInfo
from rag_arena.generation import CONTEXT_CHUNKS, generate_answer
from rag_arena.llm.base import TextLLM, Usage
from rag_arena.llm.pricing import cost_usd
from rag_arena.retrieval.backend import SearchBackend
from rag_arena.retrieval.pipelines import PIPELINE_LABELS, PIPELINES, run_pipeline
from rag_arena.retrieval.rerank import Reranker
from rag_arena.retrieval.rewrite import Rewriter

PIPELINE_DESCRIPTIONS = {
    "bm25": "Keyword search with BM25 over stemmed tokens.",
    "dense": "Nearest neighbours by cosine similarity of bge-small embeddings in pgvector.",
    "hybrid": "BM25 and dense top 50, merged with reciprocal rank fusion (k = 60).",
    "hybrid_rerank": "Hybrid top 20 re-scored by a cross-encoder.",
    "hybrid_rewrite": "The model writes 3 rewrites; BM25 and dense run for each and all lists are "
    "fused with RRF.",
}


class RuntimeLike(Protocol):
    def backend(self, set_id: str) -> SearchBackend: ...

    def chunk_sets(self) -> list[ChunkSetInfo]: ...

    @property
    def reranker(self) -> Reranker: ...

    @property
    def rewriter(self) -> Rewriter: ...

    @property
    def llm(self) -> TextLLM: ...


class ArenaService:
    def __init__(
        self,
        runtime: RuntimeLike,
        *,
        generation_enabled: bool,
        generator_model: str,
        embedding_model: str,
        reranker_model: str,
        corpus: str,
        leaderboard_path: Path,
    ) -> None:
        self.runtime = runtime
        self.generation_enabled = generation_enabled
        self.generator_model = generator_model
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.corpus = corpus
        self.leaderboard_path = leaderboard_path

    def leaderboard(self) -> dict[str, Any] | None:
        if not self.leaderboard_path.exists():
            return None
        data: dict[str, Any] = json.loads(self.leaderboard_path.read_text(encoding="utf-8"))
        return data

    def default_chunk_set(self, available: list[str]) -> str | None:
        board = self.leaderboard()
        headline = board.get("headline_chunk_set") if board else None
        for candidate in (headline, "heading"):
            if candidate in available:
                return str(candidate)
        return available[0] if available else None

    def config(self) -> ConfigResponse:
        sets = self.runtime.chunk_sets()
        ids = [s.id for s in sets]
        return ConfigResponse(
            pipelines=[
                PipelineInfo(id=p, label=PIPELINE_LABELS[p], description=PIPELINE_DESCRIPTIONS[p])
                for p in PIPELINES
            ],
            chunk_sets=[
                ChunkSetInfoOut(
                    id=s.id, chunker=s.chunker, chunk_count=s.chunk_count, params=s.params
                )
                for s in sets
            ],
            default_chunk_set=self.default_chunk_set(ids),
            generation_enabled=self.generation_enabled,
            generator_model=self.generator_model,
            embedding_model=self.embedding_model,
            reranker_model=self.reranker_model,
            corpus=self.corpus,
        )

    async def _one(
        self, pipeline: str, question: str, backend: SearchBackend, generate: bool
    ) -> PipelineResult:
        label = PIPELINE_LABELS[pipeline]
        t0 = time.perf_counter()
        try:
            retrieval = await run_pipeline(
                pipeline,
                question,
                backend,
                reranker=self.runtime.reranker if pipeline == "hybrid_rerank" else None,
                rewriter=self.runtime.rewriter if pipeline == "hybrid_rewrite" else None,
            )
        except Exception as exc:  # one pipeline failing should not sink the others
            elapsed = (time.perf_counter() - t0) * 1000
            return PipelineResult(
                pipeline=pipeline,
                label=label,
                answer=None,
                error=f"retrieval failed: {exc}",
                chunks=[],
                rewrites=[],
                latency=LatencyOut(
                    retrieval_ms=elapsed, generation_ms=None, total_ms=elapsed, stages={}
                ),
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
            )
        rows = [backend.chunks[h.chunk_id] for h in retrieval.hits]
        context = rows[:CONTEXT_CHUNKS]
        usage: Usage = retrieval.usage
        answer: AnswerOut | None = None
        error: str | None = None
        gen_ms: float | None = None
        cited: set[str] = set()
        if generate:
            try:
                ans = await generate_answer(self.runtime.llm, question, context)
            except Exception as exc:
                error = f"generation failed: {exc}"
            else:
                answer = AnswerOut(
                    text=ans.text, citations=ans.citations, abstained=ans.abstained, model=ans.model
                )
                usage = usage + ans.usage
                gen_ms = ans.latency_ms
                cited = set(ans.citations)
        chunks = [
            ChunkOut(
                id=row.id,
                rank=rank,
                score=hit.score,
                doc_id=row.document_id,
                doc_title=row.doc_title,
                heading_path=row.heading_path,
                url=row.url,
                text=row.text,
                cited=row.id in cited,
                in_context=rank <= CONTEXT_CHUNKS,
            )
            for rank, (hit, row) in enumerate(zip(retrieval.hits, rows, strict=True), start=1)
        ]
        cost = 0.0
        if usage.input_tokens or usage.output_tokens:
            cost = cost_usd(self.generator_model, usage.input_tokens, usage.output_tokens)
        return PipelineResult(
            pipeline=pipeline,
            label=label,
            answer=answer,
            error=error,
            chunks=chunks,
            rewrites=retrieval.rewrites,
            latency=LatencyOut(
                retrieval_ms=retrieval.latency_ms,
                generation_ms=gen_ms,
                total_ms=retrieval.latency_ms + (gen_ms or 0.0),
                stages=retrieval.timings,
            ),
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=cost,
        )

    async def ask(self, req: AskRequest) -> AskResponse:
        sets = [s.id for s in self.runtime.chunk_sets()]
        chunk_set = req.chunk_set or self.default_chunk_set(sets)
        if chunk_set is None or chunk_set not in sets:
            raise LookupError(f"unknown chunk set {req.chunk_set!r}; available: {sets}")
        backend = await asyncio.to_thread(self.runtime.backend, chunk_set)
        pipelines = req.pipelines or list(PIPELINES)
        if not self.generation_enabled:
            pipelines = [p for p in pipelines if p != "hybrid_rewrite"] or ["hybrid"]
        generate = req.generate and self.generation_enabled
        results = await asyncio.gather(
            *(self._one(p, req.question, backend, generate) for p in pipelines)
        )
        return AskResponse(
            question=req.question,
            chunk_set=chunk_set,
            generation_enabled=self.generation_enabled,
            results=list(results),
        )
