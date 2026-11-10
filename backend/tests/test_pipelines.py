from __future__ import annotations

import json

import pytest

from rag_arena.llm.base import Usage
from rag_arena.llm.fake import FakeLLM
from rag_arena.retrieval.backend import SearchBackend
from rag_arena.retrieval.pipelines import (
    PIPELINES,
    PipelineConfig,
    PipelineError,
    run_pipeline,
)
from rag_arena.retrieval.rerank import OverlapReranker
from rag_arena.retrieval.rewrite import LLMRewriter, RewriteResult, parse_rewrites


class StaticRewriter:
    def __init__(self, rewrites: list[str]) -> None:
        self.rewrites = rewrites

    async def rewrite(self, question: str) -> RewriteResult:
        return RewriteResult(self.rewrites, Usage(10, 5), latency_ms=250.0, model="fake")


@pytest.mark.parametrize("name", PIPELINES)
async def test_every_pipeline_returns_ranked_unique_hits(name: str, backend: SearchBackend) -> None:
    result = await run_pipeline(
        name,
        "how do I declare dependencies",
        backend,
        cfg=PipelineConfig(k=3, candidates=5, rerank_depth=5),
        reranker=OverlapReranker(),
        rewriter=StaticRewriter(["Depends dependency declaration"]),
    )
    assert result.pipeline == name
    assert result.chunk_set == "test"
    assert 0 < len(result.hits) <= 3
    assert len(set(result.chunk_ids)) == len(result.chunk_ids)
    assert result.latency_ms >= 0
    assert result.hits[0].chunk_id.startswith("t:deps")


async def test_hybrid_fuses_both_retrievers(backend: SearchBackend) -> None:
    result = await run_pipeline("hybrid", "response_model shape", backend)
    assert {"embed", "bm25", "dense"} <= result.timings.keys()
    assert result.chunk_ids[0] == "t:resp:0"


async def test_rerank_reorders_the_hybrid_candidates(backend: SearchBackend) -> None:
    cfg = PipelineConfig(k=2, candidates=5, rerank_depth=5)
    result = await run_pipeline(
        "hybrid_rerank", "background tasks", backend, cfg=cfg, reranker=OverlapReranker()
    )
    assert result.chunk_ids[0] == "t:bg:0"
    assert "rerank" in result.timings


async def test_rewrite_latency_counts_the_model_call_not_the_wait(
    backend: SearchBackend,
) -> None:
    result = await run_pipeline(
        "hybrid_rewrite", "cors", backend, rewriter=StaticRewriter(["allowed origins"])
    )
    assert result.rewrites == ["allowed origins"]
    assert result.usage == Usage(10, 5)
    assert result.timings["rewrite"] == 250.0
    assert result.latency_ms >= 250.0
    assert result.chunk_ids[0] == "t:cors:0"


async def test_pipelines_that_need_models_say_so(backend: SearchBackend) -> None:
    with pytest.raises(PipelineError, match="reranker"):
        await run_pipeline("hybrid_rerank", "q", backend)
    with pytest.raises(PipelineError, match="rewriter"):
        await run_pipeline("hybrid_rewrite", "q", backend)
    with pytest.raises(PipelineError, match="unknown pipeline"):
        await run_pipeline("colbert", "q", backend)


def test_parse_rewrites_cleans_model_output() -> None:
    text = json.dumps({"rewrites": ["  A  b ", "q", "a b", 3, "c", "d", "e"]})
    assert parse_rewrites(text, "Q", 3) == ["A b", "c", "d"]
    assert parse_rewrites("not json", "q", 3) == []
    assert parse_rewrites("[1, 2]", "q", 3) == []


async def test_llm_rewriter_sends_the_schema_and_parses_json() -> None:
    llm = FakeLLM(lambda _s, _p: json.dumps({"rewrites": ["one", "two", "three"]}))
    result = await LLMRewriter(llm).rewrite("How do I add CORS?")
    assert result.rewrites == ["one", "two", "three"]
    assert llm.calls[0]["json_schema"]["required"] == ["rewrites"]
    assert "How do I add CORS?" in llm.calls[0]["prompt"]
