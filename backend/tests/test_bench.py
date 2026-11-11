from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_arena.bench.report import build_summary, markdown_tables, write_outputs
from rag_arena.bench.runner import (
    BenchConfig,
    PipelineSpec,
    Stores,
    choose_headline,
    default_specs,
    stage_generation,
    stage_judge,
    stage_retrieval,
)
from rag_arena.evalset.schema import QAItem
from rag_arena.llm.base import Usage
from rag_arena.llm.claude import ClaudeJSON, SpendLedger
from rag_arena.llm.fake import FakeLLM
from rag_arena.retrieval.backend import SearchBackend
from rag_arena.retrieval.rerank import OverlapReranker
from rag_arena.retrieval.rewrite import RewriteResult
from tests.conftest import CHUNKS, make_backend
from tests.fakes import FakeAnthropic


def _item(qid: str, question: str, doc: str) -> QAItem:
    return QAItem(
        id=qid,
        question=question,
        answer="ref",
        question_type="factual",
        doc_id=doc,
        doc_url="u",
        section=["S"],
        evidence="ev",
        evidence_start=0,
        evidence_end=20,
        status="auto_approved",
        author_model="m",
        corpus_ref="r",
    )


ITEMS = [
    _item("q001", "how are dependencies declared", "deps"),
    _item("q002", "what shape does response_model declare", "resp"),
    _item("q003", "when do background tasks run", "bg"),
]


class CountingRewriter:
    def __init__(self) -> None:
        self.calls = 0

    async def rewrite(self, question: str) -> RewriteResult:
        self.calls += 1
        return RewriteResult(
            [question + " docs"], Usage(20, 10), latency_ms=100.0, model="gemini-3.5-flash-lite"
        )


def _judge_reply(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "faithfulness_reasoning": "ok",
        "unsupported_claims": [],
        "faithfulness": 5,
        "correctness_reasoning": "ok",
        "missing_or_wrong": [],
        "correctness": 4,
    }


async def test_full_benchmark_offline_and_resumable(tmp_path: Path) -> None:
    backends: dict[str, SearchBackend] = {
        "a": make_backend(CHUNKS, set_id="a"),
        "b": make_backend({k: v + " extra words" for k, v in CHUNKS.items()}, set_id="b"),
    }
    cfg = BenchConfig(run_dir=tmp_path / "run", chunk_sets=["a", "b"])
    stores = Stores(cfg.run_dir)
    rewriter = CountingRewriter()

    async def retrieve() -> None:
        await stage_retrieval(
            ITEMS,
            cfg,
            stores,
            backends.__getitem__,
            lambda _m: OverlapReranker(),
            lambda: rewriter,
            lambda _: None,
        )

    await retrieve()
    assert len(stores.retrieval) == 5 * 2 * 3
    assert rewriter.calls == 3  # cached per question, reused across chunk sets
    await retrieve()
    assert len(stores.retrieval) == 30
    assert rewriter.calls == 3

    head, rule = choose_headline(ITEMS, stores, cfg.chunk_sets)
    assert head in {"a", "b"}
    assert set(rule) == {"a", "b"}

    llm = FakeLLM(lambda _s, _p: "Use Depends [1].", model="gemini-3.5-flash-lite")
    ok = await stage_generation(
        ITEMS, default_specs(), head, stores, backends[head], llm, lambda _: None
    )
    assert ok
    assert len(stores.generation) == 15
    await stage_generation(
        ITEMS, default_specs(), head, stores, backends[head], llm, lambda _: None
    )
    assert len(llm.calls) == 15

    fake = FakeAnthropic(_judge_reply)
    claude = ClaudeJSON("x", "claude-opus-5-5", SpendLedger(tmp_path / "s.jsonl"), client=fake)  # type: ignore[arg-type]
    stage_judge(
        ITEMS, default_specs(), head, stores, backends[head], claude, "sync", lambda _: None
    )
    assert len(fake.sync_calls) == 15

    specs = [*default_specs(), PipelineSpec("hybrid_rerank_alt", "hybrid_rerank", "alt")]
    summary, rows = build_summary(
        ITEMS,
        stores,
        specs=specs,
        chunk_sets=cfg.chunk_sets,
        headline=head,
        headline_rule=rule,
        meta={"generator_model": "gemini-3.5-flash-lite", "judge_model": "j"},
    )
    assert [e["pipeline"] for e in summary["headline"]] == [p.id for p in default_specs()]
    first = summary["headline"][0]
    assert first["n"] == 3
    assert first["faithfulness"] == 5.0
    assert first["correctness"] == 4.0
    assert first["correct_rate"] == 1.0
    assert first["judge_failures"] == 0
    assert first["cost_per_1k_usd"] > 0
    rewrite = next(e for e in summary["headline"] if e["pipeline"] == "hybrid_rewrite")
    assert rewrite["cost_per_1k_usd"] > first["cost_per_1k_usd"]  # rewrite tokens are priced
    assert rewrite["retrieval_p50_ms"] >= 100.0
    assert len(summary["ablation"]) == 10
    assert summary["comparisons"]["winner"] in {p.id for p in default_specs()}

    write_outputs(summary, rows, tmp_path / "out")
    saved = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert saved["headline_chunk_set"] == head
    assert (tmp_path / "out" / "per_question.csv").read_text().count("\n") == len(rows) + 1
    md = markdown_tables(summary)
    assert "| Hybrid (RRF) |" in md
    assert "Chunker ablation" in md
