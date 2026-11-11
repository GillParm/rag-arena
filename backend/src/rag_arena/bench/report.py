"""Turn a run's raw records into summary.json, CSV tables and the README markdown."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_arena.bench.runner import PipelineSpec, Stores, gold_span, record_spans
from rag_arena.evalset.schema import QAItem
from rag_arena.judge import JudgeParseError, judge_custom_id, parse_judgment
from rag_arena.llm.base import Usage
from rag_arena.llm.pricing import cost_usd
from rag_arena.metrics import (
    bootstrap_ci,
    evidence_coverage,
    first_relevant_rank,
    mean,
    mrr_at_k,
    paired_bootstrap_diff,
    percentile,
    recall_at_k,
)
from rag_arena.retrieval.pipelines import PIPELINE_LABELS


@dataclass
class QuestionRow:
    qid: str
    pipeline: str
    chunk_set: str
    rank: int | None
    coverage5: float
    retrieval_ms: float
    generation_ms: float | None = None
    e2e_ms: float | None = None
    cost_usd: float | None = None
    faithfulness: int | None = None
    correctness: int | None = None
    abstained: bool | None = None
    judge_error: str | None = None


def _round(x: float | None, nd: int = 4) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(x, nd)


def question_rows(
    items: Sequence[QAItem], stores: Stores, pipeline: str, chunk_set: str, generator: str | None
) -> list[QuestionRow]:
    rows: list[QuestionRow] = []
    for it in items:
        rec = stores.retrieval.get((pipeline, chunk_set, it.id))
        if rec is None:
            continue
        spans = record_spans(rec)
        gold = gold_span(it)
        row = QuestionRow(
            qid=it.id,
            pipeline=pipeline,
            chunk_set=chunk_set,
            rank=first_relevant_rank(spans, gold),
            coverage5=evidence_coverage(spans, gold, 5),
            retrieval_ms=float(rec["latency_ms"]),
        )
        gen = stores.generation.get((pipeline, chunk_set, it.id))
        if gen is not None:
            row.generation_ms = float(gen["latency_ms"])
            row.e2e_ms = row.retrieval_ms + row.generation_ms
            row.abstained = bool(gen["abstained"])
            usage = Usage(**gen["usage"]) + Usage(**rec["usage"])
            model = generator or gen["model"]
            row.cost_usd = cost_usd(model, usage.input_tokens, usage.output_tokens)
        rows.append(row)
    return rows


def attach_judgments(rows: list[QuestionRow], judged: dict[str, Any]) -> None:
    for row in rows:
        if row.e2e_ms is None:
            continue
        res = judged.get(judge_custom_id(row.pipeline, row.chunk_set, row.qid))
        if res is None:
            row.judge_error = "not judged"
            continue
        try:
            j = parse_judgment(res)
        except JudgeParseError as exc:
            row.judge_error = str(exc)
            continue
        row.faithfulness, row.correctness = j.faithfulness, j.correctness


def summarize_rows(rows: list[QuestionRow]) -> dict[str, Any]:
    ranks = [r.rank for r in rows]
    r5 = [1.0 if r.rank is not None and r.rank <= 5 else 0.0 for r in rows]
    out: dict[str, Any] = {
        "n": len(rows),
        "recall@5": _round(recall_at_k(ranks, 5)),
        "recall@5_ci": [_round(x) for x in bootstrap_ci(r5)],
        "recall@10": _round(recall_at_k(ranks, 10)),
        "mrr@10": _round(mrr_at_k(ranks, 10)),
        "coverage@5": _round(mean([r.coverage5 for r in rows])),
        "retrieval_p50_ms": _round(percentile([r.retrieval_ms for r in rows], 50), 1),
        "retrieval_p95_ms": _round(percentile([r.retrieval_ms for r in rows], 95), 1),
    }
    gen = [r for r in rows if r.e2e_ms is not None]
    if gen:
        judged = [r for r in gen if r.faithfulness is not None and r.correctness is not None]
        faith = [float(r.faithfulness) for r in judged if r.faithfulness is not None]
        corr = [float(r.correctness) for r in judged if r.correctness is not None]
        out.update(
            {
                "answered": len(gen),
                "judged": len(judged),
                "judge_failures": len(gen) - len(judged),
                "faithfulness": _round(mean(faith), 3),
                "faithfulness_ci": [_round(x, 3) for x in bootstrap_ci(faith)],
                "faithful_rate": _round(mean([1.0 if f >= 4 else 0.0 for f in faith])),
                "correctness": _round(mean(corr), 3),
                "correctness_ci": [_round(x, 3) for x in bootstrap_ci(corr)],
                "correct_rate": _round(mean([1.0 if c >= 4 else 0.0 for c in corr])),
                "abstention_rate": _round(mean([1.0 if r.abstained else 0.0 for r in gen])),
                "generation_p50_ms": _round(
                    percentile([r.generation_ms or 0.0 for r in gen], 50), 1
                ),
                "generation_p95_ms": _round(
                    percentile([r.generation_ms or 0.0 for r in gen], 95), 1
                ),
                "e2e_p50_ms": _round(percentile([r.e2e_ms or 0.0 for r in gen], 50), 1),
                "e2e_p95_ms": _round(percentile([r.e2e_ms or 0.0 for r in gen], 95), 1),
                "cost_per_1k_usd": _round(1000 * mean([r.cost_usd or 0.0 for r in gen]), 4),
            }
        )
    return out


def _paired(
    rows_by_pipeline: dict[str, list[QuestionRow]],
    winner: str,
    value: Callable[[QuestionRow], float | None],
) -> dict[str, list[float | None]]:
    """Winner minus each other pipeline (mean and 95% paired bootstrap CI) on the questions
    both have a value for."""
    out: dict[str, list[float | None]] = {}
    base = {r.qid: r for r in rows_by_pipeline[winner]}
    for p, rows in rows_by_pipeline.items():
        if p == winner:
            continue
        a, b = [], []
        for r in rows:
            w = base.get(r.qid)
            va = value(w) if w else None
            vb = value(r)
            if va is not None and vb is not None:
                a.append(float(va))
                b.append(float(vb))
        if a:
            d, lo, hi = paired_bootstrap_diff(a, b)
            out[p] = [_round(d, 3), _round(lo, 3), _round(hi, 3)]
    return out


def build_summary(
    items: Sequence[QAItem],
    stores: Stores,
    *,
    specs: Sequence[PipelineSpec],
    chunk_sets: Sequence[str],
    headline: str,
    headline_rule: dict[str, dict[str, float]],
    meta: dict[str, Any],
) -> tuple[dict[str, Any], list[QuestionRow]]:
    judged = stores.judge.load()
    all_rows: list[QuestionRow] = []
    headline_rows: dict[str, list[QuestionRow]] = {}
    ablation: list[dict[str, Any]] = []
    headline_table: list[dict[str, Any]] = []
    generator = meta.get("generator_model")
    for set_id in chunk_sets:
        for spec in specs:
            rows = question_rows(items, stores, spec.id, set_id, generator)
            if not rows:
                continue
            attach_judgments(rows, judged)
            all_rows.extend(rows)
            summary = summarize_rows(rows)
            entry = {
                "pipeline": spec.id,
                "label": PIPELINE_LABELS.get(spec.id, spec.id),
                "base": spec.base,
                "reranker_model": spec.reranker_model,
                "chunk_set": set_id,
                **summary,
            }
            ablation.append(entry)
            if set_id == headline and "answered" in summary:
                headline_rows[spec.id] = rows
                headline_table.append(entry)
    winner = None
    comparisons: dict[str, Any] = {}
    if headline_table:
        winner = max(
            headline_table, key=lambda e: (e.get("correctness") or 0, e.get("recall@5") or 0)
        )["pipeline"]
        comparisons = {
            "winner": winner,
            "correctness_diff": _paired(
                headline_rows,
                winner,
                lambda r: float(r.correctness) if r.correctness is not None else None,
            ),
            "recall5_diff": _paired(
                headline_rows,
                winner,
                lambda r: 1.0 if r.rank is not None and r.rank <= 5 else 0.0,
            ),
        }
    summary = {
        **meta,
        "questions": len(items),
        "headline_chunk_set": headline,
        "headline_rule": "highest mean recall@5 across the five pipelines; ties by mean MRR@10",
        "headline_rule_scores": headline_rule,
        "winner_rule": "highest mean correctness on the headline chunk set; ties by recall@5",
        "headline": headline_table,
        "ablation": ablation,
        "comparisons": comparisons,
    }
    return summary, all_rows


def _fmt(x: Any, pct: bool = False, nd: int = 2) -> str:
    if x is None:
        return "n/a"
    if pct:
        return f"{100 * x:.0f}%"
    return f"{x:.{nd}f}"


def markdown_tables(summary: dict[str, Any]) -> str:
    lines = [
        f"Headline chunker: `{summary['headline_chunk_set']}`. "
        f"{summary['questions']} questions. Generator `{summary.get('generator_model')}`, "
        f"judge `{summary.get('judge_model')}`.",
        "",
        "| Pipeline | Recall@5 | Recall@10 | MRR@10 | Faithfulness | Correctness "
        "| Retrieval p50 / p95 | End-to-end p50 / p95 | Cost per 1k queries |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for e in summary["headline"]:
        lines.append(
            f"| {e['label']} | {_fmt(e['recall@5'])} | {_fmt(e['recall@10'])} "
            f"| {_fmt(e['mrr@10'])} | {_fmt(e.get('faithfulness'))} "
            f"| {_fmt(e.get('correctness'))} "
            f"| {e['retrieval_p50_ms']:.0f} / {e['retrieval_p95_ms']:.0f} ms "
            f"| {e['e2e_p50_ms']:.0f} / {e['e2e_p95_ms']:.0f} ms "
            f"| ${e['cost_per_1k_usd']:.3f} |"
        )
    lines += [
        "",
        "Chunker ablation (retrieval only):",
        "",
        "| Pipeline | Chunker | Recall@5 | Recall@10 | MRR@10 | Retrieval p50 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for e in sorted(summary["ablation"], key=lambda e: (e["pipeline"], e["chunk_set"])):
        lines.append(
            f"| {e['label']} | {e['chunk_set']} | {_fmt(e['recall@5'])} | {_fmt(e['recall@10'])} "
            f"| {_fmt(e['mrr@10'])} | {e['retrieval_p50_ms']:.0f} ms |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(summary: dict[str, Any], rows: list[QuestionRow], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(markdown_tables(summary), encoding="utf-8")
    cols = [
        "pipeline",
        "chunk_set",
        "n",
        "recall@5",
        "recall@10",
        "mrr@10",
        "coverage@5",
        "faithfulness",
        "faithful_rate",
        "correctness",
        "correct_rate",
        "abstention_rate",
        "retrieval_p50_ms",
        "retrieval_p95_ms",
        "generation_p50_ms",
        "generation_p95_ms",
        "e2e_p50_ms",
        "e2e_p95_ms",
        "cost_per_1k_usd",
        "judge_failures",
    ]
    for name, entries in (
        ("summary.csv", summary["headline"]),
        ("ablation.csv", summary["ablation"]),
    ):
        with (out_dir / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for e in entries:
                w.writerow({c: e.get(c) for c in cols})
    with (out_dir / "per_question.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(QuestionRow.__dataclass_fields__)
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})
