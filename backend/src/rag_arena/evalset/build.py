"""Build the eval set: sample sections, write candidates, filter, select, save.

Every stage caches its model results under data/eval/work/, so re-running after an
interruption never pays for the same call twice.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from rag_arena.corpus.loader import Document
from rag_arena.embedding import Embedder
from rag_arena.evalset.evidence import locate
from rag_arena.evalset.prompts import (
    FILTER_SCHEMA,
    FILTER_SYSTEM,
    FILTER_USER,
    GENERATE_SCHEMA,
    GENERATE_SYSTEM,
    GENERATE_USER,
)
from rag_arena.evalset.sampling import SectionSample, sample_sections
from rag_arena.evalset.schema import FilterVerdict, QAItem, QuestionType, write_items
from rag_arena.llm.claude import ClaudeJSON, ClaudeRequest, ClaudeResult, ResultCache

Mode = Literal["batch", "sync"]
CHECKS = ("answerable", "self_contained", "unambiguous", "answer_correct", "useful")


@dataclass
class Candidate:
    key: str
    sample: SectionSample
    question: str
    answer: str
    evidence: str
    start: int
    end: int
    question_type: str
    verdict: FilterVerdict | None = None
    dropped: str | None = None  # why it was dropped before or after filtering
    extra: dict[str, Any] = field(default_factory=dict)


def generation_requests(samples: Sequence[SectionSample]) -> list[ClaudeRequest]:
    return [
        ClaudeRequest(
            custom_id=f"gen-{i:03d}",
            system=GENERATE_SYSTEM,
            user=GENERATE_USER.format(
                title=s.doc.title, path=" > ".join(s.section.path), text=s.text.strip()
            ),
            schema=GENERATE_SCHEMA,
            max_tokens=4000,
        )
        for i, s in enumerate(samples)
    ]


def to_candidates(
    samples: Sequence[SectionSample], results: dict[str, ClaudeResult]
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Turn generation results into candidates with located evidence; log the rest."""
    kept: list[Candidate] = []
    dropped: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        cid = f"gen-{i:03d}"
        res = results.get(cid)
        if res is None or res.data is None:
            dropped.append({"key": sample.key, "reason": res.error if res else "no result"})
            continue
        data = res.data
        if data.get("skip") or not data.get("question"):
            dropped.append({"key": sample.key, "reason": "model skipped the section"})
            continue
        span = locate(
            sample.doc.text, data["evidence"], hint_start=sample.start, hint_end=sample.end
        )
        if span is None:
            dropped.append({"key": sample.key, "reason": "evidence not found in the document"})
            continue
        kept.append(
            Candidate(
                key=sample.key,
                sample=sample,
                question=" ".join(data["question"].split()),
                answer=data["answer"].strip(),
                evidence=sample.doc.text[span[0] : span[1]],
                start=span[0],
                end=span[1],
                question_type=data.get("question_type") or "factual",
            )
        )
    return kept, dropped


def drop_near_duplicates(
    candidates: list[Candidate], embedder: Embedder, threshold: float = 0.93
) -> list[Candidate]:
    """Mark later candidates whose question embedding is within `threshold` of an earlier one."""
    if not candidates:
        return []
    vecs = embedder.embed_passages([c.question for c in candidates])
    sims = vecs @ vecs.T
    kept: list[Candidate] = []
    kept_idx: list[int] = []
    for i, cand in enumerate(candidates):
        if kept_idx and float(np.max(sims[i, kept_idx])) >= threshold:
            cand.dropped = "near-duplicate question"
            continue
        kept.append(cand)
        kept_idx.append(i)
    return kept


def filter_requests(candidates: Sequence[Candidate]) -> list[ClaudeRequest]:
    return [
        ClaudeRequest(
            custom_id="flt-" + hashlib.sha1(c.key.encode()).hexdigest()[:16],
            system=FILTER_SYSTEM,
            user=FILTER_USER.format(
                title=c.sample.doc.title,
                path=" > ".join(c.sample.section.path),
                text=c.sample.text.strip(),
                question=c.question,
                answer=c.answer,
                evidence=c.evidence,
            ),
            schema=FILTER_SCHEMA,
            max_tokens=3000,
        )
        for c in candidates
    ]


def apply_verdicts(
    candidates: Sequence[Candidate],
    requests: Sequence[ClaudeRequest],
    results: dict[str, ClaudeResult],
) -> None:
    for cand, req in zip(candidates, requests, strict=True):
        res = results.get(req.custom_id)
        if res is None or res.data is None:
            cand.dropped = f"filter failed: {res.error if res else 'no result'}"
            continue
        d = res.data
        checks = {name: bool(d[name]) for name in CHECKS}
        cand.verdict = FilterVerdict(
            **checks, keep=all(checks.values()), reason=str(d["reason"]), model=res.model or "?"
        )
        if not cand.verdict.keep:
            cand.dropped = "rejected by the filter"


def select(candidates: Sequence[Candidate], target: int, seed: int = 42) -> list[Candidate]:
    """Up to `target` kept candidates, round-robin across pages so no page dominates."""
    by_doc: dict[str, list[Candidate]] = {}
    for c in candidates:
        if c.verdict is not None and c.verdict.keep:
            by_doc.setdefault(c.sample.doc.id, []).append(c)
    docs = sorted(by_doc)
    random.Random(seed).shuffle(docs)
    picked: list[Candidate] = []
    depth = max((len(v) for v in by_doc.values()), default=0)
    for round_ in range(depth):
        for doc_id in docs:
            pool = by_doc[doc_id]
            if round_ < len(pool) and len(picked) < target:
                picked.append(pool[round_])
    return picked


def to_items(picked: Sequence[Candidate], *, author_model: str, corpus_ref: str) -> list[QAItem]:
    ordered = sorted(picked, key=lambda c: (c.sample.doc.id, c.start))
    items: list[QAItem] = []
    for n, c in enumerate(ordered, start=1):
        items.append(
            QAItem(
                id=f"q{n:03d}",
                question=c.question,
                answer=c.answer,
                question_type=cast(QuestionType, c.question_type),
                doc_id=c.sample.doc.id,
                doc_url=c.sample.doc.url_for(c.sample.section),
                section=list(c.sample.section.path),
                evidence=c.evidence,
                evidence_start=c.start,
                evidence_end=c.end,
                status="auto_approved",
                human_reviewed=False,
                author_model=author_model,
                corpus_ref=corpus_ref,
                filter=c.verdict,
            )
        )
    return items


def _run(
    claude: ClaudeJSON,
    reqs: list[ClaudeRequest],
    *,
    purpose: str,
    cache: Path,
    mode: Mode,
    log: Callable[[str], None],
) -> dict[str, ClaudeResult]:
    rc = ResultCache(cache)
    if mode == "batch":
        return claude.run_batch(reqs, purpose=purpose, cache=rc, log=log)
    return claude.run_sync(reqs, purpose=purpose, cache=rc)


def build_eval_set(
    docs: Sequence[Document],
    claude: ClaudeJSON,
    embedder: Embedder,
    out_dir: Path,
    *,
    corpus_ref: str,
    n_candidates: int = 150,
    target: int = 100,
    mode: Mode = "batch",
    min_section_chars: int = 400,
    log: Callable[[str], None] = print,
) -> list[QAItem]:
    work = out_dir / "work"
    samples = sample_sections(docs, n_candidates, min_chars=min_section_chars)
    log(f"sampled {len(samples)} sections from {len({s.doc.id for s in samples})} pages")

    gen_reqs = generation_requests(samples)
    gen = _run(
        claude, gen_reqs, purpose="eval-generate", cache=work / "generate.jsonl", mode=mode, log=log
    )
    candidates, dropped = to_candidates(samples, gen)
    log(f"{len(candidates)} candidates with located evidence, {len(dropped)} dropped")

    unique = drop_near_duplicates(candidates, embedder)
    log(f"{len(unique)} after removing near-duplicate questions")

    flt_reqs = filter_requests(unique)
    flt = _run(
        claude, flt_reqs, purpose="eval-filter", cache=work / "filter.jsonl", mode=mode, log=log
    )
    apply_verdicts(unique, flt_reqs, flt)
    kept = [c for c in unique if c.verdict is not None and c.verdict.keep]
    log(f"{len(kept)} kept by the filter")

    picked = select(unique, target)
    items = to_items(picked, author_model=claude.model, corpus_ref=corpus_ref)
    write_items(out_dir / "qa.jsonl", items)

    picked_keys = {c.key for c in picked}
    log_path = out_dir / "candidates.jsonl"
    with log_path.open("w", encoding="utf-8") as f:
        for c in candidates:
            row = {
                "key": c.key,
                "question": c.question,
                "answer": c.answer,
                "evidence": c.evidence,
                "question_type": c.question_type,
                "verdict": c.verdict.model_dump() if c.verdict else None,
                "dropped": c.dropped,
                "selected": c.key in picked_keys,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        for d in dropped:
            f.write(json.dumps({**d, "selected": False}, ensure_ascii=False) + "\n")
    log(f"wrote {len(items)} items to {out_dir / 'qa.jsonl'}")
    return items
