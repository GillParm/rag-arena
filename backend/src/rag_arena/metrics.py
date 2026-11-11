"""Retrieval metrics, latency percentiles, cost and bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int

    def __len__(self) -> int:
        return max(0, self.end - self.start)

    def overlap(self, other: Span) -> int:
        if self.doc_id != other.doc_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))


def is_relevant(chunk: Span, gold: Span, *, min_fraction: float = 0.5) -> bool:
    """Relevant if the overlap covers at least `min_fraction` of the shorter of the two spans.

    A chunk that merely touches the evidence by a few characters does not count; a chunk that
    holds most of a short quote, or a chunk lying entirely inside a long quote, does.
    """
    ov = chunk.overlap(gold)
    shorter = min(len(chunk), len(gold))
    return ov > 0 and shorter > 0 and ov >= min_fraction * shorter


def first_relevant_rank(ranked: Sequence[Span], gold: Span) -> int | None:
    """1-based rank of the first relevant chunk, or None."""
    for rank, span in enumerate(ranked, start=1):
        if is_relevant(span, gold):
            return rank
    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """Share of questions with a relevant chunk in the top k (one gold span per question)."""
    if not ranks:
        return math.nan
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr_at_k(ranks: Sequence[int | None], k: int) -> float:
    if not ranks:
        return math.nan
    return sum(1.0 / r for r in ranks if r is not None and r <= k) / len(ranks)


def evidence_coverage(ranked: Sequence[Span], gold: Span, k: int) -> float:
    """Fraction of the gold evidence's characters covered by the union of the top-k chunks."""
    if len(gold) == 0:
        return math.nan
    covered = [False] * len(gold)
    for span in ranked[:k]:
        if span.doc_id != gold.doc_id:
            continue
        for i in range(max(span.start, gold.start), min(span.end, gold.end)):
            covered[i - gold.start] = True
    return sum(covered) / len(gold)


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile (numpy's default), p in [0, 100]."""
    if not values:
        return math.nan
    xs = sorted(values)
    pos = (len(xs) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def bootstrap_ci(
    values: Sequence[float],
    stat: Callable[[Sequence[float]], float] = mean,
    *,
    n: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for `stat` over questions."""
    if not values:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    k = len(values)
    stats = sorted(stat([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return (percentile(stats, 100 * alpha / 2), percentile(stats, 100 * (1 - alpha / 2)))


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], *, n: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of a - b over the same questions, with a paired bootstrap CI."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    lo, hi = bootstrap_ci(diffs, n=n, alpha=alpha, seed=seed)
    return mean(diffs), lo, hi


def cost_per_1k(costs_usd: Sequence[float]) -> float:
    """Mean cost per query times 1,000."""
    return 1000 * mean(costs_usd) if costs_usd else math.nan
