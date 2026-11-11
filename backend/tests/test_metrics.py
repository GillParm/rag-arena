from __future__ import annotations

import math

import pytest

from rag_arena.metrics import (
    Span,
    bootstrap_ci,
    cost_per_1k,
    evidence_coverage,
    first_relevant_rank,
    is_relevant,
    mrr_at_k,
    paired_bootstrap_diff,
    percentile,
    recall_at_k,
)

GOLD = Span("d", 100, 200)


@pytest.mark.parametrize(
    ("chunk", "expected"),
    [
        (Span("d", 0, 1000), True),  # contains the whole quote
        (Span("d", 150, 400), True),  # half the quote
        (Span("d", 160, 400), False),  # 40 of 100 quote chars
        (Span("d", 120, 160), True),  # short chunk inside the quote
        (Span("d", 195, 600), False),  # touches by 5 characters
        (Span("other", 100, 200), False),  # different document
        (Span("d", 200, 300), False),  # adjacent, no overlap
    ],
)
def test_relevance_needs_half_of_the_shorter_span(chunk: Span, expected: bool) -> None:
    assert is_relevant(chunk, GOLD) is expected


def test_ranks_recall_and_mrr() -> None:
    ranked = [Span("x", 0, 10), Span("d", 0, 50), Span("d", 90, 210)]
    assert first_relevant_rank(ranked, GOLD) == 3
    assert first_relevant_rank(ranked[:2], GOLD) is None
    ranks = [1, 3, None, 7]
    assert recall_at_k(ranks, 5) == 0.5
    assert recall_at_k(ranks, 10) == 0.75
    assert mrr_at_k(ranks, 10) == pytest.approx((1 + 1 / 3 + 1 / 7) / 4)
    assert mrr_at_k(ranks, 2) == pytest.approx(1 / 4)
    assert math.isnan(recall_at_k([], 5))


def test_evidence_coverage_unions_the_top_k() -> None:
    ranked = [Span("d", 100, 130), Span("d", 120, 150), Span("d", 180, 400), Span("e", 0, 999)]
    assert evidence_coverage(ranked, GOLD, 2) == pytest.approx(0.5)
    assert evidence_coverage(ranked, GOLD, 3) == pytest.approx(0.7)


def test_percentile_matches_linear_interpolation() -> None:
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(xs, 50) == 3.0
    assert percentile(xs, 95) == pytest.approx(4.8)
    assert percentile([7.0], 95) == 7.0
    assert math.isnan(percentile([], 50))


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible() -> None:
    values = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(values, n=500)
    assert lo < 0.5 < hi
    assert (lo, hi) == bootstrap_ci(values, n=500)
    assert bootstrap_ci([1.0] * 10, n=100) == (1.0, 1.0)


def test_paired_diff_and_cost() -> None:
    d, lo, hi = paired_bootstrap_diff([5, 5, 4, 5], [3, 4, 4, 2], n=500)
    assert d == pytest.approx(1.5)
    assert lo <= d <= hi
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_diff([1], [1, 2])
    assert cost_per_1k([0.001, 0.003]) == pytest.approx(2.0)
