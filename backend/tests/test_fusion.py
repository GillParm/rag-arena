from __future__ import annotations

import pytest

from rag_arena.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_scores_match_the_formula() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_rewards_agreement_between_lists() -> None:
    fused = [i for i, _ in reciprocal_rank_fusion([["x", "shared"], ["y", "shared"]])]
    assert fused[0] == "shared"


def test_rrf_breaks_ties_deterministically() -> None:
    # a and b tie on score; both best ranks are 1, so first appearance wins.
    assert [i for i, _ in reciprocal_rank_fusion([["a", "b"], ["b", "a"]])] == ["a", "b"]
    # c and d tie on score (rank 3 each); c appeared first.
    assert [i for i, _ in reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])][2:] == [
        "c",
        "d",
    ]


def test_rrf_counts_duplicates_once_per_list_and_limits() -> None:
    fused = reciprocal_rank_fusion([["a", "a", "b"]], k=0, limit=1)
    assert fused == [("a", 1.0)]


def test_rrf_handles_empty_input_and_rejects_negative_k() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError, match="non-negative"):
        reciprocal_rank_fusion([["a"]], k=-1)
