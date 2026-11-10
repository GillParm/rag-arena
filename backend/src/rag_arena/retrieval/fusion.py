"""Reciprocal rank fusion (Cormack, Clarke and Buettcher, SIGIR 2009)."""

from __future__ import annotations

from collections.abc import Sequence

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = RRF_K, limit: int | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked lists of ids: score(d) = sum over lists of 1 / (k + rank of d), rank from 1.

    Ties break on the best single rank the id reached, then on first appearance, so the
    output is deterministic.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    order: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:  # count an id once per list
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
            order.setdefault(item, len(order))
    fused = sorted(scores, key=lambda i: (-scores[i], best_rank[i], order[i]))
    if limit is not None:
        fused = fused[:limit]
    return [(i, scores[i]) for i in fused]
