from __future__ import annotations

from pathlib import Path

from rag_arena.retrieval.bm25 import BM25Index, tokenize

DOCS = {
    "c1": "Declare a dependency with Depends and FastAPI calls it for each request.",
    "c2": "Use response_model to filter the data returned by a path operation.",
    "c3": "Background tasks run after the response is sent.",
}


def test_tokenizer_stems_drops_stopwords_and_splits_snake_case() -> None:
    assert tokenize(["What does Depends do with response_model?"]) == [
        ["depend", "respons", "model"]
    ]


def test_bm25_ranks_the_matching_chunk_first() -> None:
    index = BM25Index.build(list(DOCS), list(DOCS.values()))
    assert index.search("how do dependencies work", 3)[0][0] == "c1"
    assert index.search("response model", 3)[0][0] == "c2"


def test_bm25_drops_zero_scores_and_handles_empty_queries() -> None:
    index = BM25Index.build(list(DOCS), list(DOCS.values()))
    hits = index.search("background", 3)
    assert [i for i, _ in hits] == ["c3"]
    assert index.search("the and of", 3) == []
    assert index.search("zebra", 3) == []


def test_bm25_round_trips_through_disk(tmp_path: Path) -> None:
    index = BM25Index.build(list(DOCS), list(DOCS.values()))
    index.save(tmp_path / "idx")
    loaded = BM25Index.load(tmp_path / "idx")
    assert loaded.ids == index.ids
    assert loaded.search("response model", 2) == index.search("response model", 2)
