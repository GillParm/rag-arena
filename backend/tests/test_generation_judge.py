from __future__ import annotations

import pytest

from rag_arena.db.repo import ChunkRow
from rag_arena.generation import (
    ABSTAIN_TEXT,
    build_prompt,
    generate_answer,
    is_abstention,
    parse_citations,
)
from rag_arena.judge import (
    JUDGE_SCHEMA,
    JudgeParseError,
    judge_custom_id,
    judge_request,
    load_rubric,
    parse_judgment,
)
from rag_arena.llm.claude import ClaudeResult
from rag_arena.llm.fake import FakeLLM


def _chunk(i: int, text: str) -> ChunkRow:
    return ChunkRow(
        id=f"heading:doc:{i}",
        chunk_set="heading",
        document_id="doc",
        ordinal=i,
        start=0,
        end=len(text),
        text=text,
        index_text=text,
        heading_path=["Guide", f"Part {i}"],
        url="u",
        doc_title="Guide",
    )


def test_parse_citations_handles_formats_order_and_range() -> None:
    text = "Use Depends [2]. It runs per request [1][2], see [3, 9] and [0]."
    assert parse_citations(text, 3) == [2, 1, 3]
    assert parse_citations("no citations", 3) == []


def test_prompt_numbers_passages_with_their_source() -> None:
    prompt = build_prompt("How?", [_chunk(0, "alpha"), _chunk(1, "beta")])
    assert "[1] Guide > Part 0\nalpha" in prompt
    assert "[2] Guide > Part 1\nbeta" in prompt
    assert prompt.endswith("Question: How?")


def test_abstention_detection() -> None:
    assert is_abstention(ABSTAIN_TEXT)
    assert is_abstention("Sorry. I couldn't find this in the documentation provided")
    assert not is_abstention("Use Depends.")


async def test_generate_answer_maps_citations_to_chunk_ids() -> None:
    llm = FakeLLM(lambda _s, _p: "Declare it with Depends [2], then call it [1].")
    chunks = [_chunk(0, "a"), _chunk(1, "b")]
    answer = await generate_answer(llm, "How?", chunks)
    assert answer.citations == ["heading:doc:1", "heading:doc:0"]
    assert answer.context_ids == ["heading:doc:0", "heading:doc:1"]
    assert not answer.abstained
    assert answer.usage.input_tokens > 0
    assert "only the numbered documentation passages" in llm.calls[0]["system"]


def test_rubric_loads_with_a_stable_hash() -> None:
    text, sha = load_rubric()
    assert "Faithfulness" in text
    assert "Correctness" in text
    assert len(sha) == 64
    assert load_rubric()[1] == sha


def test_judge_request_puts_rubric_in_system_and_reasoning_before_scores() -> None:
    req = judge_request(
        custom_id=judge_custom_id("hybrid", "heading", "q001"),
        rubric="RUBRIC TEXT",
        question="Q",
        reference="R",
        evidence="E",
        passages="[1] P",
        answer="A",
    )
    assert req.custom_id == "j-hybrid-heading-q001"
    assert "RUBRIC TEXT" in req.system
    assert "<answer_to_grade>\nA\n</answer_to_grade>" in req.user
    keys = list(JUDGE_SCHEMA["properties"])
    assert keys.index("faithfulness_reasoning") < keys.index("faithfulness")
    assert keys.index("correctness_reasoning") < keys.index("correctness")


def _result(data: dict[str, object] | None, error: str | None = None) -> ClaudeResult:
    return ClaudeResult("j", data, error)


def test_parse_judgment_accepts_valid_and_rejects_bad_output() -> None:
    good = {
        "faithfulness_reasoning": "ok",
        "unsupported_claims": [],
        "faithfulness": 5,
        "correctness_reasoning": "misses x",
        "missing_or_wrong": ["x"],
        "correctness": 3,
    }
    j = parse_judgment(_result(good))
    assert (j.faithfulness, j.correctness) == (5, 3)
    assert j.missing_or_wrong == ["x"]
    with pytest.raises(JudgeParseError, match="refusal"):
        parse_judgment(_result(None, "refusal (cyber)"))
    with pytest.raises(JudgeParseError, match="out of range"):
        parse_judgment(_result({**good, "correctness": 7}))
    with pytest.raises(JudgeParseError, match="missing"):
        parse_judgment(_result({"faithfulness": 4}))
