"""LLM-as-judge: one structured Claude call scores faithfulness and correctness."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_arena.config import REPO_ROOT
from rag_arena.llm.claude import ClaudeRequest, ClaudeResult

RUBRIC_PATH = REPO_ROOT / "docs" / "rubric.md"

JUDGE_SYSTEM = """\
You are grading answers produced by a retrieval-augmented question answering system over the
FastAPI documentation. Apply the rubric below exactly. Be consistent: the same answer should
always get the same scores.

<rubric>
{rubric}
</rubric>"""

JUDGE_USER = """\
<question>{question}</question>

<reference_answer>{reference}</reference_answer>

<reference_evidence>{evidence}</reference_evidence>

<passages_given_to_the_system>
{passages}
</passages_given_to_the_system>

<answer_to_grade>
{answer}
</answer_to_grade>

Grade the answer. Fill in the reasoning and claim lists before each score."""

_SCORE = {"type": "integer", "enum": [1, 2, 3, 4, 5]}

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "faithfulness_reasoning": {"type": "string"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "faithfulness": _SCORE,
        "correctness_reasoning": {"type": "string"},
        "missing_or_wrong": {"type": "array", "items": {"type": "string"}},
        "correctness": _SCORE,
    },
    "required": [
        "faithfulness_reasoning",
        "unsupported_claims",
        "faithfulness",
        "correctness_reasoning",
        "missing_or_wrong",
        "correctness",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Judgment:
    faithfulness: int
    correctness: int
    unsupported_claims: list[str]
    missing_or_wrong: list[str]
    faithfulness_reasoning: str
    correctness_reasoning: str


class JudgeParseError(ValueError):
    pass


def load_rubric(path: Path = RUBRIC_PATH) -> tuple[str, str]:
    """The rubric text and its sha256 (stored with judgments to detect rubric edits)."""
    text = path.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def judge_system(rubric: str) -> str:
    return JUDGE_SYSTEM.format(rubric=rubric.strip())


def judge_custom_id(pipeline: str, chunk_set: str, qid: str) -> str:
    return f"j-{pipeline}-{chunk_set}-{qid}"[:64]


def judge_request(
    *,
    custom_id: str,
    rubric: str,
    question: str,
    reference: str,
    evidence: str,
    passages: str,
    answer: str,
) -> ClaudeRequest:
    return ClaudeRequest(
        custom_id=custom_id,
        system=judge_system(rubric),
        user=JUDGE_USER.format(
            question=question,
            reference=reference,
            evidence=evidence,
            passages=passages,
            answer=answer,
        ),
        schema=JUDGE_SCHEMA,
        max_tokens=3000,
    )


def parse_judgment(result: ClaudeResult) -> Judgment:
    """Validate a judge result. Raises JudgeParseError for refusals and malformed output."""
    if result.data is None:
        raise JudgeParseError(result.error or "no data")
    d = result.data
    try:
        f, c = int(d["faithfulness"]), int(d["correctness"])
    except (KeyError, TypeError, ValueError) as exc:
        raise JudgeParseError(f"missing or non-integer score: {exc}") from exc
    if not (1 <= f <= 5 and 1 <= c <= 5):
        raise JudgeParseError(f"score out of range: faithfulness={f}, correctness={c}")
    return Judgment(
        faithfulness=f,
        correctness=c,
        unsupported_claims=[str(x) for x in d.get("unsupported_claims", [])],
        missing_or_wrong=[str(x) for x in d.get("missing_or_wrong", [])],
        faithfulness_reasoning=str(d.get("faithfulness_reasoning", "")),
        correctness_reasoning=str(d.get("correctness_reasoning", "")),
    )
