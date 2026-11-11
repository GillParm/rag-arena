"""The eval item stored in data/eval/qa.jsonl, plus JSONL read and write helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Status = Literal["auto_approved", "auto_rejected", "approved", "rejected"]
QuestionType = Literal["factual", "how_to", "code", "conceptual", "troubleshooting"]

# Entries the benchmark uses. Rejected entries stay in the file for the record.
USABLE_STATUSES: frozenset[str] = frozenset({"auto_approved", "approved"})


class FilterVerdict(BaseModel):
    answerable: bool
    self_contained: bool
    unambiguous: bool
    answer_correct: bool
    useful: bool
    keep: bool
    reason: str
    model: str


class QAItem(BaseModel):
    id: str
    question: str
    answer: str
    question_type: QuestionType
    doc_id: str
    doc_url: str
    section: list[str]
    evidence: str
    evidence_start: int
    evidence_end: int
    status: Status
    human_reviewed: bool = False
    edited: bool = False
    review_note: str | None = None
    author_model: str
    corpus_ref: str
    filter: FilterVerdict | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.status in USABLE_STATUSES


def read_items(path: Path) -> list[QAItem]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [QAItem.model_validate_json(line) for line in lines if line.strip()]


def write_items(path: Path, items: list[QAItem]) -> None:
    """Write atomically, so an interrupted review never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
