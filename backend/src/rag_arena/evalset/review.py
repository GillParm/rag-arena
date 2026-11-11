"""Interactive review of the eval set: approve, edit or reject each entry.

The loop is written against small input/output callables so it can be tested without a
terminal; the CLI wires it to typer prompts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rag_arena.evalset.schema import QAItem, read_items, write_items

Ask = Callable[[str, str], str]  # (prompt, default) -> answer
Say = Callable[[str], None]

HELP = "[a]pprove  [e]dit  [r]eject  [s]kip  [q]uit"


def context_window(doc_text: str | None, item: QAItem, pad: int = 240) -> str:
    """The evidence with some surrounding text, evidence marked with >>> <<<."""
    if not doc_text:
        return item.evidence
    a, b = item.evidence_start, item.evidence_end
    before = doc_text[max(0, a - pad) : a]
    after = doc_text[b : b + pad]
    return f"...{before}>>>{doc_text[a:b]}<<<{after}..."


def render(item: QAItem, position: str, doc_text: str | None) -> str:
    verdict = item.filter.reason if item.filter else "n/a"
    return "\n".join(
        [
            f"[{position}] {item.id}  {item.status}  ({item.question_type})",
            f"page:     {item.doc_url}",
            f"section:  {' > '.join(item.section)}",
            f"question: {item.question}",
            f"answer:   {item.answer}",
            f"filter:   {verdict}",
            "evidence in context:",
            context_window(doc_text, item),
        ]
    )


def edit_item(item: QAItem, ask: Ask) -> QAItem:
    question = ask("question", item.question).strip() or item.question
    answer = ask("answer", item.answer).strip() or item.answer
    note = ask("note (optional)", "").strip() or None
    changed = question != item.question or answer != item.answer
    return item.model_copy(
        update={
            "question": question,
            "answer": answer,
            "status": "approved",
            "human_reviewed": True,
            "edited": item.edited or changed,
            "review_note": note or item.review_note,
        }
    )


def review(
    path: Path,
    ask: Ask,
    say: Say,
    *,
    doc_texts: dict[str, str] | None = None,
    include_reviewed: bool = False,
) -> dict[str, int]:
    """Walk the entries, saving after every decision. Returns counts per action."""
    items = read_items(path)
    counts = {"approved": 0, "edited": 0, "rejected": 0, "skipped": 0}
    queue = [i for i, it in enumerate(items) if include_reviewed or not it.human_reviewed]
    say(f"{len(queue)} entries to review. {HELP}")
    for n, idx in enumerate(queue, start=1):
        item = items[idx]
        say(render(item, f"{n}/{len(queue)}", (doc_texts or {}).get(item.doc_id)))
        while True:
            choice = ask("action", "s").strip().lower()[:1]
            if choice in {"a", "e", "r", "s", "q"}:
                break
            say(HELP)
        if choice == "q":
            break
        if choice == "s":
            counts["skipped"] += 1
            continue
        if choice == "a":
            items[idx] = item.model_copy(update={"status": "approved", "human_reviewed": True})
            counts["approved"] += 1
        elif choice == "r":
            note = ask("reason (optional)", "").strip() or None
            items[idx] = item.model_copy(
                update={"status": "rejected", "human_reviewed": True, "review_note": note}
            )
            counts["rejected"] += 1
        elif choice == "e":
            items[idx] = edit_item(item, ask)
            counts["edited"] += 1
        write_items(path, items)
    return counts
