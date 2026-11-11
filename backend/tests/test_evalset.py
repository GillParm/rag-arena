from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from rag_arena.corpus.loader import Document
from rag_arena.embedding import HashEmbedder
from rag_arena.evalset.build import (
    Candidate,
    apply_verdicts,
    build_eval_set,
    drop_near_duplicates,
    filter_requests,
    generation_requests,
    select,
    to_candidates,
)
from rag_arena.evalset.evidence import locate
from rag_arena.evalset.review import context_window, review
from rag_arena.evalset.sampling import sample_sections, section_spans
from rag_arena.evalset.schema import QAItem, read_items, write_items
from rag_arena.llm.claude import ClaudeJSON, ClaudeResult, SpendLedger
from tests.fakes import FakeAnthropic

TEXT = "Intro.\n\nUse `Depends`   to declare\na dependency. **Bold** claim here.\n"


def test_locate_exact_whitespace_and_markup_insensitive() -> None:
    assert locate(TEXT, "Intro.") == (0, 6)
    start, end = locate(TEXT, "Use `Depends` to declare a dependency.") or (0, 0)
    assert TEXT[start:end] == "Use `Depends`   to declare\na dependency."
    start, end = locate(TEXT, "Bold claim here") or (0, 0)
    assert TEXT[start:end] == "Bold** claim here"
    assert locate(TEXT, "not in the text") is None
    assert locate(TEXT, "   ") is None


def test_locate_prefers_the_hinted_range() -> None:
    text = "same words. other. same words."
    assert locate(text, "same words.", hint_start=12) == (19, 30)


def test_section_spans_stop_at_the_next_heading(mini_docs: list[Document]) -> None:
    doc = next(d for d in mini_docs if d.id == "tutorial/first-steps")
    spans = section_spans(doc)
    assert [s.title for s, _, _ in spans] == ["First Steps", "Run it", "Check it", "Recap"]
    run_it = next((a, b) for s, a, b in spans if s.title == "Run it")
    assert doc.text[run_it[0] : run_it[1]].rstrip().endswith("```")


def test_sampling_is_deterministic_and_spread(mini_docs: list[Document]) -> None:
    a = sample_sections(mini_docs, 3, min_chars=50)
    b = sample_sections(mini_docs, 3, min_chars=50)
    assert [s.key for s in a] == [s.key for s in b]
    assert len({s.doc.id for s in a[:2]}) == 2  # first round takes one section per page


@pytest.fixture
def samples(mini_docs: list[Document]) -> list:  # type: ignore[type-arg]
    return sample_sections(mini_docs, 4, min_chars=50)


def _gen_result(cid: str, sample_text: str, **over: object) -> ClaudeResult:
    sentence = sample_text.strip().splitlines()[-1]
    data = {
        "skip": False,
        "question": "How do I start the development server?",
        "answer": "Run fastapi dev.",
        "evidence": sentence,
        "question_type": "how_to",
    }
    data.update(over)
    return ClaudeResult(cid, data, None, model="claude-opus-5-5")


def test_to_candidates_locates_evidence_and_logs_drops(samples: list) -> None:  # type: ignore[type-arg]
    results = {f"gen-{i:03d}": _gen_result(f"gen-{i:03d}", s.text) for i, s in enumerate(samples)}
    results["gen-001"] = _gen_result("gen-001", samples[1].text, evidence="invented quote")
    results["gen-002"] = _gen_result("gen-002", samples[2].text, skip=True)
    del results["gen-003"]
    kept, dropped = to_candidates(samples, results)
    assert len(kept) == 1
    c = kept[0]
    assert c.sample.doc.text[c.start : c.end] == c.evidence
    reasons = sorted(d["reason"] for d in dropped)
    assert reasons == [
        "evidence not found in the document",
        "model skipped the section",
        "no result",
    ]


def test_generation_requests_carry_the_section(samples: list) -> None:  # type: ignore[type-arg]
    reqs = generation_requests(samples)
    assert reqs[0].custom_id == "gen-000"
    assert samples[0].text.strip() in reqs[0].user
    assert reqs[0].schema["additionalProperties"] is False


def _cand(samples: list, i: int, question: str) -> Candidate:  # type: ignore[type-arg]
    s = samples[i]
    return Candidate(s.key, s, question, "a", "e", s.start, s.start + 1, "factual")


def test_near_duplicates_are_dropped(samples: list) -> None:  # type: ignore[type-arg]
    cands = [
        _cand(samples, 0, "how do I run the dev server"),
        _cand(samples, 1, "How do I run the dev server?"),
        _cand(samples, 2, "what is pydantic used for"),
    ]
    kept = drop_near_duplicates(cands, HashEmbedder(256))
    assert [c.question for c in kept] == [
        "how do I run the dev server",
        "what is pydantic used for",
    ]
    assert cands[1].dropped == "near-duplicate question"


def test_verdicts_require_every_check(samples: list) -> None:  # type: ignore[type-arg]
    cands = [_cand(samples, 0, "q0"), _cand(samples, 1, "q1")]
    reqs = filter_requests(cands)
    ok = dict.fromkeys(
        ("answerable", "self_contained", "unambiguous", "answer_correct", "useful"), True
    )
    results = {
        reqs[0].custom_id: ClaudeResult(reqs[0].custom_id, {**ok, "reason": "ok"}, None, model="m"),
        reqs[1].custom_id: ClaudeResult(
            reqs[1].custom_id, {**ok, "unambiguous": False, "reason": "vague"}, None, model="m"
        ),
    }
    apply_verdicts(cands, reqs, results)
    assert cands[0].verdict is not None
    assert cands[0].verdict.keep
    assert cands[1].verdict is not None
    assert not cands[1].verdict.keep
    assert cands[1].dropped == "rejected by the filter"
    assert select(cands, 10) == [cands[0]]


def test_build_eval_set_end_to_end_offline(mini_docs: list[Document], tmp_path: Path) -> None:
    def reply(params: dict[str, object]) -> object:
        user = str(params["messages"][0]["content"])  # type: ignore[index]
        if "<candidate>" in user:
            return {
                "answerable": True,
                "self_contained": True,
                "unambiguous": True,
                "answer_correct": True,
                "useful": True,
                "reason": "ok",
            }
        section = user.split("<section>\n", 1)[1].split("\n</section>", 1)[0]
        line = next(ln for ln in section.splitlines()[::-1] if ln.strip())
        return {
            "skip": False,
            "question": f"Question about {line[:30]}?",
            "answer": "An answer.",
            "evidence": line,
            "question_type": "factual",
        }

    fake = FakeAnthropic(reply)
    claude = ClaudeJSON("x", "claude-opus-5-5", SpendLedger(tmp_path / "s.jsonl"), client=fake)  # type: ignore[arg-type]
    items = build_eval_set(
        mini_docs,
        claude,
        HashEmbedder(256),
        tmp_path,
        corpus_ref="t",
        n_candidates=4,
        target=3,
        mode="sync",
        min_section_chars=50,
        log=lambda _: None,
    )
    assert 0 < len(items) <= 3
    assert [i.id for i in items] == [f"q{n:03d}" for n in range(1, len(items) + 1)]
    assert all(i.status == "auto_approved" and not i.human_reviewed for i in items)
    texts = {d.id: d.text for d in mini_docs}
    assert all(texts[i.doc_id][i.evidence_start : i.evidence_end] == i.evidence for i in items)
    assert read_items(tmp_path / "qa.jsonl") == items
    assert (tmp_path / "candidates.jsonl").exists()


def _item(n: int) -> QAItem:
    return QAItem(
        id=f"q{n:03d}",
        question=f"Q{n}?",
        answer=f"A{n}.",
        question_type="factual",
        doc_id="d",
        doc_url="u",
        section=["S"],
        evidence="ev",
        evidence_start=2,
        evidence_end=4,
        status="auto_approved",
        author_model="m",
        corpus_ref="r",
    )


def _scripted(answers: list[str]) -> Iterator[str]:
    yield from answers


def test_review_loop_approves_edits_rejects_and_saves(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    write_items(path, [_item(1), _item(2), _item(3), _item(4)])
    script = _scripted(["a", "e", "Better Q2?", "", "fixed wording", "x", "r", "off topic", "q"])
    out: list[str] = []
    counts = review(path, ask=lambda _p, _d: next(script), say=out.append)
    assert counts == {"approved": 1, "edited": 1, "rejected": 1, "skipped": 0}
    items = read_items(path)
    assert items[0].status == "approved"
    assert items[0].human_reviewed
    assert items[1].question == "Better Q2?"
    assert items[1].answer == "A2."
    assert items[1].edited
    assert items[1].review_note == "fixed wording"
    assert items[2].status == "rejected"
    assert items[2].review_note == "off topic"
    assert items[3].status == "auto_approved"
    assert not items[3].human_reviewed
    assert any("[a]pprove" in line for line in out)
    # A second pass only shows entries nobody has reviewed yet.
    script2 = _scripted(["s"])
    counts2 = review(path, ask=lambda _p, _d: next(script2), say=lambda _: None)
    assert counts2["skipped"] == 1


def test_context_window_marks_the_evidence() -> None:
    assert context_window("0123456789", _item(1), pad=2) == "...01>>>23<<<45..."
    assert context_window(None, _item(1)) == "ev"
