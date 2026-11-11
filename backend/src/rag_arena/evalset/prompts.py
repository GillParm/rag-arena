"""Prompts and JSON schemas for writing and filtering eval questions."""

from __future__ import annotations

from typing import Any

GENERATE_SYSTEM = """\
You write questions for an evaluation set that measures how well search systems find answers
in the FastAPI documentation. Each question should read like something a developer would type
into a docs search box or ask a colleague."""

GENERATE_USER = """\
Documentation page: {title}
Section: {path}

<section>
{text}
</section>

Write one question that this section answers.

Rules:
- A developer who has not read this section would plausibly ask it while using FastAPI.
- It must be fully answerable from this section alone, with one clear correct answer.
- It must stand on its own: no references to "this section", "the example" or "the docs".
- Use your own words. Do not copy distinctive phrases or the heading from the section, so a
  search engine cannot match the question on exact wording alone. Names of FastAPI classes,
  functions and parameters are fine where a developer would naturally use them.
- Prefer questions about behaviour, how to do something or why something works over trivia
  such as version numbers, people or where a link points.

Also give:
- answer: a reference answer of one to three sentences, complete and correct according to
  the section.
- evidence: one contiguous excerpt from the section, copied character for character, at most
  400 characters, that contains the key fact the answer needs.
- question_type: factual, how_to, code, conceptual or troubleshooting.

If the section has nothing worth asking about (only links, a list of pages or boilerplate),
set skip to true and leave the other fields as empty strings."""

GENERATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skip": {"type": "boolean"},
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "evidence": {"type": "string"},
        "question_type": {
            "type": "string",
            "enum": ["factual", "how_to", "code", "conceptual", "troubleshooting", ""],
        },
    },
    "required": ["skip", "question", "answer", "evidence", "question_type"],
    "additionalProperties": False,
}

FILTER_SYSTEM = """\
You review candidate questions for a retrieval evaluation set built from the FastAPI
documentation. Be strict: keeping a flawed question does more harm than dropping a good one."""

FILTER_USER = """\
<section page="{title}" heading="{path}">
{text}
</section>

<candidate>
Question: {question}
Reference answer: {answer}
Evidence quoted from the section: {evidence}
</candidate>

Check the candidate:
- answerable: the section alone fully answers the question.
- self_contained: the question makes sense to someone who has not seen the section; it does
  not say "this section", "the example above" or similar.
- unambiguous: there is one clear correct answer; a reasonable reader would not read the
  question in a different way that changes the answer.
- answer_correct: the reference answer is correct and complete according to the section, and
  the evidence supports it.
- useful: a real FastAPI user might ask this; it is not trivia about the documentation itself.

Give reason as one sentence naming the main problem, or "ok" if there is none."""

FILTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "self_contained": {"type": "boolean"},
        "unambiguous": {"type": "boolean"},
        "answer_correct": {"type": "boolean"},
        "useful": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": [
        "answerable",
        "self_contained",
        "unambiguous",
        "answer_correct",
        "useful",
        "reason",
    ],
    "additionalProperties": False,
}
