# Judge rubric

The judge scores every generated answer on two independent scales in one call. It sees the
question, the reference answer and its evidence quote from the docs, the passages the
generator was given, and the generated answer. The judge reads this file at run time; its
sha256 is stored with every judgment.

Score each scale from 1 to 5. Explain first, then score.

## Faithfulness: is the answer supported by the passages it was given?

Judge only against the numbered passages shown to the generator, not against the reference
answer or your own knowledge of FastAPI. A claim that is true but absent from the passages is
unsupported.

| Score | Meaning |
| --- | --- |
| 5 | Every claim is stated in or directly implied by the passages. |
| 4 | All substantive claims are supported; at most a minor detail (a wording choice, an obvious connective inference) goes beyond the passages. |
| 3 | The main point is supported, but at least one substantive claim is not. |
| 2 | Several substantive claims are unsupported, or one contradicts the passages. |
| 1 | The answer is mostly unsupported by, or contradicts, the passages. |

- An answer that says the documentation provided does not cover the question, and makes no
  other claims, scores 5: it asserts nothing unsupported.
- Code in the answer counts as claims. Code copied or minimally adapted from a passage is
  supported; invented parameters, functions or behaviour are not.
- Citation numbers are not scored here. Judge what the answer says, not which passage number
  it attaches.

## Correctness: does the answer correctly and completely answer the question?

Judge against the reference answer and its evidence quote. Other correct details do not
lower the score, and wording does not need to match.

| Score | Meaning |
| --- | --- |
| 5 | Correct and complete: it contains the key facts of the reference answer and nothing that contradicts it. |
| 4 | Correct, but misses or blurs a minor detail of the reference answer. |
| 3 | Partly correct: it gets part of the key fact right but misses a key part or adds a wrong detail that matters. |
| 2 | Mostly incorrect or very incomplete, though related to the right topic. |
| 1 | Wrong, off topic, or no answer (including saying the documentation does not cover it). |

## Reporting

- The benchmark reports the mean of each score and the share of answers scoring 4 or 5.
- Abstentions are reported separately as well, since they score 5 on faithfulness and 1 on
  correctness by design.

## Judge setup

- Model: `claude-opus-5-5`, a different model family from the generator
  (`gemini-3.5-flash-lite`), to avoid a judge preferring its own family's answers.
- Effort `low`, structured outputs with a fixed JSON schema, reasoning fields before scores.
- Requests go through the Message Batches API. A refusal or invalid result is recorded as a
  failed judgment and excluded from the means, and the count of failures is reported.
