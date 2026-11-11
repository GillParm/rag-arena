# Results

Everything here comes from real runs; nothing is edited by hand. `rag-arena bench report`
rebuilds the summaries from the raw records without calling any model.

## `runs/main/`: the published run

| File | What it holds |
| --- | --- |
| `config.json` | Chunk sets, pipelines, pipeline settings, the headline chunk set and the scores that chose it |
| `retrieval.jsonl` | One record per (pipeline, chunk set, question): ranked chunk ids with scores and character spans, per-stage timings, rewrites and their token usage |
| `rewrites.jsonl` | The query rewrites per question, shared by every chunk set |
| `generation.jsonl` | One answer per (pipeline, question) on the headline chunk set: text, passages shown, citations, tokens, latency |
| `judge.jsonl` | The judge's structured output per answer, with token usage |
| `summary.json` | Every metric, confidence intervals, paired comparisons, models and corpus versions; the API serves this as the leaderboard |
| `summary.md` | The markdown tables used in the top-level README |
| `summary.csv`, `ablation.csv` | The headline table and the chunker ablation as CSV |
| `per_question.csv` | Rank of the first relevant chunk, scores and latency for every question and pipeline |

## `spend/anthropic.jsonl`

Every Anthropic API call made for this project (eval-set writing, filtering, judging and a
few development calls), with token counts and the list-price cost. `rag-arena spend` totals
it by purpose.
