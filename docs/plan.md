# Plan

rag-arena runs five retrieval pipelines over the same documentation corpus, feeds each one's
top chunks to the same answer generator, and scores the results. This file is the plan written
before any code; the README records what actually happened.

## Goals

1. Compare BM25, dense, hybrid (RRF), hybrid + reranker and hybrid + query rewriting on one
   corpus with one eval set, and report real numbers.
2. Make every number reproducible from a clean checkout: pinned corpus, committed eval set,
   committed raw outputs, one command per stage.
3. Ship it as a small, deployable product: a FastAPI backend, a Next.js UI, Docker Compose for
   local use and Cloud Run-ready images.

Non-goals: tuning each pipeline to its best possible score, or comparing generator models. The
generator is fixed so that differences come from retrieval.

## Decisions

### Corpus

- FastAPI English docs, `docs/en/docs/**/*.md`, pinned to release tag `0.162.4`
  (commit `e10178c18d8dc2dcc21a70e14152253abc4a4e2f`). FastAPI is MIT-licensed; the download
  script checks the LICENSE file at that ref and refuses to continue if it is not MIT.
- The docs pull code samples in with include directives (`{* ../../docs_src/x.py hl[3] *}` and
  `{!../../docs_src/x.py!}`). Much of the useful content lives in those samples, so the loader
  inlines them as fenced code blocks. The processed text is the canonical document: chunk
  offsets and eval-set evidence offsets both point into it.
- Excluded pages, because they are generated, empty or not documentation: `release-notes.md`
  (a 736 KB changelog that would swamp the index), `reference/**` (mkdocstrings stubs whose
  content lives in docstrings), `fastapi-people.md`, `external-links.md`, `management.md`,
  `newsletter.md`, `_llm-test.md`, `agent-banner.md`, `translation-banner.md`.
- Heading anchors (`{ #path-parameters }`) are stripped from the text and kept as metadata, so
  every chunk can link back to the live docs page and section.
- The raw download is gitignored. `data/manifest.json` (paths and sha256 of every processed
  document) is committed so a re-download can be verified.

### Chunkers

All three chunkers return character offsets into the processed document, so a chunk is
`text[start:end]` and relevance can be computed by span overlap regardless of chunker.

| Chunker | Rule | Default size |
| --- | --- | --- |
| `fixed` | Fixed window with overlap, snapped to whitespace | 1,200 chars, 200 overlap |
| `recursive` | Split on a separator hierarchy (headings, blank lines, lines, sentences, spaces) and merge up to the target size | 1,200 chars |
| `heading` | One chunk per markdown section (headings inside code fences ignored); long sections are split with the recursive splitter; tiny sections merge into the next | up to 1,600 chars |

The heading chunker prepends the heading path (`First Steps > Path parameters`) to the text it
indexes. That is the point of the strategy, so the ablation measures it as part of the chunker.
Every chunk, whatever the chunker, stores its heading path and docs URL for display.

### Storage

- Postgres 17 + pgvector (`pgvector/pgvector:pg17`) on port 5433. psycopg 3 with explicit SQL
  and numbered `.sql` migrations applied by a small runner (no ORM).
- `chunks` is list-partitioned by chunk set, one partition per chunker config, each with its
  own HNSW index (`vector_cosine_ops`). Rebuilding one chunker drops one partition.
- BM25 (`bm25s` with a Snowball stemmer) is built from the chunk rows in Postgres and cached on
  disk next to the raw corpus.

### Retrieval pipelines

| Pipeline | Steps |
| --- | --- |
| `bm25` | bm25s top 10 |
| `dense` | `BAAI/bge-small-en-v1.5` via fastembed (384 dims, local), pgvector cosine top 10 |
| `hybrid` | BM25 top 50 + dense top 50, reciprocal rank fusion with k = 60, top 10 |
| `hybrid_rerank` | hybrid top 30 re-scored by a fastembed cross-encoder, top 10 |
| `hybrid_rewrite` | the generator model writes 3 rewrites; BM25 and dense run for the original and each rewrite; all 8 lists fused with RRF, top 10 |

The top 5 chunks go to the generator for every pipeline.

### Generation and judging

- Generator: `gemini-3.5-flash-lite` for answers and rewrites, minimal thinking. Context
  passages are numbered; the answer cites them inline as `[n]` and the API maps those to chunk
  IDs. Same prompt and settings for every pipeline.
- Judge: `claude-opus-5-5`, a different model family from the generator to avoid
  self-preference. One structured-output call per answer scores faithfulness and correctness
  (1 to 5) against `docs/rubric.md`. Judging runs through the Message Batches API (half price)
  with a synchronous fallback.

### Eval set

- About 130 candidate questions written by `claude-opus-5-5` from sampled doc sections (at most
  two per page, sections of at least 400 characters). Each candidate records its source
  document, a reference answer and a verbatim evidence quote. The quote is located in the
  processed text to get character offsets; candidates whose quote cannot be found are dropped.
- An automatic filter (a second Claude call) rejects questions that are unanswerable from the
  section, ambiguous without the section in view, or trivia. Survivors are trimmed to 100 and
  stored as `data/eval/qa.jsonl` with `status: auto_approved, human_reviewed: false`.
- `rag-arena eval review` is an interactive CLI to approve, edit or reject entries; it writes
  `status` and `human_reviewed: true` back to the same file.

### Metrics

- Relevance: a retrieved chunk is relevant if its span overlaps the gold evidence span by at
  least half of the shorter of the two. This works across chunkers and ignores one-character
  touches.
- recall@5 and recall@10: share of questions with at least one relevant chunk in the top k
  (one gold span per question, so this is also the hit rate). MRR@10.
- Faithfulness and correctness: mean judge score (1 to 5) and the share scoring 4 or 5.
- Latency p50/p95, measured in-process, for retrieval and end to end separately. Rate-limit
  backoff waits are excluded; the rewrite pipeline's retrieval time includes its LLM call.
- Cost per 1,000 queries: generator and rewrite token usage times Gemini list prices
  ($0.30 / $2.50 per million input / output tokens, checked 2026-11-10). The runs use the free
  tier, priced as if paid. Local models (embeddings, reranker, BM25) have no API cost; their
  CPU time shows up in latency, not cost.

### Benchmark protocol

1. Retrieval for every pipeline x chunker x question (cheap, local, plus cached rewrites).
2. The headline chunker is chosen by a rule fixed in advance: highest mean recall@5 across the
   five pipelines. Ties go to the higher mean MRR@10.
3. Generation and judging for every pipeline on the headline chunker only (500 answers). The
   other chunkers get retrieval metrics in an ablation table. This keeps judge spend and free
   tier usage within budget.
4. Every stage writes append-only JSONL keyed by (stage, pipeline, chunker, question), so an
   interrupted run resumes where it stopped.
5. `results/` holds the raw records, `summary.json`, `summary.csv` and the markdown tables that
   go into the README.

### API and UI

- FastAPI on port 8300: `GET /health`, `GET /api/config`, `POST /api/ask` (runs the selected
  pipelines concurrently and returns answers, cited chunks, latency and cost),
  `GET /api/leaderboard`, `GET /api/chunks/{id}`.
- Next.js 16 + TypeScript + Tailwind on port 3300: an Ask page with every pipeline's answer side
  by side, and a Leaderboard page with the metrics table and charts.

### Delivery

- pytest runs offline with fixtures and a deterministic fake embedder; DB tests are marked
  `db` and run against a pgvector service container in CI; live API tests are marked `live`
  and skip without keys.
- CI: ruff, strict mypy, pytest (unit and DB), ESLint, Prettier, tsc, `next build`, both Docker
  builds and gitleaks.
- Dockerfiles listen on `$PORT` for Cloud Run. Production needs a managed Postgres with
  pgvector (Cloud SQL for PostgreSQL supports it); documented in `docs/deployment.md`.

## Budget

Anthropic spend is capped at $20. Every Anthropic call appends its token usage and cost to
`results/spend/anthropic.jsonl`, and `rag-arena spend` totals it. Rough estimate at Opus 5.5
prices ($4 / $20 per million tokens, half price in batches): eval-set generation and filtering
about $2.50, judging 500 answers about $5, development calls about $1.

## Milestones

1. Plan (this file) and scaffold: repo layout, uv project, Next.js app, Makefile, env template.
2. Corpus download with licence check, document loader, the three chunkers.
3. Postgres schema and migrations, indexing (chunks, embeddings, HNSW, BM25 cache).
4. Retrieval pipelines and RRF.
5. Eval set generation, automatic filter, review CLI.
6. Generator, judge, metrics, resumable benchmark runner.
7. API.
8. UI: Ask and Leaderboard.
9. Full benchmark run; commit results.
10. Docker, CI, README with results and a real screenshot.

## Risks

- Gemini free-tier rate limits are not published per model. The runner uses a client-side
  rate limiter, bounded concurrency, exponential backoff on 429 and 503, and resumable caching.
- LLM-written questions tend to reuse the source's wording, which favours BM25. The generation
  prompt asks for questions phrased the way a user would ask them; the README calls out the
  remaining bias.
- One corpus, about 100 questions and one judge model: the README reports these limits next
  to the numbers rather than in a footnote.
