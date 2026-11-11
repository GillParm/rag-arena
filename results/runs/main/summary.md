Headline chunker: `heading`. 100 questions. Generator `gemini-3.5-flash-lite`, judge `claude-opus-5-5`.

| Pipeline | Recall@5 | Recall@10 | MRR@10 | Faithfulness | Correctness | Retrieval p50 / p95 | End-to-end p50 / p95 | Cost per 1k queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.75 | 0.79 | 0.57 | 4.97 | 4.15 | 0.2 / 0.2 ms | 5,382 / 8,307 ms | $0.576 |
| Dense | 0.75 | 0.87 | 0.54 | 4.96 | 4.08 | 12 / 17 ms | 4,857 / 9,010 ms | $0.577 |
| Hybrid (RRF) | 0.81 | 0.87 | 0.60 | 4.97 | 4.25 | 12 / 14 ms | 2,811 / 9,709 ms | $0.595 |
| Hybrid + rerank | 0.75 | 0.86 | 0.61 | 4.96 | 4.13 | 4,984 / 5,649 ms | 6,639 / 12,674 ms | $0.629 |
| Hybrid + rewrite | 0.82 | 0.93 | 0.62 | 4.98 | 4.24 | 3,898 / 8,080 ms | 5,905 / 12,426 ms | $0.816 |

Chunker ablation (retrieval only):

| Pipeline | Chunker | Recall@5 | Recall@10 | MRR@10 | Retrieval p50 |
| --- | --- | ---: | ---: | ---: | ---: |
| BM25 | fixed | 0.76 | 0.84 | 0.57 | 0.2 ms |
| BM25 | recursive | 0.74 | 0.80 | 0.56 | 0.2 ms |
| BM25 | heading | 0.75 | 0.79 | 0.57 | 0.2 ms |
| Dense | fixed | 0.67 | 0.77 | 0.51 | 14 ms |
| Dense | recursive | 0.67 | 0.78 | 0.51 | 15 ms |
| Dense | heading | 0.75 | 0.87 | 0.54 | 12 ms |
| Hybrid (RRF) | fixed | 0.79 | 0.87 | 0.59 | 12 ms |
| Hybrid (RRF) | recursive | 0.73 | 0.87 | 0.57 | 16 ms |
| Hybrid (RRF) | heading | 0.81 | 0.87 | 0.60 | 12 ms |
| Hybrid + rerank | fixed | 0.80 | 0.86 | 0.64 | 4,794 ms |
| Hybrid + rerank | recursive | 0.77 | 0.84 | 0.61 | 4,393 ms |
| Hybrid + rerank | heading | 0.75 | 0.86 | 0.61 | 4,984 ms |
| Hybrid + rewrite | fixed | 0.80 | 0.89 | 0.66 | 3,906 ms |
| Hybrid + rewrite | recursive | 0.84 | 0.90 | 0.60 | 3,896 ms |
| Hybrid + rewrite | heading | 0.82 | 0.93 | 0.62 | 3,898 ms |
| Hybrid + MiniLM rerank | fixed | 0.73 | 0.80 | 0.55 | 759 ms |
| Hybrid + MiniLM rerank | recursive | 0.72 | 0.82 | 0.53 | 707 ms |
| Hybrid + MiniLM rerank | heading | 0.77 | 0.86 | 0.58 | 778 ms |
| Hybrid + Jina rerank | fixed | 0.79 | 0.83 | 0.56 | 865 ms |
| Hybrid + Jina rerank | recursive | 0.72 | 0.81 | 0.55 | 754 ms |
| Hybrid + Jina rerank | heading | 0.70 | 0.81 | 0.58 | 1,007 ms |
