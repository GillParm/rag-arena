export type ChunkOut = {
  id: string;
  rank: number;
  score: number;
  doc_id: string;
  doc_title: string;
  heading_path: string[];
  url: string;
  text: string;
  cited: boolean;
  in_context: boolean;
};

export type PipelineResult = {
  pipeline: string;
  label: string;
  answer: { text: string; citations: string[]; abstained: boolean; model: string | null } | null;
  error: string | null;
  chunks: ChunkOut[];
  rewrites: string[];
  latency: {
    retrieval_ms: number;
    generation_ms: number | null;
    total_ms: number;
    stages: Record<string, number>;
  };
  input_tokens: number;
  output_tokens: number;
  cost_usd: number;
};

export type AskResponse = {
  question: string;
  chunk_set: string;
  generation_enabled: boolean;
  results: PipelineResult[];
};

export type ArenaConfig = {
  pipelines: { id: string; label: string; description: string }[];
  chunk_sets: {
    id: string;
    chunker: string;
    chunk_count: number;
    params: Record<string, unknown>;
  }[];
  default_chunk_set: string | null;
  generation_enabled: boolean;
  generator_model: string;
  embedding_model: string;
  reranker_model: string;
  corpus: string;
};

export type BoardEntry = {
  pipeline: string;
  label: string;
  base: string;
  reranker_model: string | null;
  chunk_set: string;
  n: number;
  "recall@5": number;
  "recall@5_ci": [number, number];
  "recall@10": number;
  "mrr@10": number;
  "coverage@5": number;
  retrieval_p50_ms: number;
  retrieval_p95_ms: number;
  answered?: number;
  judged?: number;
  judge_failures?: number;
  faithfulness?: number;
  faithfulness_ci?: [number, number];
  faithful_rate?: number;
  correctness?: number;
  correctness_ci?: [number, number];
  correct_rate?: number;
  abstention_rate?: number;
  generation_p50_ms?: number;
  generation_p95_ms?: number;
  e2e_p50_ms?: number;
  e2e_p95_ms?: number;
  cost_per_1k_usd?: number;
};

export type Leaderboard = {
  run: string;
  questions: number;
  human_reviewed: number;
  headline_chunk_set: string;
  headline_rule: string;
  headline_rule_scores: Record<string, { "mean_recall@5": number; "mean_mrr@10": number }>;
  winner_rule: string;
  generator_model: string;
  judge_model: string;
  embedding_model: string;
  reranker_model: string;
  corpus: { ref: string | null; commit: string | null; documents: number | null };
  code_commit: string | null;
  headline: BoardEntry[];
  ablation: BoardEntry[];
  comparisons: {
    winner?: string;
    correctness_diff?: Record<string, [number, number, number]>;
    recall5_diff?: Record<string, [number, number, number]>;
  };
};
