export const PIPELINE_ORDER = [
  "bm25",
  "dense",
  "hybrid",
  "hybrid_rerank",
  "hybrid_rewrite",
] as const;

export const PIPELINE_LABEL: Record<string, string> = {
  bm25: "BM25",
  dense: "Dense",
  hybrid: "Hybrid",
  hybrid_rerank: "Hybrid + rerank",
  hybrid_rewrite: "Hybrid + rewrite",
  hybrid_rerank_minilm: "Hybrid + MiniLM rerank",
  hybrid_rerank_jina: "Hybrid + Jina rerank",
};

/** CSS colour for a pipeline. Reranker variants share the rerank pipeline's hue. */
export function pipelineColor(id: string): string {
  const base = id.startsWith("hybrid_rerank") ? "hybrid_rerank" : id;
  return `var(--p-${base}, var(--muted))`;
}

export function pipelineLabel(id: string): string {
  return PIPELINE_LABEL[id] ?? id;
}
