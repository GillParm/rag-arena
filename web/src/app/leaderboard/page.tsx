import type { Metadata } from "next";
import { connection } from "next/server";
import { Suspense, type ReactNode } from "react";

import { CostBars } from "@/components/leaderboard/cost-bars";
import { ChunkerHeatmap } from "@/components/leaderboard/heatmap";
import { ResultsTable } from "@/components/leaderboard/results-table";
import { QualityLatency } from "@/components/leaderboard/scatter";
import { getLeaderboard } from "@/lib/api";
import { formatMs, formatScore } from "@/lib/format";
import { PIPELINE_ORDER, pipelineLabel } from "@/lib/pipelines";
import type { BoardEntry, Leaderboard } from "@/lib/types";

export const metadata: Metadata = { title: "Leaderboard" };

const REPO = "https://github.com/GillParm/rag-arena";

function Panel({ title, note, children }: { title: string; note: string; children: ReactNode }) {
  return (
    <section className="border-rule bg-surface min-w-0 rounded-lg border p-5">
      <h2 className="text-[16px] font-semibold tracking-tight">{title}</h2>
      <p className="text-ink-2 mt-1 max-w-[60ch] text-[13.5px] leading-snug">{note}</p>
      <div className="mt-4">{children}</div>
    </section>
  );
}

function verdict(board: Leaderboard): string[] {
  const rows = board.headline;
  const winner = rows.find((e) => e.pipeline === board.comparisons.winner);
  if (!winner) return [];
  const others = rows
    .filter((e) => e.pipeline !== winner.pipeline)
    .sort((a, b) => (b.correctness ?? 0) - (a.correctness ?? 0));
  const runnerUp = others[0];
  const lines: string[] = [];
  const diff = runnerUp ? board.comparisons.correctness_diff?.[runnerUp.pipeline] : undefined;
  if (runnerUp && diff) {
    const [d, lo, hi] = diff;
    lines.push(
      lo > 0
        ? `${winner.label} gave the most correct answers: ${formatScore(winner.correctness)} out of 5 on average, ${formatScore(d)} ahead of ${runnerUp.label} on the same questions (95% CI ${formatScore(lo)} to ${formatScore(hi)}).`
        : `${winner.label} had the highest mean correctness, ${formatScore(winner.correctness)} out of 5, but its ${formatScore(d)} lead over ${runnerUp.label} is within noise for ${board.questions} questions (95% CI ${formatScore(lo)} to ${formatScore(hi)}).`,
    );
  }
  const bestRecall = [...rows].sort((a, b) => b["recall@5"] - a["recall@5"])[0];
  const byRetrieval = [...rows].sort((a, b) => a.retrieval_p50_ms - b.retrieval_p50_ms);
  const fastest = byRetrieval[0];
  const slowest = byRetrieval[byRetrieval.length - 1];
  lines.push(
    `${bestRecall.label} put the evidence in its top five most often (recall@5 ${formatScore(bestRecall["recall@5"])}). Median retrieval took from ${formatMs(fastest.retrieval_p50_ms)} (${fastest.label}) to ${formatMs(slowest.retrieval_p50_ms)} (${slowest.label}).`,
  );
  return lines;
}

function Board({ board }: { board: Leaderboard }) {
  const order = (e: BoardEntry) =>
    PIPELINE_ORDER.indexOf(e.pipeline as (typeof PIPELINE_ORDER)[number]);
  const headline = [...board.headline].sort((a, b) => order(a) - order(b));
  const chunkSets = ["fixed", "recursive", "heading"].filter((c) =>
    board.ablation.some((e) => e.chunk_set === c),
  );
  const variants = board.ablation.filter(
    (e) => e.base === "hybrid_rerank" && e.chunk_set === board.headline_chunk_set,
  );
  const lines = verdict(board);
  const genMedians = headline.map((e) => e.generation_p50_ms).filter((v): v is number => v != null);
  const genSpread =
    genMedians.length > 1 && Math.max(...genMedians) > 1.5 * Math.min(...genMedians)
      ? ([Math.min(...genMedians), Math.max(...genMedians)] as const)
      : null;

  return (
    <>
      <div className="max-w-[70ch] space-y-2 text-[17px] leading-relaxed">
        {lines.map((l) => (
          <p key={l}>{l}</p>
        ))}
      </div>
      <p className="text-ink-2 mt-3 text-[13.5px]">
        {board.questions} questions over the FastAPI docs {board.corpus.ref}, chunked by{" "}
        {board.headline_chunk_set === "heading" ? "heading" : board.headline_chunk_set}. Answers by{" "}
        {board.generator_model}, judged by {board.judge_model}.{" "}
        {board.human_reviewed === 0
          ? "The questions were written and filtered by a model and have not been reviewed by a person yet."
          : `${board.human_reviewed} questions were reviewed by a person.`}
      </p>

      <div className="mt-8">
        <ResultsTable entries={headline} winner={board.comparisons.winner} />
      </div>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Panel
          title="Answer quality against retrieval time"
          note="Mean correctness per pipeline with its 95% bootstrap interval, against median retrieval latency. Up and to the left is better."
        >
          <QualityLatency entries={headline} />
        </Panel>
        <Panel
          title="Which chunker helps each pipeline"
          note="Recall@5 for every pipeline and chunker, retrieval only. The headline chunker was chosen before generation by the highest mean recall@5."
        >
          <ChunkerHeatmap
            entries={board.ablation}
            pipelines={[...PIPELINE_ORDER]}
            chunkSets={chunkSets}
            headline={board.headline_chunk_set}
          />
        </Panel>
        <Panel
          title="Cost per 1,000 questions"
          note={`Generator and rewrite tokens at ${board.generator_model} list prices. Embedding, BM25 and reranking run locally and add no API cost.`}
        >
          <CostBars entries={headline} />
        </Panel>
        {variants.length > 1 && (
          <Panel
            title="Reranker size against speed"
            note="The same hybrid candidates re-scored by three cross-encoders on CPU. Retrieval only, headline chunker."
          >
            <table className="w-full text-[13.5px]">
              <thead>
                <tr className="border-rule text-ink-2 border-b text-left text-[12.5px]">
                  <th scope="col" className="py-2 pr-3 font-medium">
                    Cross-encoder
                  </th>
                  <th scope="col" className="py-2 pr-3 font-medium">
                    Recall@5
                  </th>
                  <th scope="col" className="py-2 pr-3 font-medium">
                    MRR@10
                  </th>
                  <th scope="col" className="py-2 font-medium">
                    Retrieval p50
                  </th>
                </tr>
              </thead>
              <tbody>
                {variants.map((v) => (
                  <tr key={v.pipeline} className="border-rule border-b last:border-b-0">
                    <th
                      scope="row"
                      className="py-2 pr-3 text-left font-medium [overflow-wrap:anywhere]"
                    >
                      {v.reranker_model ?? board.reranker_model}
                      {v.pipeline === "hybrid_rerank" && (
                        <span className="text-muted block text-[12px] font-normal">
                          used in the headline run
                        </span>
                      )}
                    </th>
                    <td className="num py-2 pr-3">{formatScore(v["recall@5"])}</td>
                    <td className="num py-2 pr-3">{formatScore(v["mrr@10"])}</td>
                    <td className="num py-2">{formatMs(v.retrieval_p50_ms)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Panel>
        )}
      </div>

      <section className="text-ink-2 mt-10 max-w-[72ch] text-[14.5px] leading-relaxed">
        <h2 className="text-ink text-[16px] font-semibold">How to read this</h2>
        <ul className="mt-2 list-disc space-y-1.5 pl-5">
          <li>
            A retrieved chunk counts as relevant when it covers at least half of the question&apos;s
            evidence quote (or the quote covers half of it). Recall@k is the share of questions with
            a relevant chunk in the top k.
          </li>
          <li>
            Faithfulness asks whether the answer stays within the passages it was given; correctness
            compares it with the reference answer. Both follow the{" "}
            <a
              className="text-ink underline underline-offset-2"
              href={`${REPO}/blob/main/docs/rubric.md`}
            >
              written rubric
            </a>
            .
          </li>
          <li>
            Retrieval latency was measured on one laptop CPU, one query at a time. The reranker and
            the query rewriter dominate their pipelines&apos; time.
          </li>
          {genSpread && (
            <li>
              End-to-end latency adds one call to the shared generator. Its median ranged from{" "}
              {formatMs(genSpread[0])} to {formatMs(genSpread[1])} across pipelines although the
              model and prompt were identical, because the free tier&apos;s speed changed while the
              run was in progress. Compare pipelines on retrieval latency.
            </li>
          )}
          <li>
            One corpus, {board.questions} questions and one judge model: treat small gaps as ties.
            Raw outputs for every question are in{" "}
            <a className="text-ink underline underline-offset-2" href={`${REPO}/tree/main/results`}>
              results/
            </a>
            .
          </li>
        </ul>
      </section>
      <p className="sr-only">
        Pipelines: {headline.map((e) => pipelineLabel(e.pipeline)).join(", ")}
      </p>
    </>
  );
}

async function BoardLoader() {
  await connection();
  const board = await getLeaderboard();
  if (!board.ok) {
    return (
      <div role="alert" className="border-rule bg-surface max-w-[640px] rounded-md border p-4">
        <p className="font-semibold">No results to show yet.</p>
        <p className="text-ink-2 mt-1 text-[14px]">{board.error}</p>
      </div>
    );
  }
  return <Board board={board.data} />;
}

export default function LeaderboardPage() {
  return (
    <div className="mx-auto max-w-[1440px] px-4 py-8 sm:px-6 sm:py-10">
      <h1 className="max-w-[24ch] text-[clamp(1.9rem,3.6vw,2.9rem)] leading-[1.05] font-bold tracking-[-0.02em]">
        Which pipeline answers best?
      </h1>
      <div className="mt-4">
        <Suspense fallback={<p className="text-ink-2 text-[14px]">Loading results…</p>}>
          <BoardLoader />
        </Suspense>
      </div>
    </div>
  );
}
