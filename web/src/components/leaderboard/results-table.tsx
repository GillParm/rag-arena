import { formatMs, formatPct, formatScore, formatUsd } from "@/lib/format";
import { pipelineColor } from "@/lib/pipelines";
import type { BoardEntry } from "@/lib/types";

function Meter({ value, max }: { value: number | undefined; max: number }) {
  if (value == null) return null;
  return (
    <span aria-hidden className="bg-rule mt-1 block h-1 w-full max-w-24 rounded-full">
      <span
        className="bg-ink-2 block h-1 rounded-full"
        style={{ width: `${Math.max(0, Math.min(1, value / max)) * 100}%` }}
      />
    </span>
  );
}

function Best({ on }: { on: boolean }) {
  return on ? <span className="sr-only"> (best)</span> : null;
}

/** The headline metrics table. It is also the table view for every chart on the page. */
export function ResultsTable({ entries, winner }: { entries: BoardEntry[]; winner?: string }) {
  const best = (key: keyof BoardEntry, low = false) => {
    const vals = entries
      .map((e) => e[key] as number | undefined)
      .filter((v) => v != null) as number[];
    return low ? Math.min(...vals) : Math.max(...vals);
  };
  const bestR5 = best("recall@5");
  const bestCorr = best("correctness");
  const bestFaith = best("faithfulness");
  const bestMrr = best("mrr@10");

  const cellBest = (v: number | undefined, b: number) =>
    v != null && Math.abs(v - b) < 1e-9 ? "font-bold text-ink" : "";

  return (
    <div className="border-rule bg-surface relative overflow-x-auto rounded-lg border">
      <table className="w-full min-w-[960px] text-[14px]">
        <caption className="sr-only">
          Headline results per pipeline. Best value in each quality column is bold.
        </caption>
        <thead>
          <tr className="border-rule text-ink-2 border-b text-left align-bottom text-[12.5px]">
            <th scope="col" className="px-4 py-3 font-medium">
              Pipeline
            </th>
            <th scope="col" className="px-3 py-3 font-medium">
              Recall@5
            </th>
            <th scope="col" className="px-3 py-3 font-medium">
              Recall@10
            </th>
            <th scope="col" className="px-3 py-3 font-medium">
              MRR@10
            </th>
            <th scope="col" className="px-3 py-3 font-medium">
              Faithfulness<span className="text-muted block font-normal">1 to 5</span>
            </th>
            <th scope="col" className="px-3 py-3 font-medium">
              Correctness<span className="text-muted block font-normal">1 to 5, 95% CI</span>
            </th>
            <th scope="col" className="px-3 py-3 font-medium">
              Retrieval<span className="text-muted block font-normal">p50 / p95</span>
            </th>
            <th scope="col" className="px-3 py-3 font-medium">
              End to end<span className="text-muted block font-normal">p50 / p95</span>
            </th>
            <th scope="col" className="px-4 py-3 text-right font-medium">
              Cost per<span className="text-muted block font-normal">1,000 queries</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {entries.map((e) => (
            <tr key={e.pipeline} className="border-rule border-b align-top last:border-b-0">
              <th scope="row" className="px-4 py-3 text-left font-semibold">
                <span className="flex items-center gap-2.5">
                  <span
                    aria-hidden
                    className="h-3 w-3 shrink-0 rounded-sm"
                    style={{ background: pipelineColor(e.pipeline) }}
                  />
                  {e.label}
                </span>
                {e.pipeline === winner && (
                  <span className="text-ink-2 mt-1 ml-[22px] block text-[12px] font-normal">
                    Highest correctness
                  </span>
                )}
              </th>
              <td className={`num px-3 py-3 ${cellBest(e["recall@5"], bestR5)}`}>
                {formatScore(e["recall@5"])}
                <Best on={e["recall@5"] === bestR5} />
                <Meter value={e["recall@5"]} max={1} />
              </td>
              <td className="num px-3 py-3">{formatScore(e["recall@10"])}</td>
              <td className={`num px-3 py-3 ${cellBest(e["mrr@10"], bestMrr)}`}>
                {formatScore(e["mrr@10"])}
                <Best on={e["mrr@10"] === bestMrr} />
              </td>
              <td className={`num px-3 py-3 ${cellBest(e.faithfulness, bestFaith)}`}>
                {formatScore(e.faithfulness)}
                <span className="text-muted block text-[12px] font-normal">
                  {formatPct(e.faithful_rate)} scored 4+
                </span>
              </td>
              <td className={`num px-3 py-3 ${cellBest(e.correctness, bestCorr)}`}>
                {formatScore(e.correctness)}
                <Best on={e.correctness === bestCorr} />
                <span className="text-muted block text-[12px] font-normal">
                  {formatScore(e.correctness_ci?.[0])} to {formatScore(e.correctness_ci?.[1])}
                </span>
                <Meter value={e.correctness != null ? e.correctness - 1 : undefined} max={4} />
              </td>
              <td className="num px-3 py-3 whitespace-nowrap">
                {formatMs(e.retrieval_p50_ms)}
                <span className="text-muted block text-[12px]">{formatMs(e.retrieval_p95_ms)}</span>
              </td>
              <td className="num px-3 py-3 whitespace-nowrap">
                {formatMs(e.e2e_p50_ms)}
                <span className="text-muted block text-[12px]">{formatMs(e.e2e_p95_ms)}</span>
              </td>
              <td className="num px-4 py-3 text-right">{formatUsd(e.cost_per_1k_usd, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
