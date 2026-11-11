"use client";

import { useState } from "react";

import { formatMs, formatScore } from "@/lib/format";
import { pipelineColor, pipelineLabel } from "@/lib/pipelines";
import type { BoardEntry } from "@/lib/types";

// Sequential blue ramp, light to dark (steps 100 to 650 of the reference ramp).
const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#104281"];
const CHUNKER_NAME: Record<string, string> = {
  fixed: "Fixed size",
  recursive: "Recursive",
  heading: "By heading",
};

function rampColor(t: number): string {
  const i = Math.min(RAMP.length - 1, Math.max(0, Math.round(t * (RAMP.length - 1))));
  return RAMP[i];
}

/** Recall@5 for every pipeline and chunker, the retrieval-only ablation. */
export function ChunkerHeatmap({
  entries,
  pipelines,
  chunkSets,
  headline,
  metric = "recall@5",
}: {
  entries: BoardEntry[];
  pipelines: string[];
  chunkSets: string[];
  headline: string;
  metric?: "recall@5" | "mrr@10";
}) {
  const [hover, setHover] = useState<string | null>(null);
  const cell = (p: string, c: string) => entries.find((e) => e.pipeline === p && e.chunk_set === c);
  const values = entries.filter((e) => pipelines.includes(e.pipeline)).map((e) => e[metric]);
  const lo = Math.min(...values);
  const hi = Math.max(...values);

  return (
    <div>
      <table className="w-full border-separate border-spacing-[2px] text-[13px]">
        <caption className="sr-only">{metric} by pipeline and chunker</caption>
        <thead>
          <tr>
            <th scope="col" className="text-muted w-[38%] text-left font-normal">
              Pipeline
            </th>
            {chunkSets.map((c) => (
              <th key={c} scope="col" className="text-ink-2 pb-1 text-center font-medium">
                {CHUNKER_NAME[c] ?? c}
                {c === headline && (
                  <span className="text-muted block text-[11.5px] font-normal">headline</span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {pipelines.map((p) => (
            <tr key={p}>
              <th scope="row" className="pr-2 text-left font-medium">
                <span className="flex items-center gap-2">
                  <span
                    aria-hidden
                    className="h-2.5 w-2.5 shrink-0 rounded-sm"
                    style={{ background: pipelineColor(p) }}
                  />
                  {pipelineLabel(p)}
                </span>
              </th>
              {chunkSets.map((c) => {
                const e = cell(p, c);
                if (!e)
                  return (
                    <td key={c} className="text-muted text-center">
                      n/a
                    </td>
                  );
                const v = e[metric];
                const t = hi > lo ? (v - lo) / (hi - lo) : 0.5;
                const dark = t > 0.5;
                const key = `${p}:${c}`;
                return (
                  <td
                    key={c}
                    tabIndex={0}
                    onMouseEnter={() => setHover(key)}
                    onMouseLeave={() => setHover(null)}
                    onFocus={() => setHover(key)}
                    onBlur={() => setHover(null)}
                    className="relative h-10 rounded-[4px] text-center"
                    style={{
                      background: rampColor(t),
                      outline: hover === key ? "2px solid var(--ink)" : undefined,
                    }}
                  >
                    <span className={`num font-semibold ${dark ? "text-white" : "text-[#0b0b0b]"}`}>
                      {formatScore(v)}
                    </span>
                    {hover === key && (
                      <span className="border-rule bg-surface text-ink pointer-events-none absolute top-full left-1/2 z-10 mt-1 w-max -translate-x-1/2 rounded-md border px-2.5 py-1.5 text-left text-[12px] font-normal shadow-sm">
                        <span className="num block font-semibold">
                          recall@5 {formatScore(e["recall@5"])}, MRR@10 {formatScore(e["mrr@10"])}
                        </span>
                        <span className="num text-ink-2 block">
                          retrieval median {formatMs(e.retrieval_p50_ms)}
                        </span>
                      </span>
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="text-muted mt-3 flex items-center gap-2 text-[12px]" aria-hidden>
        <span className="num">{formatScore(lo)}</span>
        <span className="flex h-2 w-32 overflow-hidden rounded-full">
          {RAMP.map((c) => (
            <span key={c} className="flex-1" style={{ background: c }} />
          ))}
        </span>
        <span className="num">{formatScore(hi)}</span>
      </div>
    </div>
  );
}
