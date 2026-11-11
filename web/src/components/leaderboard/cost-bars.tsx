"use client";

import { useState } from "react";

import { formatUsd } from "@/lib/format";
import { pipelineColor } from "@/lib/pipelines";
import type { BoardEntry } from "@/lib/types";

/** Horizontal bars, one per pipeline, sorted cheapest first; value at the bar's tip. */
export function CostBars({ entries }: { entries: BoardEntry[] }) {
  const [hover, setHover] = useState<string | null>(null);
  const rows = entries
    .filter((e) => e.cost_per_1k_usd != null)
    .sort((a, b) => (a.cost_per_1k_usd as number) - (b.cost_per_1k_usd as number));
  const max = Math.max(...rows.map((r) => r.cost_per_1k_usd as number));
  return (
    <ul className="space-y-3">
      {rows.map((r) => {
        const v = r.cost_per_1k_usd as number;
        return (
          <li
            key={r.pipeline}
            tabIndex={0}
            onMouseEnter={() => setHover(r.pipeline)}
            onMouseLeave={() => setHover(null)}
            onFocus={() => setHover(r.pipeline)}
            onBlur={() => setHover(null)}
            className="grid grid-cols-[minmax(7.5rem,9rem)_1fr] items-center gap-3 text-[13px]"
          >
            <span className="font-medium">{r.label}</span>
            <span className="flex items-center gap-2">
              <span
                className="h-[14px] rounded-r-[4px]"
                style={{
                  width: `${Math.max((v / max) * 78, 1)}%`,
                  background: pipelineColor(r.pipeline),
                  opacity: hover && hover !== r.pipeline ? 0.45 : 1,
                }}
              />
              <span className="num text-ink shrink-0">{formatUsd(v, 3)}</span>
            </span>
          </li>
        );
      })}
    </ul>
  );
}
