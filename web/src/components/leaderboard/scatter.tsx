"use client";

import { useState } from "react";

import { formatMs, formatScore } from "@/lib/format";
import { pipelineColor } from "@/lib/pipelines";
import type { BoardEntry } from "@/lib/types";

import { useWidth } from "./use-width";

const H = 300;
const M = { top: 16, right: 24, bottom: 44, left: 48 };

function niceLogTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++) {
    for (const m of [1, 2, 5]) {
      const v = m * 10 ** p;
      if (v >= lo && v <= hi) out.push(v);
    }
  }
  return out;
}

/** Answer correctness against end-to-end median latency; whiskers are 95% bootstrap CIs. */
export function QualityLatency({ entries }: { entries: BoardEntry[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);
  const pts = entries.filter((e) => e.correctness != null && e.e2e_p50_ms != null);
  if (!pts.length) return null;

  const xs = pts.map((e) => e.e2e_p50_ms as number);
  const lows = pts.map((e) => e.correctness_ci?.[0] ?? (e.correctness as number));
  const highs = pts.map((e) => e.correctness_ci?.[1] ?? (e.correctness as number));
  const x0 = Math.min(...xs) / 1.6;
  const x1 = Math.max(...xs) * 1.6;
  const y0 = Math.max(1, Math.floor((Math.min(...lows) - 0.1) * 4) / 4);
  const y1 = Math.min(5, Math.ceil((Math.max(...highs) + 0.1) * 4) / 4);
  const iw = Math.max(width - M.left - M.right, 120);
  const ih = H - M.top - M.bottom;
  const sx = (v: number) =>
    M.left + ((Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0))) * iw;
  const sy = (v: number) => M.top + (1 - (v - y0) / (y1 - y0)) * ih;
  const yTicks: number[] = [];
  for (let v = y0; v <= y1 + 1e-9; v += 0.25) yTicks.push(Number(v.toFixed(2)));
  const xTicks = niceLogTicks(x0, x1);
  const active = pts.find((p) => p.pipeline === hover);

  return (
    <div ref={ref} className="relative">
      <svg
        width={width}
        height={H}
        role="img"
        aria-label="Answer correctness against end-to-end median latency for each pipeline"
      >
        {yTicks.map((v) => (
          <g key={`y${v}`}>
            <line x1={M.left} x2={M.left + iw} y1={sy(v)} y2={sy(v)} stroke="var(--grid)" />
            <text
              x={M.left - 8}
              y={sy(v)}
              dy="0.32em"
              textAnchor="end"
              className="num fill-muted text-[11px]"
            >
              {v.toFixed(2)}
            </text>
          </g>
        ))}
        {xTicks.map((v) => (
          <g key={`x${v}`}>
            <line x1={sx(v)} x2={sx(v)} y1={M.top} y2={M.top + ih} stroke="var(--grid)" />
            <text
              x={sx(v)}
              y={M.top + ih + 18}
              textAnchor="middle"
              className="num fill-muted text-[11px]"
            >
              {formatMs(v)}
            </text>
          </g>
        ))}
        <text x={M.left + iw / 2} y={H - 6} textAnchor="middle" className="fill-ink-2 text-[12px]">
          End-to-end latency, median (log scale)
        </text>
        <text
          transform={`translate(13 ${M.top + ih / 2}) rotate(-90)`}
          textAnchor="middle"
          className="fill-ink-2 text-[12px]"
        >
          Correctness, mean of 1 to 5
        </text>
        {pts.map((e) => {
          const cx = sx(e.e2e_p50_ms as number);
          const cy = sy(e.correctness as number);
          const color = pipelineColor(e.pipeline);
          const lo = e.correctness_ci?.[0];
          const hi = e.correctness_ci?.[1];
          const right = cx < M.left + iw * 0.72;
          return (
            <g
              key={e.pipeline}
              tabIndex={0}
              role="button"
              aria-label={`${e.label}: correctness ${formatScore(e.correctness)}, median ${formatMs(e.e2e_p50_ms)}`}
              onMouseEnter={() => setHover(e.pipeline)}
              onMouseLeave={() => setHover(null)}
              onFocus={() => setHover(e.pipeline)}
              onBlur={() => setHover(null)}
              className="cursor-default focus:outline-none"
            >
              {lo != null && hi != null && (
                <line
                  x1={cx}
                  x2={cx}
                  y1={sy(lo)}
                  y2={sy(hi)}
                  stroke={color}
                  strokeWidth={2}
                  strokeLinecap="round"
                  opacity={0.55}
                />
              )}
              <circle cx={cx} cy={cy} r={14} fill="transparent" />
              <circle
                cx={cx}
                cy={cy}
                r={hover === e.pipeline ? 7 : 5.5}
                fill={color}
                stroke="var(--surface)"
                strokeWidth={2}
              />
              <text
                x={right ? cx + 11 : cx - 11}
                y={cy}
                dy="0.32em"
                textAnchor={right ? "start" : "end"}
                className="fill-ink text-[12.5px] font-medium"
              >
                {e.label}
              </text>
            </g>
          );
        })}
      </svg>
      {active && (
        <div
          className="border-rule bg-surface pointer-events-none absolute z-10 rounded-md border px-3 py-2 text-[12.5px] shadow-sm"
          style={{
            left: Math.min(sx(active.e2e_p50_ms as number) + 12, width - 200),
            top: Math.max(sy(active.correctness as number) - 70, 0),
          }}
        >
          <p className="num text-[14px] font-semibold">
            {formatScore(active.correctness)} correctness
          </p>
          <p className="num text-ink-2">
            95% CI {formatScore(active.correctness_ci?.[0])} to{" "}
            {formatScore(active.correctness_ci?.[1])}
          </p>
          <p className="num text-ink-2">
            {formatMs(active.e2e_p50_ms)} median, {formatMs(active.e2e_p95_ms)} p95
          </p>
          <p className="text-ink-2 mt-1 flex items-center gap-1.5">
            <span
              aria-hidden
              className="inline-block h-0.5 w-3"
              style={{ background: pipelineColor(active.pipeline) }}
            />
            {active.label}
          </p>
        </div>
      )}
    </div>
  );
}
