"use client";

import { useState } from "react";

import { formatMs, formatScore } from "@/lib/format";
import { pipelineColor } from "@/lib/pipelines";
import type { BoardEntry } from "@/lib/types";

import { useWidth } from "./use-width";

const H = 300;
const M = { top: 16, right: 28, bottom: 46, left: 64 };
const LABEL_H = 15;

function niceLogTicks(lo: number, hi: number): number[] {
  const out: number[] = [];
  for (let p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++) {
    const v = 10 ** p;
    if (v >= lo && v <= hi) out.push(v);
  }
  return out;
}

type Placed = { e: BoardEntry; cx: number; cy: number; lx: number; ly: number; right: boolean };

/** Put each label beside its dot, flipping sides or stepping down when two would overlap. */
function placeLabels(points: { e: BoardEntry; cx: number; cy: number }[], width: number): Placed[] {
  const placed: Placed[] = [];
  const charW = 6.6;
  const box = (p: Placed) => {
    const w = p.e.label.length * charW;
    const x0 = p.right ? p.lx : p.lx - w;
    return { x0, x1: x0 + w, y0: p.ly - LABEL_H / 2, y1: p.ly + LABEL_H / 2 };
  };
  const clashes = (p: Placed) =>
    placed.some((q) => {
      const a = box(p);
      const b = box(q);
      return a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;
    });
  for (const pt of [...points].sort((a, b) => a.cy - b.cy)) {
    const fitsRight = pt.cx + 11 + pt.e.label.length * charW < width - 4;
    let candidate: Placed | null = null;
    for (const shift of [0, LABEL_H, -LABEL_H, 2 * LABEL_H]) {
      for (const right of fitsRight ? [true, false] : [false, true]) {
        const c = { ...pt, right, lx: right ? pt.cx + 11 : pt.cx - 11, ly: pt.cy + shift };
        if (!clashes(c)) {
          candidate = c;
          break;
        }
      }
      if (candidate) break;
    }
    placed.push(candidate ?? { ...pt, right: fitsRight, lx: pt.cx + 11, ly: pt.cy });
  }
  return placed;
}

/**
 * Mean answer correctness (with 95% bootstrap whiskers) against median retrieval latency.
 * Retrieval latency is the part each pipeline controls; generation used one shared model.
 */
export function QualityLatency({ entries }: { entries: BoardEntry[] }) {
  const [ref, width] = useWidth<HTMLDivElement>();
  const [hover, setHover] = useState<string | null>(null);
  const pts = entries.filter((e) => e.correctness != null && e.retrieval_p50_ms > 0);
  if (!pts.length) return null;

  const xs = pts.map((e) => e.retrieval_p50_ms);
  const lows = pts.map((e) => e.correctness_ci?.[0] ?? (e.correctness as number));
  const highs = pts.map((e) => e.correctness_ci?.[1] ?? (e.correctness as number));
  const x0 = Math.min(...xs) / 3;
  const x1 = Math.max(...xs) * 3;
  const y0 = Math.max(1, Math.floor((Math.min(...lows) - 0.05) * 4) / 4);
  const y1 = Math.min(5, Math.ceil((Math.max(...highs) + 0.05) * 4) / 4);
  const iw = Math.max(width - M.left - M.right, 120);
  const ih = H - M.top - M.bottom;
  const sx = (v: number) =>
    M.left + ((Math.log(v) - Math.log(x0)) / (Math.log(x1) - Math.log(x0))) * iw;
  const sy = (v: number) => M.top + (1 - (v - y0) / (y1 - y0)) * ih;
  const yTicks: number[] = [];
  for (let v = y0; v <= y1 + 1e-9; v += 0.25) yTicks.push(Number(v.toFixed(2)));
  const xTicks = niceLogTicks(x0, x1);
  const active = pts.find((p) => p.pipeline === hover);
  const placed = placeLabels(
    pts.map((e) => ({ e, cx: sx(e.retrieval_p50_ms), cy: sy(e.correctness as number) })),
    M.left + iw,
  );

  return (
    <div ref={ref} className="relative">
      <svg
        width={width}
        height={H}
        role="img"
        aria-label="Answer correctness against median retrieval latency for each pipeline"
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
          Retrieval latency, median (log scale)
        </text>
        <text
          transform={`translate(14 ${M.top + ih / 2}) rotate(-90)`}
          textAnchor="middle"
          className="fill-ink-2 text-[12px]"
        >
          Correctness (1 to 5)
        </text>
        {placed.map(({ e, cx, cy, lx, ly, right }) => {
          const color = pipelineColor(e.pipeline);
          const lo = e.correctness_ci?.[0];
          const hi = e.correctness_ci?.[1];
          return (
            <g
              key={e.pipeline}
              tabIndex={0}
              role="button"
              aria-label={`${e.label}: correctness ${formatScore(e.correctness)}, retrieval median ${formatMs(e.retrieval_p50_ms)}`}
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
                  opacity={0.5}
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
              {Math.abs(ly - cy) > 2 && (
                <line
                  x1={right ? cx + 6 : cx - 6}
                  y1={cy}
                  x2={right ? lx - 2 : lx + 2}
                  y2={ly}
                  stroke="var(--rule-strong)"
                />
              )}
              <text
                x={lx}
                y={ly}
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
            left: Math.min(sx(active.retrieval_p50_ms) + 12, width - 220),
            top: Math.max(sy(active.correctness as number) - 86, 0),
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
            Retrieval {formatMs(active.retrieval_p50_ms)} median,{" "}
            {formatMs(active.retrieval_p95_ms)} p95
          </p>
          <p className="num text-ink-2">End to end {formatMs(active.e2e_p50_ms)} median</p>
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
