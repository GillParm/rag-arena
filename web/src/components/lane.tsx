"use client";

import { useState } from "react";

import { formatMs, formatUsd } from "@/lib/format";
import { pipelineColor } from "@/lib/pipelines";
import type { ChunkOut, PipelineResult } from "@/lib/types";

import { RichText } from "./rich-text";

const CONTEXT = 5;

function Citation({
  num,
  chunk,
  color,
  active,
  onActivate,
}: {
  num: number;
  chunk: ChunkOut | undefined;
  color: string;
  active: boolean;
  onActivate: (id: string | null, scroll: boolean) => void;
}) {
  if (!chunk) {
    return <span className="text-muted">[{num}]</span>;
  }
  return (
    <button
      type="button"
      aria-label={`Source ${num}: ${chunk.heading_path.join(", ")}`}
      onMouseEnter={() => onActivate(chunk.id, false)}
      onMouseLeave={() => onActivate(null, false)}
      onFocus={() => onActivate(chunk.id, false)}
      onBlur={() => onActivate(null, false)}
      onClick={() => onActivate(chunk.id, true)}
      className="bg-surface-2 text-ink mx-px inline-flex h-[1.35em] min-w-[1.35em] translate-y-[-0.1em] items-center justify-center rounded-[3px] px-1 align-middle text-[0.72em] leading-none font-semibold"
      style={{
        boxShadow: `inset 0 -2px 0 ${color}`,
        outline: active ? `2px solid ${color}` : undefined,
      }}
    >
      {num}
    </button>
  );
}

function Source({
  chunk,
  color,
  highlighted,
}: {
  chunk: ChunkOut;
  color: string;
  highlighted: boolean;
}) {
  const [open, setOpen] = useState(false);
  const path = chunk.heading_path.length > 1 ? chunk.heading_path.slice(1) : chunk.heading_path;
  return (
    <li
      id={`src-${chunk.id}`}
      className={`relative rounded-md border py-2.5 pr-3 pl-4 transition-colors duration-150 ${
        highlighted ? "border-rule-strong bg-mark" : "border-rule bg-surface"
      }`}
    >
      {chunk.cited && (
        <span
          aria-hidden
          className="absolute top-2 bottom-2 left-0 w-[3px] rounded-r"
          style={{ background: color }}
        />
      )}
      <div className="flex items-baseline gap-2">
        <span className="num text-ink-2 w-4 shrink-0 text-[13px] font-semibold">{chunk.rank}</span>
        <div className="min-w-0 flex-1">
          <a
            href={chunk.url}
            className="text-ink decoration-rule-strong hover:decoration-ink text-[13.5px] leading-snug font-medium underline underline-offset-2"
          >
            {chunk.doc_title}
          </a>
          {path.length > 0 && path[0] !== chunk.doc_title && (
            <p className="text-ink-2 truncate text-[12.5px]" title={path.join(" / ")}>
              {path.join(" / ")}
            </p>
          )}
          <p className="text-muted mt-0.5 text-[12px]">
            {chunk.cited
              ? "Cited in the answer"
              : chunk.in_context
                ? "Given to the model"
                : "Retrieved, not used"}
          </p>
        </div>
      </div>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="text-ink-2 decoration-rule-strong hover:text-ink mt-1.5 ml-6 text-[12.5px] underline underline-offset-2"
      >
        {open ? "Hide passage" : "Show passage"}
      </button>
      {open && (
        <RichText
          source={chunk.text}
          className="border-rule text-ink-2 mt-2 ml-6 max-h-80 overflow-y-auto border-l pl-3 text-[13px] leading-relaxed"
        />
      )}
    </li>
  );
}

export function Lane({ result, description }: { result: PipelineResult; description?: string }) {
  const color = pipelineColor(result.pipeline);
  const [active, setActive] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const context = result.chunks.filter((c) => c.rank <= CONTEXT);
  const rest = result.chunks.filter((c) => c.rank > CONTEXT);

  const activate = (id: string | null, scroll: boolean) => {
    setActive(id);
    if (id && scroll) {
      document
        .getElementById(`src-${id}`)
        ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  };

  return (
    <section
      aria-labelledby={`lane-${result.pipeline}`}
      className="bg-surface flex min-w-0 flex-col border-t-[3px]"
      style={{ borderTopColor: color }}
    >
      <header className="border-rule border-b px-4 pt-3 pb-3">
        <h2 id={`lane-${result.pipeline}`} className="text-[16px] font-semibold tracking-tight">
          {result.label}
        </h2>
        {description && (
          <p className="text-muted mt-0.5 text-[12.5px] leading-snug">{description}</p>
        )}
        <dl className="mt-2.5 grid grid-cols-3 gap-2 text-[12px]">
          <div>
            <dt className="text-muted">Retrieval</dt>
            <dd className="num text-[14px] font-semibold">
              {formatMs(result.latency.retrieval_ms)}
            </dd>
          </div>
          <div>
            <dt className="text-muted">Total</dt>
            <dd className="num text-[14px] font-semibold">{formatMs(result.latency.total_ms)}</dd>
          </div>
          <div>
            <dt className="text-muted">Cost</dt>
            <dd className="num text-[14px] font-semibold">{formatUsd(result.cost_usd, 5)}</dd>
          </div>
        </dl>
      </header>

      <div className="flex-1 px-4 py-4">
        {result.error && (
          <p role="alert" className="text-danger mb-3 text-[13.5px]">
            {result.error}
          </p>
        )}
        {result.answer ? (
          <RichText
            source={result.answer.text}
            className={`text-[15px] leading-[1.6] ${result.answer.abstained ? "text-ink-2 italic" : ""}`}
            cite={(num, key) => (
              <Citation
                key={key}
                num={num}
                chunk={context[num - 1]}
                color={color}
                active={active === context[num - 1]?.id}
                onActivate={activate}
              />
            )}
          />
        ) : (
          !result.error && (
            <p className="text-muted text-[14px]">
              Answer generation is off; showing retrieval only.
            </p>
          )
        )}

        {result.rewrites.length > 0 && (
          <div className="mt-4">
            <p className="text-ink-2 text-[12.5px] font-medium">Also searched for</p>
            <ul className="text-ink-2 mt-1 space-y-0.5 text-[12.5px]">
              {result.rewrites.map((r) => (
                <li key={r} className="before:text-muted before:mr-1.5 before:content-['–']">
                  {r}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="border-rule bg-surface-2 border-t px-3 py-3">
        <h3 className="text-ink-2 px-1 text-[13px] font-semibold">Passages given to the model</h3>
        <ol className="mt-2 space-y-2">
          {context.map((c) => (
            <Source key={c.id} chunk={c} color={color} highlighted={active === c.id} />
          ))}
        </ol>
        {rest.length > 0 && (
          <>
            <button
              type="button"
              aria-expanded={showAll}
              onClick={() => setShowAll((s) => !s)}
              className="text-ink-2 decoration-rule-strong hover:text-ink mt-2 px-1 text-[12.5px] underline underline-offset-2"
            >
              {showAll ? "Hide ranks 6 to 10" : `Show ranks 6 to ${5 + rest.length}`}
            </button>
            {showAll && (
              <ol className="mt-2 space-y-2">
                {rest.map((c) => (
                  <Source key={c.id} chunk={c} color={color} highlighted={false} />
                ))}
              </ol>
            )}
          </>
        )}
      </div>
    </section>
  );
}
