"use client";

import { useRef, useState, type FormEvent } from "react";

import { PIPELINE_ORDER } from "@/lib/pipelines";
import type { ArenaConfig, AskResponse } from "@/lib/types";

import { Lane } from "./lane";

const EXAMPLES = [
  "How do I make a query parameter optional?",
  "Can a dependency run code after the response is sent?",
  "How do I return a 201 status code from a POST endpoint?",
  "Why is my background task not running when I raise an exception?",
];

const CHUNKER_HELP: Record<string, string> = {
  heading: "One chunk per doc section",
  recursive: "Split on paragraphs, then lines",
  fixed: "1,200-character windows",
};

type Status = { kind: "idle" } | { kind: "loading" } | { kind: "error"; message: string };

export function Arena({ config }: { config: ArenaConfig }) {
  const [question, setQuestion] = useState("");
  const [chunkSet, setChunkSet] = useState(config.default_chunk_set ?? config.chunk_sets[0]?.id);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [response, setResponse] = useState<AskResponse | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const descriptions = Object.fromEntries(config.pipelines.map((p) => [p.id, p.description]));

  async function ask(q: string) {
    const text = q.trim();
    if (text.length < 3) {
      setStatus({ kind: "error", message: "Type a question of at least three characters." });
      inputRef.current?.focus();
      return;
    }
    setStatus({ kind: "loading" });
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question: text, chunk_set: chunkSet }),
      });
      const body = (await res.json()) as AskResponse | { detail?: unknown };
      if (!res.ok) {
        const detail = (body as { detail?: unknown }).detail;
        const message =
          typeof detail === "string"
            ? detail
            : res.status === 422
              ? "Questions need 3 to 500 characters."
              : `The request failed with status ${res.status}.`;
        setStatus({ kind: "error", message });
        return;
      }
      const data = body as AskResponse;
      data.results.sort(
        (a, b) =>
          PIPELINE_ORDER.indexOf(a.pipeline as (typeof PIPELINE_ORDER)[number]) -
          PIPELINE_ORDER.indexOf(b.pipeline as (typeof PIPELINE_ORDER)[number]),
      );
      setResponse(data);
      setStatus({ kind: "idle" });
    } catch {
      setStatus({ kind: "error", message: "Couldn't reach the server. Check your connection." });
    }
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void ask(question);
  }

  const loading = status.kind === "loading";

  return (
    <div>
      <form onSubmit={onSubmit} className="max-w-[860px]">
        <label htmlFor="question" className="text-ink-2 text-[15px] font-medium">
          Your question about FastAPI
        </label>
        <div className="mt-2 flex flex-col gap-2 sm:flex-row sm:items-stretch">
          <textarea
            id="question"
            ref={inputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void ask(question);
              }
            }}
            rows={2}
            maxLength={500}
            placeholder="How do I declare a dependency that needs the current user?"
            className="border-rule-strong bg-surface text-ink placeholder:text-muted focus:border-focus min-h-[3.25rem] flex-1 resize-y rounded-md border px-3.5 py-3 text-[16px] leading-snug focus:outline-none"
          />
          <button
            type="submit"
            disabled={loading}
            className="bg-ink text-surface rounded-md px-5 py-3 text-[15px] font-semibold hover:opacity-90 disabled:cursor-progress disabled:opacity-60"
          >
            {loading ? "Asking…" : "Ask all five"}
          </button>
        </div>

        <fieldset className="mt-4 flex flex-wrap items-center gap-x-3 gap-y-2">
          <legend className="sr-only">Chunking strategy</legend>
          <span aria-hidden className="text-ink-2 text-[13.5px]">
            Chunks
          </span>
          {config.chunk_sets.map((cs) => (
            <label
              key={cs.id}
              className={`has-[:focus-visible]:outline-focus cursor-pointer rounded-full border px-3 py-1 text-[13.5px] has-[:focus-visible]:outline-2 ${
                chunkSet === cs.id
                  ? "border-ink bg-ink text-surface"
                  : "border-rule-strong bg-surface text-ink-2 hover:border-ink-2"
              }`}
            >
              <input
                type="radio"
                name="chunk_set"
                value={cs.id}
                checked={chunkSet === cs.id}
                onChange={() => setChunkSet(cs.id)}
                className="sr-only"
              />
              {cs.chunker === "heading"
                ? "By heading"
                : cs.chunker === "fixed"
                  ? "Fixed size"
                  : "Recursive"}
              <span className="sr-only">: {CHUNKER_HELP[cs.chunker] ?? cs.id}</span>
            </label>
          ))}
          {chunkSet && (
            <span className="text-muted text-[13px]">{CHUNKER_HELP[chunkSet] ?? ""}</span>
          )}
        </fieldset>

        {!response && (
          <div className="mt-5">
            <p className="text-ink-2 text-[13.5px]">Or try one of these</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => {
                    setQuestion(ex);
                    void ask(ex);
                  }}
                  className="border-rule bg-surface text-ink hover:border-rule-strong rounded-md border px-3 py-1.5 text-left text-[13.5px]"
                >
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}
      </form>

      <div aria-live="polite" className="mt-4 min-h-6 text-[14px]">
        {status.kind === "error" && <p className="text-danger">{status.message}</p>}
        {loading && (
          <p className="text-ink-2">
            Running five pipelines. The reranker runs on CPU, so its column takes a few seconds.
          </p>
        )}
        {!config.generation_enabled && (
          <p className="text-ink-2">
            The API has no Gemini key, so it shows retrieval results without answers.
          </p>
        )}
      </div>

      {response && (
        <div
          className={`mt-2 transition-opacity duration-200 ${loading ? "opacity-50" : "opacity-100"}`}
        >
          <p className="text-ink-2 text-[13.5px]">
            Results for <span className="text-ink font-medium">“{response.question}”</span>, chunked{" "}
            {response.chunk_set === "heading"
              ? "by heading"
              : `with the ${response.chunk_set} chunker`}
            . Every column used the same generator; only retrieval differs.
          </p>
          <div className="-mx-4 mt-3 overflow-x-auto px-4 pb-2 sm:-mx-6 sm:px-6">
            <div className="border-rule bg-rule grid min-w-[1200px] grid-cols-5 gap-px overflow-hidden rounded-lg border max-[700px]:min-w-0 max-[700px]:grid-cols-1">
              {response.results.map((r) => (
                <Lane key={r.pipeline} result={r} description={descriptions[r.pipeline]} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
