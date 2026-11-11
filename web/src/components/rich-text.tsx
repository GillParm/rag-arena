import { Fragment, type ReactNode } from "react";

/**
 * A deliberately small markdown renderer for answers and chunk text: fenced code, inline code,
 * bold, bullet lists, paragraphs and [n] citations. It builds React elements, never HTML
 * strings, so model output cannot inject markup.
 */

type Block =
  | { kind: "code"; lang: string; text: string }
  | { kind: "list"; items: string[] }
  | { kind: "para"; text: string };

export function parseBlocks(source: string): Block[] {
  const blocks: Block[] = [];
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  let para: string[] = [];
  let list: string[] | null = null;

  const flushPara = () => {
    if (para.length) blocks.push({ kind: "para", text: para.join(" ") });
    para = [];
  };
  const flushList = () => {
    if (list?.length) blocks.push({ kind: "list", items: list });
    list = null;
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const fence = line.match(/^\s*(```|~~~)\s*([\w+-]*)/);
    if (fence) {
      flushPara();
      flushList();
      const marker = fence[1];
      const body: string[] = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(marker)) {
        body.push(lines[i]);
        i++;
      }
      blocks.push({ kind: "code", lang: fence[2] ?? "", text: body.join("\n") });
      continue;
    }
    const heading = line.match(/^#{1,6}\s+(.*)$/);
    if (heading) {
      flushPara();
      flushList();
      blocks.push({ kind: "para", text: `**${heading[1].trim()}**` });
      continue;
    }
    // mkdocs admonitions ("/// tip" ... "///"): keep the label, drop the fences.
    const admonition = line.match(/^\/{3,4}\s*(?:tab\s*\|\s*)?(.*)$/);
    if (admonition) {
      flushPara();
      flushList();
      const label = admonition[1].trim();
      if (label)
        blocks.push({ kind: "para", text: `**${label[0].toUpperCase()}${label.slice(1)}**` });
      continue;
    }
    const bullet = line.match(/^\s*(?:[-*]|\d+\.)\s+(.*)$/);
    if (bullet) {
      flushPara();
      list = list ?? [];
      list.push(bullet[1]);
      continue;
    }
    if (!line.trim()) {
      flushPara();
      flushList();
      continue;
    }
    flushList();
    para.push(line.trim());
  }
  flushPara();
  flushList();
  return blocks;
}

export type CitationRenderer = (num: number, key: string) => ReactNode;

const INLINE = /(`[^`]+`|\*\*[^*]+\*\*|\[\d+(?:\s*,\s*\d+)*\])/g;

function renderInline(text: string, cite?: CitationRenderer): ReactNode[] {
  return text.split(INLINE).map((part, i) => {
    if (!part) return null;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 1) {
      return (
        <code key={i} className="bg-surface-2 rounded px-1 py-px font-mono text-[0.86em]">
          {part.slice(1, -1)}
        </code>
      );
    }
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    const citation = part.match(/^\[(\d+(?:\s*,\s*\d+)*)\]$/);
    if (citation && cite) {
      return (
        <Fragment key={i}>
          {citation[1].split(",").map((n) => cite(Number(n.trim()), `${i}-${n.trim()}`))}
        </Fragment>
      );
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

export function RichText({
  source,
  cite,
  className = "",
}: {
  source: string;
  cite?: CitationRenderer;
  className?: string;
}) {
  const blocks = parseBlocks(source);
  return (
    <div className={`space-y-3 ${className}`}>
      {blocks.map((b, i) => {
        if (b.kind === "code") {
          return (
            <pre
              key={i}
              className="border-rule bg-surface-2 overflow-x-auto rounded-md border p-3 font-mono text-[12.5px] leading-[1.55]"
            >
              <code>{b.text}</code>
            </pre>
          );
        }
        if (b.kind === "list") {
          return (
            <ul key={i} className="list-disc space-y-1 pl-5">
              {b.items.map((item, j) => (
                <li key={j}>{renderInline(item, cite)}</li>
              ))}
            </ul>
          );
        }
        return <p key={i}>{renderInline(b.text, cite)}</p>;
      })}
    </div>
  );
}
