export function formatMs(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "n/a";
  if (ms < 10) return `${ms.toFixed(1)} ms`;
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10_000 ? 2 : 1)} s`;
}

export function formatUsd(usd: number | null | undefined, digits = 4): string {
  if (usd == null || Number.isNaN(usd)) return "n/a";
  return `$${usd.toFixed(digits)}`;
}

export function formatScore(x: number | null | undefined, digits = 2): string {
  if (x == null || Number.isNaN(x)) return "n/a";
  return x.toFixed(digits);
}

export function formatPct(x: number | null | undefined): string {
  if (x == null || Number.isNaN(x)) return "n/a";
  return `${Math.round(x * 100)}%`;
}
