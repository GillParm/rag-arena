import "server-only";

import type { ArenaConfig, Leaderboard } from "./types";

export function apiUrl(path: string): string {
  const base = process.env.API_URL ?? "http://localhost:8300";
  return `${base.replace(/\/$/, "")}${path}`;
}

export type Fetched<T> = { ok: true; data: T } | { ok: false; error: string };

async function getJson<T>(path: string): Promise<Fetched<T>> {
  try {
    const res = await fetch(apiUrl(path), { cache: "no-store" });
    if (!res.ok) {
      const body = (await res.json().catch(() => null)) as { detail?: string } | null;
      return { ok: false, error: body?.detail ?? `The API returned ${res.status}.` };
    }
    return { ok: true, data: (await res.json()) as T };
  } catch {
    return { ok: false, error: `Can't reach the API at ${apiUrl("")}. Start it and reload.` };
  }
}

export const getConfig = () => getJson<ArenaConfig>("/api/config");
export const getLeaderboard = () => getJson<Leaderboard>("/api/leaderboard");
