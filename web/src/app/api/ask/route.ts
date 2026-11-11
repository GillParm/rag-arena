import { apiUrl } from "@/lib/api";

/** Proxies questions to the Python API so the browser only talks to this origin. */
export async function POST(request: Request) {
  const body = await request.text();
  if (body.length > 4_000) {
    return Response.json({ detail: "Request is too large." }, { status: 413 });
  }
  const forwardedFor = request.headers.get("x-forwarded-for");
  try {
    const res = await fetch(apiUrl("/api/ask"), {
      method: "POST",
      headers: {
        "content-type": "application/json",
        ...(forwardedFor ? { "x-forwarded-for": forwardedFor } : {}),
      },
      body,
      cache: "no-store",
      signal: AbortSignal.timeout(120_000),
    });
    return new Response(await res.text(), {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch {
    return Response.json(
      { detail: "The API did not respond. Check that it is running." },
      { status: 502 },
    );
  }
}
