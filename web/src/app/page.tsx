import { connection } from "next/server";
import { Suspense } from "react";

import { Arena } from "@/components/arena";
import { getConfig } from "@/lib/api";

async function ArenaLoader() {
  await connection();
  const config = await getConfig();
  if (!config.ok) {
    return (
      <div role="alert" className="border-rule bg-surface max-w-[640px] rounded-md border p-4">
        <p className="font-semibold">The playground can&apos;t load.</p>
        <p className="text-ink-2 mt-1 text-[14px]">{config.error}</p>
      </div>
    );
  }
  return <Arena config={config.data} />;
}

export default function AskPage() {
  return (
    <div className="mx-auto max-w-[1440px] px-4 py-8 sm:px-6 sm:py-10">
      <h1 className="max-w-[20ch] text-[clamp(1.9rem,3.6vw,2.9rem)] leading-[1.05] font-bold tracking-[-0.02em]">
        One question, five retrieval pipelines.
      </h1>
      <p className="text-ink-2 mt-3 max-w-[62ch] text-[16px] leading-relaxed">
        Each column searches the FastAPI documentation its own way, hands its top five passages to
        the same model, and shows the answer with the passages it cited. Compare what each one
        found, how long it took and what it cost.
      </p>
      <div className="mt-8">
        <Suspense fallback={<p className="text-ink-2 text-[14px]">Loading the playground…</p>}>
          <ArenaLoader />
        </Suspense>
      </div>
    </div>
  );
}
