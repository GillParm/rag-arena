// Takes the README screenshots with headless Chromium.
// Usage: start the API and the web app, then `npm run screenshots` (BASE_URL defaults to
// http://localhost:3300). Writes PNGs to ../docs/screenshots/.
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright";

const BASE = process.env.BASE_URL ?? "http://localhost:3300";
const OUT = fileURLToPath(new URL("../../docs/screenshots/", import.meta.url));
const QUESTION = process.env.QUESTION ?? "How do I make a query parameter optional?";

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch();
try {
  for (const scheme of ["light", "dark"]) {
    const page = await browser.newPage({
      viewport: { width: 1600, height: 1000 },
      deviceScaleFactor: 2,
      colorScheme: scheme,
    });

    await page.goto(`${BASE}/`, { waitUntil: "networkidle" });
    await page.getByLabel("Your question about FastAPI").fill(QUESTION);
    await page.getByRole("button", { name: "Ask all five" }).click();
    await page.getByRole("heading", { name: "Hybrid + rewrite" }).waitFor({ timeout: 120_000 });
    // Hover the first citation in the hybrid column to show the claim-to-source link.
    const citation = page
      .locator('section[aria-labelledby="lane-hybrid"] button[aria-label^="Source"]')
      .first();
    if (await citation.count()) await citation.hover();
    await page.screenshot({ path: `${OUT}ask-${scheme}.png`, fullPage: false });

    await page.goto(`${BASE}/leaderboard`, { waitUntil: "networkidle" });
    await page.getByRole("table").first().waitFor();
    await page.screenshot({ path: `${OUT}leaderboard-${scheme}.png`, fullPage: true });

    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(`${BASE}/leaderboard`, { waitUntil: "networkidle" });
    await page.screenshot({ path: `${OUT}leaderboard-mobile-${scheme}.png`, fullPage: false });
    await page.close();
  }
} finally {
  await browser.close();
}
console.log(`screenshots written to ${OUT}`);
