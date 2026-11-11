import type { Metadata } from "next";
import { JetBrains_Mono, Schibsted_Grotesk } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const sans = Schibsted_Grotesk({
  variable: "--font-schibsted",
  subsets: ["latin"],
  display: "swap",
});

const mono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: "rag-arena", template: "%s | rag-arena" },
  description:
    "Five retrieval pipelines answer the same question over the FastAPI docs, side by side, with benchmark results.",
};

const REPO_URL = "https://github.com/GillParm/rag-arena";

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} antialiased`}>
      <body className="flex min-h-dvh flex-col">
        <a
          href="#main"
          className="focus:bg-surface sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:px-3 focus:py-2"
        >
          Skip to content
        </a>
        <header className="border-rule bg-surface border-b">
          <nav
            aria-label="Main"
            className="mx-auto flex max-w-[1440px] items-center gap-6 px-4 py-3 sm:px-6"
          >
            <Link href="/" className="text-ink text-[17px] font-bold tracking-tight">
              rag-arena
            </Link>
            <div className="flex items-center gap-1 text-[15px]">
              <Link
                href="/"
                className="text-ink-2 hover:bg-surface-2 hover:text-ink rounded px-2.5 py-1.5"
              >
                Ask
              </Link>
              <Link
                href="/leaderboard"
                className="text-ink-2 hover:bg-surface-2 hover:text-ink rounded px-2.5 py-1.5"
              >
                Leaderboard
              </Link>
            </div>
            <a
              href={REPO_URL}
              className="text-ink-2 decoration-rule-strong hover:text-ink ml-auto text-[14px] underline underline-offset-4"
            >
              Source on GitHub
            </a>
          </nav>
        </header>
        <main id="main" className="flex-1">
          {children}
        </main>
        <footer className="border-rule border-t">
          <p className="text-muted mx-auto max-w-[1440px] px-4 py-5 text-[13px] sm:px-6">
            Corpus: the FastAPI documentation (MIT licence, © Sebastián Ramírez). Answers come from
            a language model and can be wrong; each one shows the passages it used.
          </p>
        </footer>
      </body>
    </html>
  );
}
