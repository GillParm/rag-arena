"""List prices used for cost reporting. USD per million tokens.

Gemini runs in this project use the free tier; costs are reported as if billed at the paid
list price so the numbers mean something for a real deployment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float  # includes thinking tokens for both providers
    batch_discount: float  # multiplier applied to both rates in batch mode
    source: str
    checked: str


PRICES: dict[str, Price] = {
    "gemini-3.5-flash-lite": Price(
        0.30, 2.50, 0.5, "https://ai.google.dev/gemini-api/docs/pricing", "2026-11-10"
    ),
    "claude-opus-5-5": Price(
        4.00, 20.00, 0.5, "https://platform.claude.com/docs/en/about-claude/pricing", "2026-11-10"
    ),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, *, batch: bool = False) -> float:
    price = PRICES.get(model)
    if price is None:
        raise KeyError(f"no list price recorded for {model!r}; add it to llm/pricing.py")
    factor = price.batch_discount if batch else 1.0
    return (
        factor
        * (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok)
        / 1_000_000
    )
