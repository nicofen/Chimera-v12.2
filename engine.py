"""
chimera/social/sentiment.py
Stocktwits message sentiment tagger.

Two-stage pipeline:
  1. Native Stocktwits sentiment label (if the API returns one)
     Messages tagged by the poster as $bullish or $bearish are used directly.
  2. Keyword fallback for untagged messages.
     A weighted keyword lexicon classifies the text when no API label exists.

Output: SentimentResult(label, confidence, bull_count, bear_count, bull_ratio)

This is intentionally lightweight — no LLM call needed here because
Stocktwits users explicitly tag a large fraction of their own messages,
and the keyword list is well-calibrated for retail trading argot.
The NewsAgent LLM handles macro/news sentiment; this module handles
pure crowd/retail price sentiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


SentimentLabel = Literal["bullish", "bearish", "neutral"]


# ── Weighted lexicons ─────────────────────────────────────────────────────────
# Positive score = bullish signal; negative = bearish.
# Weights are rough — the ratio between bull and bear counts matters more
# than individual weights.

BULL_PATTERNS: list[tuple[str, float]] = [
    (r"\bbull(?:ish)?\b",        1.5),
    (r"\blong\b",                1.2),
    (r"\bbuy\b",                 1.2),
    (r"\bbreakout\b",            1.3),
    (r"\bmoon\b",                1.4),
    (r"\brocket\b",              1.3),
    (r"\bsqueeze\b",             1.5),
    (r"\bshort\s+squeeze\b",     2.0),
    (r"\bHODL\b",                1.2),
    (r"\baccumulate\b",          1.1),
    (r"\bupside\b",              1.0),
    (r"\bmomentum\b",            0.8),
    (r"\bpump\b",                1.0),
    (r"\bATH\b",                 1.2),
    (r"🚀",                       1.4),
    (r"🔥",                       0.8),
    (r"💎\s*🙌",                  1.5),   # diamond hands
    (r"\bload(?:ing|ed)?\s+up\b",1.2),
    (r"\bno\s+brainer\b",        1.0),
    (r"\bgap\s+up\b",            1.1),
]

BEAR_PATTERNS: list[tuple[str, float]] = [
    (r"\bbear(?:ish)?\b",        1.5),
    (r"\bshort\b",               1.2),
    (r"\bsell\b",                1.2),
    (r"\bputs?\b",               1.1),   # matches "put" and "puts"
    (r"\bdump\b",                1.3),
    (r"\bcollapse\b",            1.4),
    (r"\bcrash\b",               1.5),
    (r"\bheadline\s+risk\b",     1.0),
    (r"\bdownside\b",            1.0),
    (r"\bbankrupt(?:cy)?\b",     1.8),
    (r"\bscam\b",                1.3),
    (r"\bfake\b",                0.8),
    (r"\bgap\s+down\b",          1.1),
    (r"📉",                       1.3),
    (r"🩳",                       1.2),   # shorts emoji
    (r"\bpanic\b",               1.2),
    (r"\brun\s+away\b",          1.0),
    (r"\bhalt(?:ed)?\b",         0.9),
]

_COMPILED_BULL = [(re.compile(p, re.IGNORECASE), w) for p, w in BULL_PATTERNS]
_COMPILED_BEAR = [(re.compile(p, re.IGNORECASE), w) for p, w in BEAR_PATTERNS]


@dataclass
class SentimentResult:
    label:      SentimentLabel
    confidence: float          # 0–1
    bull_score: float
    bear_score: float
    bull_ratio: float          # bull_score / (bull_score + bear_score)
    source:     str            # "api_tag" | "keyword" | "neutral_default"


def tag_message(
    text:      str,
    api_label: str | None = None,   # from Stocktwits API: "Bullish" | "Bearish" | None
) -> SentimentResult:
    """
    Classify a single Stocktwits message.

    If the API already provided a label (the poster tagged $bullish or $bearish),
    use that directly with a fixed high confidence.
    Otherwise fall back to keyword scoring.
    """
    if api_label:
        norm = api_label.lower().strip()
        if norm == "bullish":
            return SentimentResult("bullish", 0.85, 1.0, 0.0, 1.0, "api_tag")
        if norm == "bearish":
            return SentimentResult("bearish", 0.85, 0.0, 1.0, 0.0, "api_tag")

    return _keyword_score(text)


def _keyword_score(text: str) -> SentimentResult:
    bull = sum(w for pat, w in _COMPILED_BULL if pat.search(text))
    bear = sum(w for pat, w in _COMPILED_BEAR if pat.search(text))

    total = bull + bear
    if total < 0.5:
        return SentimentResult("neutral", 0.30, bull, bear, 0.5, "neutral_default")

    ratio = bull / total
    if ratio >= 0.60:
        conf = min(0.95, 0.50 + (ratio - 0.60) * 2.0)
        return SentimentResult("bullish", round(conf, 3), bull, bear, round(ratio, 3), "keyword")
    if ratio <= 0.40:
        conf = min(0.95, 0.50 + (0.40 - ratio) * 2.0)
        return SentimentResult("bearish", round(conf, 3), bull, bear, round(ratio, 3), "keyword")

    return SentimentResult("neutral", 0.40, bull, bear, round(ratio, 3), "keyword")


@dataclass
class AggregatedSentiment:
    """Rolled-up sentiment across N messages for one symbol."""
    symbol:        str
    n_messages:    int
    bull_count:    int
    bear_count:    int
    neutral_count: int
    bull_ratio:    float     # bull / (bull + bear), ignoring neutral
    confidence:    float     # weighted average confidence
    label:         SentimentLabel


def aggregate(symbol: str, results: list[SentimentResult]) -> AggregatedSentiment:
    """
    Aggregate a list of SentimentResult objects for one symbol into a
    single summary for the NewsAgent to consume.
    """
    if not results:
        return AggregatedSentiment(symbol, 0, 0, 0, 0, 0.5, 0.0, "neutral")

    bull    = sum(1 for r in results if r.label == "bullish")
    bear    = sum(1 for r in results if r.label == "bearish")
    neutral = sum(1 for r in results if r.label == "neutral")
    polar   = bull + bear

    bull_ratio = bull / polar if polar > 0 else 0.5
    avg_conf   = sum(r.confidence for r in results) / len(results)

    if polar < 3:
        label: SentimentLabel = "neutral"
    elif bull_ratio >= 0.60:
        label = "bullish"
    elif bull_ratio <= 0.40:
        label = "bearish"
    else:
        label = "neutral"

    return AggregatedSentiment(
        symbol        = symbol,
        n_messages    = len(results),
        bull_count    = bull,
        bear_count    = bear,
        neutral_count = neutral,
        bull_ratio    = round(bull_ratio, 3),
        confidence    = round(avg_conf, 3),
        label         = label,
    )
