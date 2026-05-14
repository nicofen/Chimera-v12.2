"""
chimera/agents/news_agent.py
News Agent — polls FinancialJuice and Stocktwits, runs LLM-based NLP
classification, sets confidence intervals, and raises macro veto flags.

The veto system is the most important safety mechanism in Chimera:
if a high-impact Fed / CPI / NFP event is detected, ALL technical signals
are suppressed until the dust settles (configurable cool-down window).
"""

from __future__ import annotations
import asyncio
import re
from datetime import datetime, timezone
from typing import Any

import aiohttp
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from chimera_v12.utils.state import SharedState, NewsState, Sentiment
from chimera_v12.utils.logger import setup_logger

log = setup_logger("news_agent")

# ── Veto keywords ──────────────────────────────────────────────────────────────
VETO_PATTERNS = [
    r"fed\s+(decision|meeting|statement|chair|powell)",
    r"fomc",
    r"nonfarm\s+payroll|nfp",
    r"cpi|pce|inflation\s+data",
    r"rate\s+(hike|cut|hold|decision)",
    r"emergency\s+(rate|fed|meeting)",
    r"bank\s+of\s+(england|japan|ecb|europe)\s+(decision|hike|cut)",
]
HAWKISH_KEYWORDS = ["hawkish", "rate hike", "tighten", "inflation concerns", "restrict"]
BEARISH_KEYWORDS = ["rate cut", "dovish", "recession", "layoffs", "credit event", "default"]

_VETO_RE = re.compile("|".join(VETO_PATTERNS), re.IGNORECASE)


class NewsAgent:
    """
    Autonomous News Agent.
    1. Polls REST endpoints on a configurable interval.
    2. Passes headlines through an LLM classifier (LangChain + OpenAI).
    3. Writes Sentiment + CI score to shared state.
    4. Sets veto_active=True if macro event keywords are detected.
    """

    SYSTEM_PROMPT = """You are a quantitative trading news analyst.
Classify the following financial headline(s) and return ONLY valid JSON:
{
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": <float 0.0–1.0>,
  "macro_event": <bool>,
  "macro_reason": "<string, empty if not macro>"
}
Be conservative: a single ambiguous headline should return neutral with low confidence.
High-impact macro events (FOMC, CPI, NFP) must set macro_event=true regardless of direction."""

    def __init__(self, state: SharedState, config: dict[str, Any]):
        self.state  = state
        self.config = config
        self.poll_interval = config.get("news_poll_seconds", 30)
        self.veto_cooldown = config.get("veto_cooldown_seconds", 600)  # 10 min default
        self._veto_until: datetime | None = None

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=config["openai_api_key"],
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.SYSTEM_PROMPT),
            ("human", "Headlines:\n{headlines}"),
        ])
        self._chain = prompt | llm | JsonOutputParser()

    async def run(self) -> None:
        log.info("NewsAgent started.")
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    headlines = await self._fetch_all(session)
                    if headlines:
                        await self._classify_and_update(headlines)
                except Exception as e:
                    log.warning(f"NewsAgent poll error: {e}")
                await asyncio.sleep(self.poll_interval)

    # ── Fetchers ───────────────────────────────────────────────────────────────

    async def _fetch_all(self, session: aiohttp.ClientSession) -> list[str]:
        results = await asyncio.gather(
            self._fetch_financial_juice(session),
            self._fetch_stocktwits(session),
            return_exceptions=True,
        )
        headlines: list[str] = []
        for r in results:
            if isinstance(r, list):
                headlines.extend(r)
        return headlines[:20]  # cap to keep LLM token cost low

    async def _fetch_financial_juice(self, session: aiohttp.ClientSession) -> list[str]:
        """
        FinancialJuice provides a free headline RSS / JSON feed.
        Replace the URL with their paid API endpoint if you have a key.
        """
        url = self.config.get("financialjuice_url", "https://www.financialjuice.com/feed.ashx?c=market")
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            data = await resp.json(content_type=None)
            return [item.get("title", "") for item in data.get("items", [])[:10]]

    async def _fetch_stocktwits(self, session: aiohttp.ClientSession) -> list[str]:
        """Stocktwits trending stream — no API key required for public feed."""
        url = "https://api.stocktwits.com/api/2/streams/trending.json"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            data = await resp.json()
            messages = data.get("messages", [])
            return [m.get("body", "") for m in messages[:10]]

    # ── Classification ─────────────────────────────────────────────────────────

    async def _classify_and_update(self, headlines: list[str]) -> None:
        combined = "\n".join(f"- {h}" for h in headlines if h.strip())

        # Fast regex veto check (don't spend LLM tokens if pattern match fires)
        if _VETO_RE.search(combined):
            await self._raise_veto(combined)
            return

        result: dict[str, Any] = await self._chain.ainvoke({"headlines": combined})

        sentiment_str = result.get("sentiment", "neutral").lower()
        confidence    = float(result.get("confidence", 0.5))
        macro_event   = bool(result.get("macro_event", False))
        macro_reason  = result.get("macro_reason", "")

        if macro_event:
            await self._raise_veto(macro_reason or combined)
            return

        self.state.news = NewsState(
            sentiment    = Sentiment(sentiment_str),
            confidence   = confidence,
            veto_active  = False,
            last_updated = datetime.now(timezone.utc),
        )
        log.info(f"News: {sentiment_str} CI={confidence:.2f}")

    async def _raise_veto(self, reason: str) -> None:
        from datetime import timedelta
        self._veto_until = datetime.now(timezone.utc) + timedelta(seconds=self.veto_cooldown)
        self.state.news.veto_active = True
        self.state.news.veto_reason = reason[:120]
        log.warning(f"VETO RAISED — {reason[:80]}... (cool-down {self.veto_cooldown}s)")

        # Auto-clear after cool-down
        await asyncio.sleep(self.veto_cooldown)
        self.state.news.veto_active = False
        self.state.news.veto_reason = ""
        log.info("Veto cleared — signals re-enabled.")
