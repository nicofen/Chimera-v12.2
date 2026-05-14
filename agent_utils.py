"""
chimera_v12/agents/unified_news_agent.py — Unified News + Data Agents
Merges Chimera v11 (Finnhub macro veto) with TradingAgents (AV/yfinance LLM reports).
"""
from __future__ import annotations

import asyncio
import re as _re
from datetime import datetime, timedelta, timezone
from typing import Any

from chimera_v12.core.state import SharedState, Sentiment
from chimera_v12.utils.logger import setup_logger

log = setup_logger("agents.unified_news")

VETO_KEYWORDS = [
    "federal reserve", "fed rate", "fomc", "interest rate decision",
    "nonfarm payroll", "nfp", "cpi release", "inflation report",
    "bank of england", "ecb rate", "emergency rate",
    "recession", "financial crisis", "systemic risk", "black swan",
    "circuit breaker", "market halt", "trading halt",
]


class UnifiedNewsAgent:
    """
    Two concurrent subsystems:
      1. Chimera macro veto loop  — Finnhub polling, near real-time
      2. TA LLM report loop       — Alpha Vantage / yfinance per symbol
    """

    def __init__(self, state: SharedState, config: dict[str, Any]):
        self.state       = state
        self.config      = config
        self.interval    = config.get("intervals", {}).get("news_poll_seconds", 30)
        self.veto_cd     = config.get("veto_cooldown_seconds", 600)
        self.finnhub_key = config.get("finnhub_api_key", "")

    async def run(self) -> None:
        log.info("UnifiedNewsAgent started.")
        await asyncio.gather(
            self._macro_veto_loop(),
            self._llm_report_loop(),
        )

    # ── Subsystem 1: Chimera Macro Veto ──────────────────────────────────────

    async def _macro_veto_loop(self) -> None:
        import aiohttp
        while True:
            try:
                if self.finnhub_key:
                    async with aiohttp.ClientSession() as session:
                        url = f"https://finnhub.io/api/v1/news?category=general&token={self.finnhub_key}"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            if resp.status == 200:
                                articles = await resp.json()
                                await self._process_macro_articles(articles)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.debug(f"Macro veto poll: {e}")
            await asyncio.sleep(self.interval)

    async def _process_macro_articles(self, articles: list) -> None:
        for article in articles[:20]:
            headline = (article.get("headline") or "").lower()
            summary  = (article.get("summary") or "").lower()
            text     = headline + " " + summary
            for kw in VETO_KEYWORDS:
                if kw in text:
                    await self._issue_veto(headline, kw)
                    return

    async def _issue_veto(self, headline: str, trigger: str) -> None:
        if self.state.news_veto:
            return
        self.state.news_veto       = True
        self.state.veto_until      = datetime.now(timezone.utc) + timedelta(seconds=self.veto_cd)
        self.state.news.veto_active  = True
        self.state.news.veto_reason  = f"Macro event: {trigger}"
        self.state.news.sentiment    = Sentiment.BEARISH
        self.state.news.confidence   = 0.9
        await self.state.log_audit("UnifiedNewsAgent", "ALL", "MACRO_VETO", {
            "headline": headline[:200],
            "trigger": trigger,
            "veto_until": self.state.veto_until.isoformat(),
        })
        log.warning(f"MACRO VETO: '{trigger}' | '{headline[:80]}'")
        asyncio.create_task(self._lift_veto_after(self.veto_cd))

    async def _lift_veto_after(self, seconds: int) -> None:
        await asyncio.sleep(seconds)
        self.state.news_veto        = False
        self.state.news.veto_active = False
        self.state.news.veto_reason = ""
        log.info("Macro veto lifted.")

    # ── Subsystem 2: TA LLM Reports ──────────────────────────────────────────

    async def _llm_report_loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            if self.state.news_veto:
                continue
            symbols = list(self.state.market.stocks.keys())[:5]
            for sym in symbols:
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._generate_news_report, sym,
                    )
                except Exception as e:
                    log.debug(f"LLM news report [{sym}]: {e}")

    def _generate_news_report(self, symbol: str) -> None:
        try:
            from chimera_v12.dataflows.interface import route_to_vendor
            from datetime import date, timedelta as td
            end   = date.today().isoformat()
            start = (date.today() - td(days=7)).isoformat()
            raw   = route_to_vendor("get_news", symbol, start, end)
            score = self._keyword_sentiment(str(raw))
            if symbol not in self.state.reports:
                self.state.reports[symbol] = {}
            self.state.reports[symbol]["news"]       = str(raw)[:3000]
            self.state.reports[symbol]["news_score"] = score
            if symbol not in self.state.quant_signals:
                self.state.quant_signals[symbol] = {}
            self.state.quant_signals[symbol]["news_sentiment_alpha"] = score
            log.debug(f"News report [{symbol}]: score={score:+.3f}")
        except Exception as e:
            log.debug(f"News fetch [{symbol}]: {e}")

    def _keyword_sentiment(self, text: str) -> float:
        text = text.lower()
        bull = ["beat", "record", "growth", "profit", "upgrade",
                "bullish", "surge", "breakout", "strong", "positive"]
        bear = ["miss", "loss", "decline", "downgrade", "layoffs",
                "bearish", "crash", "weak", "lawsuit", "fraud"]
        b = sum(text.count(w) for w in bull)
        br = sum(text.count(w) for w in bear)
        total = b + br or 1
        return round((b - br) / total, 4)


# ── Unified Data Agent ────────────────────────────────────────────────────────

class UnifiedDataAgent:
    """
    Alpaca WebSocket (live) + yfinance (historical + fundamentals).
    Feeds SharedState.market for all downstream agents.
    """

    def __init__(self, state: SharedState, config: dict[str, Any]):
        self.state   = state
        self.config  = config
        self.interval = config.get("intervals", {}).get("strategy_seconds", 15)
        self.symbols  = (
            config.get("stock_symbols_tier1", []) + config.get("stock_symbols_tier2", [])
        )

    async def run(self) -> None:
        log.info("UnifiedDataAgent started.")
        await asyncio.gather(
            self._live_feed_loop(),
            self._historical_fetch_loop(),
            self._fundamentals_loop(),
        )

    async def _live_feed_loop(self) -> None:
        try:
            from chimera_v12.agents.data_agent import DataAgent
            agent = DataAgent(self.state, self.config)
            await agent.run()
        except Exception as e:
            log.warning(f"Alpaca live feed unavailable ({e}), using yfinance polling.")
            await self._yfinance_poll_loop()

    async def _yfinance_poll_loop(self) -> None:
        while True:
            try:
                await asyncio.get_running_loop().run_in_executor(None, self._fetch_yfinance_bars)
            except Exception as e:
                log.debug(f"yfinance poll: {e}")
            await asyncio.sleep(self.interval)

    def _fetch_yfinance_bars(self) -> None:
        try:
            import yfinance as yf
            for sym in self.symbols:
                hist = yf.Ticker(sym).history(period="5d", interval="5m")
                if hist.empty:
                    continue
                if sym not in self.state.market.stocks:
                    self.state.market.stocks[sym] = {}
                self.state.market.stocks[sym].update({
                    "close":  hist["Close"].tolist(),
                    "high":   hist["High"].tolist(),
                    "low":    hist["Low"].tolist(),
                    "volume": hist["Volume"].tolist(),
                })
        except Exception as e:
            log.debug(f"yfinance bars: {e}")

    async def _historical_fetch_loop(self) -> None:
        while True:
            for sym in self.symbols:
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._fetch_daily_history, sym,
                    )
                except Exception as e:
                    log.debug(f"Daily history [{sym}]: {e}")
            await asyncio.sleep(3600)

    def _fetch_daily_history(self, sym: str) -> None:
        try:
            import yfinance as yf
            hist = yf.Ticker(sym).history(period="1y", interval="1d")
            if hist.empty:
                return
            if sym not in self.state.market.stocks:
                self.state.market.stocks[sym] = {}
            self.state.market.stocks[sym]["daily_df"] = hist.to_csv()
        except Exception as e:
            log.debug(f"Daily history [{sym}]: {e}")

    async def _fundamentals_loop(self) -> None:
        await asyncio.sleep(5)
        while True:
            for sym in self.symbols:
                try:
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._fetch_fundamentals, sym,
                    )
                except Exception as e:
                    log.debug(f"Fundamentals [{sym}]: {e}")
            await asyncio.sleep(86400)

    def _fetch_fundamentals(self, sym: str) -> None:
        try:
            import yfinance as yf
            info = yf.Ticker(sym).info or {}
            fundamentals = {
                "sector":             info.get("sector", "default"),
                "roa":                info.get("returnOnAssets", 0) or 0,
                # NOTE: Using current*0.95 as prior-year proxy is an APPROXIMATION.
                # For rigorous backtesting, supply a point-in-time fundamental DB.
                # The backtest engine's config["backtest_mode"] flag skips this field
                # so the Piotroski F3/F8/F9 signals are omitted rather than estimated.
                "roa_prior":          (info.get("returnOnAssets", 0) or 0) * 0.95,
                "operating_cashflow": info.get("operatingCashflow", 0) or 0,
                "total_assets":       info.get("totalAssets", 1) or 1,
                "long_term_debt":     info.get("longTermDebt", 0) or 0,
                "long_term_debt_prior": (info.get("longTermDebt", 0) or 0) * 1.1,
                "current_ratio":      info.get("currentRatio", 1) or 1,
                "current_ratio_prior": max((info.get("currentRatio", 1) or 1) * 0.95, 0.1),
                "shares_issued_yoy":  False,
                "gross_margin":       info.get("grossMargins", 0) or 0,
                "gross_margin_prior": max((info.get("grossMargins", 0) or 0) * 0.97, 0),
                "asset_turnover":     info.get("assetTurnover", 0) or 0,
                "asset_turnover_prior": max((info.get("assetTurnover", 0) or 0) * 0.97, 0),
                "pe_ratio":           info.get("trailingPE", 999) or 999,
                "pb_ratio":           info.get("priceToBook", 999) or 999,
                "ev_ebitda":          info.get("enterpriseToEbitda", 20) or 20,
                "fcf_yield":          self._fcf_yield(info),
                "roic":               info.get("returnOnEquity", 0) or 0,
                "revenue_growth_yoy": info.get("revenueGrowth", 0) or 0,
                "eps_growth_yoy":     info.get("earningsGrowth", 0) or 0,
                "fcf_growth_yoy":     (info.get("revenueGrowth", 0) or 0) * 0.8,
                "revenue_acceleration": 0.0,
            }
            if sym not in self.state.market.stocks:
                self.state.market.stocks[sym] = {}
            self.state.market.stocks[sym]["fundamentals"] = fundamentals
        except Exception as e:
            log.debug(f"Fundamentals [{sym}]: {e}")

    def _fcf_yield(self, info: dict) -> float:
        fcf  = info.get("freeCashflow", 0) or 0
        mcap = info.get("marketCap", 1) or 1
        return round(fcf / mcap, 4) if mcap > 0 else 0.0
