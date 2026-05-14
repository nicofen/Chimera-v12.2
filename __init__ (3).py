"""
chimera/social/scraper.py
Stocktwits async scraper.

Stocktwits provides a public REST API with no auth required for the
public symbol stream endpoint. The free tier is rate-limited to
~200 requests/hour across all endpoints.

Endpoints used:
  GET https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json
      ?limit=30&filter=top

Each response contains up to 30 recent messages with:
  - message body
  - optional sentiment tag (Bullish / Bearish — user-applied)
  - created_at timestamp
  - user metrics (followers, following — used for message weighting)

The scraper:
  1. Maintains a per-symbol poll queue (priority by recent spike risk).
  2. Enforces a global rate-limit token bucket (max_rph requests/hour).
  3. De-duplicates messages by ID using a per-symbol seen-set.
  4. Feeds mentions into the ZScoreEngine and text into the SentimentTagger.
  5. Writes final Z-score + aggregated sentiment into SharedState every cycle.
  6. Triggers a snapshot on the ZScoreEngine every SNAPSHOT_INTERVAL_MIN.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

import aiohttp

from chimera_v12.social.zscore import ZScoreEngine
from chimera_v12.social.sentiment import tag_message, aggregate, AggregatedSentiment
from chimera_v12.utils.logger import setup_logger

if TYPE_CHECKING:
    from chimera_v12.utils.state import SharedState

log = setup_logger("social.scraper")

BASE_URL    = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
MAX_RPH     = 180       # stay comfortably under 200 req/hour free limit
MIN_BETWEEN = 3600 / MAX_RPH   # minimum seconds between any two requests (~20 s)
POLL_INTERVAL_SECONDS = 60     # full cycle (all symbols) target


class RateLimiter:
    """Token-bucket rate limiter — ensures global request spacing."""

    def __init__(self, max_per_hour: int):
        self._min_gap  = 3600 / max_per_hour
        self._last     = 0.0
        self._lock     = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now  = time.monotonic()
            wait = self._min_gap - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()


class StocktwitsScraper:
    """
    Async agent — runs as a task inside the mainframe.
    Polls Stocktwits for all configured stock symbols, feeds
    Z-scores and aggregated sentiment into SharedState.
    """

    def __init__(self, state: "SharedState", config: dict[str, Any]):
        self.state   = state
        self.config  = config
        self.symbols = config.get("stock_symbols", [])
        self.engine  = ZScoreEngine(config)
        self._rl     = RateLimiter(config.get("stocktwits_max_rph", MAX_RPH))
        # Per-symbol message ID de-duplication sets
        self._seen: dict[str, set[int]] = {s: set() for s in self.symbols}
        # Most recent aggregated sentiment per symbol (for NewsAgent handoff)
        self._sentiment: dict[str, AggregatedSentiment] = {}
        self._snapshot_interval = config.get("social_snapshot_interval", 10)
        self._last_snapshot = 0.0

    async def run(self) -> None:
        log.info(f"StocktwitsScraper started — tracking {len(self.symbols)} symbols")
        async with aiohttp.ClientSession(
            headers={"User-Agent": "ChimeraBot/1.0"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as session:
            while True:
                try:
                    await self._cycle(session)
                except Exception as e:
                    log.warning(f"Scraper cycle error: {e}")
                await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def _cycle(self, session: aiohttp.ClientSession) -> None:
        """One full pass over all symbols."""
        # Prioritise symbols that are already showing elevated Z-scores
        ordered = sorted(
            self.symbols,
            key=lambda s: -(self.engine.zscore(s) or 0),
        )

        results: list[SentimentResult] = []

        for symbol in ordered:
            await self._rl.acquire()
            messages = await self._fetch(session, symbol)
            if messages:
                sym_results = self._process(symbol, messages)
                results.extend(sym_results)

        # Snapshot baseline every N minutes
        now = time.monotonic()
        if now - self._last_snapshot >= self._snapshot_interval * 60:
            self.engine.snapshot_all()
            self._last_snapshot = now

        # Inject Z-scores into SharedState
        self.engine.inject_into_state(self.state)

        # Log top movers
        stats = self.engine.all_stats()
        spikes = [(s, v) for s, v in stats.items() if v.get("is_spike")]
        if spikes:
            for sym, v in sorted(spikes, key=lambda x: -(x[1]["zscore"] or 0)):
                log.info(
                    f"SPIKE [{sym}] Z={v['zscore']:.2f}  "
                    f"mentions_5m={v['mentions_recent']}  "
                    f"sent={self._sentiment.get(sym, {}).label if sym in self._sentiment else '?'}"
                )

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        symbol:  str,
    ) -> list[dict] | None:
        """Fetch one page of messages for a symbol."""
        url = BASE_URL.format(symbol=symbol)
        try:
            async with session.get(
                url,
                params={"limit": 30, "filter": "top"},
            ) as resp:
                if resp.status == 429:
                    log.warning(f"Rate limited by Stocktwits — backing off 60s")
                    await asyncio.sleep(60)
                    return None
                if resp.status != 200:
                    log.debug(f"Stocktwits {symbol} → HTTP {resp.status}")
                    return None
                data = await resp.json()
                return data.get("messages", [])
        except asyncio.TimeoutError:
            log.debug(f"Timeout fetching {symbol}")
            return None
        except Exception as e:
            log.debug(f"Fetch error {symbol}: {e}")
            return None

    def _process(
        self,
        symbol:   str,
        messages: list[dict],
    ) -> list:
        """
        De-duplicate, parse timestamps, tag sentiment, feed Z-score engine.
        Returns list of SentimentResult for aggregation.
        """
        from chimera_v12.social.sentiment import SentimentResult

        new_timestamps: list[datetime] = []
        sent_results:   list[SentimentResult] = []
        seen = self._seen.setdefault(symbol, set())

        for msg in messages:
            mid = msg.get("id")
            if mid in seen:
                continue
            seen.add(mid)

            # Parse timestamp
            ts_str = msg.get("created_at", "")
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                ts = datetime.now(timezone.utc)

            new_timestamps.append(ts)

            # Sentiment
            body      = msg.get("body", "")
            api_label = (msg.get("entities", {}) or {}).get("sentiment", {})
            api_label = api_label.get("basic") if isinstance(api_label, dict) else None

            result = tag_message(body, api_label)
            sent_results.append(result)

        # Trim seen set to avoid unbounded growth (keep last 5000 IDs)
        if len(seen) > 5000:
            self._seen[symbol] = set(list(seen)[-3000:])

        if new_timestamps:
            self.engine.add_mentions_bulk(symbol, new_timestamps)

        # Aggregate and cache
        agg = aggregate(symbol, sent_results)
        self._sentiment[symbol] = agg

        # Write sentiment bull_ratio into state for potential NewsAgent use
        if symbol in self.state.market.stocks:
            self.state.market.stocks[symbol]["bull_ratio"] = agg.bull_ratio
            self.state.market.stocks[symbol]["sentiment_label"] = agg.label

        return sent_results

    def get_sentiment(self, symbol: str) -> AggregatedSentiment | None:
        return self._sentiment.get(symbol)

    def zscore(self, symbol: str) -> float | None:
        return self.engine.zscore(symbol)
