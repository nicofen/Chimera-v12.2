"""
chimera/social/monitor.py
Real-time Z-score monitor — standalone CLI tool for watching social
velocity without running the full mainframe.

Usage:
    python -m chimera.social.monitor --symbols GME AMC TSLA BBBY
    python -m chimera.social.monitor --symbols GME --interval 30

Outputs a live-updating terminal table:
    SYMBOL  MENTIONS/5m  MENTIONS/1h  Z-SCORE  SPIKE  SENTIMENT   CI
    GME          12          48        +3.42    🔥     BULLISH    0.72
    AMC           3          21        +0.81    -      NEUTRAL    0.41
    ...

Useful for:
  - Calibrating the spike threshold for your watchlist.
  - Verifying the scraper is running before wiring into the mainframe.
  - Manual scanning for squeeze candidates.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from datetime import datetime, timezone


async def _monitor(symbols: list[str], interval: int) -> None:
    import aiohttp
    from chimera_v12.social.zscore import ZScoreEngine
    from chimera_v12.social.scraper import StocktwitsScraper, RateLimiter

    config = {"stock_symbols": symbols, "stocktwits_max_rph": 120}

    class _FakeState:
        class market:
            stocks: dict = {}

    state   = _FakeState()
    engine  = ZScoreEngine(config)
    rl      = RateLimiter(120)
    seen:   dict[str, set] = {s: set() for s in symbols}
    sentiment_cache: dict  = {}

    from chimera_v12.social.scraper import BASE_URL
    from chimera_v12.social.sentiment import tag_message, aggregate

    async def fetch_one(session, symbol):
        await rl.acquire()
        url = BASE_URL.format(symbol=symbol)
        try:
            async with session.get(url, params={"limit": 30}, timeout=aiohttp.ClientTimeout(total=12)) as r:
                if r.status != 200:
                    return []
                data = await r.json()
                return data.get("messages", [])
        except Exception:
            return []

    print(f"\n  Chimera Social Monitor — {len(symbols)} symbols  (Ctrl+C to exit)\n")

    async with aiohttp.ClientSession(headers={"User-Agent": "ChimeraMonitor/1.0"}) as session:
        iteration = 0
        while True:
            iteration += 1
            for symbol in symbols:
                msgs = await fetch_one(session, symbol)
                new_ts, results = [], []
                for m in msgs:
                    mid = m.get("id")
                    if mid in seen[symbol]: continue
                    seen[symbol].add(mid)
                    ts_str = m.get("created_at", "")
                    try:
                        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    except Exception:
                        ts = datetime.now(timezone.utc)
                    new_ts.append(ts)
                    api_label = (m.get("entities", {}) or {}).get("sentiment", {})
                    api_label = api_label.get("basic") if isinstance(api_label, dict) else None
                    results.append(tag_message(m.get("body", ""), api_label))

                if new_ts:
                    engine.add_mentions_bulk(symbol, new_ts)
                if results:
                    sentiment_cache[symbol] = aggregate(symbol, results)

            if iteration % 2 == 0:
                engine.snapshot_all()

            # Clear and redraw
            os.system("cls" if sys.platform == "win32" else "clear")
            now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
            print(f"\n  Chimera Social Monitor  [{now}]  cycle #{iteration}\n")
            print(f"  {'SYMBOL':<8} {'5m':>6} {'1h':>6} {'Z-SCORE':>9} {'SPIKE':>6}  {'SENTIMENT':<10} {'CI':>5}")
            print("  " + "─" * 58)

            for sym in symbols:
                stats = engine.track(sym).stats()
                agg   = sentiment_cache.get(sym)
                z     = stats.get("zscore")
                z_str = f"{z:+.2f}" if z is not None else "  n/a"
                z_col = "\033[92m" if (z or 0) >= 2 else "\033[93m" if (z or 0) >= 1 else "\033[0m"
                spike  = "🔥" if stats.get("is_spike") else " -"
                sent   = agg.label.upper()   if agg else "—"
                ci     = f"{agg.confidence:.2f}" if agg else "—"
                sent_col = "\033[92m" if sent == "BULLISH" else "\033[91m" if sent == "BEARISH" else "\033[0m"
                print(
                    f"  {sym:<8} "
                    f"{stats['mentions_recent']:>6} "
                    f"{stats['mentions_1h']:>6} "
                    f"  {z_col}{z_str:>7}\033[0m "
                    f"{spike:>6}  "
                    f"{sent_col}{sent:<10}\033[0m "
                    f"{ci:>5}"
                )

            print(f"\n  Next refresh in {interval}s  —  drag Stocktwits Z-scores to feed Sp score")
            await asyncio.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chimera Social Monitor")
    parser.add_argument("--symbols",  nargs="+", default=["GME","AMC","TSLA","BBBY","NVDA"])
    parser.add_argument("--interval", type=int,  default=60,  help="Poll interval seconds")
    args = parser.parse_args()
    try:
        asyncio.run(_monitor(args.symbols, args.interval))
    except KeyboardInterrupt:
        print("\n  Exiting.")


if __name__ == "__main__":
    main()
