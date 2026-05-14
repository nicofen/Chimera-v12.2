"""
chimera/agents/data_agent.py
Data Agent — manages all market data ingestion.

WebSocket streams: Alpaca (stocks, forex, futures), crypto exchange WS.
REST polling:      Whale Alert, CoinMarketCap, Dune Analytics, Finviz, Stocktwits.

Writes normalized OHLCV + metadata bars into state.market.{sector}.
"""

import asyncio
import json
import time
from typing import Any

import aiohttp
import websockets

from chimera_v12.utils.state import SharedState
from chimera_v12.utils.logger import setup_logger

log = setup_logger("data_agent")

ALPACA_WS_URL    = "wss://stream.data.alpaca.markets/v2/iex"
ALPACA_CRYPTO_WS = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"




async def _poll_with_backoff(
    coro_fn,
    interval: float,
    label: str,
    max_backoff: float = 300.0,
) -> None:
    """
    Run coro_fn() on a repeating interval with exponential back-off on HTTP 429.
    Any other exception is logged and the loop continues on the next interval.

    Args:
        coro_fn:     async callable that performs one poll iteration.
        interval:    normal sleep between polls (seconds).
        label:       name for log messages.
        max_backoff: ceiling for back-off sleep (default 5 minutes).
    """
    import aiohttp
    backoff = interval
    while True:
        try:
            await coro_fn()
            backoff = interval      # reset on success
        except aiohttp.ClientResponseError as e:
            if e.status == 429:
                log.warning(f"{label}: rate-limited (429) — sleeping {backoff:.0f}s")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue
            log.warning(f"{label}: HTTP {e.status}: {e}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            log.warning(f"{label}: poll error: {e}")
        await asyncio.sleep(interval)

class DataAgent:
    """
    Runs all ingestor coroutines concurrently.
    Each ingestor writes directly into the shared state's market dicts.
    """

    def __init__(self, state: SharedState, config: dict[str, Any]):
        self.state  = state
        self.config = config

    async def run(self) -> None:
        log.info("DataAgent started.")
        await asyncio.gather(
            self._alpaca_stocks_ws(),
            self._alpaca_crypto_ws(),
            self._whale_alert_poll(),
            self._finviz_poll(),
            self._dune_poll(),
            self._alpaca_futures_poll(),
        )

    # ── Alpaca Stocks WebSocket ───────────────────────────────────────────────

    async def _alpaca_stocks_ws(self) -> None:
        symbols = self.config.get("stock_symbols", ["AAPL", "TSLA", "GME"])
        headers = {
            "APCA-API-KEY-ID":     self.config["alpaca_key"],
            "APCA-API-SECRET-KEY": self.config["alpaca_secret"],
        }
        while True:
            try:
                async with websockets.connect(ALPACA_WS_URL, extra_headers=headers) as ws:
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "bars":   symbols,
                        "trades": symbols,
                    }))
                    async for raw in ws:
                        msgs = json.loads(raw)
                        for msg in msgs:
                            if msg.get("T") == "b":   # bar message
                                sym = msg["S"]
                                bars = self.state.market.stocks.setdefault(sym, {
                                    "close": [], "high": [], "low": [], "volume": []
                                })
                                bars["close"].append(float(msg["c"]))
                                bars["high"].append(float(msg["h"]))
                                bars["low"].append(float(msg["l"]))
                                bars["volume"].append(float(msg["v"]))
                                # Keep a rolling 500-bar window
                                for k in bars:
                                    bars[k] = bars[k][-500:]
            except Exception as e:
                log.warning(f"Stocks WS error: {e} — reconnecting in 5s")
                await asyncio.sleep(5)

    # ── Alpaca Crypto WebSocket ───────────────────────────────────────────────

    async def _alpaca_crypto_ws(self) -> None:
        symbols = self.config.get("crypto_symbols", ["BTC/USD", "ETH/USD", "SOL/USD"])
        headers = {
            "APCA-API-KEY-ID":     self.config["alpaca_key"],
            "APCA-API-SECRET-KEY": self.config["alpaca_secret"],
        }
        while True:
            try:
                async with websockets.connect(ALPACA_CRYPTO_WS, extra_headers=headers) as ws:
                    await ws.send(json.dumps({"action": "subscribe", "bars": symbols}))
                    async for raw in ws:
                        msgs = json.loads(raw)
                        for msg in msgs:
                            if msg.get("T") == "b":
                                sym = msg["S"]
                                bars = self.state.market.crypto.setdefault(sym, {
                                    "close": [], "high": [], "low": [], "volume": []
                                })
                                bars["close"].append(float(msg["c"]))
                                bars["high"].append(float(msg["h"]))
                                bars["low"].append(float(msg["l"]))
                                bars["volume"].append(float(msg["v"]))
                                for k in bars:
                                    bars[k] = bars[k][-500:]
            except Exception as e:
                log.warning(f"Crypto WS error: {e} — reconnecting in 5s")
                await asyncio.sleep(5)

    # ── Whale Alert REST poll ─────────────────────────────────────────────────


    async def _whale_alert_poll(self) -> None:
        """Rate-limit-aware wrapper around __whale_alert_poll_once."""
        interval = self.config.get("intervals", {}).get(
            "_whale_alert_poll_seconds", 60
        )
        await _poll_with_backoff(
            self.__whale_alert_poll_once, interval, "WhaleAlert"
        )
    async def __whale_alert_poll_once(self) -> None:
        url      = "https://api.whale-alert.io/v1/transactions"
        api_key  = self.config.get("whale_alert_key", "")
        interval = self.config.get("whale_poll_seconds", 60)

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    params = {
                        "api_key":   api_key,
                        "cursor":    int(time.time()) - interval,
                        "min_value": 1_000_000,
                        "limit":     100,
                    }
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        data = await resp.json()
                        inflow = sum(
                            tx["amount_usd"]
                            for tx in data.get("transactions", [])
                            if tx.get("to", {}).get("owner_type") == "exchange"
                            and tx.get("symbol", "").upper() == "BTC"
                        )
                        self.state.market.crypto["btc_exchange_inflow"] = inflow
                        log.debug(f"BTC exchange inflow (last {interval}s): ${inflow:,.0f}")
                except Exception as e:
                    log.warning(f"Whale Alert poll error: {e}")
                await asyncio.sleep(interval)

    # ── Finviz screener poll (short interest + RVOL) ─────────────────────────


    async def _finviz_poll(self) -> None:
        """Rate-limit-aware wrapper around __finviz_poll_once."""
        interval = self.config.get("intervals", {}).get(
            "_finviz_poll_seconds", 60
        )
        await _poll_with_backoff(
            self.__finviz_poll_once, interval, "Finviz"
        )
    async def __finviz_poll_once(self) -> None:
        """
        Uses finviz Python library (pip install finviz) to screen for
        high short-interest, high RVOL candidates every 5 minutes.
        """
        try:
            import finviz
        except ImportError:
            log.warning("finviz not installed — skipping stock screener.")
            return

        interval = self.config.get("finviz_poll_seconds", 300)
        filters  = ["sh_short_o20", "ta_relvol_o3"]   # SI > 20%, RVOL > 3

        while True:
            try:
                results = finviz.get_screener(filters=filters, table="Performance")
                for row in results[:20]:
                    sym = row.get("Ticker", "")
                    if sym in self.state.market.stocks:
                        self.state.market.stocks[sym]["short_interest"] = (
                            float(str(row.get("Short Float", "0")).strip("%")) / 100
                        )
                        self.state.market.stocks[sym]["rvol"] = float(row.get("Rel Volume", 1.0))
            except Exception as e:
                log.warning(f"Finviz poll error: {e}")
            await asyncio.sleep(interval)

    # ── Dune Analytics poll (Solana memecoin volume) ─────────────────────────


    async def _dune_poll(self) -> None:
        """Rate-limit-aware wrapper around __dune_poll_once."""
        interval = self.config.get("intervals", {}).get(
            "_dune_poll_seconds", 60
        )
        await _poll_with_backoff(
            self.__dune_poll_once, interval, "Dune"
        )
    async def __dune_poll_once(self) -> None:
        dune_key = self.config.get("dune_api_key", "")
        query_id = self.config.get("dune_memecoin_query_id", "3152691")  # memecoin wars
        interval = self.config.get("dune_poll_seconds", 300)
        url      = f"https://api.dune.com/api/v1/query/{query_id}/results"

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    headers = {"X-Dune-API-Key": dune_key}
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                        data  = await resp.json()
                        rows  = data.get("result", {}).get("rows", [])
                        vol   = sum(r.get("volume_usd", 0) for r in rows[:10])
                        spike = vol > self.config.get("sol_memecoin_spike_threshold", 50_000_000)
                        self.state.market.crypto["sol_memecoin_vol_spike"] = spike
                        log.debug(f"Sol memecoin vol: ${vol:,.0f} spike={spike}")
                except Exception as e:
                    log.warning(f"Dune poll error: {e}")
                await asyncio.sleep(interval)

    # ── Alpaca Futures REST poll (ES1!) ───────────────────────────────────────

    async def _alpaca_futures_poll(self) -> None:
        contracts = self.config.get("futures_contracts", ["ES1!"])
        url_base  = "https://data.alpaca.markets/v2/stocks/{sym}/bars"
        headers   = {
            "APCA-API-KEY-ID":     self.config["alpaca_key"],
            "APCA-API-SECRET-KEY": self.config["alpaca_secret"],
        }
        interval = self.config.get("futures_poll_seconds", 60)

        async with aiohttp.ClientSession() as session:
            while True:
                for sym in contracts:
                    try:
                        async with session.get(
                            url_base.format(sym=sym),
                            headers=headers,
                            params={"timeframe": "5Min", "limit": 200},
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            data = await resp.json()
                            bars_raw = data.get("bars", [])
                            if bars_raw:
                                self.state.market.futures[sym] = {
                                    "close":  [b["c"] for b in bars_raw],
                                    "high":   [b["h"] for b in bars_raw],
                                    "low":    [b["l"] for b in bars_raw],
                                    "volume": [b["v"] for b in bars_raw],
                                }
                    except Exception as e:
                        log.warning(f"Futures poll error for {sym}: {e}")
                await asyncio.sleep(interval)

    async def _alpaca_forex_ws(self) -> None:
        """
        Alpaca Forex data ingestor — fills state.market.forex so the
        ForexStrategy can fire.  Without this, state.market.forex is always
        empty and the forex sector is permanently dead.

        Streams 1-minute bars for all configured forex pairs via Alpaca's
        streaming API (same format as equity bars, different feed endpoint).

        Falls back to REST polling every 60 seconds if the WebSocket
        fails or Alpaca key is not set.
        """
        import aiohttp, json
        pairs   = self.config.get("forex_pairs", ["EUR/USD", "GBP/USD", "USD/JPY"])
        api_key = self.config.get("alpaca_key", "")
        api_sec = self.config.get("alpaca_secret", "")

        if not api_key:
            log.warning("Alpaca key not set — forex using yfinance REST fallback.")
            await self._forex_rest_fallback(pairs)
            return

        # Alpaca streams forex (crypto endpoint uses same bar format)
        ws_url = "wss://stream.data.alpaca.markets/v1beta3/forex/bars"
        subscribe_msg = {
            "action": "subscribe",
            "bars": [p.replace("/", "") for p in pairs],
        }
        auth_msg = {"action": "auth", "key": api_key, "secret": api_sec}

        while True:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(
                        ws_url,
                        headers={"APCA-API-KEY-ID": api_key,
                                 "APCA-API-SECRET-KEY": api_sec},
                    ) as ws:
                        await ws.send_json(auth_msg)
                        await ws.send_json(subscribe_msg)
                        log.info(f"Forex WS connected: {pairs}")
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    bars = json.loads(msg.data)
                                    if isinstance(bars, list):
                                        for bar in bars:
                                            await self._ingest_forex_bar(bar)
                                except Exception as e:
                                    log.debug(f"Forex bar parse error: {e}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning(f"Forex WS error ({e}), retrying in 30s")
                await asyncio.sleep(30)

    async def _forex_rest_fallback(self, pairs: list[str]) -> None:
        """
        yfinance REST fallback for forex bars when Alpaca is unavailable.
        Polls every 60 seconds for 5-minute bars.
        """
        while True:
            try:
                import yfinance as yf
                for pair in pairs:
                    # yfinance uses EURUSD=X notation
                    ticker_sym = pair.replace("/", "") + "=X"
                    hist = yf.Ticker(ticker_sym).history(period="2d", interval="5m")
                    if hist.empty:
                        continue
                    symbol = pair
                    if symbol not in self.state.market.forex:
                        self.state.market.forex[symbol] = {
                            "close": [], "high": [], "low": [],
                            "volume": [], "timestamps": [],
                        }
                    mkt = self.state.market.forex[symbol]
                    mkt["close"]  = hist["Close"].tolist()
                    mkt["high"]   = hist["High"].tolist()
                    mkt["low"]    = hist["Low"].tolist()
                    mkt["volume"] = [1.0] * len(hist)
                    mkt["timestamps"] = [
                        ts.to_pydatetime() for ts in hist.index
                    ]
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.debug(f"Forex REST fallback error: {e}")
            await asyncio.sleep(60)

    async def _ingest_forex_bar(self, bar: dict) -> None:
        """Ingest a single forex bar from the Alpaca stream."""
        try:
            # Alpaca bar format: {"S": "EURUSD", "o": 1.08, "h": 1.082, "l": 1.079, "c": 1.081, "v": 0, "t": "..."}
            raw_sym = bar.get("S", "")
            if not raw_sym:
                return
            # Convert EURUSD → EUR/USD
            sym = raw_sym[:3] + "/" + raw_sym[3:] if len(raw_sym) == 6 else raw_sym
            if sym not in self.state.market.forex:
                self.state.market.forex[sym] = {
                    "close": [], "high": [], "low": [],
                    "volume": [], "timestamps": [],
                }
            mkt = self.state.market.forex[sym]
            mkt["close"].append(float(bar.get("c", 0)))
            mkt["high"].append(float(bar.get("h", 0)))
            mkt["low"].append(float(bar.get("l", 0)))
            mkt["volume"].append(float(bar.get("v", 1)))
            from datetime import datetime, timezone
            mkt["timestamps"].append(datetime.now(timezone.utc))
            # Keep only last 500 bars in memory
            for key in ("close", "high", "low", "volume", "timestamps"):
                if len(mkt[key]) > 500:
                    mkt[key] = mkt[key][-500:]
        except Exception as e:
            log.debug(f"Forex bar ingest error: {e}")
