"""
chimera/backtest/data_loader.py
Historical OHLCV data loader.

Supports three sources (tried in order):
  1. Local Parquet file  — fastest, use after first download
  2. Local CSV file      — universal fallback
  3. Alpaca REST API     — downloads and optionally caches to Parquet

All loaders return a unified dict:
    {
        symbol: pd.DataFrame(
            columns=["open","high","low","close","volume"],
            index=pd.DatetimeIndex (UTC)
        )
    }

Parquet caching: set cache_dir in config to persist downloads locally.
On subsequent runs the Parquet file is loaded directly — no API call.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger("chimera.backtest.data_loader")


class DataLoader:
    """
    Loads historical bars for one or more symbols across a date range.
    """

    def __init__(self, config: dict[str, Any]):
        self.config    = config
        self.cache_dir = Path(config.get("cache_dir", "chimera_data_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        symbols:    list[str],
        start:      str | datetime,
        end:        str | datetime,
        timeframe:  str = "1Day",
        sector:     str = "stocks",
    ) -> dict[str, pd.DataFrame]:
        """
        Load OHLCV bars for all symbols between start and end.

        timeframe: "1Min" | "5Min" | "15Min" | "1Hour" | "1Day"
        sector:    used to pick the correct Alpaca feed endpoint
        """
        start_dt = _to_dt(start)
        end_dt   = _to_dt(end)

        results: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            df = self._load_one(sym, start_dt, end_dt, timeframe, sector)
            if df is not None and not df.empty:
                results[sym] = df
                log.info(f"Loaded {len(df)} bars for {sym} ({timeframe})")
            else:
                log.warning(f"No data for {sym}")
        return results

    def _load_one(
        self,
        symbol:    str,
        start:     datetime,
        end:       datetime,
        timeframe: str,
        sector:    str,
    ) -> pd.DataFrame | None:
        # 1. Try Parquet cache
        cache_path = self._cache_path(symbol, timeframe, start, end)
        if cache_path.exists():
            log.debug(f"Cache hit: {cache_path}")
            return pd.read_parquet(cache_path)

        # 2. Try CSV in cache dir
        csv_path = cache_path.with_suffix(".csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            return df

        # 3. Fetch from Alpaca
        df = self._fetch_alpaca(symbol, start, end, timeframe, sector)
        if df is not None and not df.empty and self.config.get("cache_data", True):
            df.to_parquet(cache_path)
            log.debug(f"Cached to {cache_path}")
        return df

    def _fetch_alpaca(
        self,
        symbol:    str,
        start:     datetime,
        end:       datetime,
        timeframe: str,
        sector:    str,
    ) -> pd.DataFrame | None:
        """Fetch bars from Alpaca REST v2 (synchronous using requests)."""
        try:
            import requests
        except ImportError:
            log.error("requests not installed — pip install requests")
            return None

        base = "https://data.alpaca.markets/v2"
        if sector == "crypto":
            url = f"{base}/crypto/us/bars"
            params: dict[str, Any] = {
                "symbols":   symbol,
                "timeframe": timeframe,
                "start":     start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end":       end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit":     10000,
                "sort":      "asc",
            }
        else:
            url = f"{base}/stocks/{symbol}/bars"
            params = {
                "timeframe": timeframe,
                "start":     start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end":       end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "limit":     10000,
                "adjustment": "all",    # split + dividend adjusted
                "feed":      "sip",
                "sort":      "asc",
            }

        headers = {
            "APCA-API-KEY-ID":     self.config.get("alpaca_key", ""),
            "APCA-API-SECRET-KEY": self.config.get("alpaca_secret", ""),
        }

        all_bars: list[dict] = []
        page_token: str | None = None

        while True:
            if page_token:
                params["page_token"] = page_token
            try:
                r = requests.get(url, params=params, headers=headers, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception as e:
                log.error(f"Alpaca fetch error for {symbol}: {e}")
                return None

            # Crypto response wraps bars under the symbol key
            if sector == "crypto":
                bars = data.get("bars", {}).get(symbol, [])
            else:
                bars = data.get("bars", [])

            all_bars.extend(bars)
            page_token = data.get("next_page_token")
            if not page_token:
                break

        if not all_bars:
            return None

        df = pd.DataFrame(all_bars)
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high",
                                  "l": "low",  "c": "close", "v": "volume"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        return df

    def load_csv(self, path: str | Path, symbol: str) -> dict[str, pd.DataFrame]:
        """
        Load a single CSV file as one symbol's data.
        Expected columns: timestamp (index), open, high, low, close, volume
        """
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        log.info(f"Loaded {len(df)} bars from {path}")
        return {symbol: df}

    def _cache_path(
        self,
        symbol:    str,
        timeframe: str,
        start:     datetime,
        end:       datetime,
    ) -> Path:
        safe_sym = symbol.replace("/", "_")
        fname    = f"{safe_sym}_{timeframe}_{start.date()}_{end.date()}.parquet"
        return self.cache_dir / fname


def _to_dt(d: str | datetime) -> datetime:
    if isinstance(d, datetime):
        return d.replace(tzinfo=timezone.utc) if d.tzinfo is None else d
    return datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
