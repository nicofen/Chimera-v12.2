"""
chimera/backtest/state.py
BacktestState — a thin extension of SharedState for replay mode.

The live agents read from state.market.{sector}[symbol]["close"] etc.
BacktestState.inject_bar() writes to exactly those locations so agents
see the same data layout whether running live or in backtest.

Extra fields:
  current_bar_index  — which bar in the replay we're on
  current_dt         — datetime of the current bar
  equity_curve       — list of (dt, equity) for drawdown analysis
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from chimera_v12.utils.state import SharedState


class BacktestState(SharedState):
    """
    Extends SharedState with bar-injection and equity-curve tracking.
    All agent code reads state.market.* exactly as in live mode.
    """

    def __init__(self, initial_equity: float = 100_000.0):
        super().__init__()
        self.equity = initial_equity
        self._initial_equity = initial_equity

        # Backtest bookkeeping
        self.current_bar_index: int = 0
        self.current_dt: datetime | None = None
        self.equity_curve: list[tuple[datetime, float]] = []
        self.closed_trades: list[dict[str, Any]] = []

        # Override queues with synchronous-compatible versions
        # (backtest runs synchronously in a single loop — no asyncio needed)
        import queue
        self._sync_signal_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._sync_order_queue:  queue.SimpleQueue = queue.SimpleQueue()

    def inject_bar(
        self,
        sector: str,
        symbol: str,
        bar:    dict[str, Any],
        dt:     datetime,
        max_bars: int = 500,
    ) -> None:
        """
        Append one OHLCV bar to the correct market dict location.
        bar = {"open": float, "high": float, "low": float,
               "close": float, "volume": float}

        Agents read the rolling window of closes/highs/lows — we maintain
        that window here by keeping the last max_bars entries.
        """
        sector_map = {
            "crypto":  self.market.crypto,
            "stocks":  self.market.stocks,
            "forex":   self.market.forex,
            "futures": self.market.futures,
        }
        store = sector_map.get(sector)
        if store is None:
            raise ValueError(f"Unknown sector: {sector}")

        if symbol not in store:
            store[symbol] = {
                "open": [], "high": [], "low": [],
                "close": [], "volume": [],
                "short_interest": 0.0,
                "rvol": 1.0,
                "social_zscore": 0.0,
            }

        d = store[symbol]
        for k in ("open", "high", "low", "close", "volume"):
            d[k].append(float(bar.get(k, 0.0)))
            if len(d[k]) > max_bars:
                d[k] = d[k][-max_bars:]

        self.current_dt = dt
        self.current_bar_index += 1

    def record_equity(self) -> None:
        """Snapshot current equity to the equity curve."""
        if self.current_dt:
            self.equity_curve.append((self.current_dt, self.equity))

    def put_signal_sync(self, signal: Any) -> None:
        self.signals.append(signal)
        self._sync_signal_queue.put(signal)

    def put_order_sync(self, risk_params: Any) -> None:
        self.risk_params.append(risk_params)
        self._sync_order_queue.put(risk_params)

    def drain_signals(self) -> list:
        out = []
        while not self._sync_signal_queue.empty():
            out.append(self._sync_signal_queue.get_nowait())
        return out

    def drain_orders(self) -> list:
        out = []
        while not self._sync_order_queue.empty():
            out.append(self._sync_order_queue.get_nowait())
        return out
