"""
chimera/backtest/engine.py — bias-corrected replay engine.
See inline comments (FIX 1–6) for each audit issue addressed.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from chimera_v12.backtest.state import BacktestState
from chimera_v12.backtest.data_loader import DataLoader
from chimera_v12.backtest.simulated_oms import SimulatedOMS
from chimera_v12.backtest.performance import PerformanceReport
from chimera_v12.agents.strategy_agent import StrategyAgent
from chimera_v12.agents.risk_agent import RiskAgent
from chimera_v12.utils.logger import setup_logger

log = setup_logger("backtest.engine")

# FIX 1: sentinel offset ensures orders always fill on the NEXT bar's open
ONE_BAR_OFFSET = timedelta(seconds=1)

# Built-in approximate US macro blackout dates (month, day)
_APPROX_BLACKOUT_MD: list[tuple[int, int]] = [
    # FOMC (approx 8/year)
    (1,31),(3,20),(5,1),(6,12),(7,31),(9,18),(11,7),(12,18),
    # NFP (approx first Friday each month)
    (1,5),(2,2),(3,1),(4,5),(5,3),(6,7),(7,5),(8,2),(9,6),(10,4),(11,1),(12,6),
]


class BacktestEngine:
    """Bar-by-bar replay with all six audit biases eliminated."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.loader = DataLoader(config)

    def run(
        self,
        symbols:              dict[str, list[str]],
        start:                str | datetime,
        end:                  str | datetime,
        timeframe:            str   = "1Day",
        initial_equity:       float = 100_000.0,
        warmup_bars:          int   = 200,
        per_symbol_warmup:    bool  = False,
        half_kelly:           bool  = True,
        event_calendar:       pd.DataFrame | None = None,
        historical_si:        dict | None = None,
        historical_rvol:      dict | None = None,
        default_si_fallback:  float = 0.10,
        default_rvol_fallback:float = 1.5,
    ) -> PerformanceReport:
        """
        Run a backtest.  All six lookahead / overfitting biases addressed:

        FIX 1  Same-bar fill  — orders stamped dt+1s so fill guard is unambiguous.
        FIX 2  Full-Kelly     — _kelly_fraction() patched to return half-Kelly.
        FIX 3  Low slippage   — defaults raised to 25 bps stocks / 30 bps crypto.
        FIX 4  Stale SI/RVOL  — historical dicts injected per bar; fallback non-zero.
        FIX 5  No news veto   — event_calendar blackouts replay veto on known dates.
        FIX 6  Per-symbol warmup — global warmup: no signals until ALL syms ready.
        """
        log.info("=" * 60)
        log.info("  Project Chimera — Backtest Engine (bias-corrected v2)")
        log.info(f"  Period      : {start} → {end}  |  TF: {timeframe}")
        log.info(f"  Equity      : ${initial_equity:,.0f}  |  Half-Kelly: {half_kelly}")
        log.info(f"  Symbols     : {symbols}")
        log.info("=" * 60)

        # ── Load OHLCV ─────────────────────────────────────────────────────
        all_data: dict[str, dict[str, pd.DataFrame]] = {}
        for sector, syms in symbols.items():
            all_data[sector] = self.loader.load(syms, start, end, timeframe, sector)
        if not any(all_data.values()):
            raise RuntimeError("No data loaded — check symbols and date range.")

        timeline = _build_timeline(all_data)
        log.info(f"Timeline bars: {len(timeline)}")

        # ── FIX 5: build event blackout set ────────────────────────────────
        blackout_dates = _build_blackout_set(event_calendar)
        log.info(f"Event blackout dates: {len(blackout_dates)}")

        # ── FIX 3: apply realistic slippage defaults ────────────────────────
        cfg = _apply_slippage_defaults(dict(self.config))

        # ── Initialise agents ──────────────────────────────────────────────
        state    = BacktestState(initial_equity=initial_equity)
        strategy = StrategyAgent(state, cfg)
        risk     = RiskAgent(state, cfg)

        # FIX 2: half-Kelly patch
        if half_kelly:
            _patch_half_kelly(risk)

        oms = SimulatedOMS(state, cfg)
        _patch_state_for_backtest(state)

        # ── Replay loop ────────────────────────────────────────────────────
        bar_counts: dict[str, int] = {}
        all_symbols = [sym for syms in symbols.values() for sym in syms]

        for dt, symbol, sector, bar in timeline:
            bar_counts[symbol] = bar_counts.get(symbol, 0) + 1
            date_str = dt.strftime("%Y-%m-%d")

            # FIX 5: replay news veto for macro event dates
            state.news.veto_active = date_str in blackout_dates
            if state.news.veto_active:
                state.news.veto_reason = "Macro event blackout (backtest calendar)"

            # Always inject bar — keeps TA windows rolling
            state.inject_bar(sector, symbol, bar, dt)

            # FIX 4: inject per-bar historical SI / RVOL
            if sector == "stocks" and symbol in state.market.stocks:
                key  = (symbol, date_str)
                si   = (historical_si   or {}).get(key, default_si_fallback)
                rvol = (historical_rvol or {}).get(key, default_rvol_fallback)
                state.market.stocks[symbol]["short_interest"] = si
                state.market.stocks[symbol]["rvol"]           = rvol

            # FIX 6: global warmup — all symbols must clear threshold
            if not per_symbol_warmup:
                if any(bar_counts.get(s, 0) < warmup_bars for s in all_symbols):
                    oms.on_bar(symbol, sector, bar, dt)   # stops/TPs still checked
                    state.record_equity()
                    continue
            else:
                if bar_counts[symbol] <= warmup_bars:
                    oms.on_bar(symbol, sector, bar, dt)
                    state.record_equity()
                    continue

            # On veto or circuit open: check stops but emit no new signals
            if state.news.veto_active or state.circuit_open:
                oms.on_bar(symbol, sector, bar, dt)
                state.record_equity()
                continue

            # Generate signals on bar N's close
            _run_strategy_sync(strategy, sector, symbol, state)

            for signal in state.drain_signals():
                _run_risk_sync(risk, signal, state)

            # FIX 1: stamp orders with dt + ONE_BAR_OFFSET
            # The fill guard (queued_dt < bar_dt) will always be True on
            # the next bar, never on the current bar, regardless of symbol order.
            for rp in state.drain_orders():
                oms.accept_order(rp, dt + ONE_BAR_OFFSET)

            # Fill pending entries at THIS bar's open, check stops/TPs
            oms.on_bar(symbol, sector, bar, dt)
            state.record_equity()

        # ── Force-close on last bar ────────────────────────────────────────
        last_prices = {
            sym: _last_close(all_data, sym)
            for sd in all_data.values() for sym in sd
        }
        last_dt = timeline[-1][0] if timeline else datetime.now(timezone.utc)
        oms.close_all(last_prices, last_dt)

        vetoed = sum(1 for dt, *_ in timeline if dt.strftime("%Y-%m-%d") in blackout_dates)
        log.info(f"Backtest complete — {len(state.closed_trades)} trades")
        log.info(f"Final equity: ${state.equity:,.2f}")
        log.info(f"Vetoed bars (macro calendar): {vetoed}/{len(timeline)} "
                 f"({vetoed/max(len(timeline),1)*100:.1f}%)")

        return PerformanceReport(
            trades         = state.closed_trades,
            equity_curve   = state.equity_curve,
            initial_equity = initial_equity,
            config         = cfg,
        )


# ── Fix helpers ───────────────────────────────────────────────────────────────

def _patch_half_kelly(risk: RiskAgent) -> None:
    """FIX 2: Wrap _kelly_fraction() to return half the computed value."""
    _orig = risk._kelly_fraction
    def _half() -> float:
        return _orig() * 0.5
    risk._kelly_fraction = _half
    log.info("Half-Kelly applied: kelly_fraction × 0.5")


def _apply_slippage_defaults(cfg: dict) -> dict:
    """
    FIX 3: raise slippage defaults to realistic levels for momentum strategies.

    Literature benchmarks for retail-accessible strategies:
      Stocks  (squeeze names): 15–35 bps per side (market impact + spread)
      Crypto  (non BTC/ETH):   20–50 bps per side
      Forex   (major pairs):   3–8 bps
      Futures (ES, NQ):        1–3 bps (very liquid)

    Using conservative mid-range values.
    """
    if "slippage_bps" not in cfg:
        cfg["slippage_bps"] = {}
    defaults = {"stocks": 25, "crypto": 30, "forex": 5, "futures": 3}
    for sector, bps in defaults.items():
        cfg["slippage_bps"].setdefault(sector, bps)
    log.info(f"Slippage (bps): {cfg['slippage_bps']}")
    return cfg


def _build_blackout_set(calendar: pd.DataFrame | None) -> set[str]:
    """FIX 5: Build a set of ISO date strings that trigger the news veto."""
    if calendar is not None:
        dates = pd.to_datetime(calendar["date"]).dt.strftime("%Y-%m-%d")
        return set(dates.tolist())
    # Approximate built-in calendar for years 2010–2030
    out: set[str] = set()
    for year in range(2010, 2031):
        for m, d in _APPROX_BLACKOUT_MD:
            try:
                out.add(f"{year}-{m:02d}-{d:02d}")
            except ValueError:
                pass
    return out


def _run_strategy_sync(strategy: StrategyAgent, sector: str,
                       symbol: str, state: BacktestState) -> None:
    method = {
        "crypto": strategy._sector_crypto,
        "stocks": strategy._sector_stocks,
        "forex":  strategy._sector_forex,
        "futures":strategy._sector_futures,
    }.get(sector)
    if method:
        _sync_run(method())


def _run_risk_sync(risk: RiskAgent, signal: Any, state: BacktestState) -> None:
    _sync_run(risk._process(signal))


def _sync_run(coro) -> None:
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(coro)
    finally:
        loop.close()


def _patch_state_for_backtest(state: BacktestState) -> None:
    async def _put_sig(s):  state.put_signal_sync(s)
    async def _put_ord(r):  state.put_order_sync(r)
    state.put_signal = _put_sig
    state.put_order  = _put_ord


def _build_timeline(all_data: dict) -> list[tuple[datetime, str, str, dict]]:
    rows = []
    for sector, sd in all_data.items():
        for sym, df in sd.items():
            for dt, row in df.iterrows():
                rows.append((
                    dt.to_pydatetime() if hasattr(dt, "to_pydatetime") else dt,
                    sym, sector,
                    {"open": float(row["open"]), "high": float(row["high"]),
                     "low":  float(row["low"]),  "close":float(row["close"]),
                     "volume": float(row["volume"])},
                ))
    rows.sort(key=lambda x: x[0])
    return rows


def _last_close(all_data: dict, symbol: str) -> float:
    for sd in all_data.values():
        if symbol in sd and not sd[symbol].empty:
            return float(sd[symbol]["close"].iloc[-1])
    return 0.0
