"""
chimera_v12/backtest/monte_carlo.py
Monte Carlo simulator — resamples trade R-multiples to estimate return
distribution under path-dependency.

All Sharpe/Sortino metrics use R-multiples and correct tpy annualisation
(not sqrt(252)) matching performance.py.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from chimera_v12.backtest.performance import (
    rolling_sortino_r,
    rolling_sharpe_r,
    DEFAULT_RISK_FREE_RATE,
    DEFAULT_SORTINO_WINDOW,
)
from chimera_v12.utils.logger import setup_logger

log = setup_logger("backtest.monte_carlo")


@dataclass
class MonteCarloResult:
    n_simulations:     int
    initial_equity:    float
    # Equity distribution
    terminal_mean:     float
    terminal_median:   float
    terminal_p5:       float      # 5th percentile (bad case)
    terminal_p95:      float      # 95th percentile (good case)
    # R-metric distribution
    sortino_mean:      float
    sortino_p5:        float
    sortino_p95:       float
    sharpe_mean:       float
    # Drawdown distribution
    max_dd_mean:       float
    max_dd_p95:        float
    # Ruin
    ruin_rate:         float      # fraction of paths below ruin threshold
    ruin_threshold_pct: float


class MonteCarloEngine:
    """
    Bootstraps the closed trade R-multiples from a completed backtest
    to simulate N alternative equity paths. Reports percentile distribution
    of terminal equity, Sortino (R-based), and max drawdown.

    Usage:
        engine = MonteCarloEngine(config)
        result = engine.run(trades, n_simulations=2000)
        engine.print_summary(result)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def run(
        self,
        trades:        list[dict[str, Any]],
        initial_equity: float           = 100_000.0,
        n_simulations:  int             = 1_000,
        backtest_days:  int             = 252,
        ruin_threshold_pct: float       = 50.0,
        seed:           int | None      = None,
    ) -> MonteCarloResult:
        """
        Args:
            trades:             closed_trades list from BacktestEngine
            initial_equity:     starting account size
            n_simulations:      bootstrap paths to simulate
            backtest_days:      original backtest duration (for tpy estimate)
            ruin_threshold_pct: account drawdown % below which = "ruin"
            seed:               RNG seed for reproducibility
        """
        if not trades:
            raise ValueError("No trades to simulate.")

        rng = random.Random(seed)
        rs  = [t["r_multiple"]   for t in trades]
        pnls_raw = [t["realised_pnl"] for t in trades]
        n_trades = len(rs)
        avg_risk = abs(sum(pnls_raw) / sum(rs)) if sum(rs) != 0 else initial_equity * 0.01

        tpy    = max(n_trades / max(backtest_days / 365.25, 0.01), 1.0)
        rfr_pt = self.config.get("risk_free_rate", DEFAULT_RISK_FREE_RATE) / tpy
        window = self.config.get("sortino_window", DEFAULT_SORTINO_WINDOW)
        ruin_floor = initial_equity * (1 - ruin_threshold_pct / 100)

        log.info(
            f"Monte Carlo: {n_simulations} sims × {n_trades} trades  "
            f"tpy={tpy:.0f}  rfr/trade={rfr_pt:.5f}"
        )

        terminals  = []
        sortinos   = []
        sharpes    = []
        max_dds    = []
        ruin_count = 0

        for _ in range(n_simulations):
            sampled_rs   = rng.choices(rs, k=n_trades)
            equity       = initial_equity
            peak         = initial_equity
            max_dd_frac  = 0.0

            for r in sampled_rs:
                pnl     = r * avg_risk
                equity  = max(0.0, equity + pnl)
                peak    = max(peak, equity)
                dd_frac = (peak - equity) / peak if peak > 0 else 0.0
                max_dd_frac = max(max_dd_frac, dd_frac)

            terminals.append(equity)
            max_dds.append(max_dd_frac * 100)
            if equity <= ruin_floor:
                ruin_count += 1

            sortinos.append(rolling_sortino_r(sampled_rs, rfr_pt, len(sampled_rs), tpy))
            sharpes.append(rolling_sharpe_r(sampled_rs,  rfr_pt, len(sampled_rs), tpy))

        terminals.sort()
        max_dds.sort()
        sortinos.sort()

        def pct(lst: list[float], p: float) -> float:
            idx = int(len(lst) * p / 100)
            return round(lst[min(idx, len(lst) - 1)], 4)

        result = MonteCarloResult(
            n_simulations      = n_simulations,
            initial_equity     = initial_equity,
            terminal_mean      = round(sum(terminals) / len(terminals), 2),
            terminal_median    = round(terminals[len(terminals) // 2], 2),
            terminal_p5        = round(pct(terminals, 5), 2),
            terminal_p95       = round(pct(terminals, 95), 2),
            sortino_mean       = round(sum(sortinos) / len(sortinos), 4),
            sortino_p5         = round(pct(sortinos, 5), 4),
            sortino_p95        = round(pct(sortinos, 95), 4),
            sharpe_mean        = round(sum(sharpes) / len(sharpes), 4),
            max_dd_mean        = round(sum(max_dds) / len(max_dds), 3),
            max_dd_p95         = round(pct(max_dds, 95), 3),
            ruin_rate          = round(ruin_count / n_simulations, 4),
            ruin_threshold_pct = ruin_threshold_pct,
        )

        log.info(
            f"MC complete — terminal median ${result.terminal_median:,.0f}  "
            f"ruin={result.ruin_rate:.1%}  "
            f"sortino_mean={result.sortino_mean:.3f}"
        )
        return result

    def print_summary(self, r: MonteCarloResult) -> None:
        w   = 54
        sep = "─" * w

        def row(label: str, val: Any, width: int = 34) -> str:
            return f"  {label:<{width}} {val}"

        print(f"\n╔{'═' * w}╗")
        print(f"║{'MONTE CARLO SIMULATION REPORT':^{w}}║")
        print(f"╚{'═' * w}╝")
        print(f"\n{sep}\n  SETUP\n{sep}")
        print(row("Simulations:", f"{r.n_simulations:,}"))
        print(row("Initial equity:", f"${r.initial_equity:,.2f}"))
        print(row("Ruin threshold:", f"{r.ruin_threshold_pct:.0f}% drawdown"))

        print(f"\n{sep}\n  TERMINAL EQUITY DISTRIBUTION\n{sep}")
        print(row("Mean:", f"${r.terminal_mean:,.2f}"))
        print(row("Median:", f"${r.terminal_median:,.2f}"))
        print(row("5th pct (bad case):", f"${r.terminal_p5:,.2f}"))
        print(row("95th pct (good case):", f"${r.terminal_p95:,.2f}"))

        print(f"\n{sep}\n  ROLLING SORTINO (R-based, downside-only)\n{sep}")
        print(row("Mean Sortino:", f"{r.sortino_mean:.4f}"))
        print(row("5th pct Sortino:", f"{r.sortino_p5:.4f}"))
        print(row("95th pct Sortino:", f"{r.sortino_p95:.4f}"))
        print(row("Mean Sharpe (reference):", f"{r.sharpe_mean:.4f}"))

        print(f"\n{sep}\n  DRAWDOWN DISTRIBUTION\n{sep}")
        print(row("Mean max drawdown:", f"{r.max_dd_mean:.2f}%"))
        print(row("95th pct max drawdown:", f"{r.max_dd_p95:.2f}%"))
        print(row("Ruin probability:", f"{r.ruin_rate:.1%}"))
        if r.ruin_rate > 0.05:
            print(f"\n  ⚠️  Ruin > 5% — reduce position sizing before live trading")
        print()
