"""
chimera_v12/backtest/performance.py
═══════════════════════════════════════════════════════════════════════════════
PerformanceReport — all standard quant metrics from a completed backtest.

METRIC DESIGN DECISIONS
───────────────────────
R-MULTIPLES over dollar PnL
    Raw PnL mixes sizing luck with edge. R-multiples (trade PnL ÷ initial risk)
    isolate the strategy's edge from position-sizing noise. A +3R trade is +3R
    regardless of whether risk was $100 or $10,000.

ROLLING SORTINO over Sharpe
    1. Sharpe penalises upside volatility equally with downside — wrong.
    2. Sortino penalises ONLY negative R trades (downside semi-deviation).
    3. Rolling window (default 30 trades) surfaces edge degradation early.
    Formula: (mean_R − rfr_pt) / std(R_neg) × √(trades_per_year)
    where R_neg = [R for R in window if R < rfr_pt]

CORRECT ANNUALISATION: √(trades_per_year) not √252
    √252 assumes daily data. For trade-based R-series, annualise by
    √(expected trades per year). Hardcoding √252 inflates Sharpe by 2–5×
    for weekly/monthly systems.

RISK-FREE RATE
    Configurable via config["risk_free_rate"] (default 5%/yr).
    Per-trade equivalent: rfr_pt = annual_rfr / trades_per_year.
═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any

DEFAULT_SORTINO_WINDOW = 30
DEFAULT_RISK_FREE_RATE = 0.05
SORTINO_CAP            = 10.0


class PerformanceReport:
    def __init__(
        self,
        trades:         list[dict[str, Any]],
        equity_curve:   list[tuple[datetime, float]],
        initial_equity: float,
        config:         dict[str, Any] | None = None,
    ):
        self.trades         = trades
        self.equity_curve   = equity_curve
        self.initial_equity = initial_equity
        self.config         = config or {}

    @property
    def _rfr_annual(self) -> float:
        return float(self.config.get("risk_free_rate", DEFAULT_RISK_FREE_RATE))

    @property
    def _sortino_window(self) -> int:
        return int(self.config.get("sortino_window", DEFAULT_SORTINO_WINDOW))

    def _trades_per_year(self, n_trades: int, n_days: int) -> float:
        """Estimate annual trade frequency from observed count and duration."""
        if n_days <= 0:
            return 252.0
        return max(n_trades / max(n_days / 365.25, 0.01), 1.0)

    def compute(self) -> dict[str, Any]:
        if not self.trades:
            return {"error": "No closed trades."}

        m: dict[str, Any] = {}
        pnls   = [t["realised_pnl"] for t in self.trades]
        rs     = [t["r_multiple"]   for t in self.trades]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        # ── Core stats ────────────────────────────────────────────────────────
        m["total_trades"]    = len(self.trades)
        m["winning_trades"]  = len(wins)
        m["losing_trades"]   = len(losses)
        m["win_rate"]        = round(len(wins) / len(pnls), 4)
        m["gross_profit"]    = round(sum(wins), 2)
        m["gross_loss"]      = round(abs(sum(losses)), 2)
        m["net_profit"]      = round(sum(pnls), 2)
        m["profit_factor"]   = round(
            sum(wins) / abs(sum(losses)) if losses else float("inf"), 3
        )
        m["commission_total"] = round(
            sum(t.get("commission", 0) for t in self.trades), 2
        )

        # ── Return metrics ────────────────────────────────────────────────────
        final_eq = self.initial_equity + m["net_profit"]
        m["initial_equity"]   = round(self.initial_equity, 2)
        m["final_equity"]     = round(final_eq, 2)
        m["total_return_pct"] = round((final_eq / self.initial_equity - 1) * 100, 3)

        n_days = 0
        if self.equity_curve and len(self.equity_curve) >= 2:
            n_days = (self.equity_curve[-1][0] - self.equity_curve[0][0]).days
            years  = max(n_days / 365.25, 1 / 365.25)
            m["backtest_days"] = n_days
            m["cagr_pct"]      = round(
                ((final_eq / self.initial_equity) ** (1 / years) - 1) * 100, 3
            ) if final_eq > 0 and self.initial_equity > 0 else 0.0
        else:
            m["backtest_days"] = 0
            m["cagr_pct"]      = 0.0

        # ── R-multiple stats ──────────────────────────────────────────────────
        win_rs  = [r for r in rs if r > 0]
        loss_rs = [r for r in rs if r <= 0]
        m["avg_r"]          = round(_mean(rs), 4)
        m["avg_win_r"]      = round(_mean(win_rs),  4) if win_rs  else 0.0
        m["avg_loss_r"]     = round(_mean(loss_rs), 4) if loss_rs else 0.0
        m["expectancy_r"]   = round(
            m["win_rate"] * m["avg_win_r"]
            + (1 - m["win_rate"]) * m["avg_loss_r"], 4
        )
        m["best_trade_r"]   = round(max(rs), 4)
        m["worst_trade_r"]  = round(min(rs), 4)
        m["std_r"]          = round(_std(rs), 4)

        # ── Drawdown ──────────────────────────────────────────────────────────
        if self.equity_curve:
            dd_pct, dd_dur        = _max_drawdown(self.equity_curve)
            m["max_drawdown_pct"] = round(dd_pct * 100, 3)
            m["max_drawdown_days"]= dd_dur
            m["calmar_ratio"]     = round(
                m["cagr_pct"] / m["max_drawdown_pct"]
                if m["max_drawdown_pct"] > 0 else 0.0, 3
            )
        else:
            m["max_drawdown_pct"] = 0.0
            m["max_drawdown_days"]= 0
            m["calmar_ratio"]     = 0.0

        # ── Rolling Sortino + Sharpe on R-multiples ───────────────────────────
        tpy    = self._trades_per_year(len(rs), n_days)
        rfr_pt = self._rfr_annual / tpy
        window = self._sortino_window

        if len(rs) >= 5:
            m["sortino_ratio"]      = round(rolling_sortino_r(rs, rfr_pt, len(rs), tpy), 4)
            m["sharpe_ratio"]       = round(rolling_sharpe_r(rs,  rfr_pt, len(rs), tpy), 4)
            m["rolling_sortino_r"]  = round(rolling_sortino_r(rs, rfr_pt, window,  tpy), 4)
        else:
            m["sortino_ratio"]      = 0.0
            m["sharpe_ratio"]       = 0.0
            m["rolling_sortino_r"]  = 0.0

        m["rolling_sortino_series"] = _rolling_sortino_series(rs, rfr_pt, window, tpy)
        m["trades_per_year_est"]    = round(tpy, 1)
        m["rfr_annual_pct"]         = round(self._rfr_annual * 100, 2)
        m["sortino_window_trades"]  = window

        # ── Streaks ───────────────────────────────────────────────────────────
        m["max_win_streak"]  = _max_streak(pnls, positive=True)
        m["max_loss_streak"] = _max_streak(pnls, positive=False)

        # ── Per-sector breakdown ──────────────────────────────────────────────
        by_sector: dict[str, list] = defaultdict(list)
        for t in self.trades:
            by_sector[t.get("sector", "unknown")].append(t)

        sector_stats = {}
        for sec, st in by_sector.items():
            sec_pnls = [t["realised_pnl"] for t in st]
            sec_rs   = [t["r_multiple"]   for t in st]
            sec_wins = sum(1 for p in sec_pnls if p > 0)
            sec_tpy  = self._trades_per_year(len(sec_rs), n_days)
            sec_rfr  = self._rfr_annual / sec_tpy
            sector_stats[sec] = {
                "trades":       len(st),
                "win_rate":     round(sec_wins / len(st), 3),
                "avg_r":        round(_mean(sec_rs), 3),
                "net_pnl":      round(sum(sec_pnls), 2),
                "expectancy_r": round(
                    (sec_wins / len(st)) * _mean([r for r in sec_rs if r > 0])
                    + (1 - sec_wins / len(st)) * _mean([r for r in sec_rs if r <= 0]),
                    3,
                ),
                "sortino_r": round(
                    rolling_sortino_r(sec_rs, sec_rfr, len(sec_rs), sec_tpy), 3
                ) if len(sec_rs) >= 3 else 0.0,
            }
        m["by_sector"] = sector_stats

        # ── Exit reasons ──────────────────────────────────────────────────────
        by_reason: dict[str, int] = defaultdict(int)
        for t in self.trades:
            by_reason[t.get("close_reason", "unknown")] += 1
        m["close_reasons"] = dict(by_reason)

        return m

    def print_summary(self) -> None:
        m = self.compute()
        if "error" in m:
            print(m["error"])
            return
        w   = 56
        sep = "─" * w

        def row(label: str, val: Any, width: int = 34) -> str:
            return f"  {label:<{width}} {val}"

        print(f"\n╔{'═' * w}╗")
        print(f"║{'PROJECT CHIMERA — BACKTEST REPORT':^{w}}║")
        print(f"╚{'═' * w}╝")

        print(f"\n{sep}\n  OVERVIEW\n{sep}")
        print(row("Backtest period:",  f"{m['backtest_days']} days"))
        print(row("Est. trades/year:", f"~{m['trades_per_year_est']:.0f}"))
        print(row("Initial equity:",   f"${m['initial_equity']:,.2f}"))
        print(row("Final equity:",     f"${m['final_equity']:,.2f}"))
        print(row("Net profit:",       f"${m['net_profit']:,.2f}"))
        print(row("Total return:",     f"{m['total_return_pct']:.2f}%"))
        print(row("CAGR:",             f"{m['cagr_pct']:.2f}%"))
        print(row("Commissions:",      f"${m['commission_total']:,.2f}"))

        print(f"\n{sep}\n  TRADE STATISTICS (R-multiples)\n{sep}")
        print(row("Total trades:",    m['total_trades']))
        print(row("Win rate:",        f"{m['win_rate']*100:.1f}%  "
                                      f"({m['winning_trades']}W / {m['losing_trades']}L)"))
        print(row("Profit factor:",   f"{m['profit_factor']:.3f}"))
        print(row("Expectancy:",      f"{m['expectancy_r']:+.4f}R per trade"))
        print(row("Avg R:",           f"{m['avg_r']:+.4f}R"))
        print(row("Avg win R:",       f"{m['avg_win_r']:+.4f}R"))
        print(row("Avg loss R:",      f"{m['avg_loss_r']:+.4f}R"))
        print(row("Std R:",           f"{m['std_r']:.4f}"))
        print(row("Best trade:",      f"{m['best_trade_r']:+.3f}R"))
        print(row("Worst trade:",     f"{m['worst_trade_r']:+.3f}R"))
        print(row("Max win streak:",  m['max_win_streak']))
        print(row("Max loss streak:", m['max_loss_streak']))

        print(f"\n{sep}\n  RISK-ADJUSTED METRICS\n{sep}")
        print(row("Max drawdown:",
                  f"{m['max_drawdown_pct']:.2f}%  ({m['max_drawdown_days']} days)"))
        print(row("Calmar ratio:", f"{m['calmar_ratio']:.3f}"))
        print(f"\n  ── Rolling Sortino (R-based, downside-only) ──────")
        print(row("Full-history Sortino:",
                  f"{m['sortino_ratio']:.4f}  "
                  f"[rfr={m['rfr_annual_pct']:.1f}% "
                  f"tpy≈{m['trades_per_year_est']:.0f}]"))
        w_n = m['sortino_window_trades']
        print(row(f"Rolling Sortino (last {w_n}):", f"{m['rolling_sortino_r']:.4f}"))
        print(f"\n  ── Sharpe (R-based, symmetric volatility) ────────")
        print(row("Full-history Sharpe:", f"{m['sharpe_ratio']:.4f}"))
        print(f"  Note: use Sortino — Sharpe penalises winning trades")

        print(f"\n{sep}\n  BY SECTOR\n{sep}")
        for sec, stats in m["by_sector"].items():
            print(row(f"  {sec.upper()}:",
                      f"{stats['trades']}tr  "
                      f"WR={stats['win_rate']*100:.0f}%  "
                      f"E={stats['expectancy_r']:+.3f}R  "
                      f"Sortino={stats['sortino_r']:.3f}  "
                      f"P&L=${stats['net_pnl']:,.0f}"))

        print(f"\n{sep}\n  EXIT REASONS\n{sep}")
        for reason, count in m["close_reasons"].items():
            print(row(f"  {reason}:", count))
        print()


# ══════════════════════════════════════════════════════════════════════════════
# Core metric functions (importable standalone)
# ══════════════════════════════════════════════════════════════════════════════

def rolling_sortino_r(
    rs:     list[float],
    rfr_pt: float,
    window: int,
    tpy:    float,
) -> float:
    """
    Rolling Sortino on R-multiples.

    Args:
        rs:     R-multiples list (full history or window slice)
        rfr_pt: per-trade risk-free rate = annual_rfr / trades_per_year
        window: how many of the most recent trades to include
        tpy:    estimated trades per year (annualisation denominator)

    Formula:
        window_rs = rs[-window:]
        mean_R    = mean(window_rs)
        R_neg     = [r for r in window_rs if r < rfr_pt]
        ds_std    = sample_std(R_neg)
        sortino   = (mean_R - rfr_pt) / ds_std × sqrt(tpy)

    Returns float capped to [-SORTINO_CAP, +SORTINO_CAP].
    """
    if not rs:
        return 0.0
    w_rs = rs[-window:]
    if len(w_rs) < 2:
        return 0.0
    mean_r = _mean(w_rs)
    neg_rs = [r for r in w_rs if r < rfr_pt]
    if not neg_rs:
        return SORTINO_CAP if mean_r > rfr_pt else 0.0
    ds_std = _std(neg_rs)
    if ds_std == 0:
        return SORTINO_CAP if mean_r > rfr_pt else 0.0
    raw = (mean_r - rfr_pt) / ds_std * math.sqrt(max(tpy, 1.0))
    return round(max(-SORTINO_CAP, min(SORTINO_CAP, raw)), 4)


def rolling_sharpe_r(
    rs:     list[float],
    rfr_pt: float,
    window: int,
    tpy:    float,
) -> float:
    """
    Rolling Sharpe on R-multiples (kept for comparability; prefer Sortino).
    Uses tpy not 252 — avoids daily-data inflation for low-frequency systems.
    """
    if not rs:
        return 0.0
    w_rs   = rs[-window:]
    if len(w_rs) < 2:
        return 0.0
    excess = [r - rfr_pt for r in w_rs]
    mu     = _mean(excess)
    std    = _std(excess)
    if std == 0:
        return 0.0
    raw = mu / std * math.sqrt(max(tpy, 1.0))
    return round(max(-10.0, min(10.0, raw)), 4)


def _rolling_sortino_series(
    rs:     list[float],
    rfr_pt: float,
    window: int,
    tpy:    float,
) -> list[float]:
    """One Sortino value per trade for the rolling chart sparkline."""
    series = []
    for i in range(len(rs)):
        if i < max(window // 2, 2):
            series.append(0.0)
            continue
        w_slice = rs[max(0, i - window + 1): i + 1]
        series.append(rolling_sortino_r(w_slice, rfr_pt, len(w_slice), tpy))
    return series


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: list[float]) -> float:
    """Bessel-corrected sample std (n-1 denominator)."""
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1))


def _max_drawdown(curve: list[tuple[datetime, float]]) -> tuple[float, int]:
    peak        = curve[0][1]
    peak_dt     = curve[0][0]
    max_dd      = 0.0
    max_dd_days = 0
    for dt, eq in curve[1:]:
        if eq > peak:
            peak, peak_dt = eq, dt
        dd = (peak - eq) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd      = dd
            max_dd_days = (dt - peak_dt).days
    return max_dd, max_dd_days


def _max_streak(pnls: list[float], positive: bool) -> int:
    max_s = cur_s = 0
    for p in pnls:
        if (positive and p > 0) or (not positive and p <= 0):
            cur_s += 1
            max_s  = max(max_s, cur_s)
        else:
            cur_s = 0
    return max_s
