"""
chimera/backtest/run_monte_carlo.py
CLI entry point — run Monte Carlo stress-testing on a saved backtest result.

Usage:
    # Run on a saved backtest JSON
    python -m chimera.backtest.run_monte_carlo \\
        --trades results.json \\
        --n 5000 \\
        --equity 100000

    # Block bootstrap mode (preserves streaks)
    python -m chimera.backtest.run_monte_carlo \\
        --trades results.json \\
        --mode block --block-size 5

    # Save output for visualiser
    python -m chimera.backtest.run_monte_carlo \\
        --trades results.json \\
        --output mc_result.json

    # Use a fixed seed for reproducible results
    python -m chimera.backtest.run_monte_carlo \\
        --trades results.json \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Chimera — Monte Carlo stress tester"
    )
    parser.add_argument("--trades",      required=True, help="Path to backtest JSON (from --output)")
    parser.add_argument("--n",           type=int,   default=5_000,    help="Number of simulations")
    parser.add_argument("--equity",      type=float, default=100_000,  help="Initial equity")
    parser.add_argument("--ruin",        type=float, default=0.50,     help="Ruin threshold (fraction of initial)")
    parser.add_argument("--mode",        choices=["bootstrap", "block"], default="bootstrap")
    parser.add_argument("--block-size",  type=int,   default=5,        help="Block size for block bootstrap")
    parser.add_argument("--seed",        type=int,   default=None,     help="Random seed")
    parser.add_argument("--output",      type=str,   default=None,     help="Save JSON output to file")
    args = parser.parse_args()

    # ── Load trades ───────────────────────────────────────────────────────────
    trades_path = Path(args.trades)
    if not trades_path.exists():
        print(f"ERROR: file not found: {trades_path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(trades_path.read_text())

    # Accept either a full report dict (with "trades" key) or a raw trade list
    if isinstance(data, dict) and "trades" in data:
        trades = data["trades"]
        if args.equity == 100_000 and "initial_equity" in data:
            args.equity = float(data["initial_equity"])
    elif isinstance(data, list):
        trades = data
    else:
        print("ERROR: JSON must be a trade list or a report dict with a 'trades' key.", file=sys.stderr)
        sys.exit(1)

    if not trades:
        print("ERROR: No trades found in input file.", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Project Chimera — Monte Carlo Simulator")
    print(f"  ────────────────────────────────────────")
    print(f"  Trades loaded  : {len(trades)}")
    print(f"  Simulations    : {args.n:,}")
    print(f"  Initial equity : ${args.equity:,.0f}")
    print(f"  Mode           : {args.mode}")
    print(f"  Ruin threshold : {args.ruin * 100:.0f}% of initial")
    if args.seed:
        print(f"  Seed           : {args.seed}")
    print()

    # ── Run ───────────────────────────────────────────────────────────────────
    from chimera_v12.backtest.monte_carlo import MonteCarloSimulator

    simulator = MonteCarloSimulator()
    result = simulator.run(
        trades          = trades,
        initial_equity  = args.equity,
        n_simulations   = args.n,
        ruin_threshold  = args.ruin,
        mode            = args.mode,
        block_size      = args.block_size,
        seed            = args.seed,
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    s = result.summary()
    w = 42
    sep = "─" * w

    def row(lbl, val, w=26):
        return f"  {lbl:<{w}} {val}"

    print(f"\n  {sep}")
    print(f"  TERMINAL EQUITY DISTRIBUTION")
    print(f"  {sep}")
    print(row("5th  percentile:", f"${s['terminal_p05']:>12,.2f}"))
    print(row("25th percentile:", f"${s['terminal_p25']:>12,.2f}"))
    print(row("50th (median):", f"${s['terminal_p50']:>12,.2f}"))
    print(row("75th percentile:", f"${s['terminal_p75']:>12,.2f}"))
    print(row("95th percentile:", f"${s['terminal_p95']:>12,.2f}"))
    print(row("Mean:", f"${s['mean_terminal']:>12,.2f}"))
    print(row("Std deviation:", f"${s['std_terminal']:>12,.2f}"))

    print(f"\n  {sep}")
    print(f"  RISK METRICS")
    print(f"  {sep}")
    print(row("Probability of profit:", f"{s['prob_profit_pct']:>7.1f}%"))
    print(row("Probability of ruin:", f"{s['ruin_prob_pct']:>7.1f}%"))
    print(row("Median max drawdown:", f"{s['median_max_dd_pct']:>7.2f}%"))
    print(row("CVaR (worst 5%):", f"${s['cvar_5pct']:>12,.2f}"))
    print()

    # ── Save output ───────────────────────────────────────────────────────────
    if args.output:
        output = {
            "summary":        s,
            "curve_p05":      result.curve_p05,
            "curve_p25":      result.curve_p25,
            "curve_p50":      result.curve_p50,
            "curve_p75":      result.curve_p75,
            "curve_p95":      result.curve_p95,
            "curve_original": result.curve_original,
        }
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"  Results saved to {args.output}")


if __name__ == "__main__":
    main()
