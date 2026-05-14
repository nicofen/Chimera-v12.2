"""
chimera/backtest/run_backtest.py
CLI entry point — run a backtest from the command line.

Usage examples:
    # Stocks — last 2 years, daily bars
    python -m chimera.backtest.run_backtest \\
        --sector stocks \\
        --symbols GME AMC TSLA \\
        --start 2022-01-01 --end 2024-01-01 \\
        --timeframe 1Day \\
        --equity 100000

    # Crypto — 6 months, 4-hour bars
    python -m chimera.backtest.run_backtest \\
        --sector crypto \\
        --symbols BTC/USD ETH/USD \\
        --start 2023-06-01 --end 2024-01-01 \\
        --timeframe 4Hour \\
        --equity 50000

    # Multi-sector
    python -m chimera.backtest.run_backtest \\
        --sector stocks crypto \\
        --symbols GME BTC/USD \\
        --start 2023-01-01 --end 2024-01-01

    # Save results to JSON
    python -m chimera.backtest.run_backtest \\
        --symbols GME --sector stocks \\
        --start 2023-01-01 --end 2024-01-01 \\
        --output results.json

    # Load from local CSV instead of Alpaca
    python -m chimera.backtest.run_backtest \\
        --csv ./data/GME_daily.csv --symbol GME --sector stocks \\
        --start 2023-01-01 --end 2024-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Chimera — Backtest Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--symbols",   nargs="+", required=False, default=[])
    parser.add_argument("--symbol",    type=str,  help="Single symbol (for --csv mode)")
    parser.add_argument("--sector",    nargs="+", default=["stocks"],
                        choices=["stocks", "crypto", "forex", "futures"])
    parser.add_argument("--start",     required=True,  help="Start date YYYY-MM-DD")
    parser.add_argument("--end",       required=True,  help="End date YYYY-MM-DD")
    parser.add_argument("--timeframe", default="1Day",
                        choices=["1Min","5Min","15Min","1Hour","4Hour","1Day"])
    parser.add_argument("--equity",    type=float, default=100_000.0)
    parser.add_argument("--warmup",    type=int,   default=200,
                        help="Warmup bars before signals start")
    parser.add_argument("--csv",       type=str,   default=None,
                        help="Path to local CSV file (skips Alpaca fetch)")
    parser.add_argument("--output",    type=str,   default=None,
                        help="Save JSON results to this file path")
    parser.add_argument("--no-cache",  action="store_true",
                        help="Disable local Parquet caching")
    args = parser.parse_args()

    # Load config (reads from .env / environment)
    try:
        from chimera_v12.config.settings import load_config
        config = load_config()
    except Exception as e:
        print(f"Config error: {e}")
        print("Set ALPACA_KEY, ALPACA_SECRET, OPENAI_API_KEY in .env")
        sys.exit(1)

    if args.no_cache:
        config["cache_data"] = False

    from chimera_v12.backtest.engine import BacktestEngine
    engine = BacktestEngine(config)

    # ── CSV mode ──────────────────────────────────────────────────────────
    if args.csv:
        sym    = args.symbol or Path(args.csv).stem
        sec    = args.sector[0]
        loader = engine.loader
        data   = loader.load_csv(args.csv, sym)
        # Inject directly into engine's loader cache
        from chimera_v12.backtest.state import BacktestState
        from chimera_v12.backtest.simulated_oms import SimulatedOMS
        from chimera_v12.backtest.performance import PerformanceReport
        from chimera_v12.agents.strategy_agent import StrategyAgent
        from chimera_v12.agents.risk_agent import RiskAgent
        from chimera_v12.backtest.engine import (
            _build_timeline, _run_strategy_sync, _run_risk_sync,
            _patch_state_for_backtest, _last_close,
        )

        all_data = {sec: data}
        timeline = _build_timeline(all_data)
        state    = BacktestState(initial_equity=args.equity)
        strategy = StrategyAgent(state, config)
        risk     = RiskAgent(state, config)
        oms      = SimulatedOMS(state, config)
        _patch_state_for_backtest(state, strategy, risk)

        bar_counts: dict[str, int] = {}
        for dt, symbol, sector, bar in timeline:
            bar_counts[symbol] = bar_counts.get(symbol, 0) + 1
            state.inject_bar(sector, symbol, bar, dt)
            if bar_counts[symbol] <= args.warmup:
                continue
            _run_strategy_sync(strategy, sector, symbol, state)
            for signal in state.drain_signals():
                _run_risk_sync(risk, signal, state)
            for rp in state.drain_orders():
                oms.accept_order(rp, dt)
            oms.on_bar(symbol, sector, bar, dt)
            state.record_equity()

        last_prices = {sym: _last_close(all_data, sym)}
        import datetime as _dt
        oms.close_all(last_prices, timeline[-1][0] if timeline else _dt.datetime.now(timezone.utc))

        report = PerformanceReport(
            trades=state.closed_trades,
            equity_curve=state.equity_curve,
            initial_equity=args.equity,
            config=config,
        )

    else:
        # ── Alpaca mode ───────────────────────────────────────────────────
        if not args.symbols:
            parser.error("--symbols required (or use --csv for local data)")

        # Map symbols to sectors — if multiple sectors given, first sector
        # applies to all symbols (for finer control use the Python API directly)
        symbols_map: dict[str, list[str]] = {}
        for sec in args.sector:
            symbols_map[sec] = args.symbols

        report = engine.run(
            symbols        = symbols_map,
            start          = args.start,
            end            = args.end,
            timeframe      = args.timeframe,
            initial_equity = args.equity,
            warmup_bars    = args.warmup,
        )

    # ── Output ────────────────────────────────────────────────────────────
    report.print_summary()

    if args.output:
        metrics = report.compute()
        # Make JSON serialisable (convert datetime keys in equity_curve)
        out = {k: v for k, v in metrics.items()}
        Path(args.output).write_text(json.dumps(out, indent=2, default=str))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
