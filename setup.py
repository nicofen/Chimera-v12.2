"""
chimera/config/settings.py
Load configuration from environment variables or a .env file.

ALWAYS start with mode="paper" until you have 3+ months of
consistent paper-trading P&L. Use mode="live" only when you
are prepared to manage real capital drawdowns.
"""

import os
from typing import Any


def load_config() -> dict[str, Any]:
    """
    Returns the Chimera configuration dict.
    Secrets are read from environment variables — never hardcode them.

    Required env vars:
      ALPACA_KEY, ALPACA_SECRET, OPENAI_API_KEY

    Optional:
      WHALE_ALERT_KEY, CMC_API_KEY, DUNE_API_KEY, FINANCIALJUICE_URL
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    return {
        # ── Mode ──────────────────────────────────────────────────────────────
        "mode": os.getenv("CHIMERA_MODE", "paper"),   # "paper" | "live"

        # ── API Keys ──────────────────────────────────────────────────────────
        "alpaca_key":    os.getenv("ALPACA_KEY",    ""),   # required for live trading
        "alpaca_secret": os.getenv("ALPACA_SECRET", ""),   # required for live trading
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""), # required for LLM agents
        "whale_alert_key":  os.getenv("WHALE_ALERT_KEY", ""),
        "cmc_api_key":      os.getenv("CMC_API_KEY", ""),
        "dune_api_key":     os.getenv("DUNE_API_KEY", ""),
        "financialjuice_url": os.getenv(
            "FINANCIALJUICE_URL",
            "https://www.financialjuice.com/feed.ashx?c=market",
        ),

        # ── Symbols ───────────────────────────────────────────────────────────
        "stock_symbols":      ["AAPL", "TSLA", "GME", "AMC", "BBBY"],
        "crypto_symbols":     ["BTC/USD", "ETH/USD", "SOL/USD"],
        "forex_pairs":        ["EUR/USD", "GBP/USD", "USD/JPY"],
        "futures_contracts":  ["ES1!"],

        # ── Risk ──────────────────────────────────────────────────────────────
        "base_risk_pct":    0.01,   # 1% of equity per trade (paper trading default)
        "kelly_lookback":   50,     # trades to include in Kelly calculation
        "avg_win_r":        1.5,    # expected win in R-multiples
        "avg_loss_r":       1.0,    # expected loss in R-multiples

        # ── Strategy thresholds ───────────────────────────────────────────────
        "min_sp_score":            0.60,   # minimum Sp to act on a squeeze
        "btc_inflow_threshold":    1_000_000,  # USD — risk-off trigger
        "sol_memecoin_spike_threshold": 50_000_000,  # USD 24h vol

        # ── Polling intervals (seconds) ───────────────────────────────────────
        "news_poll_seconds":    30,
        "strategy_interval_seconds": 15,
        "whale_poll_seconds":   60,
        "finviz_poll_seconds":  300,
        "dune_poll_seconds":    300,
        "futures_poll_seconds": 60,

        # ── News / veto ───────────────────────────────────────────────────────
        "veto_cooldown_seconds": 600,   # 10 minutes of silence after macro event


        # ── Regime classifier
        "adx_trending":           float(os.getenv("REGIME_ADX_TREND",   "25")),
        "adx_ranging":            float(os.getenv("REGIME_ADX_RANGE",   "20")),
        "vix_proxy_high_vol":     float(os.getenv("REGIME_VIX_PROXY",   "0.025")),
        "breadth_bull":           float(os.getenv("REGIME_BREADTH_BULL","0.60")),
        "breadth_bear":           float(os.getenv("REGIME_BREADTH_BEAR","0.40")),
        "update_interval_min":    int(os.getenv("REGIME_UPDATE_MIN",    "15")),
        "smoothing_periods":      int(os.getenv("REGIME_SMOOTH",        "3")),

        # ── Alerts (Telegram + Discord)
        "telegram_bot_token":   os.getenv("TELEGRAM_BOT_TOKEN", ""),
        "telegram_chat_id":     os.getenv("TELEGRAM_CHAT_ID", ""),
        "discord_webhook_url":  os.getenv("DISCORD_WEBHOOK_URL", ""),
        "discord_mention_critical": os.getenv("DISCORD_MENTION_CRITICAL","false").lower()=="true",
        "alert_heartbeat_interval": int(os.getenv("ALERT_HEARTBEAT_INTERVAL","3600")),

        # ── Circuit breaker
        "daily_loss_limit_pct":   float(os.getenv("CB_DAILY_LOSS_PCT",    "0.05")),  # 5%
        "drawdown_limit_pct":     float(os.getenv("CB_DRAWDOWN_PCT",      "0.10")),  # 10%
        "loss_streak_limit":      int(os.getenv("CB_STREAK_LIMIT",        "4")),

        # ── Social / Stocktwits Z-score
        "stocktwits_max_rph":          int(os.getenv("STOCKTWITS_MAX_RPH", "180")),
        "social_short_window_min":     5,
        "social_baseline_hours":       24,
        "social_snapshot_interval":    10,
        "social_min_baseline_points":  6,
        "social_spike_threshold":      2.0,

        # ── API server
        "api_host": os.getenv("CHIMERA_API_HOST", "0.0.0.0"),
        "api_port": int(os.getenv("CHIMERA_API_PORT", "8765")),

        # ── Dune query IDs ────────────────────────────────────────────────────
        "dune_memecoin_query_id": "3152691",
    }

def api_defaults() -> dict:
    """Merge these into load_config() return value."""
    return {
        "api_host": os.getenv("CHIMERA_API_HOST", "0.0.0.0"),
        "api_port": int(os.getenv("CHIMERA_API_PORT", "8765")),
    }
