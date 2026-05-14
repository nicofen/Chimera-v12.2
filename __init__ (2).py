"""
chimera/server/serializer.py
Converts live SharedState into a JSON-serializable snapshot dict.

Rules:
- Every field must be JSON-safe (no datetime objects, no numpy types).
- Positions come from state.open_positions (Order objects).
- Signals are the last N from state.signals list.
- The snapshot is diffed against the previous one; only changed top-level
  keys are broadcast to save bandwidth on busy ticks.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from chimera_v12.utils.state import SharedState


def _f(v: float | None, digits: int = 4) -> float | None:
    """Round a float, return None for NaN/Inf."""
    if v is None:
        return None
    if not math.isfinite(v):
        return None
    return round(v, digits)


def _dt(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def serialize_state(state: "SharedState") -> dict[str, Any]:
    """Full snapshot — sent on first connect."""
    return {
        "type":       "snapshot",
        "ts":         datetime.now(timezone.utc).isoformat(),
        "equity":     _f(state.equity, 2),
        "news":       _serialize_news(state),
        "regime":     state.regime.value,
        "positions":  _serialize_positions(state),
        "signals":    _serialize_signals(state),
        "agents":     _serialize_agents(),
        "breaker":    _serialize_breaker(state),
        "regime":     _serialize_regime(state),
    }


def serialize_diff(prev: dict, curr: dict) -> dict[str, Any]:
    """
    Returns only the top-level keys that changed between two snapshots.
    Always includes "ts" and "type".
    """
    diff: dict[str, Any] = {"type": "diff", "ts": curr["ts"]}
    for key in ("equity", "news", "regime", "positions", "signals", "agents"):
        if curr.get(key) != prev.get(key):
            diff[key] = curr[key]
    return diff


# ── Section serializers ────────────────────────────────────────────────────────

def _serialize_news(state: "SharedState") -> dict:
    n = state.news
    return {
        "sentiment":    n.sentiment.value,
        "confidence":   _f(n.confidence, 3),
        "veto_active":  n.veto_active,
        "veto_reason":  n.veto_reason,
        "multiplier":   _f(state.news_multiplier(), 3),
        "last_updated": _dt(n.last_updated),
    }


def _serialize_positions(state: "SharedState") -> list[dict]:
    out = []
    for sym, order in state.open_positions.items():
        if sym.startswith("_"):          # skip internal keys like _daily_loss_usd
            continue
        if not hasattr(order, "symbol"):  # guard against non-Order values
            continue
        out.append({
            "symbol":        order.symbol,
            "sector":        order.sector,
            "side":          order.side.value,
            "qty":           _f(order.qty, 4),
            "entry_price":   _f(order.fill_price or order.entry_price, 4),
            "stop_price":    _f(order.stop_price, 4),
            "initial_stop":  _f(order.initial_stop, 4),
            "take_profit":   _f(order.take_profit, 4),
            "unrealised_pnl": _f(order.unrealised_pnl, 2),
            "r_multiple":    _f(order.r_multiple, 3),
            "kelly_fraction": _f(order.kelly_fraction, 3),
            "sp_score":      _f(order.sp_score, 3),
            "atr":           _f(order.atr, 4),
            "status":        order.status.value,
            "filled_at":     _dt(order.filled_at),
            "client_order_id": order.client_order_id,
        })
    return out


def _serialize_signals(state: "SharedState", limit: int = 40) -> list[dict]:
    out = []
    for sig in state.signals[-limit:]:
        out.append({
            "sector":             sig.sector,
            "symbol":             sig.symbol,
            "direction":          sig.direction,
            "confidence":         _f(sig.confidence, 3),
            "sp_score":           _f(sig.sp_score, 3),
            "adx":                _f(sig.adx, 1),
            "rsi_divergence":     sig.rsi_divergence,
            "bb_squeeze":         sig.bb_squeeze,
            "ema_ribbon_aligned": sig.ema_ribbon_aligned,
            "atr":                _f(sig.atr, 4),
            "timestamp":          _dt(sig.timestamp),
        })
    return list(reversed(out))   # newest first for the feed


def _serialize_agents() -> list[dict]:
    """
    Static agent registry — in a full implementation this would pull
    heartbeat timestamps from a health-check dict in SharedState.
    """
    return [
        {"name": "DataAgent",     "status": "running"},
        {"name": "NewsAgent",     "status": "running"},
        {"name": "StrategyAgent", "status": "running"},
        {"name": "RiskAgent",     "status": "running"},
        {"name": "OrderManager",  "status": "running"},
        {"name": "TradeLogger",   "status": "running"},
    ]


def _serialize_breaker(state) -> dict:
    b = getattr(state, "breaker", None)
    if b is None:
        return {"state": "uninitialised", "allows_trading": True}
    return {
        "state":              b.state.value,
        "trip_reason":        b.trip_reason.value if b.trip_reason else None,
        "tripped_at":         _dt(b.tripped_at),
        "daily_loss_usd":     _f(b.daily_loss_usd, 2),
        "drawdown_pct":       _f(b.drawdown_pct * 100, 3),
        "consecutive_losses": b.consecutive_losses,
        "trip_count_today":   b.trip_count_today,
        "allows_trading":     b.allows_trading,
        "high_water_mark":    _f(b.high_water_mark, 2),
    }


def _serialize_regime(state) -> dict:
    rs = getattr(state, "regime_state", None)
    if rs is None:
        return {"regime": "neutral", "confidence": 0.0, "reason": "not yet classified"}
    return {
        "regime":     rs.regime.value,
        "confidence": _f(rs.confidence, 3),
        "adx":        _f(rs.adx, 1),
        "ema_bull":   rs.ema_bull,
        "ema_bear":   rs.ema_bear,
        "vix_proxy":  _f(rs.vix_proxy, 4),
        "reason":     rs.reason,
        "updated_at": rs.updated_at,
    }
