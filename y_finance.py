"""
chimera/alerts/models.py
Alert event dataclasses — the shared vocabulary for all alert senders.

Every event in Chimera that is worth notifying about is represented as
an AlertEvent with a priority level. The dispatcher routes them to the
correct sender and applies rate limiting per priority tier.

Priority tiers:
  CRITICAL  — circuit breaker trip, ruin risk, force-close.  Never throttled.
  HIGH      — order filled, stop hit, TP hit, veto raised.   Max 1/min.
  NORMAL    — new signal emitted, daily P&L summary.         Max 3/min.
  LOW       — heartbeat, agent status, info.                 Max 1/5min.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any


class Priority(IntEnum):
    CRITICAL = 0    # fires immediately, never dropped
    HIGH     = 1
    NORMAL   = 2
    LOW      = 3


class EventType(str):
    # Trading events
    SIGNAL_EMITTED    = "signal_emitted"
    ORDER_FILLED      = "order_filled"
    STOP_HIT          = "stop_hit"
    TP_HIT            = "tp_hit"
    TRAILING_STOP_MV  = "trailing_stop_moved"
    POSITION_CLOSED   = "position_closed"
    FORCE_CLOSE_ALL   = "force_close_all"

    # Risk events
    CIRCUIT_TRIP      = "circuit_breaker_tripped"
    CIRCUIT_RESET     = "circuit_breaker_reset"
    DAILY_LOSS_WARN   = "daily_loss_warning"    # at 80% of daily limit
    DRAWDOWN_WARN     = "drawdown_warning"       # at 80% of drawdown limit
    STREAK_WARN       = "loss_streak_warning"

    # System events
    VETO_RAISED       = "veto_raised"
    VETO_CLEARED      = "veto_cleared"
    MAINFRAME_START   = "mainframe_started"
    MAINFRAME_STOP    = "mainframe_stopped"
    AGENT_ERROR       = "agent_error"
    HEARTBEAT         = "heartbeat"
    DAILY_SUMMARY     = "daily_summary"


# Priority map — determines throttling tier
EVENT_PRIORITIES: dict[str, Priority] = {
    EventType.SIGNAL_EMITTED:    Priority.NORMAL,
    EventType.ORDER_FILLED:      Priority.HIGH,
    EventType.STOP_HIT:          Priority.HIGH,
    EventType.TP_HIT:            Priority.HIGH,
    EventType.TRAILING_STOP_MV:  Priority.LOW,
    EventType.POSITION_CLOSED:   Priority.HIGH,
    EventType.FORCE_CLOSE_ALL:   Priority.CRITICAL,
    EventType.CIRCUIT_TRIP:      Priority.CRITICAL,
    EventType.CIRCUIT_RESET:     Priority.HIGH,
    EventType.DAILY_LOSS_WARN:   Priority.HIGH,
    EventType.DRAWDOWN_WARN:     Priority.HIGH,
    EventType.STREAK_WARN:       Priority.HIGH,
    EventType.VETO_RAISED:       Priority.HIGH,
    EventType.VETO_CLEARED:      Priority.NORMAL,
    EventType.MAINFRAME_START:   Priority.NORMAL,
    EventType.MAINFRAME_STOP:    Priority.HIGH,
    EventType.AGENT_ERROR:       Priority.HIGH,
    EventType.HEARTBEAT:         Priority.LOW,
    EventType.DAILY_SUMMARY:     Priority.NORMAL,
}


@dataclass
class AlertEvent:
    """
    One alertable event from anywhere in the Chimera system.
    Created by helper factory functions (see factories below).
    """
    event_type:   str
    priority:     Priority
    title:        str           # short one-liner (for notification preview)
    body:         str           # full formatted message body
    data:         dict[str, Any] = field(default_factory=dict)
    ts:           datetime       = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def emoji(self) -> str:
        _map = {
            EventType.CIRCUIT_TRIP:    "🚨",
            EventType.FORCE_CLOSE_ALL: "🚨",
            EventType.CIRCUIT_RESET:   "✅",
            EventType.ORDER_FILLED:    "📈",
            EventType.STOP_HIT:        "🛑",
            EventType.TP_HIT:          "🎯",
            EventType.POSITION_CLOSED: "💰",
            EventType.VETO_RAISED:     "⏸️",
            EventType.VETO_CLEARED:    "▶️",
            EventType.SIGNAL_EMITTED:  "📡",
            EventType.DAILY_LOSS_WARN: "⚠️",
            EventType.DRAWDOWN_WARN:   "⚠️",
            EventType.STREAK_WARN:     "⚠️",
            EventType.MAINFRAME_START: "🟢",
            EventType.MAINFRAME_STOP:  "🔴",
            EventType.HEARTBEAT:       "💓",
            EventType.DAILY_SUMMARY:   "📊",
            EventType.AGENT_ERROR:     "❌",
        }
        return _map.get(self.event_type, "ℹ️")


# ── Event factory functions ────────────────────────────────────────────────────
# Call these throughout the codebase instead of constructing AlertEvent directly.

def evt_circuit_trip(reason: str, equity: float, daily_loss: float,
                     drawdown_pct: float) -> AlertEvent:
    title = f"CIRCUIT BREAKER TRIPPED — {reason.upper()}"
    body  = (
        f"*Reason:* {reason}\n"
        f"*Equity:* ${equity:,.2f}\n"
        f"*Daily loss:* ${daily_loss:,.2f}\n"
        f"*Drawdown:* {drawdown_pct:.2f}%\n\n"
        f"All positions are being force-closed.\n"
        f"Manual reset required: `POST /api/breaker/reset`"
    )
    return AlertEvent(EventType.CIRCUIT_TRIP, Priority.CRITICAL, title, body,
                      {"reason": reason, "equity": equity})


def evt_circuit_reset(note: str, equity: float) -> AlertEvent:
    title = "Circuit breaker reset"
    body  = f"*Note:* {note}\n*Equity:* ${equity:,.2f}\nTrading resumed."
    return AlertEvent(EventType.CIRCUIT_RESET, Priority.HIGH, title, body)


def evt_order_filled(symbol: str, side: str, qty: float, fill_price: float,
                     stop: float, tp: float, sector: str) -> AlertEvent:
    title = f"Filled: {side.upper()} {qty} {symbol}"
    body  = (
        f"*Symbol:* {symbol}  ({sector.upper()})\n"
        f"*Side:* {side.upper()}\n"
        f"*Qty:* {qty:.2f}\n"
        f"*Fill:* ${fill_price:.4f}\n"
        f"*Stop:* ${stop:.4f}\n"
        f"*Target:* ${tp:.4f}"
    )
    return AlertEvent(EventType.ORDER_FILLED, Priority.HIGH, title, body,
                      {"symbol": symbol, "fill_price": fill_price})


def evt_position_closed(symbol: str, pnl: float, r_multiple: float,
                        reason: str, sector: str) -> AlertEvent:
    sign  = "+" if pnl >= 0 else ""
    title = f"Closed {symbol}: {sign}${pnl:.2f} ({sign}{r_multiple:.2f}R)"
    body  = (
        f"*Symbol:* {symbol}  ({sector.upper()})\n"
        f"*P&L:* {sign}${pnl:.2f}\n"
        f"*R-multiple:* {sign}{r_multiple:.2f}R\n"
        f"*Reason:* {reason}"
    )
    pri = Priority.HIGH
    return AlertEvent(EventType.POSITION_CLOSED, pri, title, body,
                      {"symbol": symbol, "pnl": pnl, "r": r_multiple})


def evt_veto_raised(reason: str, cooldown_sec: int) -> AlertEvent:
    title = f"News veto active — {reason[:40]}"
    body  = (
        f"*Reason:* {reason}\n"
        f"*Cool-down:* {cooldown_sec // 60} minutes\n"
        "Signal emission suspended."
    )
    return AlertEvent(EventType.VETO_RAISED, Priority.HIGH, title, body)


def evt_veto_cleared() -> AlertEvent:
    return AlertEvent(
        EventType.VETO_CLEARED, Priority.NORMAL,
        "News veto cleared — signals resumed",
        "Macro cool-down window expired. Signal emission re-enabled.",
    )


def evt_signal(symbol: str, sector: str, direction: str,
               confidence: float, adx: float, sp: float | None) -> AlertEvent:
    sp_str = f"  Sp={sp:.2f}" if sp else ""
    title  = f"Signal: {direction.upper()} {symbol}"
    body   = (
        f"*Symbol:* {symbol}  ({sector.upper()})\n"
        f"*Direction:* {direction.upper()}\n"
        f"*Confidence:* {confidence:.2f}\n"
        f"*ADX:* {adx:.1f}{sp_str}"
    )
    return AlertEvent(EventType.SIGNAL_EMITTED, Priority.NORMAL, title, body)


def evt_daily_summary(equity: float, day_pnl: float, win_rate: float,
                      n_trades: int, max_dd: float) -> AlertEvent:
    sign  = "+" if day_pnl >= 0 else ""
    title = f"Daily summary: {sign}${day_pnl:.2f}"
    body  = (
        f"*Equity:* ${equity:,.2f}\n"
        f"*Day P&L:* {sign}${day_pnl:.2f}\n"
        f"*Trades:* {n_trades}\n"
        f"*Win rate:* {win_rate*100:.1f}%\n"
        f"*Max DD today:* {max_dd:.2f}%"
    )
    return AlertEvent(EventType.DAILY_SUMMARY, Priority.NORMAL, title, body)


def evt_warning(event_type: str, message: str) -> AlertEvent:
    return AlertEvent(event_type, Priority.HIGH, message, message)


def evt_heartbeat(equity: float, open_positions: int,
                  veto_active: bool, breaker_state: str) -> AlertEvent:
    title = f"Heartbeat — ${equity:,.0f}  {open_positions} open"
    body  = (
        f"*Equity:* ${equity:,.2f}\n"
        f"*Open positions:* {open_positions}\n"
        f"*Veto:* {'Active' if veto_active else 'Clear'}\n"
        f"*Breaker:* {breaker_state}"
    )
    return AlertEvent(EventType.HEARTBEAT, Priority.LOW, title, body)
