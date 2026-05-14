"""
chimera/alerts/agent.py
AlertAgent — the bridge between SharedState and the AlertDispatcher.

Runs as an asyncio task. On each tick it compares the current state
snapshot against the previous one and fires alert events for anything
that changed meaningfully.

Watched state transitions:
  - New signal in state.signals  → SIGNAL_EMITTED
  - New closed trade             → POSITION_CLOSED (stop/TP/trailing)
  - New fill in open_positions   → ORDER_FILLED
  - circuit_open becomes True    → already fired by CircuitBreaker directly
  - news.veto_active changes     → VETO_RAISED / VETO_CLEARED
  - daily_loss crosses 80% warn  → DAILY_LOSS_WARN
  - drawdown crosses 80% warn    → DRAWDOWN_WARN
  - Midnight                     → DAILY_SUMMARY

The agent does NOT duplicate events already fired by CircuitBreaker.
It only watches state diffs for events the other agents don't self-report.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from chimera_v12.alerts.models import (
    evt_signal, evt_position_closed, evt_order_filled,
    evt_veto_raised, evt_veto_cleared, evt_daily_summary,
    evt_warning, evt_heartbeat, EventType,
)

if TYPE_CHECKING:
    from chimera_v12.utils.state import SharedState
    from chimera_v12.alerts.dispatcher import AlertDispatcher

log = logging.getLogger("chimera.alerts.agent")


class AlertAgent:
    """
    Watches SharedState diffs every POLL_INTERVAL seconds and fires
    alert events to the dispatcher.
    """

    POLL_INTERVAL = 5.0   # seconds

    def __init__(self, state: "SharedState", dispatcher: "AlertDispatcher",
                 config: dict[str, Any]):
        self.state      = state
        self.dispatcher = dispatcher
        self.config     = config

        # Previous state snapshots for diff detection
        self._prev_signal_count  = 0
        self._prev_trade_count   = 0
        self._prev_position_syms: set[str] = set()
        self._prev_veto          = False
        self._prev_circuit_open  = False
        self._daily_loss_warned  = False
        self._drawdown_warned    = False
        self._last_summary_day   = ""
        self._heartbeat_interval = config.get("alert_heartbeat_interval", 3600)
        self._last_heartbeat     = 0.0

    async def run(self) -> None:
        log.info("AlertAgent started.")
        import time
        while True:
            try:
                await self._tick()
            except Exception as e:
                log.warning(f"AlertAgent tick error: {e}")
            await asyncio.sleep(self.POLL_INTERVAL)

    async def _tick(self) -> None:
        import time

        now = datetime.now(timezone.utc)

        # ── New signals ──────────────────────────────────────────────────
        curr_signals = len(self.state.signals)
        if curr_signals > self._prev_signal_count:
            new_sigs = list(self.state.signals)[self._prev_signal_count:]
            for sig in new_sigs[-3:]:   # max 3 signal alerts per tick
                if hasattr(sig, "symbol"):
                    self.dispatcher.send_nowait(evt_signal(
                        symbol     = sig.symbol,
                        sector     = sig.sector,
                        direction  = sig.direction,
                        confidence = sig.confidence,
                        adx        = sig.adx,
                        sp         = sig.sp_score if sig.sp_score else None,
                    ))
            self._prev_signal_count = curr_signals

        # ── New fills (order filled) ──────────────────────────────────────
        curr_syms = set(
            sym for sym, pos in self.state.open_positions.items()
            if not sym.startswith("_") and hasattr(pos, "fill_price")
            and getattr(pos, "fill_price", 0) > 0
        )
        new_fills = curr_syms - self._prev_position_syms
        for sym in new_fills:
            pos = self.state.open_positions.get(sym)
            if pos and hasattr(pos, "fill_price"):
                self.dispatcher.send_nowait(evt_order_filled(
                    symbol     = sym,
                    side       = getattr(pos, "side", type("", (), {"value":"buy"})()).value,
                    qty        = getattr(pos, "qty", 0),
                    fill_price = getattr(pos, "fill_price", 0),
                    stop       = getattr(pos, "stop_price", 0),
                    tp         = getattr(pos, "take_profit", 0),
                    sector     = getattr(pos, "sector", "unknown"),
                ))
        self._prev_position_syms = curr_syms

        # ── Closed trades ─────────────────────────────────────────────────
        curr_trades = len(self.state.closed_trades) if hasattr(self.state, "closed_trades") else 0
        if curr_trades > self._prev_trade_count:
            new_trades = list(self.state.closed_trades or [])[self._prev_trade_count:]
            for t in new_trades[-3:]:
                if isinstance(t, dict):
                    self.dispatcher.send_nowait(evt_position_closed(
                        symbol    = t.get("symbol", "?"),
                        pnl       = t.get("realised_pnl", 0),
                        r_multiple= t.get("r_multiple", 0),
                        reason    = t.get("close_reason", "unknown"),
                        sector    = t.get("sector", "unknown"),
                    ))
            self._prev_trade_count = curr_trades

        # ── Veto state change ─────────────────────────────────────────────
        curr_veto = self.state.news.veto_active
        if curr_veto and not self._prev_veto:
            self.dispatcher.send_nowait(evt_veto_raised(
                reason       = self.state.news.veto_reason or "macro event",
                cooldown_sec = self.config.get("veto_cooldown_seconds", 600),
            ))
        elif not curr_veto and self._prev_veto:
            self.dispatcher.send_nowait(evt_veto_cleared())
        self._prev_veto = curr_veto

        # ── Circuit breaker warnings ──────────────────────────────────────
        breaker = getattr(self.state, "breaker", None)
        if breaker:
            eq_start = breaker._equity_start_of_day if hasattr(breaker, "_equity_start_of_day") else self.state.equity
            if eq_start > 0:
                loss_pct = abs(min(breaker.status.daily_loss_usd, 0)) / eq_start
                dd_pct   = breaker.status.drawdown_pct

                warn_thresh = 0.80   # warn at 80% of limit
                if (not self._daily_loss_warned and
                        loss_pct > breaker.status.daily_loss_limit * warn_thresh):
                    self.dispatcher.send_nowait(evt_warning(
                        EventType.DAILY_LOSS_WARN,
                        f"Daily loss at {loss_pct*100:.1f}% — "
                        f"limit is {breaker.status.daily_loss_limit*100:.0f}%"
                    ))
                    self._daily_loss_warned = True
                elif loss_pct < breaker.status.daily_loss_limit * 0.5:
                    self._daily_loss_warned = False   # reset after recovery

                if (not self._drawdown_warned and
                        dd_pct > breaker.status.drawdown_limit_pct * warn_thresh):
                    self.dispatcher.send_nowait(evt_warning(
                        EventType.DRAWDOWN_WARN,
                        f"Drawdown at {dd_pct*100:.1f}% — "
                        f"limit is {breaker.status.drawdown_limit_pct*100:.0f}%"
                    ))
                    self._drawdown_warned = True
                elif dd_pct < breaker.status.drawdown_limit_pct * 0.5:
                    self._drawdown_warned = False

        # ── Daily summary at midnight ─────────────────────────────────────
        today = now.strftime("%Y-%m-%d")
        if today != self._last_summary_day and now.hour == 0 and now.minute < 5:
            from chimera_v12.oms.trade_logger import TradeLogger
            try:
                db_path = self.config.get("db_path", "chimera_trades.db")
                logger  = TradeLogger(db_path)
                summary = logger.daily_pnl_summary()
                self.dispatcher.send_nowait(evt_daily_summary(
                    equity   = self.state.equity,
                    day_pnl  = summary.get("total_pnl", 0),
                    win_rate = summary.get("win_rate", 0),
                    n_trades = summary.get("trade_count", 0),
                    max_dd   = 0.0,   # would need to be tracked separately
                ))
                self._last_summary_day = today
            except Exception as e:
                log.debug(f"Daily summary error: {e}")

        # ── Periodic heartbeat ────────────────────────────────────────────
        import time as _time
        if (self._heartbeat_interval > 0 and
                _time.monotonic() - self._last_heartbeat > self._heartbeat_interval):
            breaker_state = "unknown"
            if hasattr(self.state, "breaker") and self.state.breaker:
                breaker_state = self.state.breaker.state.value
            self.dispatcher.send_nowait(evt_heartbeat(
                equity         = self.state.equity,
                open_positions = len(self.state.open_positions),
                veto_active    = self.state.news.veto_active,
                breaker_state  = breaker_state,
            ))
            self._last_heartbeat = _time.monotonic()
