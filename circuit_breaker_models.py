"""
chimera/risk/circuit_breaker.py
CircuitBreaker — evaluates three independent trip conditions on every tick
and transitions the system into SAFE MODE when any condition is met.

Three trip conditions (all configurable, any one fires the breaker):
──────────────────────────────────────────────────────────────────────
1. DAILY LOSS LIMIT
   Sum of (realised P&L + unrealised P&L) since 00:00 UTC.
   Default: -5% of starting-day equity.
   Resets automatically at midnight UTC.

2. PEAK-TO-TROUGH DRAWDOWN
   (high_water_mark - current_equity) / high_water_mark.
   High-water mark is the highest equity seen since the last reset.
   Default: -10% drawdown.
   Does NOT auto-reset — requires manual re-arm after investigation.

3. CONSECUTIVE LOSS STREAK
   N closed trades in a row, each with negative P&L.
   Default: 4 consecutive losses.
   Resets when a winning trade closes or on manual reset.

When any condition fires:
  1. state.circuit_open  = True   →  StrategyAgent & RiskAgent suppress output
  2. OMS.force_close_all()        →  all open positions liquidated at market
  3. BreakerEvent logged to SQLite and state.breaker_events
  4. Dashboard receives updated BreakerStatus via the WebSocket

Manual reset (required after drawdown or manual trip):
  CircuitBreaker.reset(operator_note="investigated, resuming")
  Also exposed via POST /api/breaker/reset in the API server.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, TYPE_CHECKING

from chimera_v12.risk.circuit_breaker_models import (
    BreakerState, BreakerStatus, BreakerEvent, TripReason
)
from chimera_v12.utils.logger import setup_logger

if TYPE_CHECKING:
    from chimera_v12.utils.state import SharedState
    from chimera_v12.oms.order_manager import OrderManager

log = setup_logger("risk.circuit_breaker")


class CircuitBreaker:
    """
    Runs as an asyncio task inside the mainframe.
    Evaluates all three trip conditions every EVAL_INTERVAL seconds.
    """

    EVAL_INTERVAL = 5.0   # seconds between evaluations

    def __init__(
        self,
        state:         "SharedState",
        order_manager: "OrderManager",
        config:        dict[str, Any],
        dispatcher:    "Any | None" = None,
    ):
        self.state         = state
        self.oms           = order_manager
        self.config        = config
        self.status        = BreakerStatus()

        # Limits from config
        self.status.daily_loss_limit   = config.get("daily_loss_limit_pct",    0.05)
        self.status.drawdown_limit_pct = config.get("drawdown_limit_pct",       0.10)
        self.status.loss_streak_limit  = config.get("loss_streak_limit",        4)

        # Internal state
        self._equity_start_of_day: float  = 0.0
        self._last_day: str               = ""   # YYYY-MM-DD
        self._last_trade_count: int       = 0
        self._db_path: str                = config.get("db_path", "chimera_trades.db")
        self._dispatcher = dispatcher   # AlertDispatcher, injected after construction

    # ── Main loop ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        log.info("CircuitBreaker armed — monitoring active.")
        while True:
            try:
                await self._evaluate()
            except Exception as e:
                log.exception(f"CircuitBreaker evaluation error: {e}")
            await asyncio.sleep(self.EVAL_INTERVAL)

    async def _evaluate(self) -> None:
        """Run one evaluation cycle."""
        self._maybe_midnight_reset()
        self._update_high_water_mark()
        self._update_daily_loss()
        self._update_loss_streak()
        self._sync_to_state()

        if self.status.is_open:
            return   # already tripped — don't re-evaluate trip conditions

        reason = self._check_trip_conditions()
        if reason:
            await self._trip(reason)

    # ── Trip condition checks ──────────────────────────────────────────────

    def _check_trip_conditions(self) -> TripReason | None:
        eq    = self.state.equity
        limit = self._equity_start_of_day * self.status.daily_loss_limit

        # 1. Daily loss
        if self.status.daily_loss_usd <= -limit and limit > 0:
            return TripReason.DAILY_LOSS

        # 2. Drawdown
        if self.status.drawdown_pct >= self.status.drawdown_limit_pct:
            return TripReason.DRAWDOWN

        # 3. Loss streak
        if self.status.consecutive_losses >= self.status.loss_streak_limit:
            return TripReason.LOSS_STREAK

        return None

    # ── Trip action ────────────────────────────────────────────────────────

    async def _trip(self, reason: TripReason) -> None:
        old = self.status.state
        self.status.state       = BreakerState.OPEN
        self.status.trip_reason = reason
        self.status.tripped_at  = _now()
        self.status.trip_count_today += 1

        detail = self._trip_detail(reason)
        event  = BreakerEvent(
            ts              = self.status.tripped_at,
            old_state       = old,
            new_state       = BreakerState.OPEN,
            reason          = reason,
            detail          = detail,
            equity_at_trip  = self.state.equity,
            daily_loss_usd  = self.status.daily_loss_usd,
            drawdown_pct    = self.status.drawdown_pct,
            streak          = self.status.consecutive_losses,
        )
        self.status.events.append(event)
        self._sync_to_state()

        log.critical(
            f"╔══════════════════════════════════════════╗\n"
            f"║  CIRCUIT BREAKER TRIPPED                  ║\n"
            f"║  Reason  : {reason.value:<31}║\n"
            f"║  Detail  : {detail[:31]:<31}║\n"
            f"║  Equity  : ${self.state.equity:>10,.2f}               ║\n"
            f"╚══════════════════════════════════════════╝"
        )

        # Block signal/order emission immediately
        self.state.circuit_open = True

        # Force-close all open positions
        from chimera_v12.oms.models import CloseReason
        await self.oms.force_close_all(CloseReason.VETO)

        # Persist to SQLite
        await asyncio.to_thread(self._persist_event, event)

        # Fire alert
        if self._dispatcher:
            from chimera_v12.alerts.models import evt_circuit_trip
            self._dispatcher.send_nowait(evt_circuit_trip(
                reason      = reason.value,
                equity      = self.state.equity,
                daily_loss  = self.status.daily_loss_usd,
                drawdown_pct= self.status.drawdown_pct * 100,
            ))

        # Transition to COOLDOWN (positions closed, awaiting manual reset)
        self.status.state = BreakerState.COOLDOWN
        self._sync_to_state()
        log.warning("Breaker in COOLDOWN — manual reset required via API or CLI.")

    def _trip_detail(self, reason: TripReason) -> str:
        if reason == TripReason.DAILY_LOSS:
            return (
                f"Daily loss ${abs(self.status.daily_loss_usd):,.2f} "
                f"exceeded {self.status.daily_loss_limit*100:.1f}% limit"
            )
        if reason == TripReason.DRAWDOWN:
            return (
                f"Drawdown {self.status.drawdown_pct*100:.2f}% "
                f"exceeded {self.status.drawdown_limit_pct*100:.1f}% limit"
            )
        if reason == TripReason.LOSS_STREAK:
            return (
                f"{self.status.consecutive_losses} consecutive losses "
                f"(limit: {self.status.loss_streak_limit})"
            )
        return reason.value

    # ── Manual reset ───────────────────────────────────────────────────────

    def reset(self, operator_note: str = "") -> bool:
        """
        Manually re-arm the circuit breaker after investigation.
        Returns True if reset was successful, False if state doesn't allow it.

        Daily loss limit auto-resets at midnight; this is only needed for
        drawdown and manual/streak trips.
        """
        if self.status.state == BreakerState.CLOSED:
            log.info("Breaker already closed — nothing to reset.")
            return False

        old = self.status.state
        self.status.state             = BreakerState.CLOSED
        self.status.trip_reason       = None
        self.status.reset_at          = _now()
        self.status.consecutive_losses = 0
        self.state.circuit_open       = False

        event = BreakerEvent(
            ts              = self.status.reset_at,
            old_state       = old,
            new_state       = BreakerState.CLOSED,
            reason          = TripReason.MANUAL,
            detail          = f"Manual reset. Note: {operator_note[:80]}",
            equity_at_trip  = self.state.equity,
            daily_loss_usd  = self.status.daily_loss_usd,
            drawdown_pct    = self.status.drawdown_pct,
            streak          = 0,
        )
        self.status.events.append(event)
        self._sync_to_state()

        log.info(
            f"Circuit breaker RESET by operator. "
            f"Note: {operator_note or '(none)'}"
        )
        if self._dispatcher:
            from chimera_v12.alerts.models import evt_circuit_reset
            self._dispatcher.send_nowait(evt_circuit_reset(operator_note, self.state.equity))
        return True

    def manual_trip(self, reason: str = "operator") -> None:
        """Force-trip the breaker manually (e.g. from the dashboard)."""
        asyncio.create_task(self._trip(TripReason.MANUAL))

    # ── Running metric updates ─────────────────────────────────────────────

    def _update_high_water_mark(self) -> None:
        eq = self.state.equity
        if eq > self.status.high_water_mark:
            self.status.high_water_mark = eq
        if self.status.high_water_mark > 0:
            self.status.drawdown_pct = (
                (self.status.high_water_mark - eq) / self.status.high_water_mark
            )

    def _update_daily_loss(self) -> None:
        """
        Daily loss = (current equity - start-of-day equity).
        Includes unrealised P&L on open positions so the breaker fires
        before a bad trade fully closes.
        """
        if self._equity_start_of_day <= 0:
            return
        unrealised = sum(
            getattr(p, "unrealised_pnl", 0)
            for p in self.state.open_positions.values()
            if hasattr(p, "unrealised_pnl")
        )
        self.status.daily_loss_usd = (
            self.state.equity + unrealised - self._equity_start_of_day
        )

    def _drain_outcomes(self) -> None:
        """
        Non-blocking drain of all pending _TradeOutcome objects from
        outcome_queue.  Called every evaluation cycle.

        Uses outcome_queue (not signals list) so:
          - No O(n) scan of an ever-growing list
          - No risk of reading TechnicalSignals objects as trade outcomes
          - Reliable: every closed trade posts exactly one outcome object
        """
        while True:
            try:
                outcome = self.state.outcome_queue.get_nowait()
            except Exception:   # asyncio.QueueEmpty
                break
            r = getattr(outcome, "r_multiple", None)
            if r is None:
                continue
            if r > 0:
                self.status.consecutive_losses = 0
            else:
                self.status.consecutive_losses += 1

    def _update_loss_streak(self) -> None:
        """Delegate to _drain_outcomes (kept for API compatibility)."""
        self._drain_outcomes()

    def _maybe_midnight_reset(self) -> None:
        """Auto-reset daily counters at UTC midnight."""
        today = _now().strftime("%Y-%m-%d")
        if today != self._last_day:
            self._last_day            = today
            self._equity_start_of_day = self.state.equity if self.state.equity > 0 else self._equity_start_of_day
            self.status.daily_loss_usd  = 0.0
            self.status.trip_count_today = 0
            # Auto-rearm if tripped only on daily loss (not on drawdown/streak)
            if (self.status.state == BreakerState.COOLDOWN
                    and self.status.trip_reason == TripReason.DAILY_LOSS):
                self.reset("Automatic midnight reset after daily loss trip")
            log.info(f"New trading day {today} — daily loss counter reset. Equity: ${self._equity_start_of_day:,.2f}")

    # ── State sync ─────────────────────────────────────────────────────────

    def _sync_to_state(self) -> None:
        """Write BreakerStatus snapshot into SharedState for dashboard/API."""
        self.state.breaker = self.status

    # ── SQLite persistence ─────────────────────────────────────────────────

    def _persist_event(self, event: BreakerEvent) -> None:
        """Write a trip/reset event to SQLite for audit trail."""
        import sqlite3, json
        try:
            conn = sqlite3.connect(self._db_path, timeout=10)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS breaker_events (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts             TEXT,
                    old_state      TEXT,
                    new_state      TEXT,
                    reason         TEXT,
                    detail         TEXT,
                    equity_at_trip REAL,
                    daily_loss_usd REAL,
                    drawdown_pct   REAL,
                    streak         INTEGER
                )
            """)
            conn.execute("""
                INSERT INTO breaker_events
                (ts,old_state,new_state,reason,detail,equity_at_trip,daily_loss_usd,drawdown_pct,streak)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                event.ts.isoformat(),
                event.old_state.value,
                event.new_state.value,
                event.reason.value,
                event.detail,
                event.equity_at_trip,
                event.daily_loss_usd,
                event.drawdown_pct,
                event.streak,
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning(f"Failed to persist breaker event: {e}")


def _now() -> datetime:
    return datetime.now(timezone.utc)
