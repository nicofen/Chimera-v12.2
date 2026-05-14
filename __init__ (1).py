"""
chimera/risk/circuit_breaker_models.py
Dataclasses and enums for the circuit breaker.

Kept separate so other modules can import these types without
pulling in the full CircuitBreaker logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class BreakerState(str, Enum):
    CLOSED   = "closed"    # normal — trading allowed
    OPEN     = "open"      # tripped — trading halted, positions closing
    COOLDOWN = "cooldown"  # positions closed, waiting for manual reset
    RESET    = "reset"     # manually re-armed, monitoring resumed


class TripReason(str, Enum):
    DAILY_LOSS    = "daily_loss_limit"
    DRAWDOWN      = "drawdown_limit"
    LOSS_STREAK   = "loss_streak"
    MANUAL        = "manual_trip"
    EXTERNAL      = "external_signal"   # e.g. broker margin call webhook


@dataclass
class BreakerEvent:
    """Immutable record of a breaker state transition."""
    ts:           datetime
    old_state:    BreakerState
    new_state:    BreakerState
    reason:       TripReason
    detail:       str       # human-readable context
    equity_at_trip: float
    daily_loss_usd: float
    drawdown_pct:   float
    streak:         int


@dataclass
class BreakerStatus:
    """
    Current snapshot of the circuit breaker — serialised for the dashboard
    and the API server.
    """
    state:              BreakerState = BreakerState.CLOSED
    trip_reason:        Optional[TripReason] = None
    tripped_at:         Optional[datetime]   = None
    reset_at:           Optional[datetime]   = None

    # Running counters (reset on midnight / manual reset)
    daily_loss_usd:     float = 0.0
    daily_loss_limit:   float = 0.0
    drawdown_pct:       float = 0.0
    drawdown_limit_pct: float = 0.0
    consecutive_losses: int   = 0
    loss_streak_limit:  int   = 0
    high_water_mark:    float = 0.0

    # History
    trip_count_today:   int = 0
    events:             list[BreakerEvent] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        return self.state in (BreakerState.OPEN, BreakerState.COOLDOWN)

    @property
    def allows_trading(self) -> bool:
        return self.state == BreakerState.CLOSED
