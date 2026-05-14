"""
chimera/regime/models.py
Market regime definitions and the strategy permission matrix.

Five regimes, each with a different set of allowed strategies:

  TRENDING_BULL   — Strong uptrend (ADX > 25, EMA9 > EMA20 > EMA50 > EMA200,
                    VIX proxy low). Best for: momentum longs, squeeze longs.

  TRENDING_BEAR   — Strong downtrend (ADX > 25, inverted EMA ribbon,
                    price below EMA200). Best for: shorts only. Suppress
                    all long squeeze signals; only allow forex shorts and
                    futures short mean-reversion.

  HIGH_VOLATILITY — Elevated volatility (VIX proxy > threshold, wide ATR,
                    BTC dominance rising). Best for: crypto scalps, veto
                    most equity strategies. The risk-off + high-vol combo.

  MEAN_REVERTING  — Low ADX (< 20), price oscillating around EMA200,
                    narrow ATR. Best for: futures Value Area trades,
                    forex range trades. Suppress all momentum strategies.

  NEUTRAL         — Transitional / ambiguous. Apply no regime-specific
                    suppression; rely on signal confidence thresholds.

The StrategyGate encodes which (sector, direction) combinations are
allowed in each regime. Signals that violate the gate are suppressed
before reaching the RiskAgent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class Regime(str, Enum):
    TRENDING_BULL   = "trending_bull"
    TRENDING_BEAR   = "trending_bear"
    HIGH_VOLATILITY = "high_volatility"
    MEAN_REVERTING  = "mean_reverting"
    NEUTRAL         = "neutral"


# Direction literals
Direction = Literal["long", "short", "both", "none"]


@dataclass(frozen=True)
class SectorPermission:
    """Which directions are allowed for a sector in a given regime."""
    sector:    str
    direction: Direction   # "long" | "short" | "both" | "none"


# ── Permission matrix ──────────────────────────────────────────────────────────
# For each regime, what is each sector allowed to do?
#
# Design philosophy:
#   - When in doubt, block rather than allow.
#   - Never allow longs in a bear trend.
#   - Never allow momentum strategies in a mean-reverting market.
#   - Crypto scalps only during elevated volatility or bull trend.
#   - Forex NLP momentum works in any trending regime.
#   - Futures Value Area mean-reversion only works in mean-reverting or neutral.

REGIME_PERMISSIONS: dict[Regime, list[SectorPermission]] = {
    Regime.TRENDING_BULL: [
        SectorPermission("stocks",  "long"),    # squeeze works in bull
        SectorPermission("crypto",  "long"),    # crypto follows risk-on
        SectorPermission("forex",   "both"),    # NLP can catch either direction
        SectorPermission("futures", "long"),    # trend-following on ES
    ],
    Regime.TRENDING_BEAR: [
        SectorPermission("stocks",  "short"),   # only short squeezes in bear
        SectorPermission("crypto",  "none"),    # too volatile to long, no shorting
        SectorPermission("forex",   "short"),   # USD strength → short other currencies
        SectorPermission("futures", "short"),   # short futures trend
    ],
    Regime.HIGH_VOLATILITY: [
        SectorPermission("stocks",  "none"),    # equity squeezes unpredictable in HV
        SectorPermission("crypto",  "both"),    # crypto is the HV play
        SectorPermission("forex",   "none"),    # spreads too wide
        SectorPermission("futures", "none"),    # gap risk too high
    ],
    Regime.MEAN_REVERTING: [
        SectorPermission("stocks",  "none"),    # no momentum in ranging market
        SectorPermission("crypto",  "none"),    # crypto needs trend
        SectorPermission("forex",   "both"),    # range-bound forex pairs OK
        SectorPermission("futures", "both"),    # Value Area is a mean-rev play
    ],
    Regime.NEUTRAL: [
        SectorPermission("stocks",  "both"),    # no regime filter — use signal confidence
        SectorPermission("crypto",  "both"),
        SectorPermission("forex",   "both"),
        SectorPermission("futures", "both"),
    ],
}


def is_signal_allowed(regime: Regime, sector: str, direction: str) -> bool:
    """
    Returns True if the given signal is permitted in the current regime.

    Args:
        regime    : current detected regime
        sector    : signal sector ("stocks" | "crypto" | "forex" | "futures")
        direction : signal direction ("long" | "short")
    """
    permissions = REGIME_PERMISSIONS.get(regime, REGIME_PERMISSIONS[Regime.NEUTRAL])
    for perm in permissions:
        if perm.sector == sector:
            if perm.direction == "none":
                return False
            if perm.direction == "both":
                return True
            return perm.direction == direction
    return True   # unknown sector — allow by default


@dataclass
class RegimeState:
    """
    Current regime assessment with confidence and contributing signals.
    Stored in SharedState.regime_state.
    """
    regime:     Regime  = Regime.NEUTRAL
    confidence: float   = 0.0        # 0–1, how clear the regime is
    adx:        float   = 0.0
    ema_bull:   bool    = False       # EMA ribbon aligned bullish
    ema_bear:   bool    = False       # EMA ribbon aligned bearish
    vix_proxy:  float   = 0.0        # ATR/price ratio used as VIX proxy
    btc_dom:    float   = 0.0        # BTC dominance (0–1)
    updated_at: str     = ""
    reason:     str     = ""         # human-readable explanation
