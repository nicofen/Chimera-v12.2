"""
chimera/agents/risk_agent.py
Risk Agent — consumes TechnicalSignals from the queue, computes optimal
position size (Kelly Criterion), sets ATR-based stops, and dispatches
RiskParameters to the Order Queue for the OMS.

The position sizing formula:
    Position Size = (Account Equity × Risk%) / (ATR × ATR_Multiplier)

Kelly fraction is computed from the agent's rolling win/loss history and
used as a ceiling on position size — never bet more than Kelly suggests.
"""

import asyncio
from collections import deque
from datetime import datetime
from typing import Any

from chimera_v12.utils.state import SharedState, TechnicalSignals, RiskParameters
from chimera_v12.utils.logger import setup_logger

log = setup_logger("risk_agent")


# ── Hard limits — override these in config, never remove them ─────────────────
MAX_RISK_PER_TRADE  = 0.02    # 2% of equity maximum
MAX_KELLY_FRACTION  = 0.25    # never bet more than 25% Kelly even if math allows
ATR_MULTIPLIER      = 2.0     # stop = entry ± ATR × 2 (1R = 2×ATR)
TP_MULTIPLIER       = 4.0     # take-profit = entry ± ATR × 4  (2:1 R:R)
# Research basis: at 40% win rate, 2:1 R:R gives E = 0.40×2 - 0.60×1 = +0.2R/trade
# vs 3.0 TP: at 40% WR, E = 0.40×1.5 - 0.60×1 = 0.0R/trade (break even only)
PARTIAL_PROFIT_R    = 2.0     # close 50% of position at 2R (banked profit)
PARTIAL_PROFIT_PCT  = 0.50    # fraction to close at partial target
MIN_CONFIDENCE      = 0.50    # discard signals below this confidence threshold


class RiskAgent:
    """
    Sits between StrategyAgent and the OMS.

    Flow:
        TechnicalSignals queue → Kelly sizing → ATR stop/TP → RiskParameters queue
    """

    def __init__(self, state: SharedState, config: dict[str, Any]):
        self.state  = state
        self.config = config

        # Rolling R-multiple history for Kelly Criterion
        # Using R-multiples (not raw wins) eliminates position-sizing noise:
        # a +0.1R and a +5R win are NOT equivalent for Kelly estimation.
        _lookback = config.get("kelly_lookback", 50)
        self._r_history: deque[float] = deque(maxlen=_lookback)

    async def run(self) -> None:
        log.info("RiskAgent started — waiting for signals.")
        while True:
            try:
                signal: TechnicalSignals = await asyncio.wait_for(
                    self.state.signal_queue.get(), timeout=5.0
                )
                await self._process(signal)
            except asyncio.TimeoutError:
                pass   # no signal — loop back
            except Exception as e:
                log.warning(f"RiskAgent error: {e}")

    async def _process(self, signal: TechnicalSignals) -> None:
        if getattr(self.state, "circuit_open", False):
            log.info(f"Order blocked for {signal.symbol} — circuit breaker open.")
            return
        # Re-check veto (signal could have been queued just before veto raised)
        if self.state.is_vetoed():
            log.info(f"Dropping signal {signal.symbol} — veto active.")
            return

        if signal.confidence < MIN_CONFIDENCE:
            log.debug(f"Dropping {signal.symbol} — confidence {signal.confidence:.2f} below threshold.")
            return

        equity      = self.state.equity
        if equity <= 0:
            log.warning("Equity not set in state — skipping.")
            return

        kelly       = self._kelly_fraction()
        risk_pct    = min(
            self.config.get("base_risk_pct", 0.01),
            MAX_RISK_PER_TRADE,
            kelly,
        )
        risk_pct   *= self.state.news_multiplier()   # News Agent multiplier applied here

        # Apply signal confidence as a further scalar
        risk_pct   *= signal.confidence

        atr = signal.atr
        if atr <= 0:
            log.warning(f"ATR=0 for {signal.symbol} — cannot size position.")
            return

        position_size = (equity * risk_pct) / (atr * ATR_MULTIPLIER)

        # We don't have a live entry price here — the OMS will fill it at market.
        # We estimate entry from the latest close in market data.
        entry_est = self._latest_price(signal)
        if entry_est <= 0:
            return

        if signal.direction == "long":
            stop  = entry_est - atr * ATR_MULTIPLIER
            tp    = entry_est + atr * TP_MULTIPLIER
        else:
            stop  = entry_est + atr * ATR_MULTIPLIER
            tp    = entry_est - atr * TP_MULTIPLIER

        # Partial profit target: close PARTIAL_PROFIT_PCT at 2R
        if signal.direction == "long":
            partial_tp = entry_est + atr * ATR_MULTIPLIER * PARTIAL_PROFIT_R
        else:
            partial_tp = entry_est - atr * ATR_MULTIPLIER * PARTIAL_PROFIT_R

        rp = RiskParameters(
            symbol         = signal.symbol,
            position_size  = round(position_size, 4),
            entry_price    = entry_est,
            stop_price     = round(stop, 4),
            take_profit    = round(tp, 4),
            kelly_fraction = kelly,
            max_loss_usd   = round(equity * risk_pct, 2),
        )
        # Attach partial TP metadata (consumed by SimulatedOMS / live OMS)
        rp.partial_take_profit    = round(partial_tp, 4)
        rp.partial_close_fraction = PARTIAL_PROFIT_PCT

        log.info(
            f"Risk approved → {rp.symbol} size={rp.position_size} "
            f"entry~{rp.entry_price:.4f} stop={rp.stop_price:.4f} "
            f"TP={rp.take_profit:.4f} maxLoss=${rp.max_loss_usd:.2f}"
        )
        await self.state.put_order(rp)

    def _kelly_fraction(self) -> float:
        """
        Kelly Criterion using R-magnitude history.

        Classic Kelly: f* = (b*p - q) / b
        We compute b (win/loss ratio) and p (win rate) from actual R-multiples
        rather than config constants. This adapts to the strategy's current edge
        instead of assuming static win/loss ratios.

        R-multiples eliminate position-sizing noise:
        a +0.1R win and a +5R win are NOT equivalent — using booleans treats
        them identically, systematically overstating or understating Kelly.
        """
        if len(self._r_history) < 10:
            return 0.10   # conservative default until sufficient history

        rs     = list(self._r_history)
        win_rs = [r for r in rs if r > 0]
        los_rs = [r for r in rs if r <= 0]

        if not win_rs or not los_rs:
            return 0.05   # all-win or all-loss streaks → minimal sizing

        p = len(win_rs) / len(rs)
        q = 1.0 - p
        b = abs(sum(win_rs) / len(win_rs)) / abs(sum(los_rs) / len(los_rs))

        if b <= 0:
            return 0.01

        numerator = b * p - q
        if numerator <= 0:
            return 0.01   # negative expectancy — near-zero sizing

        kelly = numerator / b
        return min(kelly, MAX_KELLY_FRACTION)

    def record_trade_outcome(self, r_multiple: float) -> None:
        """
        Called by the OMS after each trade closes.
        Stores R-multiple (not bool) for accurate Kelly estimation.
        """
        self._r_history.append(r_multiple)

    def _latest_price(self, signal: TechnicalSignals) -> float:
        sector_map = {
            "crypto":  self.state.market.crypto,
            "stocks":  self.state.market.stocks,
            "forex":   self.state.market.forex,
            "futures": self.state.market.futures,
        }
        data = sector_map.get(signal.sector, {})
        bars = data.get(signal.symbol, {})
        closes = bars.get("close", [])
        return float(closes[-1]) if closes else 0.0
