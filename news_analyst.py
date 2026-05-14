"""
chimera/agents/strategy_agent.py
Strategy Agent — The Brain's signal engine.

Sector strategies:
  A. Crypto  — Exchange inflow/outflow + memecoin volume spike filter
  B. Stocks  — Squeeze Probability Score (Sp)
  C. Forex   — NLP momentum bias on EMA
  D. Futures — Value Area mean reversion via Volume Profile + AVWAP

All signals are gated by the News Agent veto before being enqueued.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

import numpy as np

from chimera_v12.utils.state import SharedState, TechnicalSignals, Sentiment
from chimera_v12.utils.ta import ema, rsi, adx, atr_value, bollinger_squeeze, detect_rsi_divergence
from chimera_v12.regime.models import is_signal_allowed, Regime
from chimera_v12.utils.logger import setup_logger

log = setup_logger("strategy_agent")


# ══════════════════════════════════════════════════════════════════════════════
# Technical Analysis helpers (vectorized with numpy for speed)
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# Squeeze Probability Score (Stocks, Sector B)
# ══════════════════════════════════════════════════════════════════════════════

def squeeze_probability_score(
    short_interest: float,   # SI as decimal (e.g. 0.25 = 25%)
    rvol: float,             # Relative Volume (e.g. 3.5)
    sentiment_zscore: float, # Z-score of social mention velocity
) -> float:
    """
    Sp = (SI × 0.4) + (Vvelocity × 0.3) + (Ssentiment × 0.3)
    All inputs are normalised to 0–1 before weighting.
    """
    si_norm   = min(short_interest / 0.50, 1.0)   # cap at 50% SI → 1.0
    rvol_norm = min((rvol - 1.0) / 9.0, 1.0)      # RVOL 1→10 mapped to 0→1
    sent_norm = min(max(sentiment_zscore / 5.0, 0.0), 1.0)  # Z-score 0→5 → 0→1

    sp = (si_norm * 0.4) + (rvol_norm * 0.3) + (sent_norm * 0.3)
    return round(sp, 4)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy Agent
# ══════════════════════════════════════════════════════════════════════════════

class StrategyAgent:
    """
    Consumes market data from shared state, runs all four sector strategies,
    and publishes TechnicalSignals to the signal queue.

    Signals are suppressed (not emitted) when the News Agent veto is active.
    """

    def __init__(self, state: SharedState, config: dict[str, Any]):
        self.state  = state
        self.config = config
        self.interval = config.get("strategy_interval_seconds", 15)

    async def run(self) -> None:
        log.info("StrategyAgent started.")
        while True:
            try:
                await self._evaluate_all_sectors()
            except Exception as e:
                log.warning(f"StrategyAgent error: {e}")
            await asyncio.sleep(self.interval)

    async def _evaluate_all_sectors(self) -> None:
        await asyncio.gather(
            self._sector_crypto(),
            self._sector_stocks(),
            self._sector_forex(),
            self._sector_futures(),
        )

    # ── Sector A: Crypto ──────────────────────────────────────────────────────

    async def _sector_crypto(self) -> None:
        d = self.state.market.crypto
        if not d:
            return

        for symbol, bars in d.items():
            closes = np.array(bars["close"], dtype=float)
            highs  = np.array(bars["high"],  dtype=float)
            lows   = np.array(bars["low"],   dtype=float)
            if len(closes) < 200:
                continue

            # Exchange inflow/outflow edge
            exchange_inflow = d.get("btc_exchange_inflow", 0)
            sol_memecoin_vol_spike = d.get("sol_memecoin_vol_spike", False)

            regime = "risk_off" if exchange_inflow > self.config.get("btc_inflow_threshold", 1000) else "neutral"
            if sol_memecoin_vol_spike and regime == "risk_off":
                regime = "high_volatility_scalp"

            signal = self._build_signal("crypto", symbol, closes, highs, lows)
            signal.sp_score = 0.0  # not applicable for crypto

            # Directional override based on on-chain regime
            if regime == "risk_off":
                signal.direction = "flat"
            elif regime == "high_volatility_scalp":
                signal.direction = "long"  # momentum scalp on low-cap

            await self._emit(signal)

    # ── Sector B: Stocks (Short Squeeze) ─────────────────────────────────────

    async def _sector_stocks(self) -> None:
        d = self.state.market.stocks
        if not d:
            return

        for symbol, bars in d.items():
            closes = np.array(bars["close"], dtype=float)
            highs  = np.array(bars["high"],  dtype=float)
            lows   = np.array(bars["low"],   dtype=float)
            if len(closes) < 200:
                continue

            si   = bars.get("short_interest", 0.0)   # e.g. 0.25
            rvol = bars.get("rvol", 1.0)
            zscore = bars.get("social_zscore", 0.0)

            sp = squeeze_probability_score(si, rvol, zscore)

            # Only act on high-conviction squeezes
            if sp < self.config.get("min_sp_score", 0.60):
                continue

            signal = self._build_signal("stocks", symbol, closes, highs, lows)
            signal.sp_score = sp
            signal.direction = "long" if sp > 0.75 else signal.direction

            await self._emit(signal)

    # ── Sector C: Forex (NLP momentum) ───────────────────────────────────────

    async def _sector_forex(self) -> None:
        d = self.state.market.forex
        if not d:
            return

        for pair, bars in d.items():
            closes = np.array(bars["close"], dtype=float)
            highs  = np.array(bars["high"],  dtype=float)
            lows   = np.array(bars["low"],   dtype=float)
            if len(closes) < 20:
                continue

            ema20 = ema(closes, 20)
            below_ema = closes[-1] < ema20[-1]

            signal = self._build_signal("forex", pair, closes, highs, lows)

            # Hawkish USD + price below 20 EMA → momentum long USD
            if self.state.news.sentiment == Sentiment.BULLISH and below_ema:
                signal.direction = "long"
                signal.confidence *= self.state.news_multiplier()

            await self._emit(signal)

    # ── Sector D: Futures (Value Area mean reversion) ────────────────────────

    async def _sector_futures(self) -> None:
        d = self.state.market.futures
        if not d:
            return

        for contract, bars in d.items():
            closes  = np.array(bars["close"],  dtype=float)
            highs   = np.array(bars["high"],   dtype=float)
            lows    = np.array(bars["low"],    dtype=float)
            volumes = np.array(bars["volume"], dtype=float)
            if len(closes) < 50:
                continue

            # Anchored VWAP approximation (session open)
            cum_vol_price = np.cumsum(closes * volumes)
            cum_vol       = np.cumsum(volumes)
            avwap         = cum_vol_price / (cum_vol + 1e-9)

            val_hi = bars.get("value_area_high", avwap[-1] * 1.002)
            val_lo = bars.get("value_area_low",  avwap[-1] * 0.998)
            price  = closes[-1]

            signal = self._build_signal("futures", contract, closes, highs, lows)

            if price <= val_lo:
                signal.direction  = "long"   # mean-revert from value low
                signal.confidence = 0.75
            elif price >= val_hi:
                signal.direction  = "short"  # mean-revert from value high
                signal.confidence = 0.75

            await self._emit(signal)

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _build_signal(
        self,
        sector: str,
        symbol: str,
        closes: np.ndarray,
        highs:  np.ndarray,
        lows:   np.ndarray,
    ) -> TechnicalSignals:
        adx_val  = adx(highs, lows, closes)
        atr_val  = atr_value(highs, lows, closes)
        rsi_val  = rsi(closes)
        squeeze, _ = bollinger_squeeze(closes)

        ema9   = ema(closes, 9)
        ema20  = ema(closes, 20)
        ema200 = ema(closes, 200)
        ribbon = (closes[-1] > ema9[-1] > ema20[-1] > ema200[-1])  # bullish ribbon

        direction = "long" if ribbon and adx_val > 25 else "flat"

        return TechnicalSignals(
            sector            = sector,
            symbol            = symbol,
            direction         = direction,
            confidence        = min(adx_val / 50.0, 1.0),
            adx               = adx_val,
            rsi_divergence    = rsi_val < 30 or rsi_val > 70,
            bb_squeeze        = squeeze,
            ema_ribbon_aligned= ribbon,
            atr               = atr_val,
            timestamp         = datetime.now(timezone.utc),
        )

    async def _emit(self, signal: TechnicalSignals) -> None:
        if getattr(self.state, "circuit_open", False):
            log.debug(f"Signal for {signal.symbol} blocked — circuit breaker open.")
            return
        if self.state.is_vetoed():
            log.debug(f"Signal for {signal.symbol} suppressed — veto active.")
            return
        # Regime gate — check if this (sector, direction) is allowed
        regime_state = getattr(self.state, "regime_state", None)
        current_regime = regime_state.regime if regime_state else Regime.NEUTRAL
        if not is_signal_allowed(current_regime, signal.sector, signal.direction):
            log.debug(
                f"Signal for {signal.symbol} suppressed by regime gate "
                f"[{current_regime.value}]: {signal.sector}/{signal.direction} not allowed"
            )
            return

        if signal.direction == "flat":
            return
        log.info(
            f"Signal → {signal.sector}/{signal.symbol} {signal.direction} "
            f"ADX={signal.adx:.1f} Sp={signal.sp_score:.2f} "
            f"[regime={current_regime.value}]"
        )
        await self.state.put_signal(signal)
