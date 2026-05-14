"""
chimera_v12/options/wheel_engine.py — Options Wheel Trading System
See README Section 5 for full strategy explanation.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from chimera_v12.core.state import SharedState, WheelPosition
from chimera_v12.utils.logger import setup_logger

log = setup_logger("options.wheel")


# ── Black-Scholes Greeks ──────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_greeks(
    S: float, K: float, T: float, r: float, sigma: float, option_type: str = "put",
) -> dict:
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "price": 0.0}

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        delta = _norm_cdf(d1)
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        delta = _norm_cdf(d1) - 1.0
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    gamma = math.exp(-d1 ** 2 / 2) / (S * sigma * math.sqrt(2 * math.pi * T))
    vega  = S * math.sqrt(T) * math.exp(-d1 ** 2 / 2) / math.sqrt(2 * math.pi) / 100
    theta = (
        -(S * sigma * math.exp(-d1 ** 2 / 2)) / (2 * math.sqrt(2 * math.pi * T))
        - r * K * math.exp(-r * T) * (_norm_cdf(d2) if option_type == "call" else _norm_cdf(-d2))
    ) / 365

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 6),
        "theta": round(theta, 4),
        "vega":  round(vega,  4),
        "price": round(max(price, 0.0), 4),
    }


def iv_rank(current_iv: float, iv_52w_low: float, iv_52w_high: float) -> float:
    if iv_52w_high == iv_52w_low:
        return 50.0
    return ((current_iv - iv_52w_low) / (iv_52w_high - iv_52w_low)) * 100.0


def find_target_strike(
    stock_price: float,
    iv: float,
    dte: int,
    target_delta: float = 0.30,
    r: float = 0.05,
    option_type: str = "put",
    step: float = 0.50,
) -> tuple:
    T = dte / 365.0
    best_strike = stock_price
    best_diff   = float("inf")
    best_greeks: dict = {}
    search_range = stock_price * 0.30
    strike = stock_price - search_range
    while strike <= stock_price + search_range:
        g = black_scholes_greeks(stock_price, strike, T, r, iv, option_type)
        diff = abs(abs(g["delta"]) - target_delta)
        if diff < best_diff:
            best_diff   = diff
            best_strike = strike
            best_greeks = g
        strike += step
    return round(best_strike, 2), best_greeks


# ── Wheel Signal ──────────────────────────────────────────────────────────────

@dataclass
class WheelSignal:
    symbol:      str
    action:      str
    strike:      float
    expiry:      str
    option_type: str
    delta:       float
    premium_bid: float
    iv_rank:     float
    dte:         int
    reason:      str
    timestamp:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Wheel Engine ──────────────────────────────────────────────────────────────

class WheelEngine:
    def __init__(self, state: SharedState, config: dict[str, Any]):
        self.state   = state
        self.cfg     = config.get("options_wheel", {})
        self.interval = config.get("intervals", {}).get("options_scan_seconds", 120)
        self.r        = 0.05
        self.candidates = config.get("wheel_candidates", [])

    async def run(self) -> None:
        log.info("WheelEngine started.")
        while True:
            try:
                await self._scan_all()
            except Exception as e:
                log.warning(f"WheelEngine error: {e}")
            await asyncio.sleep(self.interval)

    async def _scan_all(self) -> None:
        for symbol in self.candidates:
            try:
                await self._evaluate_symbol(symbol)
            except Exception as e:
                log.debug(f"Wheel scan {symbol}: {e}")

    async def _evaluate_symbol(self, symbol: str) -> None:
        odata = self.state.market.options.get(symbol)
        sdata = self.state.market.stocks.get(symbol)
        if not odata or not sdata:
            return

        current_price = sdata.get("close", [0])[-1]
        current_iv    = odata.get("atm_iv", 0.25)
        ivr           = iv_rank(
            current_iv,
            odata.get("iv_52w_low", 0.15),
            odata.get("iv_52w_high", 0.60),
        )

        min_ivr = self.cfg.get("min_iv_rank", 30)
        max_ivr = self.cfg.get("max_iv_rank", 85)
        if not (min_ivr <= ivr <= max_ivr):
            return

        existing = self.state.wheel_positions.get(symbol)
        if existing is None:
            await self._enter_csp(symbol, current_price, current_iv, ivr)
        elif existing.phase == "cash_secured_put":
            await self._manage_csp(symbol, existing, current_price, current_iv)
        elif existing.phase in ("assigned", "covered_call"):
            await self._manage_covered_call(symbol, existing, current_price, current_iv, ivr)

    def _days_until(self, expiry_str: str) -> int:
        expiry = datetime.strptime(expiry_str, "%Y-%m-%d").date()
        return max((expiry - datetime.now(timezone.utc).date()).days, 0)

    async def _enter_csp(self, symbol: str, price: float, iv: float, ivr: float) -> None:
        dte       = self.cfg.get("dte_target", 30)
        strike, g = find_target_strike(price, iv, dte, self.cfg.get("target_delta_put", 0.30), self.r, "put")
        premium   = g["price"]
        min_prem  = self.cfg.get("min_premium_pct", 0.01) * strike
        if premium < min_prem:
            return
        expiry = (datetime.now(timezone.utc) + timedelta(days=dte)).strftime("%Y-%m-%d")
        signal = WheelSignal(
            symbol=symbol, action="OPEN_CSP",
            strike=strike, expiry=expiry, option_type="put",
            delta=g["delta"], premium_bid=premium, iv_rank=ivr, dte=dte,
            reason=f"IVR={ivr:.1f}, delta={g['delta']:.2f}, prem=${premium:.2f}",
        )
        await self.state.order_queue.put(signal)
        log.info(f"WHEEL CSP: {symbol} {strike}P {expiry} @ ${premium:.2f}  IVR={ivr:.1f}")
        await self.state.log_audit("WheelEngine", symbol, "OPEN_CSP", {
            "strike": strike, "expiry": expiry, "premium": premium, "ivr": ivr,
        })

    async def _manage_csp(self, symbol: str, pos: WheelPosition, price: float, iv: float) -> None:
        T       = self._days_until(pos.expiry)
        g       = black_scholes_greeks(price, pos.strike, T / 365.0, self.r, iv, "put")
        curr_v  = g["price"]
        open_v  = pos.premium
        profit  = (open_v - curr_v) / open_v if open_v > 0 else 0.0

        if profit >= self.cfg.get("profit_target_pct", 0.50):
            await self.state.order_queue.put(WheelSignal(
                symbol=symbol, action="CLOSE_EARLY", strike=pos.strike,
                expiry=pos.expiry, option_type="put", delta=g["delta"],
                premium_bid=curr_v, iv_rank=0, dte=T,
                reason=f"50% profit target ({profit:.0%})",
            ))
            return

        if T <= self.cfg.get("dte_min_close", 7):
            await self.state.order_queue.put(WheelSignal(
                symbol=symbol, action="ROLL", strike=pos.strike,
                expiry=pos.expiry, option_type="put", delta=g["delta"],
                premium_bid=curr_v, iv_rank=0, dte=T,
                reason=f"DTE={T} at roll threshold",
            ))
            return

        loss = (curr_v - open_v) / open_v if open_v > 0 else 0.0
        if loss >= self.cfg.get("loss_limit_pct", 2.00):
            await self.state.order_queue.put(WheelSignal(
                symbol=symbol, action="CLOSE_EARLY", strike=pos.strike,
                expiry=pos.expiry, option_type="put", delta=g["delta"],
                premium_bid=curr_v, iv_rank=0, dte=T,
                reason=f"Hard stop: loss {loss:.0%}",
            ))
            log.warning(f"WHEEL hard stop CSP {symbol}")

    async def _manage_covered_call(
        self, symbol: str, pos: WheelPosition, price: float, iv: float, ivr: float,
    ) -> None:
        if pos.phase == "assigned":
            dte       = self.cfg.get("dte_target", 30)
            strike, g = find_target_strike(price, iv, dte, self.cfg.get("target_delta_call", 0.30), self.r, "call")
            expiry    = (datetime.now(timezone.utc) + timedelta(days=dte)).strftime("%Y-%m-%d")
            await self.state.order_queue.put(WheelSignal(
                symbol=symbol, action="OPEN_CC", strike=strike, expiry=expiry,
                option_type="call", delta=g["delta"], premium_bid=g["price"],
                iv_rank=ivr, dte=dte,
                reason=f"Assigned at ${pos.strike:.2f}, CC at ${strike:.2f}",
            ))
            log.info(f"WHEEL CC: {symbol} {strike}C {expiry}")
            return

        T       = self._days_until(pos.expiry)
        g       = black_scholes_greeks(price, pos.strike, T / 365.0, self.r, iv, "call")
        curr_v  = g["price"]
        open_v  = pos.premium
        profit  = (open_v - curr_v) / open_v if open_v > 0 else 0.0
        if profit >= self.cfg.get("profit_target_pct", 0.50):
            await self.state.order_queue.put(WheelSignal(
                symbol=symbol, action="CLOSE_EARLY", strike=pos.strike,
                expiry=pos.expiry, option_type="call", delta=g["delta"],
                premium_bid=curr_v, iv_rank=ivr, dte=T,
                reason=f"50% profit ({profit:.0%})",
            ))
            log.info(f"WHEEL close CC {symbol}: {profit:.0%}")

    def expected_annual_return(self, symbol: str, stock_price: float, monthly_pct: float) -> dict:
        return {
            "symbol":              symbol,
            "stock_price":         stock_price,
            "monthly_premium_pct": monthly_pct,
            "est_annual_put_yield": f"{monthly_pct * 12:.1%}",
            "est_annual_cc_yield":  f"{monthly_pct * 10:.1%}",
            "est_total_annual":     f"{monthly_pct * 22:.1%}",
        }
