"""
chimera/backtest/simulated_oms.py
SimulatedOMS — replaces the live OrderManager during backtesting.

Fill logic (bar-by-bar):
  - Market orders fill at open of the *next* bar + slippage.
  - Stop-loss: fills at stop_price if low (long) or high (short) crosses it.
    If the bar gaps through the stop, fill at open (gap risk, realistic).
  - Take-profit: fills at take_profit if high (long) or low (short) reaches it.
  - Trailing stop: ratcheted by TrailingStopManager on every bar close,
    then checked against the *next* bar's low/high.

Slippage model: flat bps per sector (configurable).
Commission model: flat $ per trade or bps of notional (configurable).

All fills are written to state.closed_trades for PerformanceReport.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TYPE_CHECKING

from chimera_v12.oms.models import Order, OrderSide, OrderStatus, CloseReason
from chimera_v12.oms.trailing_stop import TrailingStopManager

if TYPE_CHECKING:
    from chimera_v12.backtest.state import BacktestState
    from chimera_v12.utils.state import RiskParameters

log = logging.getLogger("chimera.backtest.simulated_oms")

# Default slippage in basis points per sector
DEFAULT_SLIPPAGE_BPS = {
    "crypto":  10,
    "stocks":  5,
    "forex":   3,
    "futures": 2,
}


@dataclass
class BacktestTrade:
    """Closed trade record for reporting."""
    symbol:        str
    sector:        str
    side:          str
    qty:           float
    entry_price:   float
    exit_price:    float
    entry_dt:      datetime
    exit_dt:       datetime
    stop_price:    float
    take_profit:   float
    realised_pnl:  float
    r_multiple:    float
    close_reason:  str
    commission:    float
    slippage_cost: float
    sp_score:      float = 0.0
    kelly_fraction: float = 0.0



def _copy_order_for_partial(order, partial_qty: float):
    """Create a lightweight Order copy for recording a partial close."""
    import copy
    p          = copy.copy(order)
    p.qty      = partial_qty
    p.order_id = order.order_id + "_partial"
    return p

class SimulatedOMS:
    """
    Bar-by-bar order execution simulator.

    Usage inside the backtest loop:
        oms.on_bar(symbol, sector, bar, dt)   — call once per bar per symbol
        oms.process_pending_orders(dt)         — fill next-bar entries
    """

    def __init__(self, state: "BacktestState", config: dict[str, Any]):
        self.state    = state
        self.config   = config
        self._trailing = TrailingStopManager(config)

        # Pending entries waiting for next-bar fill
        self._pending: list[tuple[Order, datetime]] = []
        # Open positions: symbol → Order
        self._open: dict[str, Order] = {}

        self._slippage_bps = {
            **DEFAULT_SLIPPAGE_BPS,
            **config.get("slippage_bps", {}),
        }
        self._commission_per_trade = config.get("commission_per_trade", 1.00)

    # ── Main hooks called by the BacktestEngine ────────────────────────────

    def accept_order(self, rp: "RiskParameters", dt: datetime) -> None:
        """
        Convert a RiskParameters object into a pending Order.
        Will fill at next bar's open (simulates market order latency).
        """
        # Determine direction from the most recent signal for this symbol
        side = OrderSide.BUY
        for sig in reversed(self.state.signals):
            if sig.symbol == rp.symbol:
                side = OrderSide.BUY if sig.direction == "long" else OrderSide.SELL
                break

        if rp.symbol in self._open:
            log.debug(f"Skipping {rp.symbol} — position already open")
            return
        if len(self._open) >= self.config.get("max_open_positions", 5):
            log.debug(f"Max positions reached — skipping {rp.symbol}")
            return

        order = Order(
            symbol         = rp.symbol,
            sector         = next((s.sector for s in reversed(self.state.signals) if s.symbol == rp.symbol), "unknown"),
            side           = side,
            qty            = rp.position_size,
            entry_price    = rp.entry_price,
            stop_price     = rp.stop_price,
            initial_stop   = rp.stop_price,
            take_profit    = rp.take_profit,
            atr            = next((s.atr for s in reversed(self.state.signals) if s.symbol == rp.symbol), 0.0),
            kelly_fraction = rp.kelly_fraction,
            sp_score       = next((s.sp_score for s in reversed(self.state.signals) if s.symbol == rp.symbol), 0.0),
            created_at     = dt,
        )
        self._pending.append((order, dt))
        log.debug(f"Queued {side.value} {rp.symbol} @ ~{rp.entry_price:.4f}")

    def on_bar(
        self,
        symbol: str,
        sector: str,
        bar:    dict[str, float],
        dt:     datetime,
    ) -> None:
        """
        Process one OHLCV bar for a symbol:
          1. Fill any pending entry orders at bar open.
          2. Check stop / TP / trailing stop for open positions.
        """
        # 1. Fill pending entries at this bar's open
        #
        # FIX 1 (lookahead): the engine stamps orders with dt + ONE_BAR_OFFSET
        # (1 second). queued_dt is always > signal_bar_dt, so this guard is
        # unambiguous even when multiple symbols share the same bar timestamp:
        #   signal bar N:   queued_dt = bar_dt + 1s  →  queued_dt >= dt → defer
        #   next bar N+1:   bar_dt > signal_bar_dt   →  queued_dt < dt  → fill
        still_pending = []
        for order, queued_dt in self._pending:
            if order.symbol != symbol:
                still_pending.append((order, queued_dt))
                continue
            if queued_dt >= dt:   # not yet — still on signal bar or earlier next bar
                still_pending.append((order, queued_dt))
                continue
            self._fill_entry(order, bar["open"], sector, dt)
        self._pending = still_pending

        # 2. Manage open position for this symbol
        if symbol not in self._open:
            return

        order = self._open[symbol]
        o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]

        # Check for gap-through stop at open
        if self._trailing.stop_hit(order, o):
            self._close(order, o, dt, CloseReason.STOP_HIT)
            return

        # Intrabar stop check
        stop_breached = (order.side == OrderSide.BUY and l <= order.stop_price) or \
                        (order.side == OrderSide.SELL and h >= order.stop_price)
        if stop_breached:
            fill = order.stop_price   # assume exact fill at stop (conservative)
            self._close(order, fill, dt, CloseReason.TRAILING_STOP
                        if order.stop_price != order.initial_stop
                        else CloseReason.STOP_HIT)
            return

        # Partial TP: close 50% of position at 2R — banks profit early
        if self._trailing.partial_tp_hit(order, h if order.side == OrderSide.BUY else l):
            ptp  = order.partial_take_profit
            frac = getattr(order, "partial_close_fraction", 0.5)
            partial_qty       = round(order.qty * frac, 6)
            order.qty        -= partial_qty
            order.partial_closed = True
            # Book the partial as a separate closed trade
            partial_order = _copy_order_for_partial(order, partial_qty)
            self._close(partial_order, ptp, dt, CloseReason.TP_HIT)
            log.info(f"PARTIAL TP [{order.symbol}]: {frac:.0%} @ {ptp:.4f}")
            # Remaining position continues with trailing stop

        # Full TP check
        tp_hit = (order.side == OrderSide.BUY  and h >= order.take_profit > 0) or \
                 (order.side == OrderSide.SELL and l <= order.take_profit > 0)
        if tp_hit:
            self._close(order, order.take_profit, dt, CloseReason.TP_HIT)
            return

        # Ratchet trailing stop on bar close
        new_stop = self._trailing.evaluate(order, c)
        if new_stop:
            order.stop_price = new_stop

        # Update unrealised P&L
        order.update_unrealised(c)

    # ── Fill helpers ───────────────────────────────────────────────────────

    def _fill_entry(
        self,
        order:  Order,
        open_price: float,
        sector: str,
        dt:     datetime,
    ) -> None:
        slippage_bps  = self._slippage_bps.get(sector, 5)
        slippage_mult = slippage_bps / 10_000
        if order.side == OrderSide.BUY:
            fill = open_price * (1 + slippage_mult)
        else:
            fill = open_price * (1 - slippage_mult)

        order.fill_price = round(fill, 6)
        order.status     = OrderStatus.FILLED
        order.filled_at  = dt

        commission = self._commission_per_trade
        self.state.equity -= commission

        self._open[order.symbol] = order
        self.state.open_positions[order.symbol] = order
        log.debug(f"FILLED {order.side.value} {order.qty:.2f} {order.symbol} @ {fill:.4f} [{dt.date()}]")

    def _close(
        self,
        order:  Order,
        price:  float,
        dt:     datetime,
        reason: CloseReason,
    ) -> None:
        # Apply exit slippage symmetrically — both entries AND exits incur
        # market impact.  Exiting a long fills at the bid (below price);
        # exiting a short fills at the ask (above price).
        slippage_bps  = self._slippage_bps.get(order.sector, 5)
        slippage_mult = slippage_bps / 10_000
        if order.side == OrderSide.BUY:   # exiting a long → sell at bid
            exit_fill = price * (1 - slippage_mult)
            pnl       = (exit_fill - order.fill_price) * order.qty
        else:                             # exiting a short → buy at ask
            exit_fill = price * (1 + slippage_mult)
            pnl       = (order.fill_price - exit_fill) * order.qty

        commission = self._commission_per_trade
        net_pnl    = pnl - commission

        order.realised_pnl = net_pnl
        order.compute_r_multiple()
        order.status       = OrderStatus.CLOSED
        order.close_reason = reason
        order.closed_at    = dt

        self.state.equity += net_pnl
        self._open.pop(order.symbol, None)
        self.state.open_positions.pop(order.symbol, None)

        # Slippage cost now already baked into net_pnl via exit_fill.
        slippage_cost = abs(price - exit_fill) * order.qty  # for reporting

        trade = BacktestTrade(
            symbol        = order.symbol,
            sector        = order.sector,
            side          = order.side.value,
            qty           = order.qty,
            entry_price   = order.fill_price,
            exit_price    = exit_fill,
            entry_dt      = order.filled_at,
            exit_dt       = dt,
            stop_price    = order.initial_stop,
            take_profit   = order.take_profit,
            realised_pnl  = net_pnl,
            r_multiple    = order.r_multiple,
            close_reason  = reason.value,
            commission    = commission * 2,
            slippage_cost = slippage_cost,
            sp_score      = order.sp_score,
            kelly_fraction= order.kelly_fraction,
        )
        self.state.closed_trades.append(vars(trade))

        log.debug(
            f"CLOSED {order.symbol} [{reason.value}] "
            f"PnL=${net_pnl:.2f} R={order.r_multiple:.2f} [{dt.date()}]"
        )

    def close_all(self, current_prices: dict[str, float], dt: datetime) -> None:
        """Force-close all open positions at end of backtest."""
        for sym, order in list(self._open.items()):
            price = current_prices.get(sym, order.fill_price)
            self._close(order, price, dt, CloseReason.MANUAL)

    @property
    def open_count(self) -> int:
        return len(self._open)
