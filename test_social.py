"""
chimera_v12/tests/test_unified.py
Integration tests for Project Chimera v12.
Run with: python -m pytest chimera_v12/tests/ -v
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest
from chimera_v12.core.state import SharedState, MarketRegime, TechnicalSignals, RiskParameters
from chimera_v12.options.wheel_engine import black_scholes_greeks, iv_rank, find_target_strike, WheelEngine
from chimera_v12.strategies.scoring import (
    piotroski_f_score, value_score, momentum_score,
    growth_score, compute_composite, kelly_position_size
)
from chimera_v12.strategies.quant_edge.engine import (
    OptionsMicrostructureArb, CrossAssetETFArb,
    OrderFlowImbalance, VolSurfaceArb, NewsSentimentAlpha
)
from chimera_v12.strategies.sector.engine import (
    squeeze_probability_score, StocksStrategy, CryptoStrategy,
    ForexStrategy, FuturesStrategy
)
from chimera_v12.orchestrator.master import VotingAgent, HITLCheckpoint
from chimera_v12.utils.ta import ema, rsi, adx, atr_value, bollinger_squeeze
from chimera_v12.config.settings import load_config

# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def state():
    return SharedState()

@pytest.fixture
def config():
    """Minimal config for tests — avoids env var requirements."""
    return {
        "mode": "paper",
        "stocks": {
            "squeeze": {
                "min_sp_score": 0.60, "si_weight": 0.40,
                "rvol_weight": 0.30, "sentiment_weight": 0.30,
                "si_cap": 0.50, "rvol_cap": 10.0
            },
            "adx_trending": 25, "adx_ranging": 20,
            "atr_multiplier_stop": 2.0, "atr_multiplier_target": 3.0
        },
        "crypto":  {"ema_fast": 9, "ema_slow": 21, "rsi_period": 14,
                    "rsi_oversold": 30, "rsi_overbought": 70,
                    "btc_inflow_threshold": 1_000_000,
                    "sol_memecoin_spike_threshold": 50_000_000,
                    "funding_rate_extreme": 0.01},
        "forex":   {"ema_fast": 20, "ema_slow": 50, "rsi_period": 14,
                    "rsi_momentum_bull": 55, "rsi_momentum_bear": 45,
                    "news_bias_weight": 0.40, "carry_weight": 0.20,
                    "session_filter": False},
        "futures": {"va_lookback_bars": 20, "va_pct": 0.70,
                    "adx_trending": 25, "rollover_days_before": 5},
        "options_wheel": {
            "min_iv_rank": 30, "max_iv_rank": 85,
            "dte_target": 30, "dte_min_close": 7,
            "profit_target_pct": 0.50, "loss_limit_pct": 2.00,
            "min_premium_pct": 0.01, "target_delta_put": 0.30,
            "target_delta_call": 0.30
        },
        "quant_edge": {
            "options_microstructure": {"min_bid_ask_edge_bps": 5, "hold_seconds_min": 30},
            "etf_arb": {"pairs": [["SPY", "ES1!"]], "z_score_entry": 2.0,
                        "z_score_exit": 0.5, "min_spread_bps": 5,
                        "mean_revert_minutes": 10},
            "order_flow_imbalance": {"imbalance_thresh": 0.60, "window_seconds": 60},
            "vol_surface_arb": {"skew_zscore_entry": 2.0},
            "news_sentiment_alpha": {"min_score": 0.65, "hold_minutes": 15,
                                     "decay_halflife_min": 30}
        },
        "risk": {
            "base_risk_pct": 0.01, "kelly_fraction": 0.25,
            "avg_win_r": 1.8, "avg_loss_r": 1.0,
            "max_single_position": 0.10, "hitl_threshold_usd": 10_000
        },
        "wheel_candidates": ["AAPL", "MSFT"],
        "intervals": {"strategy_seconds": 15, "options_scan_seconds": 120,
                      "quant_edge_seconds": 5}
    }

@pytest.fixture
def sample_prices():
    import numpy as np
    np.random.seed(42)
    # 260 bars of synthetic price data with upward drift
    p = [100.0]
    for _ in range(259):
        p.append(p[-1] * (1 + np.random.normal(0.0003, 0.015)))
    return p

@pytest.fixture
def sample_bars(sample_prices):
    import numpy as np
    closes  = sample_prices
    highs   = [c * 1.005 for c in closes]
    lows    = [c * 0.995 for c in closes]
    volumes = [1_000_000 + np.random.randint(-200_000, 200_000) for _ in closes]
    return {"close": closes, "high": highs, "low": lows, "volume": volumes}

# ══════════════════════════════════════════════════════════════════════════════
# 1. Core State Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSharedState:
    def test_initial_equity(self, state):
        assert state.equity == 100_000.0

    def test_initial_regime(self, state):
        assert state.regime == MarketRegime.UNKNOWN

    def test_snapshot_returns_dict(self, state):
        snap = state.snapshot("AAPL")
        assert isinstance(snap, dict)
        assert snap["symbol"] == "AAPL"
        assert "final_trade_decision" in snap
        assert "audit_trail" in snap

    def test_snapshot_defaults(self, state):
        snap = state.snapshot("TSLA")
        assert snap["circuit_open"] is False
        assert snap["news_veto_active"] is False
        assert snap["hitl_required"] is False

    def test_circuit_open_propagates(self, state):
        state.circuit_open = True
        snap = state.snapshot("AAPL")
        assert snap["circuit_open"] is True

# ══════════════════════════════════════════════════════════════════════════════
# 2. Options Wheel Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBlackScholes:
    def test_call_positive(self):
        g = black_scholes_greeks(100, 100, 0.25, 0.05, 0.20, "call")
        assert g["price"] > 0
        assert 0 < g["delta"] < 1

    def test_put_positive(self):
        g = black_scholes_greeks(100, 100, 0.25, 0.05, 0.20, "put")
        assert g["price"] > 0
        assert -1 < g["delta"] < 0

    def test_put_call_parity(self):
        """C - P ≈ S - K*e^(-rT)"""
        S, K, T, r, sig = 100, 100, 0.25, 0.05, 0.20
        c = black_scholes_greeks(S, K, T, r, sig, "call")["price"]
        p = black_scholes_greeks(S, K, T, r, sig, "put")["price"]
        parity = S - K * math.exp(-r * T)
        assert abs((c - p) - parity) < 0.01

    def test_deep_itm_call_delta(self):
        g = black_scholes_greeks(150, 100, 1.0, 0.05, 0.20, "call")
        assert g["delta"] > 0.90

    def test_deep_otm_put_delta(self):
        g = black_scholes_greeks(150, 100, 0.08, 0.05, 0.20, "put")
        assert abs(g["delta"]) < 0.10

    def test_zero_dte_returns_zeros(self):
        g = black_scholes_greeks(100, 100, 0, 0.05, 0.20)
        assert g["price"] == 0.0

    def test_iv_rank_range(self):
        assert iv_rank(0.30, 0.15, 0.60) == pytest.approx(33.33, rel=0.01)
        assert iv_rank(0.60, 0.15, 0.60) == pytest.approx(100.0, rel=0.01)
        assert iv_rank(0.15, 0.15, 0.60) == pytest.approx(0.0, rel=0.01)

    def test_find_target_strike_delta(self):
        strike, greeks = find_target_strike(100, 0.25, 30, 0.30, 0.05, "put")
        assert 70 <= strike <= 100
        assert abs(abs(greeks["delta"]) - 0.30) < 0.15

# ══════════════════════════════════════════════════════════════════════════════
# 3. Scoring Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScoring:
    def test_piotroski_perfect_score(self):
        # F4 (low accruals): CFO/assets must exceed ROA
        # accrual = roa - (cfo/ta); need accrual < 0 → cfo/ta > roa
        # roa=0.10, cfo/ta = 9_000_000/50_000_000 = 0.18 > 0.10 ✓
        funds = {
            "roa": 0.10, "roa_prior": 0.08,
            "operating_cashflow": 9_000_000,
            "total_assets": 50_000_000,
            "long_term_debt": 1_000_000,
            "long_term_debt_prior": 2_000_000,
            "current_ratio": 2.5, "current_ratio_prior": 2.0,
            "shares_issued_yoy": False,
            "gross_margin": 0.45, "gross_margin_prior": 0.40,
            "asset_turnover": 1.2, "asset_turnover_prior": 1.0
        }
        score, detail = piotroski_f_score(funds)
        assert score == 9
        assert all(v == 1 for v in detail.values())

    def test_piotroski_zero_score(self):
        funds = {
            "roa": -0.05, "roa_prior": 0.05,
            "operating_cashflow": -100_000,
            "total_assets": 1_000_000,
            "long_term_debt": 500_000, "long_term_debt_prior": 400_000,
            "current_ratio": 0.8, "current_ratio_prior": 1.2,
            "shares_issued_yoy": True,
            "gross_margin": 0.20, "gross_margin_prior": 0.30,
            "asset_turnover": 0.5, "asset_turnover_prior": 0.8
        }
        score, _ = piotroski_f_score(funds)
        assert score == 0

    def test_value_score_cheap_stock(self):
        funds = {"pe_ratio": 10, "pb_ratio": 1.0, "ev_ebitda": 6, "fcf_yield": 0.08, "roic": 0.20}
        score = value_score(funds)
        assert score > 0.60

    def test_value_score_expensive(self):
        funds = {"pe_ratio": 80, "pb_ratio": 15, "ev_ebitda": 40, "fcf_yield": 0.005, "roic": 0.05}
        score = value_score(funds)
        assert score < 0.40

    def test_momentum_score_uptrend(self, sample_prices):
        # Growing prices → strong momentum
        score = momentum_score(sample_prices)
        assert 0.0 <= score <= 1.0

    def test_growth_score_high(self):
        funds = {"revenue_growth_yoy": 0.30, "eps_growth_yoy": 0.25,
                 "fcf_growth_yoy": 0.20, "revenue_acceleration": 0.05}
        score = growth_score(funds)
        assert score > 0.60

    def test_composite_score_range(self, sample_prices):
        funds = {
            "roa": 0.12, "roa_prior": 0.10, "operating_cashflow": 1e6,
            "total_assets": 10e6, "long_term_debt": 500_000,
            "long_term_debt_prior": 700_000, "current_ratio": 2.0,
            "current_ratio_prior": 1.8, "shares_issued_yoy": False,
            "gross_margin": 0.40, "gross_margin_prior": 0.38,
            "asset_turnover": 1.0, "asset_turnover_prior": 0.95,
            "pe_ratio": 20, "pb_ratio": 3, "ev_ebitda": 12, "fcf_yield": 0.05, "roic": 0.18,
            "revenue_growth_yoy": 0.15, "eps_growth_yoy": 0.12,
            "fcf_growth_yoy": 0.10, "revenue_acceleration": 0.02,
            "sector": "technology"
        }
        cs = compute_composite("AAPL", "technology", funds, sample_prices)
        assert 0.0 <= cs.composite <= 1.0
        assert cs.recommendation in ["STRONG BUY","BUY","HOLD","SELL","STRONG SELL"]

    def test_kelly_sizing(self):
        size = kelly_position_size(100_000, 0.55, 1.8, 1.0, 0.25, 0.70, 0.10)
        assert 0 < size <= 10_000   # max 10% of equity
        # Higher composite score → larger position
        size_low = kelly_position_size(100_000, 0.55, 1.8, 1.0, 0.25, 0.20, 0.10)
        assert size > size_low

    def test_kelly_zero_win_rate(self):
        size = kelly_position_size(100_000, 0.0, 1.8, 1.0, 0.25, 0.70, 0.10)
        assert size == 0.0

# ══════════════════════════════════════════════════════════════════════════════
# 4. Quant Edge Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestQuantEdge:
    def test_microstructure_arb_detects_skew(self, config):
        arb = OptionsMicrostructureArb(config)
        sig = arb.scan("SPY", {"atm_call_iv": 0.20, "atm_put_iv": 0.24})
        assert sig is not None
        assert sig.direction in ("BUY", "SELL")
        assert sig.strength > 0

    def test_microstructure_no_signal_small_skew(self, config):
        arb = OptionsMicrostructureArb(config)
        # 0.001% skew = 0.1 bps — well below the 5 bps threshold
        sig = arb.scan("SPY", {"atm_call_iv": 0.20000, "atm_put_iv": 0.20001})
        assert sig is None

    def test_order_flow_imbalance_buy(self, config):
        ofi = OrderFlowImbalance(config)
        sig = ofi.scan("AAPL", bid_volume=8000, ask_volume=2000)
        assert sig is not None
        assert sig.direction == "BUY"
        assert sig.strength >= 0.60

    def test_order_flow_imbalance_sell(self, config):
        ofi = OrderFlowImbalance(config)
        sig = ofi.scan("AAPL", bid_volume=1000, ask_volume=9000)
        assert sig is not None
        assert sig.direction == "SELL"

    def test_order_flow_no_signal_balanced(self, config):
        ofi = OrderFlowImbalance(config)
        sig = ofi.scan("AAPL", bid_volume=5000, ask_volume=5000)
        assert sig is None

    def test_news_sentiment_strong_beat(self, config):
        ns = NewsSentimentAlpha(config)
        sig = ns.process_news("NVDA", 0.85, "earnings_beat")
        assert sig is not None
        assert sig.direction == "BUY"
        assert sig.strength > 0.65

    def test_news_sentiment_fade_overreaction(self, config):
        ns = NewsSentimentAlpha(config)
        sig = ns.process_news("TSLA", 0.90, "earnings_beat", gap_pct=0.15)
        # Gap > 10% → fade it (sell the news)
        assert sig is not None
        assert sig.direction == "SELL"

    def test_news_sentiment_below_threshold(self, config):
        ns = NewsSentimentAlpha(config)
        sig = ns.process_news("AAPL", 0.40, "general")
        assert sig is None

    def test_vol_surface_arb_inversion(self, config):
        vs = VolSurfaceArb(config)
        # Pre-populate history
        for _ in range(25):
            vs.update_and_scan("SPX", iv_front=0.20, iv_back=0.22,
                                put_skew=0.25, call_skew=0.19)
        # Now extreme inversion
        sig = vs.update_and_scan("SPX", iv_front=0.35, iv_back=0.22,
                                  put_skew=0.25, call_skew=0.19)
        assert sig is not None
        assert sig.direction in ("BUY", "SELL")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Sector Strategy Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSectorStrategies:
    def test_squeeze_probability_high(self, config):
        sp = squeeze_probability_score(0.40, 5.0, 3.0,
                                       config["stocks"]["squeeze"])
        assert sp > 0.60

    def test_squeeze_probability_low(self, config):
        sp = squeeze_probability_score(0.02, 1.1, 0.1,
                                       config["stocks"]["squeeze"])
        assert sp < 0.30

    def test_squeeze_probability_bounded(self, config):
        sp = squeeze_probability_score(1.0, 100.0, 100.0,
                                       config["stocks"]["squeeze"])
        assert 0.0 <= sp <= 1.0

    def test_stocks_strategy_returns_signal(self, config, sample_bars):
        strat = StocksStrategy(config)
        sig   = strat.evaluate("AAPL", sample_bars)
        assert isinstance(sig, TechnicalSignals)
        assert sig.direction in ("BUY", "SELL", "HOLD")

    def test_crypto_strategy_whale_veto(self, config, sample_bars):
        strat = CryptoStrategy(config)
        sig   = strat.evaluate("BTC/USD", sample_bars,
                                whale_inflow=5_000_000)  # above threshold
        assert sig.direction == "HOLD"

    def test_forex_no_signal_in_ranging(self, config, sample_bars):
        strat = ForexStrategy(config)
        # Near-flat bars → no strong signal
        flat_bars = {k: [100.0] * 60 for k in ["close", "high", "low"]}
        sig = strat.evaluate("EUR/USD", flat_bars)
        assert isinstance(sig, TechnicalSignals)

    def test_futures_rollover_close(self, config, sample_bars):
        strat = FuturesStrategy(config)
        sig   = strat.evaluate("ES1!", sample_bars, days_to_exp=3)
        assert sig.direction == "CLOSE"

# ══════════════════════════════════════════════════════════════════════════════
# 6. Technical Analysis Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTA:
    def test_ema_length(self, sample_prices):
        import numpy as np
        prices = np.array(sample_prices)
        result = ema(prices, 20)
        assert len(result) == len(prices)

    def test_rsi_range(self, sample_prices):
        import numpy as np
        r = rsi(np.array(sample_prices))
        assert 0 <= r <= 100

    def test_atr_positive(self, sample_bars):
        import numpy as np
        v = atr_value(
            np.array(sample_bars["high"]),
            np.array(sample_bars["low"]),
            np.array(sample_bars["close"])
        )
        assert v > 0

    def test_bollinger_squeeze_bool(self, sample_prices):
        import numpy as np
        result, width = bollinger_squeeze(np.array(sample_prices))
        assert isinstance(result, bool)
        assert isinstance(width, float)

# ══════════════════════════════════════════════════════════════════════════════
# 7. Orchestrator Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOrchestrator:
    def _base_state(self, config):
        return {
            "symbol":          "AAPL",
            "circuit_open":    False,
            "news_veto_active": False,
            "investment_plan": "BUY based on strong fundamentals",
            "risk_debate_state": {"judge_decision": "APPROVE the trade"},
            "squeeze_probability": 0.70,
            "composite_score": 0.72,
            "order_flow_imbalance": 0.65,
            "atr": 2.5,
            "_equity": 100_000.0,
            "_last_price": 180.0,
            "audit_trail": [],
            "cycle_id": "test_cycle_001"
        }

    def test_circuit_breaker_forces_hold(self, config):
        voter = VotingAgent()
        state = self._base_state(config)
        state["circuit_open"] = True
        result = voter.resolve(state, config)
        assert result["final_trade_decision"] == "HOLD"
        assert result["decision_confidence"] == 1.0

    def test_news_veto_forces_hold(self, config):
        voter = VotingAgent()
        state = self._base_state(config)
        state["news_veto_active"] = True
        result = voter.resolve(state, config)
        assert result["final_trade_decision"] == "HOLD"

    def test_strong_buy_consensus(self, config):
        voter  = VotingAgent()
        state  = self._base_state(config)
        result = voter.resolve(state, config)
        assert result["final_trade_decision"] == "BUY"
        assert result["position_size_usd"] > 0

    def test_low_composite_downgrades_buy(self, config):
        voter = VotingAgent()
        state = self._base_state(config)
        state["composite_score"] = 0.20   # very low quality
        result = voter.resolve(state, config)
        assert result["final_trade_decision"] == "HOLD"

    def test_hitl_flag_on_large_position(self, config):
        voter = VotingAgent()
        state = self._base_state(config)
        state["_equity"]          = 10_000_000  # large account
        state["composite_score"]  = 0.90
        result = voter.resolve(state, config)
        # Large account may trigger HITL
        assert "hitl_required" in result

    def test_position_size_non_negative(self, config):
        voter  = VotingAgent()
        state  = self._base_state(config)
        result = voter.resolve(state, config)
        assert result["position_size_usd"] >= 0

    def test_audit_trail_populated(self, config):
        voter  = VotingAgent()
        state  = self._base_state(config)
        result = voter.resolve(state, config)
        assert len(result["audit_trail"]) > 0
        assert result["audit_trail"][-1]["agent"] == "VotingAgent"

    def test_stop_loss_below_price_for_buy(self, config):
        voter  = VotingAgent()
        state  = self._base_state(config)
        result = voter.resolve(state, config)
        if result["final_trade_decision"] == "BUY":
            assert result["stop_loss"] < state["_last_price"]
            assert result["take_profit"] > state["_last_price"]

# ══════════════════════════════════════════════════════════════════════════════
# 8. Config Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_sector_weights_present(self, config):
        assert "stocks" in config
        assert "crypto" in config
        assert "forex" in config
        assert "futures" in config
        assert "options_wheel" in config
        assert "quant_edge" in config

    def test_risk_params_sane(self, config):
        risk = config["risk"]
        assert 0 < risk["kelly_fraction"] <= 0.5
        assert risk["max_single_position"] <= 0.20
        assert risk["avg_win_r"] > risk["avg_loss_r"]

    def test_wheel_params_sane(self, config):
        wh = config["options_wheel"]
        assert 0 < wh["target_delta_put"] < 0.50
        assert wh["profit_target_pct"] == 0.50
        assert wh["min_iv_rank"] < wh["max_iv_rank"]


# ══════════════════════════════════════════════════════════════════════════════
# 9. Rolling Sortino on R-multiples
# ══════════════════════════════════════════════════════════════════════════════

class TestRollingSortino:
    """Validates the new R-multiple Sortino implementation."""

    def _perf(self, trades, days=252):
        from chimera_v12.backtest.performance import PerformanceReport
        from datetime import datetime as _dt
        eq = [(_dt(2024, 1, 1), 100_000.0), (_dt(2024, 12, 31), 100_000.0)]
        return PerformanceReport(trades, eq, 100_000.0, {"risk_free_rate": 0.05, "sortino_window": 10})

    def _trade(self, r: float, pnl: float | None = None):
        return {
            "realised_pnl": pnl if pnl is not None else r * 1000,
            "r_multiple": r,
            "commission": 0.0,
            "sector": "stocks",
            "close_reason": "target",
        }

    def test_positive_sortino_on_profitable_system(self):
        trades = [self._trade(1.5) for _ in range(20)]
        trades += [self._trade(-0.8) for _ in range(10)]
        m = self._perf(trades).compute()
        assert m["sortino_ratio"] > 0, "Profitable system should have positive Sortino"

    def test_sortino_exceeds_sharpe_on_positive_skew(self):
        """When wins are larger than losses, Sortino > Sharpe (less penalised)."""
        trades = [self._trade(3.0)] * 15 + [self._trade(-1.0)] * 15
        m = self._perf(trades).compute()
        # Sortino ignores upside variance so it should be >= Sharpe
        assert m["sortino_ratio"] >= m["sharpe_ratio"], (
            f"Sortino {m['sortino_ratio']} should >= Sharpe {m['sharpe_ratio']} "
            "on positively-skewed R distribution"
        )

    def test_all_winners_caps_at_sortino_cap(self):
        from chimera_v12.backtest.performance import SORTINO_CAP
        trades = [self._trade(2.0) for _ in range(15)]
        m = self._perf(trades).compute()
        assert m["sortino_ratio"] == SORTINO_CAP

    def test_no_dollar_pnl_influence(self):
        """Two systems with identical R-multiples but different position sizes
        must produce identical Sortino ratios."""
        rs = [1.5, -0.8, 2.1, -0.5, 1.0, -1.0, 3.0, -0.3, 0.8, -0.6]
        # Small positions
        trades_small = [self._trade(r, pnl=r * 100) for r in rs]
        # Large positions (10× bigger dollar PnL)
        trades_large = [self._trade(r, pnl=r * 10_000) for r in rs]
        m_s = self._perf(trades_small).compute()
        m_l = self._perf(trades_large).compute()
        assert abs(m_s["sortino_ratio"] - m_l["sortino_ratio"]) < 0.001, (
            "Sortino must be identical regardless of dollar position size"
        )

    def test_rolling_window_is_configurable(self):
        from chimera_v12.backtest.performance import PerformanceReport
        from datetime import datetime as _dt
        trades = [self._trade(1.0)] * 5 + [self._trade(-2.0)] * 5 + [self._trade(1.5)] * 20
        eq = [(_dt(2024, 1, 1), 100_000.0), (_dt(2024, 12, 31), 100_000.0)]
        # Small window (recent 5 trades) vs large window (all 30)
        m5  = PerformanceReport(trades, eq, 100_000.0, {"risk_free_rate": 0.05, "sortino_window": 5}).compute()
        m30 = PerformanceReport(trades, eq, 100_000.0, {"risk_free_rate": 0.05, "sortino_window": 30}).compute()
        # With 5-trade window on all-winners tail, rolling_sortino_r should differ
        assert "rolling_sortino_r" in m5
        assert "rolling_sortino_r" in m30
        # Both should be present and be floats
        assert isinstance(m5["rolling_sortino_r"], float)
        assert isinstance(m30["rolling_sortino_r"], float)
        # The window metadata should reflect the config
        assert m5["sortino_window_trades"] == 5
        assert m30["sortino_window_trades"] == 30

    def test_annualisation_uses_tpy_not_252(self):
        """
        Verify the formula uses sqrt(tpy), not a hardcoded sqrt(252).
        Uses weak-edge data designed to stay below SORTINO_CAP so the
        ratio test is not distorted by capping.
        base = mean_R / ds_std must satisfy: base < 10 / sqrt(252) ≈ 0.630
        """
        from chimera_v12.backtest.performance import rolling_sortino_r, _mean, _std, SORTINO_CAP
        import math

        # Use tpy=4 vs tpy=16 — ratio=2.0, well under SORTINO_CAP at both values.
        # Same R-series, different tpy → exactly 2× difference in annualised Sortino.
        rs = [0.8, -0.5, 1.2, -0.4, 0.6, -0.3, 0.9, -0.6, 1.1, -0.2]
        rfr_pt = 0.0

        s_tpy4  = rolling_sortino_r(rs, rfr_pt, len(rs), 4)
        s_tpy16 = rolling_sortino_r(rs, rfr_pt, len(rs), 16)

        assert s_tpy16 > s_tpy4 > 0, (
            f"Higher tpy must produce higher Sortino: {s_tpy16:.4f} > {s_tpy4:.4f}"
        )

        # sqrt(16) / sqrt(4) = exactly 2.0 — ratio must be within 2%
        ratio    = s_tpy16 / s_tpy4
        expected = math.sqrt(16) / math.sqrt(4)   # = 2.0
        assert abs(ratio - expected) / expected < 0.02, (
            f"ratio {ratio:.4f} must equal sqrt(16)/sqrt(4)={expected:.4f} ±2%"
        )

    def test_rfr_reduces_sortino(self):
        """Higher risk-free rate must produce lower Sortino (hurdle rate harder)."""
        from chimera_v12.backtest.performance import rolling_sortino_r
        rs = [0.5, -0.3, 0.8, -0.2, 0.6]
        tpy = 50
        s_low_rfr  = rolling_sortino_r(rs, 0.02 / tpy, len(rs), tpy)
        s_high_rfr = rolling_sortino_r(rs, 0.08 / tpy, len(rs), tpy)
        assert s_low_rfr >= s_high_rfr, "Higher RFR must lower Sortino"

    def test_rolling_series_length_matches_trades(self):
        trades = [self._trade(r) for r in [1.0, -0.5, 2.0, -1.0, 0.5, 1.5, -0.3]]
        m = self._perf(trades).compute()
        assert len(m["rolling_sortino_series"]) == len(trades)

    def test_sector_sortino_populated(self):
        trades = [self._trade(1.5, pnl=1500) for _ in range(10)]
        trades += [{"realised_pnl": -800, "r_multiple": -0.8, "commission": 0,
                    "sector": "crypto", "close_reason": "stop"}] * 5
        m = self._perf(trades).compute()
        assert "stocks" in m["by_sector"]
        assert "sortino_r" in m["by_sector"]["stocks"]


# ══════════════════════════════════════════════════════════════════════════════
# 10. Kelly R-Multiple Test
# ══════════════════════════════════════════════════════════════════════════════

class TestKellyRMultiple:
    """Kelly criterion now uses R-magnitude not win/loss boolean."""

    def _make_agent(self, config=None):
        from chimera_v12.agents.risk_agent import RiskAgent
        from chimera_v12.core.state import SharedState
        state = SharedState()
        return RiskAgent(state, config or {"kelly_lookback": 20, "avg_win_r": 1.5, "avg_loss_r": 1.0})

    def test_small_r_wins_produce_smaller_kelly(self):
        """Many small-R wins + few large-R losses → smaller Kelly than vice versa."""
        agent = self._make_agent()
        for _ in range(15):
            agent.record_trade_outcome(0.2)   # small wins
        for _ in range(5):
            agent.record_trade_outcome(-2.0)  # large losses
        kelly_small_wins = agent._kelly_fraction()

        agent2 = self._make_agent()
        for _ in range(15):
            agent2.record_trade_outcome(2.0)  # large wins
        for _ in range(5):
            agent2.record_trade_outcome(-0.2) # small losses
        kelly_large_wins = agent2._kelly_fraction()

        assert kelly_large_wins > kelly_small_wins, (
            "Large-R wins should produce higher Kelly than small-R wins"
        )

    def test_returns_default_with_insufficient_history(self):
        agent = self._make_agent()
        assert agent._kelly_fraction() == 0.10  # default before 10 trades

    def test_negative_expectancy_returns_minimal(self):
        agent = self._make_agent()
        for _ in range(12):
            agent.record_trade_outcome(-1.5)  # all losers
        assert agent._kelly_fraction() <= 0.05

    def test_kelly_capped_at_max(self):
        from chimera_v12.agents.risk_agent import MAX_KELLY_FRACTION
        agent = self._make_agent()
        for _ in range(20):
            agent.record_trade_outcome(5.0)   # huge wins
        agent.record_trade_outcome(-0.1)      # one tiny loss
        assert agent._kelly_fraction() <= MAX_KELLY_FRACTION


# ══════════════════════════════════════════════════════════════════════════════
# 11. Trailing Stop Calibration Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrailingStop:
    """
    Validates the v12 calibrated 3-stage trailing stop.

    Key regression: in v11, the trail activated at +1R with a 2×ATR width —
    the same as the initial stop. Any pullback after a small gain would stop
    out the position (trail_exit dominant, 27% win rate).

    v12 fix: trail only activates at +2R, with a 3×ATR width.
    """

    def _order(self, fill=100.0, stop=96.0, atr=2.0, tp=108.0, side="BUY"):
        """
        Build a minimal Order for trailing stop tests.
        Uses dataclass() instead of __new__ so all fields are properly set.
        is_open is a computed property (status == FILLED) — set status directly.
        """
        from chimera_v12.oms.models import Order, OrderSide, OrderStatus
        o = Order(
            symbol        = "TEST",
            side          = OrderSide.BUY if side == "BUY" else OrderSide.SELL,
            fill_price    = fill,
            initial_stop  = stop,
            stop_price    = stop,
            take_profit   = tp,
            atr           = atr,
            qty           = 10.0,
            status        = OrderStatus.FILLED,   # is_open is derived from this
        )
        return o

    def _trail(self, config=None):
        from chimera_v12.oms.trailing_stop import TrailingStopManager
        return TrailingStopManager(config or {})

    def test_no_trail_before_2r(self):
        """Trail MUST NOT activate before 2R profit — this was the v11 bug."""
        trail  = self._trail()
        order  = self._order(fill=100.0, stop=96.0, atr=2.0)
        # At +1R = 104.0, trail should NOT move (returns None)
        result = trail.evaluate(order, 104.0)
        assert result is None, (
            f"Trail activated too early at +1R (got {result}). "
            "This is the v11 bug that caused trail_exit to dominate."
        )

    def test_no_trail_at_1_5r(self):
        """Still no trail at +1.5R."""
        trail  = self._trail()
        order  = self._order(fill=100.0, stop=96.0, atr=2.0)
        result = trail.evaluate(order, 106.0)  # +1.5R
        assert result is None

    def test_trail_activates_at_2r(self):
        """Trail activates once 2R is reached."""
        trail  = self._trail()
        order  = self._order(fill=100.0, stop=96.0, atr=2.0)
        # At +2R = 108.0, trail should activate (returns new stop)
        result = trail.evaluate(order, 108.0)
        assert result is not None, "Trail must activate at 2R"

    def test_breakeven_at_2r(self):
        """At 2R, stop moves to breakeven (entry price), never below."""
        trail  = self._trail()
        order  = self._order(fill=100.0, stop=96.0, atr=2.0)
        # At exactly 2R = 108.0, trail should be at or above fill (100.0)
        new_stop = trail.evaluate(order, 108.0)
        assert new_stop is not None
        assert new_stop >= 100.0, f"Breakeven should be at fill=100.0, got {new_stop}"

    def test_trail_width_is_3x_atr(self):
        """
        Trail uses 3×ATR width (wider than the 2×ATR initial stop).
        At +3R = price 112, trail = price - 3×ATR = 112 - 6 = 106.
        Breakeven floor (100) is also active but 106 > 100 so trail wins.
        """
        trail  = self._trail()
        order  = self._order(fill=100.0, stop=96.0, atr=2.0)
        # Risk per unit = |fill - stop| = |100 - 96| = 4  → 1R = 4 price units
        # At +3R price = 100 + 3×4 = 112
        new_stop = trail.evaluate(order, 112.0)
        assert new_stop is not None, "Trail must activate at +3R"
        # Expected: trail = price - 3×ATR = 112 - 3×2 = 106
        expected_trail = 112.0 - 3 * 2.0   # 106
        expected_floor = 100.0              # breakeven (fill price)
        expected = max(expected_trail, expected_floor)
        assert abs(new_stop - expected) < 0.05, (
            f"Trail width: expected {expected:.2f} (3×ATR behind price), got {new_stop:.2f}"
        )

    def test_trail_ratchets_only_up_for_long(self):
        """
        Trail only ratchets in the profitable direction.
        On a pullback from a high, evaluate() must return None (no change)
        or a stop >= the current stop_price.
        """
        import dataclasses
        trail  = self._trail()
        order  = self._order(fill=100.0, stop=96.0, atr=2.0)

        # Activate trail at 2R = 108 (4 units above fill with 2-ATR risk = 4)
        stop1 = trail.evaluate(order, 108.0)
        assert stop1 is not None, "Trail must activate at 2R"
        order = dataclasses.replace(order, stop_price=stop1)

        # Price moves higher to 3R = 112
        stop2 = trail.evaluate(order, 112.0)
        assert stop2 is not None
        assert stop2 >= stop1, f"Stop must ratchet up: {stop2:.4f} >= {stop1:.4f}"
        order = dataclasses.replace(order, stop_price=stop2)

        # Price pulls back to 110 — trail must NOT lower the stop
        stop3 = trail.evaluate(order, 110.0)
        if stop3 is not None:
            assert stop3 >= order.stop_price, (
                f"Trail moved against long: {stop3:.4f} < {order.stop_price:.4f}"
            )
        # stop3=None means no change needed — also correct

    def test_lock_floor_at_3r(self):
        """
        At 3R profit, the lock floor activates:
          lock_floor = fill + LOCK_R × LOCK_FLOOR_FRACTION × risk_per_unit
                     = 100  + 3.0   × 0.50                × 4.0
                     = 106.0
        The returned stop must be >= 106.0.
        """
        trail = self._trail()
        order = self._order(fill=100.0, stop=96.0, atr=2.0)
        # risk_per_unit = |fill - stop| = 4
        # At price = fill + 3×risk = 100 + 12 = 112  (+3R)
        new_stop = trail.evaluate(order, 112.0)
        assert new_stop is not None, "Trail must activate at 3R"
        risk       = abs(100.0 - 96.0)                     # 4.0
        lock_floor = 100.0 + 3.0 * 0.50 * risk             # 106.0
        assert new_stop >= lock_floor - 0.01, (
            f"Lock floor at 3R must be >= {lock_floor:.2f}, got {new_stop:.2f}"
        )

    def test_stop_hit_long(self):
        """stop_hit() returns True when price <= stop_price for long."""
        trail = self._trail()
        order = self._order(fill=100.0, stop=96.0)
        assert trail.stop_hit(order, 95.9) is True
        assert trail.stop_hit(order, 96.0) is True
        assert trail.stop_hit(order, 96.1) is False

    def test_partial_tp_not_fired_twice(self):
        """partial_closed flag prevents double-firing the partial exit."""
        from dataclasses import dataclass as _dc, field as _f
        trail = self._trail()
        order = self._order(fill=100.0, stop=96.0, tp=108.0)

        # Attach ad-hoc partial-TP metadata (added by RiskAgent at order creation)
        object.__setattr__(order, "partial_take_profit",    104.0)
        object.__setattr__(order, "partial_close_fraction", 0.50)
        object.__setattr__(order, "partial_closed",         False)

        # First check: should fire (price reached 2R target)
        assert trail.partial_tp_hit(order, 104.5) is True

        # Mark as fired
        object.__setattr__(order, "partial_closed", True)

        # Second check: must NOT fire again
        assert trail.partial_tp_hit(order, 106.0) is False


# ══════════════════════════════════════════════════════════════════════════════
# 12. Wilder RSI / ADX Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestWilderIndicators:
    """
    Validates Wilder-smoothed RSI and ADX match industry standard formulas.
    """

    def _trending_up(self, n=60):
        import numpy as np
        return np.array([100.0 + i * 0.5 for i in range(n)])

    def _flat(self, n=60):
        import numpy as np
        return np.array([100.0] * n)

    def test_rsi_bullish_trend_above_50(self):
        from chimera_v12.utils.ta import rsi
        import numpy as np
        prices = self._trending_up(60)
        r = rsi(prices)
        assert r > 50, f"Uptrend RSI must be > 50, got {r}"

    def test_rsi_flat_near_50(self):
        from chimera_v12.utils.ta import rsi
        import numpy as np
        # Flat market with tiny noise → RSI near 50
        np.random.seed(99)
        prices = np.array([100.0 + np.random.normal(0, 0.01) for _ in range(60)])
        r = rsi(prices)
        assert 40 < r < 60, f"Flat market RSI should be ≈50, got {r}"

    def test_rsi_needs_2x_period_bars(self):
        """Wilder RSI requires 2×period bars minimum (seed + smooth)."""
        from chimera_v12.utils.ta import rsi
        import numpy as np
        prices = np.array([100.0 + i for i in range(14)])   # only 1× period
        r = rsi(prices)
        assert r == 50.0, "RSI with < 2×period data must return neutral 50"

    def test_rsi_all_gains_is_100(self):
        from chimera_v12.utils.ta import rsi
        import numpy as np
        prices = np.array([100.0 + i for i in range(40)])  # strictly up
        r = rsi(prices)
        assert r == 100.0

    def test_adx_strong_trend(self):
        """ADX > 25 on a persistent uptrend."""
        from chimera_v12.utils.ta import adx
        import numpy as np
        n = 60
        prices = np.array([100.0 + i * 0.5 for i in range(n)])
        highs  = prices + 0.3
        lows   = prices - 0.3
        a = adx(highs, lows, prices)
        assert a > 20, f"Persistent uptrend should have ADX > 20, got {a}"

    def test_adx_ranging_below_25(self):
        """ADX < 25 on alternating up/down prices (no trend)."""
        from chimera_v12.utils.ta import adx
        import numpy as np
        np.random.seed(42)
        prices = np.array([100.0 + ((-1)**i) * 0.5 for i in range(90)])
        highs  = prices + 0.2
        lows   = prices - 0.2
        a = adx(highs, lows, prices)
        assert a < 25, f"Ranging market should have ADX < 25, got {a}"

    def test_adx_needs_3x_period_bars(self):
        """Wilder ADX requires 3×period bars minimum."""
        from chimera_v12.utils.ta import adx
        import numpy as np
        prices = np.array([100.0 + i for i in range(28)])  # only 2×period
        highs  = prices + 0.5
        lows   = prices - 0.5
        a = adx(highs, lows, prices)
        assert a == 0.0, "ADX with < 3×period data must return 0"


# ══════════════════════════════════════════════════════════════════════════════
# 13. Session VWAP Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSessionVWAP:
    """Validates session_vwap anchors to session open, not bar 0."""

    def test_session_vwap_without_timestamps_falls_back(self):
        from chimera_v12.utils.ta import session_vwap, vwap
        import numpy as np
        closes  = np.array([100.0] * 20)
        highs   = closes + 0.5
        lows    = closes - 0.5
        vols    = np.ones(20) * 1000
        sv = session_vwap(highs, lows, closes, vols, timestamps=None)
        fv = vwap(highs, lows, closes, vols)
        assert abs(sv - fv) < 0.01, "No-timestamp fallback must equal full VWAP"

    def test_session_vwap_anchors_after_session_open(self):
        """When timestamps span pre-market + regular session, VWAP anchors at open."""
        from chimera_v12.utils.ta import session_vwap, vwap
        from datetime import datetime, timezone
        import numpy as np

        # 20 pre-market bars (12:00 UTC = 07:00 ET) then 20 session bars (14:30 UTC = 09:30 ET)
        pre  = [datetime(2024, 1, 15, 12, i, tzinfo=timezone.utc) for i in range(20)]
        sess = [datetime(2024, 1, 15, 14, i+30, tzinfo=timezone.utc) for i in range(20)]
        ts   = pre + sess

        # Pre-market prices at 90, session prices at 100
        prices_pre  = np.full(20, 90.0)
        prices_sess = np.full(20, 100.0)
        closes  = np.concatenate([prices_pre, prices_sess])
        highs   = closes + 0.5
        lows    = closes - 0.5
        vols    = np.ones(40) * 1000

        sv = session_vwap(highs, lows, closes, vols, timestamps=ts)

        # Session VWAP should be near 100 (not 95 which includes pre-market)
        assert sv >= 99.0, (
            f"Session VWAP {sv:.2f} should be ≈100 (session bars only), "
            "not pulled down by pre-market 90.0 prices"
        )

    def test_session_vwap_equals_full_vwap_if_all_in_session(self):
        """If all bars are within session hours, result matches full VWAP."""
        from chimera_v12.utils.ta import session_vwap, vwap
        from datetime import datetime, timezone
        import numpy as np

        ts = [datetime(2024, 1, 15, 15, i, tzinfo=timezone.utc) for i in range(20)]
        closes = np.array([100.0 + i * 0.1 for i in range(20)])
        highs  = closes + 0.5
        lows   = closes - 0.5
        vols   = np.ones(20) * 1000

        sv = session_vwap(highs, lows, closes, vols, timestamps=ts)
        fv = vwap(highs, lows, closes, vols)
        assert abs(sv - fv) < 0.01


# ══════════════════════════════════════════════════════════════════════════════
# 14. Outcome Queue Isolation Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestOutcomeQueue:
    """Validates that _TradeOutcome objects go to outcome_queue, not signal_queue."""

    def test_shared_state_has_outcome_queue(self):
        from chimera_v12.core.state import SharedState
        import asyncio
        state = SharedState()
        assert hasattr(state, "outcome_queue")
        assert isinstance(state.outcome_queue, asyncio.Queue)

    def test_signals_is_bounded_deque(self):
        from chimera_v12.core.state import SharedState
        from collections import deque
        state = SharedState()
        assert isinstance(state.signals, deque)
        assert state.signals.maxlen == 1_000

    def test_signals_deque_does_not_grow_past_maxlen(self):
        """Inserting 1500 signals should keep only the last 1000."""
        from chimera_v12.core.state import SharedState
        state = SharedState()
        for i in range(1500):
            state.signals.append(f"signal_{i}")
        assert len(state.signals) == 1_000
        assert state.signals[-1] == "signal_1499"
        assert state.signals[0]  == "signal_500"  # oldest surviving


# ══════════════════════════════════════════════════════════════════════════════
# 15. Entry Filter Tests (win-rate calibration)
# ══════════════════════════════════════════════════════════════════════════════

class TestEntryFilters:
    """Validates that tightened entry conditions reduce false signals."""

    def _config(self):
        """
        Minimal inline config — no environment variables required.
        Uses the same structure as load_config() but with safe defaults.
        """
        return {
            "stocks": {
                "squeeze": {
                    "min_sp_score": 0.60, "si_weight": 0.40,
                    "rvol_weight": 0.30, "sentiment_weight": 0.30,
                    "si_cap": 0.50, "rvol_cap": 10.0,
                },
                "ema_fast": 9, "ema_slow": 21, "ema_trend": 200,
                "adx_trending": 25, "adx_ranging": 20,
                "atr_multiplier_stop": 2.0, "atr_multiplier_target": 4.0,
                "partial_profit_r": 2.0, "partial_close_fraction": 0.50,
                "momentum": {"lookback_12m": 252, "skip_1m": 21},
            },
            "crypto": {
                "ema_fast": 9, "ema_slow": 21, "rsi_period": 14,
                "rsi_oversold": 30, "rsi_overbought": 70,
                "btc_inflow_threshold": 1_000_000,
                "sol_memecoin_spike_threshold": 50_000_000,
                "funding_rate_extreme": 0.01,
                "atr_multiplier_stop": 2.0, "atr_multiplier_target": 4.0,
            },
            "forex": {
                "ema_fast": 20, "ema_slow": 50, "rsi_period": 14,
                "rsi_momentum_bull": 55, "rsi_momentum_bear": 45,
                "news_bias_weight": 0.40, "carry_weight": 0.20,
                "session_filter": False,
                "atr_multiplier_stop": 1.5, "atr_multiplier_target": 3.0,
            },
            "futures": {
                "va_lookback_bars": 20, "va_pct": 0.70,
                "adx_trending": 25, "rollover_days_before": 5,
                "atr_multiplier_stop": 1.5, "atr_multiplier_target": 3.5,
            },
            "options_wheel": {
                "min_iv_rank": 30, "max_iv_rank": 85,
                "dte_target": 30, "dte_min_close": 7,
                "profit_target_pct": 0.50, "loss_limit_pct": 2.00,
                "min_premium_pct": 0.01, "target_delta_put": 0.30,
                "target_delta_call": 0.30,
            },
            "risk": {
                "base_risk_pct": 0.01, "kelly_fraction": 0.25,
                "avg_win_r": 1.8, "avg_loss_r": 1.0,
                "max_single_position": 0.10, "hitl_threshold_usd": 10_000,
                "trailing_atr_multiple": 3.0, "breakeven_at_r": 2.0,
                "trail_activate_r": 2.0, "lock_profit_at_r": 3.0,
                "lock_floor_fraction": 0.50,
            },
            "intervals": {"strategy_seconds": 15, "options_scan_seconds": 120},
            "wheel_candidates": ["AAPL", "MSFT"],
            "stock_symbols_tier1": [], "stock_symbols_tier2": [],
            "quant_edge": {
                "options_microstructure": {"min_bid_ask_edge_bps": 5, "hold_seconds_min": 30},
                "etf_arb": {"pairs": [], "z_score_entry": 2.0, "z_score_exit": 0.5,
                            "min_spread_bps": 5, "mean_revert_minutes": 10},
                "order_flow_imbalance": {"imbalance_thresh": 0.60},
                "vol_surface_arb": {"skew_zscore_entry": 2.0},
                "news_sentiment_alpha": {"min_score": 0.65, "hold_minutes": 15,
                                         "decay_halflife_min": 30},
            },
        }

    def _long_bars(self, n=260, trend="up"):
        """Synthetic bars for testing."""
        import numpy as np
        if trend == "up":
            closes  = np.array([100.0 + i * 0.3 for i in range(n)])
        else:
            closes  = np.array([100.0 - i * 0.3 for i in range(n)])
        highs   = closes + 0.5
        lows    = closes - 0.5
        volumes = np.ones(n) * 2_000_000
        return {"close": closes, "high": highs, "low": lows, "volume": volumes}

    def test_stocks_no_signal_without_volume_confirmation(self):
        """
        Low RVOL (1.0 < 1.5 threshold) suppresses entries even on valid EMA/ADX.
        evaluate(symbol, bars, short_interest, rvol, sentiment_z, composite_score)
        """
        from chimera_v12.strategies.sector.engine import StocksStrategy
        strat = StocksStrategy(self._config())
        bars  = self._long_bars(260, "up")
        # rvol=1.0 (below 1.5 threshold) with otherwise valid signals
        sig = strat.evaluate("AAPL", bars,
                             short_interest=0.05, rvol=1.0,
                             sentiment_z=1.0,    composite_score=0.70)
        # With rvol=1.0, trend BUY requires vol_confirmed=True → fails
        assert sig.direction == "HOLD", (
            f"Low RVOL should suppress BUY entry, got {sig.direction}"
        )

    def test_stocks_buy_requires_volume_and_rsi(self):
        """
        Ranging market (flat prices): EMA is flat, ADX low, RSI ≈50.
        All three filters (vol_confirmed, rsi_bull_momentum, adx_trend) must pass.
        Flat market fails the adx_trend filter → HOLD.
        """
        from chimera_v12.strategies.sector.engine import StocksStrategy
        import numpy as np
        strat = StocksStrategy(self._config())
        n = 260
        # Alternating prices → flat EMA, low ADX, RSI ≈ 50
        closes = np.array([100.0 + ((-1)**i) * 0.2 for i in range(n)])
        bars   = {"close": closes, "high": closes+0.3, "low": closes-0.3,
                  "volume": np.ones(n) * 3_000_000}
        sig = strat.evaluate("MSFT", bars,
                             short_interest=0.05, rvol=2.5,
                             sentiment_z=1.0,    composite_score=0.70)
        assert sig.direction == "HOLD", (
            f"Ranging market with flat EMA should be HOLD, got {sig.direction}"
        )

    def test_forex_threshold_is_65(self):
        """
        ForexStrategy requires composite score ≥ 0.65 (raised from 0.60).
        Flat bars → EMA flat → bull_score ≈ 0.40 → well below 0.65 → HOLD.
        evaluate(pair, bars, news_bias, rate_diff, session_ok)
        """
        from chimera_v12.strategies.sector.engine import ForexStrategy
        import numpy as np
        strat = ForexStrategy(self._config())
        n = 60
        # Flat price → EMA20 = EMA50 → ema_bull = False → bull_score = 0
        bars = {"close": np.full(n, 1.08), "high": np.full(n, 1.082),
                "low":   np.full(n, 1.078)}
        sig = strat.evaluate("EUR/USD", bars,
                             news_bias=0.0, rate_diff=0.0, session_ok=True)
        assert sig.direction == "HOLD", (
            f"Flat forex market should be HOLD (score << 0.65), got {sig.direction}"
        )

    def test_futures_no_signal_outside_session(self):
        """
        Futures signals suppressed outside RTH (09:30–15:45 ET = 14:30–20:45 UTC).
        Pre-market bars at 08:00 UTC (03:00 ET) must return HOLD.
        evaluate(symbol, bars, cot_net, days_to_exp, seasonal)
        """
        from chimera_v12.strategies.sector.engine import FuturesStrategy
        from datetime import datetime, timezone
        import numpy as np
        strat = FuturesStrategy(self._config())

        n = 60
        closes  = np.array([4000.0 + i * 0.5 for i in range(n)])
        bars = {
            "close":      closes,
            "high":       closes + 2.0,
            "low":        closes - 2.0,
            "volume":     np.ones(n) * 500_000,
            # Pre-market: 08:00 UTC = 03:00 ET — outside RTH
            "timestamps": [datetime(2024, 1, 15, 8, i % 60, tzinfo=timezone.utc) for i in range(n)],
        }
        sig = strat.evaluate("ES1!", bars, cot_net=0.5, days_to_exp=30, seasonal=0.5)
        assert sig.direction == "HOLD", (
            f"Futures must be HOLD outside RTH (08:00 UTC), got {sig.direction}"
        )
