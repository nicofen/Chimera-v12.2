"""
chimera_v12/core/state.py
═══════════════════════════════════════════════════════════════════════════════
UNIFIED GLOBAL STATE  —  The Market Blackboard

All agents (Chimera + TradingAgents) read/write to this single object.
No agent ever calls another agent directly — all communication flows through
the shared state.  This enforces clean isolation and testability.

Architecture: Two layers
  1. ChimeraState (TypedDict)  — immutable snapshot passed through LangGraph
  2. SharedState (asyncio-safe mutable) — live data updated by DataAgent / feeds

LangGraph nodes receive ChimeraState dicts; async agents mutate SharedState.
The Orchestrator bridges them: it snapshots SharedState → ChimeraState at
each decision cycle and writes results back.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any
from typing_extensions import TypedDict

# ══════════════════════════════════════════════════════════════════════════════
# Enumerations
# ══════════════════════════════════════════════════════════════════════════════

class MarketRegime(str, Enum):
    BULL_TREND   = "bull_trend"
    BEAR_TREND   = "bear_trend"
    HIGH_VOL     = "high_volatility"
    LOW_VOL      = "low_volatility"
    RANGING      = "ranging"
    CRISIS       = "crisis"
    UNKNOWN      = "unknown"

class SignalDirection(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"

class AssetClass(str, Enum):
    STOCK   = "stock"
    CRYPTO  = "crypto"
    FOREX   = "forex"
    FUTURES = "futures"
    OPTIONS = "options"

class TradeDecision(str, Enum):
    BUY    = "BUY"
    SELL   = "SELL"
    HOLD   = "HOLD"
    CLOSE  = "CLOSE"
    WHEEL_SELL_PUT  = "WHEEL_SELL_PUT"
    WHEEL_SELL_CALL = "WHEEL_SELL_CALL"
    WHEEL_ASSIGNED  = "WHEEL_ASSIGNED"

# ══════════════════════════════════════════════════════════════════════════════
# LangGraph-compatible TypedDict state (immutable snapshots for agent nodes)
# ══════════════════════════════════════════════════════════════════════════════

class InvestDebateState(TypedDict):
    bull_history:     Annotated[str, "Bullish conversation history"]
    bear_history:     Annotated[str, "Bearish conversation history"]
    history:          Annotated[str, "Full debate history"]
    current_response: Annotated[str, "Latest response"]
    judge_decision:   Annotated[str, "Final judge decision"]
    count:            Annotated[int, "Round count"]

class RiskDebateState(TypedDict):
    aggressive_history:         Annotated[str, "Aggressive analyst history"]
    conservative_history:       Annotated[str, "Conservative analyst history"]
    neutral_history:            Annotated[str, "Neutral analyst history"]
    history:                    Annotated[str, "Full debate history"]
    latest_speaker:             Annotated[str, "Last speaker"]
    current_aggressive_response:  Annotated[str, "Last aggressive response"]
    current_conservative_response:Annotated[str, "Last conservative response"]
    current_neutral_response:   Annotated[str, "Last neutral response"]
    judge_decision:             Annotated[str, "Risk judge decision"]
    count:                      Annotated[int, "Round count"]

class WheelState(TypedDict):
    """Options Wheel position tracking."""
    phase:           Annotated[str, "cash_secured_put | covered_call | assigned"]
    symbol:          Annotated[str, "Underlying ticker"]
    strike:          Annotated[float, "Strike price"]
    expiry:          Annotated[str, "YYYY-MM-DD expiry"]
    premium_collected: Annotated[float, "Total premium collected this cycle"]
    cost_basis:      Annotated[float, "Adjusted cost basis if assigned"]
    shares_held:     Annotated[int, "Shares held (0 if in CSP phase)"]
    open_date:       Annotated[str, "When position was opened"]

class ChimeraState(TypedDict):
    """
    The canonical LangGraph state snapshot.
    Every field uses Annotated[T, description] for graph introspection.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    symbol:      Annotated[str, "Primary ticker under analysis"]
    asset_class: Annotated[str, "Asset class (stock/crypto/forex/futures/options)"]
    trade_date:  Annotated[str, "ISO date of this decision cycle"]
    cycle_id:    Annotated[str, "Unique ID for this decision cycle"]

    # ── Market Data ───────────────────────────────────────────────────────────
    market_regime:    Annotated[str, "Current market regime"]
    vix:              Annotated[float, "VIX or proxy volatility level"]
    sector_etf_data:  Annotated[dict, "Sector ETF prices and relative strength"]

    # ── Research Reports (from TradingAgents analysts) ────────────────────────
    market_report:       Annotated[str, "Technical market analysis"]
    sentiment_report:    Annotated[str, "Social media sentiment analysis"]
    news_report:         Annotated[str, "News and macro analysis"]
    fundamentals_report: Annotated[str, "Fundamental analysis report"]

    # ── Chimera Technical Signals ─────────────────────────────────────────────
    squeeze_probability: Annotated[float, "Sp score 0-1 for short squeeze"]
    rsi:                 Annotated[float, "14-period RSI"]
    adx:                 Annotated[float, "ADX trend strength"]
    atr:                 Annotated[float, "ATR for position sizing"]
    ema_signal:          Annotated[str, "EMA crossover signal"]
    vol_surface_skew:    Annotated[float, "IV skew put/call ratio"]

    # ── Quant Edge Signals ────────────────────────────────────────────────────
    order_flow_imbalance:  Annotated[float, "Book pressure -1 to +1"]
    etf_arb_spread:        Annotated[float, "ETF vs futures spread bps"]
    vol_arb_signal:        Annotated[float, "Volatility surface dislocation score"]
    news_sentiment_alpha:  Annotated[float, "LLM-parsed news alpha score"]

    # ── Scoring (long-term value factors) ────────────────────────────────────
    quality_score:   Annotated[float, "Piotroski F-Score 0-9"]
    value_score:     Annotated[float, "Composite value score (P/E, P/B, EV/EBITDA)"]
    momentum_score:  Annotated[float, "12-1 month price momentum"]
    growth_score:    Annotated[float, "Revenue/earnings growth score"]
    composite_score: Annotated[float, "Weighted composite final score 0-1"]

    # ── Debate States ─────────────────────────────────────────────────────────
    investment_debate_state: Annotated[InvestDebateState, "Bull/Bear debate state"]
    investment_plan:         Annotated[str, "Research team investment plan"]
    trader_investment_plan:  Annotated[str, "Trader's refined plan"]
    risk_debate_state:       Annotated[RiskDebateState, "Risk team debate state"]

    # ── Final Decision ────────────────────────────────────────────────────────
    final_trade_decision:  Annotated[str, "BUY/SELL/HOLD/CLOSE"]
    decision_confidence:   Annotated[float, "0-1 confidence in decision"]
    position_size_usd:     Annotated[float, "Dollar position size from Kelly"]
    stop_loss:             Annotated[float, "Stop loss price"]
    take_profit:           Annotated[float, "Take profit price"]

    # ── Options Wheel ─────────────────────────────────────────────────────────
    wheel_state: Annotated[WheelState | None, "Active wheel position if any"]

    # ── HITL / Audit ─────────────────────────────────────────────────────────
    hitl_required:     Annotated[bool, "True if human approval required"]
    hitl_threshold:    Annotated[float, "USD threshold requiring HITL"]
    audit_trail:       Annotated[list[dict], "Full agent decision log"]
    agent_votes:       Annotated[dict[str, str], "Per-agent vote registry"]

    # ── Memory & Context ──────────────────────────────────────────────────────
    past_context:      Annotated[str, "Memory log: prior decisions for this ticker"]
    news_veto_active:  Annotated[bool, "True if NewsAgent has issued macro veto"]
    circuit_open:      Annotated[bool, "True if circuit breaker has tripped"]

# ══════════════════════════════════════════════════════════════════════════════
# Live Mutable Shared State (asyncio-safe, used by real-time agents)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TechnicalSignals:
    symbol:    str
    direction: str = "HOLD"
    rsi:       float = 50.0
    adx:       float = 0.0
    atr:       float = 0.0
    squeeze_prob: float = 0.0
    vol_skew:  float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class RiskParameters:
    symbol:        str
    direction:     str
    size_usd:      float
    stop_loss:     float
    take_profit:   float
    asset_class:   str = "stock"
    source_agent:  str = "unknown"
    timestamp:     datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    audit_id:      str = ""

@dataclass
class MarketDataBucket:
    stocks:  dict[str, dict] = field(default_factory=dict)
    crypto:  dict[str, dict] = field(default_factory=dict)
    forex:   dict[str, dict] = field(default_factory=dict)
    futures: dict[str, dict] = field(default_factory=dict)
    options: dict[str, dict] = field(default_factory=dict)

@dataclass
class WheelPosition:
    """Live options wheel position."""
    symbol:     str
    phase:      str   # "cash_secured_put" | "covered_call" | "assigned"
    strike:     float
    expiry:     str
    premium:    float
    cost_basis: float = 0.0
    shares:     int   = 0
    opened_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class SharedState:
    """
    Thread-safe, asyncio-compatible live state store.
    Instantiated once by the Mainframe and passed to every agent.

    Key design decisions:
    - asyncio.Queue for signal passing (non-blocking)
    - asyncio.Lock guards all mutable dicts
    - .snapshot() produces a frozen ChimeraState dict for LangGraph
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

        # Live market data
        self.market = MarketDataBucket()

        # Signal pipeline (DataAgent → StrategyAgent → RiskAgent → OMS)
        # Signal pipeline queues
        self.signal_queue: asyncio.Queue[TechnicalSignals]   = asyncio.Queue(maxsize=512)
        self.order_queue:  asyncio.Queue[RiskParameters]     = asyncio.Queue(maxsize=128)
        # Dedicated outcome queue: OMS → RiskAgent + CircuitBreaker
        # Keeps _TradeOutcome objects OUT of signal_queue so RiskAgent
        # never accidentally processes one as a TechnicalSignals object.
        self.outcome_queue: asyncio.Queue = asyncio.Queue(maxsize=256)

        # Portfolio state
        self.equity:         float = 100_000.0
        self.high_water:     float = 100_000.0
        self.open_positions: dict[str, Any] = {}
        self.daily_pnl:      float = 0.0
        self.loss_streak:    int   = 0

        # Regime & risk flags
        self.regime:         MarketRegime = MarketRegime.UNKNOWN
        self.vix:            float = 18.0
        self.news_veto:      bool  = False
        self.veto_until:     datetime | None = None
        self.circuit_open:   bool  = False
        self.breaker_events: list[dict] = []

        # Agent reports (populated by TA analyst agents)
        self.reports: dict[str, dict[str, str]] = {}   # symbol → {market, sentiment, news, fundamentals}

        # Options wheel positions
        self.wheel_positions: dict[str, WheelPosition] = {}

        # Quant edge signals (updated by QuantEdgeAgent)
        self.quant_signals: dict[str, dict] = {}

        # Audit trail
        self.audit_log: list[dict] = []

        # Agent votes registry
        self.agent_votes: dict[str, dict[str, str]] = {}  # cycle_id → {agent: vote}

    async def log_audit(self, agent: str, symbol: str, action: str, detail: dict) -> None:
        async with self._lock:
            self.audit_log.append({
                "ts": datetime.now(timezone.utc).isoformat(),
                "agent":  agent,
                "symbol": symbol,
                "action": action
                **detail
            })

    async def record_vote(self, cycle_id: str, agent: str, vote: str) -> None:
        async with self._lock:
            if cycle_id not in self.agent_votes:
                self.agent_votes[cycle_id] = {}
            self.agent_votes[cycle_id][agent] = vote

    def snapshot(self, symbol: str) -> dict:
        """Return a frozen dict compatible with ChimeraState for LangGraph."""
        reports = self.reports.get(symbol, {})
        qsig    = self.quant_signals.get(symbol, {})
        return {
            "symbol":      symbol,
            "asset_class": "stock",
            "trade_date":  datetime.now(timezone.utc).date().isoformat(),
            "cycle_id":    f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
            "market_regime":    self.regime.value,
            "vix":              self.vix,
            "sector_etf_data":  {},
            "market_report":       reports.get("market",       ""),
            "sentiment_report":    reports.get("sentiment",    ""),
            "news_report":         reports.get("news",         ""),
            "fundamentals_report": reports.get("fundamentals", ""),
            "squeeze_probability":  0.0,
            "rsi":  0.0, "adx": 0.0, "atr": 0.0,
            "ema_signal": "HOLD",
            "vol_surface_skew": 0.0,
            "order_flow_imbalance": qsig.get("order_flow_imbalance", 0.0),
            "etf_arb_spread":       qsig.get("etf_arb_spread", 0.0),
            "vol_arb_signal":       qsig.get("vol_arb_signal", 0.0),
            "news_sentiment_alpha": qsig.get("news_sentiment_alpha", 0.0),
            "quality_score":   0.0, "value_score":  0.0,
            "momentum_score":  0.0, "growth_score": 0.0,
            "composite_score": 0.0,
            "investment_debate_state": {
                "bull_history": "", "bear_history": "", "history": "",
                "current_response": "", "judge_decision": "", "count": 0
            },
            "investment_plan":       "",
            "trader_investment_plan": "",
            "risk_debate_state": {
                "aggressive_history": "", "conservative_history": "",
                "neutral_history": "", "history": "",
                "latest_speaker": "", "current_aggressive_response": "",
                "current_conservative_response": "", "current_neutral_response": "",
                "judge_decision": "", "count": 0
            },
            "final_trade_decision": "HOLD",
            "decision_confidence":  0.0,
            "position_size_usd":    0.0,
            "stop_loss":    0.0,
            "take_profit":  0.0,
            "wheel_state":  None,
            "hitl_required": False,
            "hitl_threshold": 10_000.0,
            "audit_trail":  self.audit_log[-50:],
            "agent_votes":  {},
            "past_context": "",
            "news_veto_active": self.news_veto,
            "circuit_open":     self.circuit_open
        }

# ══════════════════════════════════════════════════════════════════════════════
# Patch SharedState with all v11-compatible attributes and methods
# ══════════════════════════════════════════════════════════════════════════════

class Sentiment(str, Enum):
    """Unified Sentiment enum — used by NewsState and all v11 agents."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

def _patch_shared_state():
    """
    Inject v11-compatible attributes and methods into SharedState so all
    ported v11 agents and tests work without modification.
    """
    import queue as _queue
    from enum import Enum as _E

    class _NewsState:
        def __init__(self):
            self.sentiment   = Sentiment.NEUTRAL
            self.confidence  = 0.0
            self.veto_active = False
            self.veto_reason = ""
            self.last_updated = None

    # Patch __init__ to add v11 fields
    _orig_init = SharedState.__init__

    def _new_init(self):
        _orig_init(self)
        # v11 attributes
        self.news        = _NewsState()
        # Bounded deque — prevents O(n) scan growing forever during live sessions
        from collections import deque as _dq
        self.signals:     _dq  = _dq(maxlen=1_000)
        self.risk_params: list = []
        self.breaker     = None
        # BacktestState sync queues (used when asyncio loop not running)
        self._sync_signal_queue = _queue.SimpleQueue()
        self._sync_order_queue  = _queue.SimpleQueue()

    SharedState.__init__ = _new_init

    # v11 methods
    async def put_signal(self, signal) -> None:
        async with self._lock:
            self.signals.append(signal)
        await self.signal_queue.put(signal)

    async def put_order(self, risk_params) -> None:
        async with self._lock:
            self.risk_params.append(risk_params)
        await self.order_queue.put(risk_params)

    def is_vetoed(self) -> bool:
        return self.news.veto_active

    def news_multiplier(self) -> float:
        if self.news.veto_active:
            return 0.0
        base = self.news.confidence
        if self.news.sentiment == Sentiment.BULLISH:
            return min(1.0, base * 1.1)
        if self.news.sentiment == Sentiment.BEARISH:
            return base * 0.5
        return base

    SharedState.put_signal      = put_signal
    SharedState.put_order       = put_order
    SharedState.is_vetoed       = is_vetoed
    SharedState.news_multiplier = news_multiplier

    # Expose Sentiment class as class attribute for v11 imports
    return _NewsState

_NS = _patch_shared_state()
