"""
chimera_v12/agents/bridges/chimera_bridge.py
═══════════════════════════════════════════════════════════════════════════════
CHIMERA v11 → v12 BRIDGE ADAPTERS

Wraps every v11 agent class to use the v12 SharedState.
This allows a clean migration path — v11 agents run unchanged inside v12.

Pattern: Each Bridge class:
  1. Imports the original v11 class
  2. Translates v12 SharedState to the v11 format expected
  3. Writes results back into v12 SharedState format

Eliminates the need to rewrite all v11 agents immediately — they can be
migrated to native v12 format incrementally.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from chimera_v12.core.state import SharedState
from chimera_v12.utils.logger import setup_logger

log = setup_logger("bridges.chimera")

class _BaseBridge:
    """Common base for all v11→v12 bridge adapters."""

    def __init__(self, state: SharedState, config: dict[str, Any]):
        self.state  = state
        self.config = config
        self._agent = None   # set by subclass

    async def run(self) -> None:
        if self._agent and hasattr(self._agent, "run"):
            await self._agent.run()

# ── Data Agent Bridge ──────────────────────────────────────────────────────────

class DataAgentBridge(_BaseBridge):
    """
    Wraps chimera.agents.data_agent.DataAgent.
    Translates v12 SharedState.market into the v11 format.
    """
    def __init__(self, state: SharedState, config: dict):
        super().__init__(state, config)
        try:
            from chimera.agents.data_agent import DataAgent
            from chimera.utils.state import SharedState as V11State

            # Create a v11 state shim that reads/writes to our v12 state
            v11_state       = _V11StateShim(state)
            self._agent     = DataAgent(v11_state, config)
            log.info("DataAgentBridge: loaded v11 DataAgent")
        except ImportError:
            log.warning("DataAgentBridge: chimera v11 not installed, using stub")
            self._agent = _StubAgent("DataAgent")

class NewsAgentBridge(_BaseBridge):
    def __init__(self, state: SharedState, config: dict):
        super().__init__(state, config)
        try:
            from chimera.agents.news_agent import NewsAgent
            v11_state   = _V11StateShim(state)
            self._agent = NewsAgent(v11_state, config)
            log.info("NewsAgentBridge: loaded v11 NewsAgent")
        except ImportError:
            self._agent = _StubAgent("NewsAgent")

class StrategyAgentBridge(_BaseBridge):
    def __init__(self, state: SharedState, config: dict):
        super().__init__(state, config)
        try:
            from chimera.agents.strategy_agent import StrategyAgent
            v11_state   = _V11StateShim(state)
            self._agent = StrategyAgent(v11_state, config)
            log.info("StrategyAgentBridge: loaded v11 StrategyAgent")
        except ImportError:
            self._agent = _StubAgent("StrategyAgent")

class RiskAgentBridge(_BaseBridge):
    def __init__(self, state: SharedState, config: dict):
        super().__init__(state, config)
        try:
            from chimera.agents.risk_agent import RiskAgent
            v11_state   = _V11StateShim(state)
            self._agent = RiskAgent(v11_state, config)
            log.info("RiskAgentBridge: loaded v11 RiskAgent")
        except ImportError:
            self._agent = _StubAgent("RiskAgent")

class OrderManagerBridge(_BaseBridge):
    def __init__(self, state: SharedState, config: dict):
        super().__init__(state, config)
        try:
            from chimera.oms.order_manager import OrderManager
            v11_state   = _V11StateShim(state)
            self._agent = OrderManager(v11_state, config)
            log.info("OrderManagerBridge: loaded v11 OrderManager")
        except ImportError:
            self._agent = _StubAgent("OrderManager")

class SocialScraperBridge(_BaseBridge):
    def __init__(self, state: SharedState, config: dict):
        super().__init__(state, config)
        try:
            from chimera.social.scraper import StocktwitsScraper
            v11_state   = _V11StateShim(state)
            self._agent = StocktwitsScraper(v11_state, config)
        except ImportError:
            self._agent = _StubAgent("SocialScraper")

class CircuitBreakerBridge(_BaseBridge):
    def __init__(self, state: SharedState, config: dict):
        super().__init__(state, config)
        try:
            from chimera.risk.circuit_breaker import CircuitBreaker
            from chimera.oms.order_manager import OrderManager
            v11_state   = _V11StateShim(state)
            v11_oms     = OrderManager(v11_state, config)
            self._agent = CircuitBreaker(v11_state, v11_oms, config)
        except ImportError:
            self._agent = _StubAgent("CircuitBreaker")

class AlertDispatcherBridge(_BaseBridge):
    def __init__(self, state: SharedState, config: dict):
        super().__init__(state, config)
        try:
            from chimera.alerts.dispatcher import build_dispatcher
            v11_state   = _V11StateShim(state)
            self._agent = build_dispatcher(v11_state, config)
        except ImportError:
            self._agent = _StubAgent("AlertDispatcher")

class RegimeClassifierBridge(_BaseBridge):
    def __init__(self, state: SharedState, config: dict):
        super().__init__(state, config)
        try:
            from chimera.regime.classifier import RegimeClassifier
            v11_state   = _V11StateShim(state)
            self._agent = RegimeClassifier(v11_state, config)
        except ImportError:
            self._agent = _StubAgent("RegimeClassifier")

class APIServerBridge(_BaseBridge):
    def __init__(self, state: SharedState, config: dict):
        super().__init__(state, config)
        try:
            from chimera.server.runner import APIServer
            from chimera.oms.trade_logger import TradeLogger
            v11_state   = _V11StateShim(state)
            logger      = TradeLogger(config.get("db_path", "chimera_v12.db"))
            self._agent = APIServer(v11_state, logger, config)
        except ImportError:
            self._agent = _StubAgent("APIServer")

# ══════════════════════════════════════════════════════════════════════════════
# V11 State Shim
# ══════════════════════════════════════════════════════════════════════════════

class _V11StateShim:
    """
    Presents the v12 SharedState with the interface expected by v11 agents.
    Properties proxy through to the underlying v12 state object.

    This shim is how we get zero code changes in v11 agents while running
    them inside the v12 architecture.
    """

    def __init__(self, v12_state: SharedState):
        self._s = v12_state

    # ── Market data (v11 used state.market.crypto etc.) ─────────────────────
    @property
    def market(self):
        return self._s.market

    # ── Signal queues ─────────────────────────────────────────────────────────
    @property
    def signal_queue(self):
        return self._s.signal_queue

    @property
    def order_queue(self):
        return self._s.order_queue

    # ── Portfolio ─────────────────────────────────────────────────────────────
    @property
    def equity(self):
        return self._s.equity

    @equity.setter
    def equity(self, val):
        self._s.equity = val

    @property
    def high_water(self):
        return self._s.high_water

    @high_water.setter
    def high_water(self, val):
        self._s.high_water = val

    @property
    def open_positions(self):
        return self._s.open_positions

    @property
    def daily_pnl(self):
        return self._s.daily_pnl

    @daily_pnl.setter
    def daily_pnl(self, val):
        self._s.daily_pnl = val

    @property
    def loss_streak(self):
        return self._s.loss_streak

    @loss_streak.setter
    def loss_streak(self, val):
        self._s.loss_streak = val

    # ── Risk flags ────────────────────────────────────────────────────────────
    @property
    def news_veto(self):
        return self._s.news_veto

    @news_veto.setter
    def news_veto(self, val):
        self._s.news_veto = val

    @property
    def circuit_open(self):
        return self._s.circuit_open

    @circuit_open.setter
    def circuit_open(self, val):
        self._s.circuit_open = val

    @property
    def breaker_events(self):
        return self._s.breaker_events

    @property
    def veto_until(self):
        return self._s.veto_until

    @veto_until.setter
    def veto_until(self, val):
        self._s.veto_until = val

    @property
    def regime(self):
        return self._s.regime

    @regime.setter
    def regime(self, val):
        self._s.regime = val

    @property
    def vix(self):
        return self._s.vix

    @vix.setter
    def vix(self, val):
        self._s.vix = val

# ══════════════════════════════════════════════════════════════════════════════
# Stub Agent (no-op, used when v11 not installed)
# ══════════════════════════════════════════════════════════════════════════════

class _StubAgent:
    """No-op agent stub for graceful degradation."""

    def __init__(self, name: str):
        self.name = name
        log.debug(f"StubAgent created for {name}")

    async def run(self) -> None:
        log.info(f"[{self.name}] Running as stub (v11 not installed). ",
                 "Install chimera_v11 deps to enable.")
        while True:
            await asyncio.sleep(3600)  # sleep indefinitely, do nothing

# ══════════════════════════════════════════════════════════════════════════════
# TradingAgents LangGraph Bridge
# ══════════════════════════════════════════════════════════════════════════════

class TradingAgentsBridge:
    """
    Wraps the TradingAgents LangGraph graph so it can be called synchronously
    from the MasterOrchestrator pipeline.

    Usage:
        bridge = TradingAgentsBridge(llm, config)
        state  = bridge.run_analyst_panel(chimera_state_snapshot)
    """

    def __init__(self, llm: Any, config: dict):
        self.llm    = llm
        self.config = config
        self._analysts_ready = False
        self._graph = None

        try:
            self._build_graph()
        except ImportError:
            log.warning("TradingAgentsBridge: langgraph/tradingagents not installed. ",
                        "LLM debate panel will be skipped.")

    def _build_graph(self) -> None:
        from tradingagents.agents.analysts.market_analyst import create_market_analyst
        from tradingagents.agents.analysts.fundamentals_analyst import create_fundamentals_analyst
        from tradingagents.agents.analysts.news_analyst import create_news_analyst
        from tradingagents.agents.analysts.social_media_analyst import create_social_media_analyst
        from tradingagents.agents.researchers.bull_researcher import create_bull_researcher
        from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
        from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
        from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
        from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
        from tradingagents.agents.trader.trader import create_trader
        from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager

        self._analysts = {
            "market":       create_market_analyst(self.llm),
            "fundamentals": create_fundamentals_analyst(self.llm),
            "news":         create_news_analyst(self.llm),
            "social":       create_social_media_analyst(self.llm)
        }
        self._bull       = create_bull_researcher(self.llm)
        self._bear       = create_bear_researcher(self.llm)
        self._trader     = create_trader(self.llm)
        self._aggressive = create_aggressive_debator(self.llm)
        self._conservative = create_conservative_debator(self.llm)
        self._neutral    = create_neutral_debator(self.llm)
        self._pm         = create_portfolio_manager(self.llm)
        self._analysts_ready = True
        log.info("TradingAgentsBridge: LangGraph analyst panel loaded.")

    def run_analyst_panel(self, state: dict) -> dict:
        """
        Run all four analysts sequentially and populate research reports.
        Called in a thread executor from the async orchestrator.
        """
        if not self._analysts_ready:
            return state

        try:
            state = self._analysts["market"](state)
            state = self._analysts["fundamentals"](state)
            state = self._analysts["news"](state)
            state = self._analysts["social"](state)
        except Exception as e:
            log.warning(f"Analyst panel partial failure: {e}")
        return state

    def run_debate_pipeline(self, state: dict, rounds: int = 3) -> dict:
        """
        Run Bull/Bear debate → Trader synthesis → Risk team debate.
        """
        if not self._analysts_ready:
            return state

        # Bull/Bear debate
        for _ in range(rounds):
            try:
                state = self._bull(state)
                state = self._bear(state)
            except Exception as e:
                log.warning(f"Debate round error: {e}")
                break

        # Trader synthesis
        try:
            state = self._trader(state)
        except Exception as e:
            log.warning(f"Trader synthesis error: {e}")

        # Risk team debate
        for _ in range(rounds):
            try:
                state = self._aggressive(state)
                state = self._conservative(state)
                state = self._neutral(state)
            except Exception as e:
                log.warning(f"Risk debate round error: {e}")
                break

        # Portfolio manager final
        try:
            state = self._pm(state)
        except Exception as e:
            log.warning(f"PM final error: {e}")

        return state
