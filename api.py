"""
chimera/server/runner.py
Starts the FastAPI/uvicorn server in a daemon thread so it runs alongside
the asyncio trading agents without blocking or competing with them.

Why a thread and not asyncio?
──────────────────────────────
uvicorn runs its own asyncio event loop internally. Running it inside the
mainframe's event loop via `asyncio.create_task` would cause two event loops
to fight over the same thread. The correct pattern is:

    Thread 1 (main):  asyncio.run(mainframe.run())
                        └─ DataAgent, NewsAgent, StrategyAgent,
                           RiskAgent, OrderManager, StatePublisher
    Thread 2 (daemon): uvicorn.run(app, ...)
                        └─ FastAPI HTTP + WebSocket server

The StatePublisher.queue is an asyncio.Queue created in Thread 1's loop.
FastAPI's broadcaster coroutine runs in Thread 2's loop and cannot await
Thread 1's queue directly — we bridge them with a thread-safe
asyncio.Queue clone using asyncio.run_coroutine_threadsafe.

Actually, the cleaner approach (used here) is to run uvicorn with the
`loop="none"` option and let it share Thread 1's event loop via
`server.serve()` as a coroutine. This is the recommended pattern when
embedding uvicorn inside an existing asyncio application.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import uvicorn

from chimera_v12.server.api import build_app
from chimera_v12.server.publisher import StatePublisher

if TYPE_CHECKING:
    from chimera_v12.utils.state import SharedState
    from chimera_v12.oms.trade_logger import TradeLogger

log = logging.getLogger("chimera.server.runner")


class APIServer:
    """
    Wraps the FastAPI app and uvicorn server.
    Call `.run()` as an asyncio task from the mainframe.

    Usage in mainframe.py:
        self.api_server = APIServer(self.state, trade_logger, config)
        tasks.append(asyncio.create_task(self.api_server.run(), name="APIServer"))
    """

    def __init__(
        self,
        state:        "SharedState",
        trade_logger: "TradeLogger",
        config:       dict[str, Any],
    ):
        self.state        = state
        self.trade_logger = trade_logger
        self.config       = config
        self.host         = config.get("api_host", "0.0.0.0")
        self.port         = config.get("api_port", 8765)

        self.publisher = StatePublisher(state)
        self.app       = build_app(self.publisher, trade_logger)

    async def run(self) -> None:
        """
        Runs the StatePublisher and uvicorn server as concurrent tasks
        within the mainframe's asyncio event loop.
        """
        uv_config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self.port,
            log_level="warning",    # keep uvicorn quiet; we have our own logger
            access_log=False,
            loop="none",            # critical: reuse the existing event loop
        )
        server = uvicorn.Server(uv_config)

        log.info(f"API server starting on http://{self.host}:{self.port}")
        log.info(f"  WebSocket : ws://{self.host}:{self.port}/ws/state")
        log.info(f"  REST docs : http://{self.host}:{self.port}/docs")

        await asyncio.gather(
            self.publisher.run(),
            server.serve(),
        )
