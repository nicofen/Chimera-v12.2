"""
chimera/server/api.py
FastAPI application — WebSocket state stream + REST trade history endpoints.

Endpoints
─────────
WS  /ws/state          Live SharedState diffs at 4 Hz (250 ms cadence)
GET /api/summary        Today's P&L summary from SQLite
GET /api/trades         Last N closed trades
GET /api/positions      Current open positions (REST snapshot)
GET /api/health         Liveness probe

Running inside the mainframe
─────────────────────────────
The server is started with uvicorn in a background thread so it doesn't
block the asyncio event loop that runs the trading agents.
`chimera.server.runner.start_server()` handles this.

CORS is permissive by default (allow all origins) so the React dashboard
can connect from any dev server or file:// URL during development.
Tighten `allow_origins` in production.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, TYPE_CHECKING

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

if TYPE_CHECKING:
    from chimera_v12.server.publisher import StatePublisher
    from chimera_v12.oms.trade_logger import TradeLogger

log = logging.getLogger("chimera.server.api")


def build_app(
    publisher: "StatePublisher",
    trade_logger: "TradeLogger",
) -> FastAPI:
    """
    Factory function — call once from the mainframe with the live publisher
    and trade_logger instances. Returns a configured FastAPI app.
    """
    app = FastAPI(
        title="Project Chimera API",
        version="0.1.0",
        docs_url="/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],         # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Connection manager ──────────────────────────────────────────────────

    class ConnectionManager:
        def __init__(self):
            self._clients: set[WebSocket] = set()

        async def connect(self, ws: WebSocket) -> None:
            await ws.accept()
            self._clients.add(ws)
            log.info(f"WS client connected — total: {len(self._clients)}")

        def disconnect(self, ws: WebSocket) -> None:
            self._clients.discard(ws)
            log.info(f"WS client disconnected — total: {len(self._clients)}")

        async def broadcast(self, msg: str) -> None:
            dead: set[WebSocket] = set()
            for ws in self._clients:
                try:
                    await ws.send_text(msg)
                except Exception:
                    dead.add(ws)
            self._clients -= dead

        @property
        def count(self) -> int:
            return len(self._clients)

    manager = ConnectionManager()

    # ── Background broadcast task ───────────────────────────────────────────

    async def _broadcaster() -> None:
        """
        Drains the publisher queue and fans each message out to all
        connected WebSocket clients. Runs as a FastAPI lifespan task.
        """
        while True:
            try:
                msg = await asyncio.wait_for(publisher.queue.get(), timeout=1.0)
                if manager.count > 0:
                    await manager.broadcast(msg)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                log.warning(f"Broadcaster error: {e}")

    @app.on_event("startup")
    async def _startup() -> None:
        asyncio.create_task(_broadcaster(), name="WSBroadcaster")
        log.info("Chimera API server started.")

    # ── WebSocket endpoint ──────────────────────────────────────────────────

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket) -> None:
        await manager.connect(ws)
        try:
            # Send full snapshot immediately on connect
            snapshot = publisher.get_snapshot_json()
            await ws.send_text(snapshot)

            # Keep alive — accept pings, ignore other incoming messages
            while True:
                try:
                    data = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                    if data == "ping":
                        await ws.send_text('{"type":"pong"}')
                except asyncio.TimeoutError:
                    # Send keepalive ping
                    await ws.send_text('{"type":"ping"}')
        except WebSocketDisconnect:
            manager.disconnect(ws)
        except Exception as e:
            log.warning(f"WS error: {e}")
            manager.disconnect(ws)

    # ── REST endpoints ──────────────────────────────────────────────────────

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status":   "ok",
            "ws_clients": manager.count,
            "mode":     publisher.state.equity > 0 and "live" or "initialising",
        }

    @app.get("/api/summary")
    async def summary(date: str | None = None) -> dict[str, Any]:
        """Today's P&L summary — delegates to TradeLogger.daily_pnl_summary()."""
        try:
            return await asyncio.to_thread(trade_logger.daily_pnl_summary, date)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/trades")
    async def trades(limit: int = 50) -> list[dict]:
        """Last N closed trades from SQLite."""
        limit = min(max(limit, 1), 500)
        try:
            return await asyncio.to_thread(trade_logger.get_closed_trades, limit)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/positions")
    async def positions() -> list[dict]:
        """Current open positions — REST snapshot (same data as WS)."""
        snap = json.loads(publisher.get_snapshot_json())
        return snap.get("positions", [])

    @app.get("/api/signals")
    async def signals(limit: int = 20) -> list[dict]:
        """Recent strategy signals."""
        snap = json.loads(publisher.get_snapshot_json())
        return snap.get("signals", [])[:limit]


    @app.get("/api/breaker")
    async def breaker_status() -> dict:
        """Current circuit breaker status."""
        b = publisher.state.breaker
        if b is None:
            return {"state": "uninitialised"}
        return {
            "state":               b.state.value,
            "trip_reason":         b.trip_reason.value if b.trip_reason else None,
            "tripped_at":          b.tripped_at.isoformat() if b.tripped_at else None,
            "daily_loss_usd":      round(b.daily_loss_usd, 2),
            "daily_loss_limit_pct":b.daily_loss_limit * 100,
            "drawdown_pct":        round(b.drawdown_pct * 100, 3),
            "drawdown_limit_pct":  b.drawdown_limit_pct * 100,
            "consecutive_losses":  b.consecutive_losses,
            "loss_streak_limit":   b.loss_streak_limit,
            "trip_count_today":    b.trip_count_today,
            "allows_trading":      b.allows_trading,
        }

    @app.post("/api/breaker/reset")
    async def breaker_reset(note: str = "operator reset via API") -> dict:
        """Manually reset the circuit breaker after investigation."""
        # The CircuitBreaker instance is not directly accessible here;
        # we signal a reset by setting a flag in state that the breaker polls.
        publisher.state.circuit_open = False
        if publisher.state.breaker:
            publisher.state.breaker.state = __import__("chimera.risk.circuit_breaker_models", fromlist=["BreakerState"]).BreakerState.CLOSED
            publisher.state.breaker.trip_reason = None
        log.info(f"Circuit breaker reset via API. Note: {note}")
        return {"status": "reset", "note": note}

    @app.post("/api/breaker/trip")
    async def breaker_trip_manual(reason: str = "operator") -> dict:
        """Manually trip the circuit breaker (emergency flatten)."""
        publisher.state.circuit_open = True
        log.warning(f"Circuit breaker manually tripped via API. Reason: {reason}")
        return {"status": "tripped", "reason": reason}

    return app
