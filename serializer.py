"""
chimera/server/publisher.py
StatePublisher — the bridge between SharedState and the WebSocket layer.

Runs as an asyncio task inside the mainframe process.
Every POLL_INTERVAL seconds it:
  1. Serializes the current SharedState into a snapshot dict.
  2. Diffs it against the previous snapshot.
  3. If anything changed, puts the diff onto a broadcast queue.
  4. The WebSocket handler drains that queue and sends to all clients.

This decouples the publishing rate from the agent cycle rates —
agents can update state at any frequency; the dashboard always gets
a clean 250 ms cadence with no duplicate sends.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from chimera_v12.server.serializer import serialize_state, serialize_diff

if TYPE_CHECKING:
    from chimera_v12.utils.state import SharedState

log = logging.getLogger("chimera.server.publisher")

POLL_INTERVAL = 0.25   # seconds — 4 Hz tick rate to the dashboard


class StatePublisher:
    """
    Owned by the WebSocket server. Call `.run()` as an asyncio task.
    Consumers read from `.queue` (asyncio.Queue of JSON strings).
    """

    def __init__(self, state: "SharedState"):
        self.state  = state
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self._prev_snapshot: dict = {}

    async def run(self) -> None:
        log.info("StatePublisher started — polling every %.0fms", POLL_INTERVAL * 1000)
        while True:
            try:
                await self._tick()
            except Exception as e:
                log.warning(f"StatePublisher tick error: {e}")
            await asyncio.sleep(POLL_INTERVAL)

    async def _tick(self) -> None:
        curr = serialize_state(self.state)

        if not self._prev_snapshot:
            # First tick — queue a full snapshot for any waiting clients
            msg = json.dumps(curr)
            await self._enqueue(msg)
        else:
            diff = serialize_diff(self._prev_snapshot, curr)
            # Only broadcast if something actually changed (beyond ts)
            if len(diff) > 2:
                msg = json.dumps(diff)
                await self._enqueue(msg)

        self._prev_snapshot = curr

    async def _enqueue(self, msg: str) -> None:
        try:
            self.queue.put_nowait(msg)
        except asyncio.QueueFull:
            # Drop oldest message to make room — dashboard will catch up
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self.queue.put_nowait(msg)

    def get_snapshot_json(self) -> str:
        """Synchronous snapshot for new WebSocket clients joining mid-session."""
        import json
        snap = serialize_state(self.state)
        return json.dumps(snap)
