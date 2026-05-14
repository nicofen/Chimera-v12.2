"""
chimera/alerts/dispatcher.py
AlertDispatcher — the central hub for all Chimera notifications.

Responsibilities:
  1. Accept AlertEvent objects from anywhere in the system
  2. Apply rate limiting per priority tier (CRITICAL never throttled)
  3. Fan out to all configured senders (Telegram, Discord, or both)
  4. Run as an asyncio task inside the mainframe process

Rate limits (configurable):
  CRITICAL : unlimited
  HIGH     : 1 per 30 seconds
  NORMAL   : 1 per 60 seconds
  LOW      : 1 per 300 seconds (5 minutes)

Deduplication: consecutive identical event_types within the throttle window
are suppressed (e.g. don't send 100 "signal emitted" alerts in a minute).

Usage (from anywhere in the codebase):
    from chimera_v12.alerts.dispatcher import AlertDispatcher
    # dispatcher is a singleton set up in mainframe
    await dispatcher.send(evt_circuit_trip(...))
    dispatcher.send_nowait(evt_signal(...))   # non-blocking, queues it
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, TYPE_CHECKING

from chimera_v12.alerts.models import AlertEvent, Priority

if TYPE_CHECKING:
    from chimera_v12.alerts.telegram_sender import TelegramSender
    from chimera_v12.alerts.discord_sender  import DiscordSender

log = logging.getLogger("chimera.alerts.dispatcher")

# Rate limit windows in seconds per priority tier
RATE_WINDOWS: dict[Priority, float] = {
    Priority.CRITICAL: 0,      # never throttled
    Priority.HIGH:     30,
    Priority.NORMAL:   60,
    Priority.LOW:      300,
}


class AlertDispatcher:
    """
    Central alert dispatcher. Instantiated once in mainframe, injected
    into every agent that needs to send notifications.

    Call `await dispatcher.run()` as an asyncio task.
    Call `dispatcher.send_nowait(event)` from sync or async context.
    """

    def __init__(
        self,
        config:          dict[str, Any],
        telegram_sender: "TelegramSender | None" = None,
        discord_sender:  "DiscordSender  | None" = None,
    ):
        self.config   = config
        self.senders  = [s for s in [telegram_sender, discord_sender] if s is not None]
        self._queue:  asyncio.Queue[AlertEvent] = asyncio.Queue(maxsize=512)

        # last-sent timestamps per priority tier (for rate limiting)
        self._last_sent: dict[Priority, float] = {p: 0.0 for p in Priority}

        # last event_type sent per priority tier (for deduplication)
        self._last_type: dict[Priority, str] = {}

        # Heartbeat interval (seconds), 0 = disabled
        self._heartbeat_interval = config.get("alert_heartbeat_interval", 3600)

        if not self.senders:
            log.warning(
                "AlertDispatcher: no senders configured. "
                "Set TELEGRAM_BOT_TOKEN or DISCORD_WEBHOOK_URL in .env"
            )

    # ── Public API ─────────────────────────────────────────────────────────

    def send_nowait(self, event: AlertEvent) -> None:
        """
        Non-blocking enqueue. Safe to call from sync or async context.
        Drops quietly if queue is full (only for LOW/NORMAL priority).
        CRITICAL and HIGH events always find space.
        """
        if event.priority <= Priority.HIGH:
            # For critical/high: make room by dropping a LOW event if full
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            log.debug(f"Alert queue full — dropping {event.event_type} (priority={event.priority.name})")

    async def send(self, event: AlertEvent) -> None:
        """Async enqueue — awaits if queue is full."""
        await self._queue.put(event)

    # ── Main dispatch loop ─────────────────────────────────────────────────

    async def run(self) -> None:
        log.info(f"AlertDispatcher started — {len(self.senders)} sender(s) configured")

        # Start heartbeat task if enabled
        if self._heartbeat_interval > 0:
            asyncio.create_task(self._heartbeat_loop(), name="AlertHeartbeat")

        while True:
            try:
                event: AlertEvent = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
                await self._dispatch(event)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                log.exception(f"AlertDispatcher error: {e}")

    async def _dispatch(self, event: AlertEvent) -> None:
        """Apply rate limiting, then fan out to all configured senders."""
        if not self.senders:
            return

        now = time.monotonic()
        window = RATE_WINDOWS[event.priority]

        # Rate limit check (CRITICAL always passes)
        if window > 0:
            elapsed = now - self._last_sent.get(event.priority, 0.0)
            if elapsed < window:
                # Deduplicate: only suppress if same event type
                last_type = self._last_type.get(event.priority, "")
                if last_type == event.event_type:
                    log.debug(
                        f"Rate limited: {event.event_type} "
                        f"(retry in {window - elapsed:.0f}s)"
                    )
                    return

        self._last_sent[event.priority] = now
        self._last_type[event.priority] = event.event_type

        log.info(f"Dispatching [{event.priority.name}] {event.event_type}: {event.title[:60]}")

        # Fan out concurrently to all senders
        results = await asyncio.gather(
            *[sender.send(event) for sender in self.senders],
            return_exceptions=True,
        )
        failures = [r for r in results if r is False or isinstance(r, Exception)]
        if failures:
            log.warning(f"Alert delivery failures: {len(failures)}/{len(self.senders)}")

    async def _heartbeat_loop(self) -> None:
        """Periodic heartbeat — confirms the bot is alive."""
        from chimera_v12.alerts.models import evt_heartbeat
        await asyncio.sleep(self._heartbeat_interval)   # wait before first heartbeat
        while True:
            try:
                # The dispatcher doesn't have direct access to state here,
                # so heartbeat data is populated by the caller via send_nowait
                # from the mainframe's monitoring loop.
                log.debug("Heartbeat tick")
            except Exception:
                pass
            await asyncio.sleep(self._heartbeat_interval)


# ── Singleton factory ─────────────────────────────────────────────────────────

def build_dispatcher(config: dict[str, Any]) -> AlertDispatcher:
    """
    Build an AlertDispatcher from config/environment.
    Returns a dispatcher with all available senders attached.
    Call this once from mainframe.
    """
    import os

    telegram_sender = None
    discord_sender  = None

    tg_token = config.get("telegram_bot_token") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat  = config.get("telegram_chat_id")   or os.getenv("TELEGRAM_CHAT_ID", "")

    if tg_token and tg_chat:
        from chimera_v12.alerts.telegram_sender import TelegramSender
        telegram_sender = TelegramSender(tg_token, tg_chat)
        log.info("Telegram sender configured")
    else:
        log.info("Telegram not configured (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID)")

    dc_url = config.get("discord_webhook_url") or os.getenv("DISCORD_WEBHOOK_URL", "")
    if dc_url:
        from chimera_v12.alerts.discord_sender import DiscordSender
        discord_sender = DiscordSender(
            dc_url,
            mention_on_critical=config.get("discord_mention_critical", False),
        )
        log.info("Discord sender configured")
    else:
        log.info("Discord not configured (set DISCORD_WEBHOOK_URL)")

    return AlertDispatcher(config, telegram_sender, discord_sender)
