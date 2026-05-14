"""
chimera/alerts/telegram_sender.py
Telegram alert sender using the Bot API (sendMessage endpoint).

Setup:
  1. Create a bot via @BotFather → get BOT_TOKEN
  2. Start a conversation with your bot or add it to a group → get CHAT_ID
     (send /start, then: curl https://api.telegram.org/bot<TOKEN>/getUpdates)
  3. Set in .env:
       TELEGRAM_BOT_TOKEN=1234567890:ABCdef...
       TELEGRAM_CHAT_ID=-1001234567890   # negative = group, positive = private

Message format: Markdown V2 (bold, italic, code, pre blocks).
Long messages are automatically truncated at Telegram's 4096-char limit.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from chimera_v12.alerts.models import AlertEvent, Priority
from chimera_v12.utils.logger import setup_logger

log = setup_logger("alerts.telegram")

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
MAX_MSG_LEN  = 4096
RETRY_DELAYS = [1, 3, 10]   # seconds between retries on failure


class TelegramSender:
    """
    Sends AlertEvent messages to a Telegram chat via Bot API.

    Usage:
        sender = TelegramSender(bot_token="...", chat_id="...")
        await sender.send(event)
    """

    def __init__(self, bot_token: str, chat_id: str):
        if not bot_token or not chat_id:
            raise ValueError("Telegram bot_token and chat_id are required.")
        self.bot_token = bot_token
        self.chat_id   = str(chat_id)
        self._url      = TELEGRAM_API.format(token=bot_token)

    async def send(self, event: AlertEvent) -> bool:
        """
        Send one alert event. Returns True on success, False on failure.
        Retries up to 3 times with exponential backoff for transient errors.
        """
        text = self._format(event)
        payload = {
            "chat_id":                  self.chat_id,
            "text":                     text[:MAX_MSG_LEN],
            "parse_mode":               "MarkdownV2",
            "disable_web_page_preview": True,
            "disable_notification":     event.priority == Priority.LOW,
        }

        for attempt, delay in enumerate([0] + RETRY_DELAYS):
            if delay:
                await asyncio.sleep(delay)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self._url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        body = await resp.json()
                        if resp.status == 200 and body.get("ok"):
                            log.debug(f"Telegram sent: {event.event_type}")
                            return True
                        # Handle known permanent errors (don't retry)
                        err_code = body.get("error_code", 0)
                        if err_code in (400, 403):
                            log.error(
                                f"Telegram permanent error {err_code}: "
                                f"{body.get('description')} — check CHAT_ID"
                            )
                            return False
                        log.warning(
                            f"Telegram attempt {attempt+1} failed: "
                            f"{resp.status} {body.get('description','')}"
                        )
            except asyncio.TimeoutError:
                log.warning(f"Telegram timeout (attempt {attempt+1})")
            except aiohttp.ClientError as e:
                log.warning(f"Telegram network error: {e}")

        log.error(f"Telegram: all retries exhausted for {event.event_type}")
        return False

    def _format(self, event: AlertEvent) -> str:
        """
        Build a MarkdownV2-formatted message.
        MarkdownV2 requires escaping: _ * [ ] ( ) ~ ` > # + - = | { } . !
        """
        ts = event.ts.strftime("%H:%M:%S UTC")

        # Header line
        header = f"{event.emoji} *{_escape(event.title)}*"

        # Priority badge for CRITICAL
        badge = ""
        if event.priority == Priority.CRITICAL:
            badge = "\n🔴 *CRITICAL — IMMEDIATE ACTION REQUIRED*"

        # Body — escape all MarkdownV2 special chars
        body_escaped = _escape_body(event.body)

        # Timestamp footer
        footer = f"\n\\_\\_{_escape(ts)}\\_{_escape('chimera')}__"

        return f"{header}{badge}\n\n{body_escaped}{footer}"


def _escape(text: str) -> str:
    """Escape MarkdownV2 special characters."""
    special = r"\_*[]()~`>#+-=|{}.!"
    return "".join("\\" + c if c in special else c for c in str(text))


def _escape_body(text: str) -> str:
    """
    Escape a multi-line body that may contain *bold* and `code` markers.
    We preserve intentional Markdown markers (single * and `) and only
    escape other special characters.
    """
    # This is a simplified approach: escape everything except * and `
    # which are used for formatting in the body strings
    special = r"\_[]()~>#+-=|{}.!"
    result = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in special:
            result.append("\\" + c)
        else:
            result.append(c)
        i += 1
    return "".join(result)
