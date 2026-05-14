"""
chimera/alerts/discord_sender.py
Discord alert sender using Webhook API (no bot token required).

Setup:
  1. In your Discord server: Channel Settings → Integrations → Webhooks → New Webhook
  2. Copy the webhook URL
  3. Set in .env:
       DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/1234/abcdef...

Messages use Discord Embeds for rich formatting:
  - Colour-coded by priority (red=critical, orange=high, blue=normal, grey=low)
  - Timestamp in embed footer
  - Fields for structured data
  - Mention @here for CRITICAL events (optional, set mention_on_critical=True)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from chimera_v12.alerts.models import AlertEvent, Priority
from chimera_v12.utils.logger import setup_logger

log = setup_logger("alerts.discord")

MAX_EMBED_DESC = 4096
RETRY_DELAYS   = [1, 3, 10]

# Embed colours per priority (Discord uses decimal integers)
PRIORITY_COLOURS = {
    Priority.CRITICAL: 0xE03050,   # red
    Priority.HIGH:     0xE8A030,   # amber
    Priority.NORMAL:   0x3888E8,   # blue
    Priority.LOW:      0x5A6870,   # grey
}


class DiscordSender:
    """
    Sends AlertEvent messages to a Discord channel via webhook.

    Usage:
        sender = DiscordSender(webhook_url="https://discord.com/api/webhooks/...")
        await sender.send(event)
    """

    def __init__(self, webhook_url: str, mention_on_critical: bool = False):
        if not webhook_url:
            raise ValueError("Discord webhook_url is required.")
        self.webhook_url         = webhook_url
        self.mention_on_critical = mention_on_critical

    async def send(self, event: AlertEvent) -> bool:
        """Send one alert. Returns True on success."""
        payload = self._build_payload(event)

        for attempt, delay in enumerate([0] + RETRY_DELAYS):
            if delay:
                await asyncio.sleep(delay)
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status in (200, 204):
                            log.debug(f"Discord sent: {event.event_type}")
                            return True
                        # 429 = rate limited — Discord gives a Retry-After header
                        if resp.status == 429:
                            body = await resp.json()
                            wait = body.get("retry_after", 5)
                            log.warning(f"Discord rate limited — waiting {wait}s")
                            await asyncio.sleep(wait)
                            continue
                        # 400/404 = permanent error
                        if resp.status in (400, 404):
                            text = await resp.text()
                            log.error(f"Discord permanent error {resp.status}: {text[:200]}")
                            return False
                        log.warning(f"Discord attempt {attempt+1}: HTTP {resp.status}")
            except asyncio.TimeoutError:
                log.warning(f"Discord timeout (attempt {attempt+1})")
            except aiohttp.ClientError as e:
                log.warning(f"Discord network error: {e}")

        log.error(f"Discord: all retries exhausted for {event.event_type}")
        return False

    def _build_payload(self, event: AlertEvent) -> dict[str, Any]:
        # Mention @here for critical events
        content = ""
        if event.priority == Priority.CRITICAL and self.mention_on_critical:
            content = "@here "

        # Parse body into embed fields (lines starting with *Key:* pattern)
        description, fields = _parse_body(event.body)

        embed = {
            "title":       f"{event.emoji}  {event.title}",
            "description": description[:MAX_EMBED_DESC] if description else None,
            "color":       PRIORITY_COLOURS.get(event.priority, 0x3888E8),
            "fields":      fields,
            "footer":      {
                "text": f"Chimera  •  {event.ts.strftime('%H:%M:%S UTC')}"
            },
            "timestamp":   event.ts.isoformat(),
        }

        # Remove None values
        embed = {k: v for k, v in embed.items() if v is not None}
        if not fields:
            embed.pop("fields", None)

        return {
            "content": content or None,
            "username": "Project Chimera",
            "embeds": [embed],
        }


def _parse_body(body: str) -> tuple[str, list[dict]]:
    """
    Parse a body string into (description, fields).
    Lines matching '*Key:* Value' become inline embed fields.
    Other lines go into the description.
    """
    import re
    field_pattern = re.compile(r"^\*(.+?):\*\s*(.+)$")
    fields: list[dict] = []
    desc_lines: list[str] = []

    for line in body.split("\n"):
        m = field_pattern.match(line.strip())
        if m:
            fields.append({
                "name":   m.group(1),
                "value":  m.group(2),
                "inline": True,
            })
        else:
            desc_lines.append(line)

    description = "\n".join(l for l in desc_lines if l.strip())
    return description, fields
