import logging
import os
from datetime import datetime, timezone
from typing import Optional

import httpx


log = logging.getLogger("llamaherd")


class TelegramNotifier:
    """Sends Ollama Cloud usage notifications to Telegram.

    Configured via environment variables (no DB, no UI):
      LLAMAHERD_TELEGRAM_BOT_TOKEN  — Bot token from @BotFather
      LLAMAHERD_TELEGRAM_CHAT_ID    — Chat/group ID (negative for groups)
      LLAMAHERD_TELEGRAM_TOPIC_ID   — Topic ID for groups with topics (optional)
      LLAMAHERD_TELEGRAM_INTERVAL   — Seconds between notifications (default: 1800 = 30min)

    If BOT_TOKEN or CHAT_ID is not set, the notifier is disabled.
    """

    def __init__(self):
        self.bot_token = os.environ.get("LLAMAHERD_TELEGRAM_BOT_TOKEN", "") or os.environ.get("LLAMAHERD_TELEGRAM_TOKEN", "")
        self.chat_id = os.environ.get("LLAMAHERD_TELEGRAM_CHAT_ID", "")
        self.topic_id = os.environ.get("LLAMAHERD_TELEGRAM_TOPIC_ID", "")
        self.interval = int(os.environ.get("LLAMAHERD_TELEGRAM_INTERVAL", "1800"))
        self.enabled = bool(self.bot_token and self.chat_id)
        self._client = httpx.AsyncClient(timeout=30.0) if self.enabled else None

    def _make_progress_bar(self, pct: float, width: int = 16) -> str:
        """Create a text progress bar: ████░░░░░░░░░░░░ 25%"""
        if pct < 0:
            return "░" * width + " ?"
        pct = max(0, min(100, pct))
        filled = int(pct / 100 * width)
        return "█" * filled + "░" * (width - filled) + f" {pct:.1f}%"

    def _status_emoji(self, pct: float) -> str:
        """🟢 < 50%, 🟡 50-80%, 🔴 > 80%"""
        if pct < 0:
            return "⚪"
        if pct >= 80:
            return "🔴"
        if pct >= 50:
            return "🟡"
        return "🟢"

    def _format_reset_time(self, resets_at: Optional[str]) -> str:
        """Format reset time as 'in X hours/days'."""
        if not resets_at:
            return "unknown"
        try:
            end = datetime.fromisoformat(resets_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            remaining = end - now
            if remaining.total_seconds() <= 0:
                return "now"
            hours = remaining.total_seconds() / 3600
            if hours < 1:
                return f"in {int(remaining.total_seconds() / 60)} min"
            if hours < 24:
                return f"in {int(hours)} hours"
            return f"in {int(hours / 24)} days"
        except (ValueError, TypeError):
            return "unknown"

    def format_message(self, keys: list) -> str:
        """Format the notification message matching the dashboard layout."""
        now_str = datetime.now(timezone.utc).strftime("%d.%m.%Y, %H:%M:%S")

        # Count statuses by weekly usage
        red = sum(1 for k in keys if k.weekly_usage_pct >= 80)
        yellow = sum(1 for k in keys if 50 <= k.weekly_usage_pct < 80)
        green = sum(1 for k in keys if 0 <= k.weekly_usage_pct < 50)

        lines = ["☁️ Ollama Cloud Monitor", now_str, ""]
        lines.append(f"🔴{red} 🟡{yellow} 🟢{green}")

        for k in keys:
            # Key header: email (plan)
            plan = k.plan or "?"
            email = k.account_email or k.label or "Unknown"
            lines.append(f"\n• {email} ({plan})")

            # Slots: in_flight / max_concurrent
            slots = f"{k.in_flight}/{k.max_concurrent}"
            lines.append(f"  Slots: {slots}")

            # Session usage with elapsed %
            s_pct = k.session_usage_pct
            s_emoji = self._status_emoji(s_pct)
            s_bar = self._make_progress_bar(s_pct)
            s_reset = self._format_reset_time(k.session_resets_at)
            s_elapsed = k._session_elapsed_pct()
            s_elapsed_str = f" ({s_elapsed:.0f}% elapsed)" if s_elapsed >= 0 else ""
            lines.append(f"  {s_emoji} Session: {s_bar}{s_elapsed_str} → {s_reset}")

            # Weekly usage with elapsed %
            w_pct = k.weekly_usage_pct
            w_emoji = self._status_emoji(w_pct)
            w_bar = self._make_progress_bar(w_pct)
            w_reset = self._format_reset_time(k.weekly_resets_at)
            w_elapsed = k._weekly_elapsed_pct()
            w_elapsed_str = f" ({w_elapsed:.0f}% elapsed)" if w_elapsed >= 0 else ""
            lines.append(f"  {w_emoji} Weekly: {w_bar}{w_elapsed_str} → {w_reset}")

            # Billing (progress bar shows remaining %)
            billing = k.period_remaining_pct
            b_bar = self._make_progress_bar(billing)
            lines.append(f"  💰 Billing: {b_bar} left")

            # Requests & 429s
            lines.append(f"  Requests: {k.total_requests}")
            lines.append(f"  429s: {k.total_429s}")

        # Summary: total slots
        total_in_flight = sum(k.in_flight for k in keys)
        total_max = sum(k.max_concurrent for k in keys)
        lines.append(f"\n📊 Total slots: {total_in_flight}/{total_max}")

        return "\n".join(lines)

    async def send(self, text: str) -> bool:
        """Send a message to Telegram. Returns True on success."""
        if not self.enabled or not self._client:
            return False
        try:
            payload: dict = {
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML",
            }
            if self.topic_id:
                payload["message_thread_id"] = int(self.topic_id)
            r = await self._client.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json=payload,
            )
            if r.status_code != 200:
                log.warning(f"Telegram send failed: {r.status_code} {r.text[:200]}")
                return False
            return True
        except Exception as e:
            log.warning(f"Telegram send error: {e}")
            return False

    async def send_usage_notification(self, manager) -> bool:
        """Format and send a usage notification for all keys."""
        if not self.enabled or not manager:
            return False
        text = self.format_message(manager.keys)
        return await self.send(text)

    async def close(self):
        if self._client:
            await self._client.aclose()

