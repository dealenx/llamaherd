import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any


log = logging.getLogger("llamaherd")


class UsageScraper:
    """Scrapes session/weekly usage from ollama.com/settings using browser cookies.
    
    Requires __Secure-session, aid, and cf_clearance cookies per key.
    Falls back gracefully if cookies are missing or expired.
    """

    def __init__(self, keys_config: list[dict]):
        self.cookie_map: dict[str, dict] = {}  # label -> {secure_session, aid, cf_clearance}
        for kc in keys_config:
            label = kc.get("label", "")
            cookies = kc.get("cookies", {})
            if cookies and cookies.get("secure_session"):
                self.cookie_map[label] = {
                    "secure_session": cookies["secure_session"],
                    "aid": cookies.get("aid", ""),
                    "cf_clearance": cookies.get("cf_clearance", ""),
                    "stripe_mid": cookies.get("stripe_mid", ""),
                }

    def scrape_usage(self, key: Any) -> dict | None:
        """Scrape usage data for a single key. Returns dict or None."""
        if key.label not in self.cookie_map:
            return None
        cookies = self.cookie_map[key.label]
        if not cookies.get("secure_session"):
            return None

        try:
            import cloudscraper
            from bs4 import BeautifulSoup
        except ImportError:
            log.warning("cloudscraper or beautifulsoup4 not installed — usage scraping disabled")
            return None

        scraper = None
        try:
            scraper = cloudscraper.create_scraper()
            scraper.cookies.set("__Secure-session", cookies["secure_session"], domain="ollama.com")
            if cookies.get("aid"):
                scraper.cookies.set("aid", cookies["aid"], domain="ollama.com")
            if cookies.get("cf_clearance"):
                scraper.cookies.set("cf_clearance", cookies["cf_clearance"], domain="ollama.com")
            if cookies.get("stripe_mid"):
                scraper.cookies.set("__stripe_mid", cookies["stripe_mid"], domain="ollama.com")

            resp = scraper.get(
                "https://ollama.com/settings",
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                timeout=15,
            )
            resp.raise_for_status()

            if "Just a moment" in resp.text:
                log.warning(f"Usage scrape for {key.label}: Cloudflare challenge — need fresh cf_clearance cookie")
                return None
            if "Sign in" in resp.text[:3000]:
                log.warning(f"Usage scrape for {key.label}: auth required — __Secure-session cookie expired")
                return None

            soup = BeautifulSoup(resp.text, "html.parser")

            result: dict = {
                "session_usage_pct": -1.0,
                "weekly_usage_pct": -1.0,
                "session_resets_at": None,
                "weekly_resets_at": None,
                "session_models": {},
                "weekly_models": {},
            }

            # Parse top-level usage percentages and reset times
            pct_divs = soup.find_all("div", class_="flex justify-between mb-2")
            for i, div in enumerate(pct_divs):
                spans = div.find_all("span")
                if len(spans) >= 2:
                    label = spans[0].get_text(strip=True)
                    value = spans[1].get_text(strip=True)
                    pct_str = value.replace("% used", "").replace("%", "").strip()
                    try:
                        pct = float(pct_str)
                    except ValueError:
                        continue
                    if "Session" in label:
                        result["session_usage_pct"] = pct
                    elif "Weekly" in label:
                        result["weekly_usage_pct"] = pct

            for i, div in enumerate(soup.find_all("div", class_="local-time")):
                iso_time = div.get("data-time", "")
                text = div.get_text(strip=True)
                if i == 0:
                    result["session_resets_at"] = iso_time or text
                elif i == 1:
                    result["weekly_resets_at"] = iso_time or text

            # Parse per-model usage bars: session (first meter), weekly (second meter)
            meters = soup.find_all("div", attrs={"data-usage-meter": True})
            for window, meter in [("session", meters[0] if len(meters) > 0 else None),
                                  ("weekly", meters[1] if len(meters) > 1 else None)]:
                if not meter:
                    continue
                window_key = f"{window}_models"
                for seg in meter.find_all("button", attrs={"data-usage-segment": True}):
                    model = seg.get("data-model", "")
                    reqs = seg.get("data-requests", "")
                    if not model:
                        continue
                    try:
                        requests = int(re.sub(r"[^0-9]", "", str(reqs))) if reqs else 0
                    except ValueError:
                        requests = 0
                    pct_str = "0"
                    style = str(seg.get("style", ""))
                    m = re.search(r"width:\s*([\d.]+)%", style)
                    if m:
                        pct_str = m.group(1)
                    try:
                        bar_pct = float(pct_str)
                    except ValueError:
                        bar_pct = 0.0
                    result[window_key][model] = {
                        "requests": requests,
                        "bar_pct": round(bar_pct, 3),
                    }

            log.info(f"Usage scrape for {key.label}: "
                     f"session {result['session_usage_pct']}% "
                     f"weekly {result['weekly_usage_pct']}% "
                     f"models={len(result['session_models'])}/{len(result['weekly_models'])}")
            return result

        except Exception as e:
            log.warning(f"Usage scrape error for {key.label}: {e}")
            return None
        finally:
            if scraper is not None:
                scraper.close()

    def scrape_all(self, keys: list[Any]) -> dict[str, dict]:
        """Scrape usage for all keys with cookies configured."""
        results = {}
        for key in keys:
            data = self.scrape_usage(key)
            if data:
                key.session_usage_pct = data["session_usage_pct"]
                key.session_resets_at = data["session_resets_at"]
                key.weekly_usage_pct = data["weekly_usage_pct"]
                key.weekly_resets_at = data["weekly_resets_at"]
                key.session_models = data.get("session_models", {})
                key.weekly_models = data.get("weekly_models", {})
                results[key.label] = data
        return results


class ActivityUsageRefresher:
    """Debounce per-account usage scrapes after upstream activity."""

    def __init__(
        self,
        scraper: UsageScraper,
        *,
        debounce_seconds: float = 30,
        min_interval_seconds: float = 300,
        on_update: Callable[[], Awaitable[None]] | None = None,
    ):
        self.scraper = scraper
        self.debounce_seconds = debounce_seconds
        self.min_interval_seconds = min_interval_seconds
        self.on_update = on_update
        self._last_scraped: dict[str, float] = {}
        self._pending: dict[str, asyncio.Task] = {}

    def mark_scraped(self, labels: list[str] | tuple[str, ...] | set[str]) -> None:
        """Record externally completed scrapes, such as the startup refresh."""
        now = time.monotonic()
        for label in labels:
            self._last_scraped[label] = now

    def schedule(self, key: Any) -> None:
        """Schedule one bounded refresh for an account that handled a request."""
        label = key.label
        if label not in self.scraper.cookie_map or label in self._pending:
            return
        self._pending[label] = asyncio.create_task(
            self._refresh_after_delay(key),
            name=f"llamaherd-usage-refresh:{label}",
        )

    async def _refresh_after_delay(self, key: Any) -> None:
        label = key.label
        try:
            elapsed = time.monotonic() - self._last_scraped.get(label, 0)
            delay = max(self.debounce_seconds, self.min_interval_seconds - elapsed)
            if delay > 0:
                await asyncio.sleep(delay)

            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(None, self.scraper.scrape_usage, key)
            self._last_scraped[label] = time.monotonic()
            if not data:
                return

            key.session_usage_pct = data["session_usage_pct"]
            key.session_resets_at = data["session_resets_at"]
            key.weekly_usage_pct = data["weekly_usage_pct"]
            key.weekly_resets_at = data["weekly_resets_at"]
            key.session_models = data.get("session_models", {})
            key.weekly_models = data.get("weekly_models", {})
            log.info("Activity-triggered usage scrape updated: %s", label)
            if self.on_update:
                await self.on_update()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Activity-triggered usage scrape failed: %s", label)
        finally:
            if self._pending.get(label) is asyncio.current_task():
                self._pending.pop(label, None)

    async def close(self) -> None:
        """Cancel and reap pending refreshes during application shutdown."""
        tasks = list(self._pending.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._pending.clear()
