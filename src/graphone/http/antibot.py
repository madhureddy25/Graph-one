"""Bot-protection detection and the headless-browser escalation tier.

Strategy (see docs/architecture.md, 'Anti-bot'):
  Tier 0  official API / RSS / sitemap / embedded JSON (never touches the WAF)
  Tier 1  aiohttp with a consistent browser header set, per-host politeness
          budget, AIMD throttling and optional rotating proxy gateway
  Tier 2  real Chromium via Playwright: executes the JS challenge
          (Cloudflare 'Just a moment…' / DataDome device check) the same way a
          visitor's browser does, with a persistent context per host so the
          clearance cookie is reused for the rest of the crawl
  Tier 3  give up on the host for this run, mark it 'blocked' and fall back to
          an alternative source for the same entity. Interactive CAPTCHAs are
          never solved or outsourced.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlsplit

from ..logging_setup import METRICS, log

logger = logging.getLogger("graphone.antibot")

_CF_MARKERS = (
    "cf-browser-verification", "challenge-platform", "cf-chl-", "_cf_chl_opt",
    "<title>just a moment...</title>", "attention required! | cloudflare",
    "cf-turnstile", "checking your browser before accessing",
)
_DATADOME_MARKERS = ("captcha-delivery.com", "geo.captcha-delivery", "datadome")
_PX_MARKERS = ("px-captcha", "_pxappid", "perimeterx")
_GENERIC = ("access to this page has been denied", "are you a robot", "verify you are human")


def detect_challenge(status: int, headers: dict[str, str], body: str) -> str | None:
    """Return the vendor name if the response is a bot challenge, else None."""
    h = {k.lower(): v.lower() for k, v in headers.items()}
    b = body[:20000].lower() if body else ""
    server = h.get("server", "")
    if "x-datadome" in h or "datadome" in h.get("set-cookie", ""):
        if status in (401, 403, 429) or any(m in b for m in _DATADOME_MARKERS):
            return "datadome"
    if any(m in b for m in _DATADOME_MARKERS) and status in (403, 429):
        return "datadome"
    if (status in (403, 429, 503) and "cloudflare" in server) or "cf-mitigated" in h:
        return "cloudflare"
    if any(m in b for m in _CF_MARKERS):
        return "cloudflare"
    if any(m in b for m in _PX_MARKERS):
        return "perimeterx"
    if status in (403, 429) and any(m in b for m in _GENERIC):
        return "generic"
    return None


@dataclass
class RenderResult:
    url: str
    status: int
    html: str
    challenge: str | None


class BrowserRenderer:
    """Lazy, shared Playwright Chromium. One persistent context per host."""

    def __init__(self, user_agent: str, concurrency: int = 4, proxy_url: str | None = None) -> None:
        self.user_agent = user_agent
        self.proxy_url = proxy_url
        self._sem = asyncio.Semaphore(concurrency)
        self._pw = None
        self._browser = None
        self._contexts: dict[str, object] = {}
        self._start_lock = asyncio.Lock()
        self.available: bool | None = None  # unknown until first use
        self.blocked_hosts: set[str] = set()

    async def _ensure(self) -> bool:
        async with self._start_lock:
            if self.available is not None:
                return self.available
            try:
                from playwright.async_api import async_playwright  # optional dependency

                self._pw = await async_playwright().start()
                launch: dict = {"headless": True}
                if self.proxy_url:
                    launch["proxy"] = {"server": self.proxy_url}
                self._browser = await self._pw.chromium.launch(**launch)
                self.available = True
            except Exception as exc:  # not installed / no browser binary
                log(logger, logging.WARNING, "browser tier unavailable", error=str(exc)[:200])
                self.available = False
            return self.available

    async def _context(self, host: str):
        ctx = self._contexts.get(host)
        if ctx is None:
            ctx = await self._browser.new_context(
                user_agent=self.user_agent, locale="en-US", viewport={"width": 1366, "height": 900},
            )
            # Save bandwidth: we only need the DOM.
            await ctx.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in {"image", "media", "font"}
                else route.continue_(),
            )
            self._contexts[host] = ctx
        return ctx

    async def render(self, url: str, timeout_s: float = 30.0) -> RenderResult | None:
        host = urlsplit(url).netloc
        if host in self.blocked_hosts or not await self._ensure():
            return None
        async with self._sem:
            METRICS.inc("browser.render")
            page = await (await self._context(host)).new_page()
            try:
                resp = await page.goto(url, wait_until="domcontentloaded", timeout=timeout_s * 1000)
                status = resp.status if resp else 0
                html = await page.content()
                # JS challenges usually auto-resolve and redirect within a few seconds.
                for _ in range(int(timeout_s // 2)):
                    if not detect_challenge(status, {}, html):
                        break
                    await page.wait_for_timeout(2000)
                    html = await page.content()
                challenge = detect_challenge(200, {}, html)
                if challenge:
                    # Interactive CAPTCHA (or hard block): do not try to solve it.
                    self.blocked_hosts.add(host)
                    METRICS.inc(f"browser.blocked.{challenge}")
                    log(logger, logging.WARNING, "host still challenged after render; marking blocked",
                        host=host, vendor=challenge)
                else:
                    METRICS.inc("browser.cleared")
                return RenderResult(url=page.url, status=status, html=html, challenge=challenge)
            except Exception as exc:
                METRICS.inc("browser.error")
                log(logger, logging.WARNING, "browser render failed", url=url, error=str(exc)[:200])
                return None
            finally:
                await page.close()

    async def close(self) -> None:
        for ctx in self._contexts.values():
            try:
                await ctx.close()
            except Exception:
                pass
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
