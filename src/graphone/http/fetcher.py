"""Async, polite, fault-tolerant HTTP fetcher.

* global connection pool (HTTP_CONCURRENCY) + per-host semaphore and token bucket
* AIMD per-host throttling: 429/503 halve that host's rate, successes restore it
* retries with exponential backoff + jitter, honours Retry-After
* bot-challenge detection -> escalation to the Playwright tier
* optional robots.txt compliance and proxy gateway
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import aiohttp

from ..config import Settings
from ..logging_setup import METRICS, log
from .antibot import BrowserRenderer, detect_challenge
from .backoff import TokenBucket, backoff_delay, parse_retry_after

logger = logging.getLogger("graphone.fetcher")

RETRYABLE = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524}


_SOFT_BLOCK = (403, 405)  # escalated to the browser tier once


class FetchError(Exception):
    def __init__(self, url: str, status: int | None, reason: str) -> None:
        super().__init__(f"{reason} ({status}) {url}")
        self.url, self.status, self.reason = url, status, reason


class NotFound(FetchError):
    pass


@dataclass
class FetchResult:
    url: str
    final_url: str
    status: int
    text: str
    headers: dict[str, str] = field(default_factory=dict)
    via: str = "http"  # http | browser

    def json(self):
        return json.loads(self.text)


class _Host:
    def __init__(self, concurrency: int, rps: float) -> None:
        self.sem = asyncio.Semaphore(concurrency)
        self.bucket = TokenBucket(rps, capacity=max(1.0, rps))


class Fetcher:
    def __init__(self, settings: Settings, host_overrides: dict[str, float] | None = None) -> None:
        self.s = settings
        self._hosts: dict[str, _Host] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self._robots_lock = asyncio.Lock()
        self.host_overrides = host_overrides or {}
        self.browser = (
            BrowserRenderer(settings.user_agent, settings.browser_concurrency, settings.proxy_url)
            if settings.browser_fallback else None
        )
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "Fetcher":
        connector = aiohttp.TCPConnector(limit=self.s.http_concurrency, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.s.request_timeout, sock_connect=10),
            headers={
                "User-Agent": self.s.user_agent,
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate",
            },
            trust_env=True,  # honour HTTPS_PROXY etc.
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self.session:
            await self.session.close()
        if self.browser:
            await self.browser.close()

    # ------------------------------------------------------------------ helpers
    def _host(self, url: str) -> _Host:
        host = urlsplit(url).netloc.lower()
        h = self._hosts.get(host)
        if h is None:
            rps = self.host_overrides.get(host, self.s.per_host_rps)
            h = self._hosts[host] = _Host(self.s.per_host_concurrency, rps)
        return h

    async def _allowed(self, url: str) -> bool:
        if not self.s.respect_robots:
            return True
        parts = urlsplit(url)
        key = f"{parts.scheme}://{parts.netloc}"
        async with self._robots_lock:
            if key not in self._robots:
                rp: RobotFileParser | None = RobotFileParser()
                try:
                    async with self.session.get(key + "/robots.txt", proxy=self.s.proxy_url,
                                                timeout=aiohttp.ClientTimeout(total=10)) as r:
                        if r.status == 200:
                            rp.parse((await r.text(errors="ignore")).splitlines())
                        else:
                            rp = None  # no robots.txt -> allowed
                except Exception:
                    rp = None
                self._robots[key] = rp
        rp = self._robots[key]
        return True if rp is None else rp.can_fetch(self.s.user_agent, url)

    # ------------------------------------------------------------------ public
    async def get(
        self,
        url: str,
        *,
        params: dict | None = None,
        headers: dict | None = None,
        accept: str = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        allow_browser: bool = True,
        check_robots: bool = True,
        method: str = "GET",
        json_body: dict | None = None,
    ) -> FetchResult:
        assert self.session is not None, "use `async with Fetcher(...)`"
        if check_robots and not await self._allowed(url):
            METRICS.inc("http.robots_disallowed")
            raise FetchError(url, None, "disallowed by robots.txt")
        host = self._host(url)
        hdrs = {"Accept": accept, **(headers or {})}
        last_status: int | None = None
        for attempt in range(self.s.max_retries + 1):
            try:
                async with host.sem:
                    await host.bucket.acquire()
                    METRICS.inc("http.requests")
                    async with self.session.request(
                        method, url, params=params, headers=hdrs, json=json_body, proxy=self.s.proxy_url,
                        allow_redirects=True,
                    ) as resp:
                        status = resp.status
                        last_status = status
                        text = await resp.text(errors="replace")
                        rh = dict(resp.headers)
                        final = str(resp.url)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                METRICS.inc("http.network_error")
                delay = backoff_delay(attempt)
                log(logger, logging.DEBUG, "network error; retrying", url=url, attempt=attempt, error=type(exc).__name__)
                await asyncio.sleep(delay)
                continue

            challenge = detect_challenge(status, rh, text)
            if challenge:
                METRICS.inc(f"http.challenge.{challenge}")
                if allow_browser and self.browser:
                    rendered = await self.browser.render(url)
                    if rendered and not rendered.challenge:
                        return FetchResult(url, rendered.url, rendered.status or 200, rendered.html, {}, "browser")
                raise FetchError(url, status, f"blocked by {challenge}")

            # GitHub-style rate limiting arrives as 403 + x-ratelimit headers
            if status == 403 and (rh.get("X-RateLimit-Remaining") == "0" or "rate limit" in text[:500].lower()):
                reset = rh.get("X-RateLimit-Reset")
                wait = parse_retry_after(rh.get("Retry-After")) or (
                    max(1.0, float(reset) - time.time()) if reset and reset.isdigit() else backoff_delay(attempt, 5))
                METRICS.inc("http.rate_limited_403")
                if wait > 120:
                    raise FetchError(url, status, f"rate limited for {wait:.0f}s")
                await asyncio.sleep(wait)
                continue
            if status in RETRYABLE:
                METRICS.inc(f"http.status.{status}")
                if status in (429, 503):
                    host.bucket.penalize()
                wait = parse_retry_after(rh.get("Retry-After")) or backoff_delay(attempt)
                if wait > 120:
                    raise FetchError(url, status, f"retry-after too long ({wait:.0f}s)")
                log(logger, logging.DEBUG, "retryable status", url=url, status=status, wait=round(wait, 2))
                await asyncio.sleep(wait)
                continue
            if status in _SOFT_BLOCK and method == "GET" and allow_browser and self.browser:
                # WAFs that block bots without a recognisable challenge page (e.g. 405 on GET)
                METRICS.inc(f"http.soft_block.{status}")
                rendered = await self.browser.render(url)
                if rendered and not rendered.challenge and (rendered.status or 200) < 400:
                    return FetchResult(url, rendered.url, rendered.status or 200, rendered.html, {}, "browser")
                raise FetchError(url, status, "blocked (soft)")
            if status in (404, 410):
                METRICS.inc("http.not_found")
                raise NotFound(url, status, "not found")
            if status >= 400:
                METRICS.inc(f"http.status.{status}")
                raise FetchError(url, status, "http error")

            host.bucket.reward()
            METRICS.inc("http.ok")
            return FetchResult(url, final, status, text, rh, "http")

        METRICS.inc("http.gave_up")
        raise FetchError(url, last_status, "retries exhausted")

    async def get_json(self, url: str, **kw):
        kw.setdefault("accept", "application/json")
        kw.setdefault("allow_browser", False)
        kw.setdefault("check_robots", False)  # documented public APIs
        res = await self.get(url, **kw)
        return res.json()
