"""Backoff + rate limiting primitives shared by the crawler and the LLM layer."""
from __future__ import annotations

import asyncio
import random
import re
import time
from email.utils import parsedate_to_datetime


def backoff_delay(attempt: int, base: float = 1.0, cap: float = 60.0, rng: random.Random | None = None) -> float:
    """Exponential backoff with 'equal jitter': half deterministic, half random.

    Never returns ~0 (unlike full jitter), and still de-synchronises thousands of
    concurrent workers that all hit a 429 at the same instant (thundering herd).
    """
    r = rng or random
    ceiling = min(cap, base * (2 ** attempt))
    return ceiling / 2 + r.uniform(0, ceiling / 2)


_RETRY_IN = re.compile(
    r"(?:try again|retry) in\s+(?:(?P<h>\d+)h)?\s*(?:(?P<m>\d+)m(?!s))?\s*(?:(?P<v>[\d.]+)\s*(?P<u>ms|s))?", re.I)
_RETRY_DELAY = re.compile(r'"retryDelay"\s*:\s*"([\d.]+)s"')


def parse_retry_after(header: str | None = None, body: str | None = None) -> float | None:
    """Seconds to wait, from a Retry-After header (seconds or HTTP-date) or from
    provider error bodies ('Please try again in 7.66s', Gemini RetryInfo)."""
    if header:
        h = header.strip()
        try:
            return max(0.0, float(h))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(h).timestamp() - time.time())
            except (TypeError, ValueError):
                pass
    if body:
        m = _RETRY_DELAY.search(body)
        if m:
            return float(m.group(1))
        m = _RETRY_IN.search(body)
        if m and any(m.group(g) for g in ("h", "m", "v")):
            val = float(m.group("v") or 0)
            if m.group("u") == "ms":
                val /= 1000
            return float(m.group("h") or 0) * 3600 + float(m.group("m") or 0) * 60 + val
    return None


class TokenBucket:
    """Async token bucket. `rate` tokens/sec refill, `capacity` burst.

    Used for per-host request rates (tokens = requests) and for LLM TPM budgets
    (tokens = estimated prompt tokens). Supports AIMD adaptation: `penalize()` on
    429 halves the rate, `reward()` slowly restores it.
    """

    def __init__(self, rate: float, capacity: float | None = None) -> None:
        self.max_rate = rate
        self.rate = rate
        self.capacity = capacity if capacity is not None else max(1.0, rate)
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
        self._last = now

    async def acquire(self, tokens: float = 1.0) -> None:
        tokens = min(tokens, self.capacity)  # oversize requests are the chunker's job
        async with self._lock:
            while True:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                await asyncio.sleep((tokens - self._tokens) / self.rate)

    def penalize(self, factor: float = 0.5, floor: float = 0.05) -> None:
        self.rate = max(floor, self.rate * factor)

    def reward(self, step: float = 0.05) -> None:
        self.rate = min(self.max_rate, self.rate + self.max_rate * step)
