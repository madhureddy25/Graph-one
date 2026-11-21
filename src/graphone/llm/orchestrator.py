"""Multi-tier LLM orchestrator: fallback chain, budgets, backoff, circuit breakers.

Per request, for each tier in order (e.g. Gemini Flash -> Groq Llama 3 -> DeepSeek):
  1. skip tier if its circuit breaker is open
  2. size the payload to the tier's token budget with semantic truncation
     (so 413s are prevented up-front, not just handled)
  3. wait on the tier's RPM and TPM token buckets (shared by all coroutines)
  4. call; on
       413 / context_length -> halve the budget and retry same tier (no sleep)
       429 rate_limit       -> sleep Retry-After or exp. backoff + jitter, retry;
                               if the wait is too long, open breaker -> next tier
       quota / auth         -> open breaker for the run -> next tier
       5xx / network        -> backoff, retry, then next tier
       invalid JSON/schema  -> next tier
  5. validate (schema + enum + grounding); cache result by content hash
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass

import aiohttp

from ..http.backoff import backoff_delay
from ..logging_setup import METRICS, log
from ..text import sha1
from .chunking import estimate_tokens, fit_to_budget
from .providers import LLMError, Provider
from .tasks import ExtractionTask

logger = logging.getLogger("graphone.llm")


class AllProvidersFailed(Exception):
    pass


@dataclass
class ExtractionResult:
    data: dict
    provider: str
    attempts: int
    truncated: bool
    cached: bool = False


class CircuitBreaker:
    def __init__(self, threshold: int = 5) -> None:
        self.threshold = threshold
        self.failures = 0
        self.open_until = 0.0

    @property
    def is_open(self) -> bool:
        return time.monotonic() < self.open_until

    def trip(self, seconds: float) -> None:
        self.open_until = time.monotonic() + seconds

    def failure(self, cooldown: float = 60.0) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.trip(cooldown)
            self.failures = 0

    def success(self) -> None:
        self.failures = 0


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def parse_json_object(raw: str) -> dict:
    s = _FENCE.sub("", raw.strip())
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)  # tolerate prose around the object
        if not m:
            raise
        obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("expected JSON object")
    return obj


class LLMOrchestrator:
    def __init__(self, providers: list[Provider], store=None, concurrency: int = 8,
                 max_attempts_per_tier: int = 4, max_rate_limit_wait: float = 45.0,
                 sleep=asyncio.sleep) -> None:
        self.providers = providers
        self.store = store
        self.breakers = {p.name: CircuitBreaker() for p in providers}
        self.max_attempts = max_attempts_per_tier
        self.max_wait = max_rate_limit_wait
        self._sem = asyncio.Semaphore(concurrency)
        self._sleep = sleep
        self._session: aiohttp.ClientSession | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    async def __aenter__(self):
        self._session = aiohttp.ClientSession(trust_env=True)
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    async def extract(self, task: ExtractionTask, text: str) -> ExtractionResult:
        cache_key = f"llm:{task.name}:{task.version}:{sha1(text)}"
        if self.store is not None:
            hit = await self.store.kv_get(cache_key)
            if hit:
                METRICS.inc("llm.cache_hit")
                return ExtractionResult(hit["data"], hit["provider"], 0, hit.get("truncated", False), cached=True)
        async with self._sem:
            res = await self._extract_uncached(task, text)
        if self.store is not None:
            await self.store.kv_set(cache_key, {"data": res.data, "provider": res.provider, "truncated": res.truncated})
        return res

    async def _extract_uncached(self, task: ExtractionTask, text: str) -> ExtractionResult:
        total_attempts = 0
        errors: list[str] = []
        overhead = estimate_tokens(task.system) + estimate_tokens(task.render(""))
        for p in self.providers:
            br = self.breakers[p.name]
            if br.is_open:
                METRICS.inc(f"llm.{p.name}.skipped_open_circuit")
                continue
            shrink = 1.0
            for attempt in range(self.max_attempts):
                total_attempts += 1
                budget = max(256, int((p.max_input_tokens - overhead) * shrink))
                payload, truncated = fit_to_budget(text, budget, list(task.query_terms))
                prompt = task.render(payload)
                est = estimate_tokens(task.system) + estimate_tokens(prompt) + p.max_output_tokens
                await p.rpm_bucket.acquire(1)
                await p.tpm_bucket.acquire(est)
                METRICS.inc(f"llm.{p.name}.calls")
                try:
                    raw = await p.complete(self._session, task.system, prompt)
                except LLMError as e:
                    METRICS.inc(f"llm.{p.name}.{e.kind}")
                    errors.append(str(e))
                    if e.kind in ("payload_too_large", "context_length"):
                        shrink *= 0.5
                        log(logger, logging.INFO, "payload too large; shrinking", provider=p.name,
                            new_budget=int((p.max_input_tokens - overhead) * shrink))
                        if shrink < 0.1:
                            break
                        continue
                    if e.kind == "rate_limit":
                        p.tpm_bucket.penalize(0.7)
                        wait = e.retry_after if e.retry_after is not None else backoff_delay(attempt, base=2.0, cap=40)
                        wait += backoff_delay(0, base=1.0)  # jitter on top of server hint
                        if wait > self.max_wait:
                            br.trip(wait)
                            log(logger, logging.WARNING, "rate limit wait too long; failing over",
                                provider=p.name, wait=round(wait, 1))
                            break
                        await self._sleep(wait)
                        continue
                    if e.kind in ("quota", "auth"):
                        br.trip(3600 if e.kind == "quota" else 10**9)
                        log(logger, logging.WARNING, "tier disabled", provider=p.name, reason=e.kind)
                        break
                    if e.kind == "server":
                        br.failure()
                        await self._sleep(backoff_delay(attempt, base=1.0, cap=20))
                        if attempt >= 1:
                            break
                        continue
                    break  # bad_request / bad_response -> next tier
                try:
                    obj = parse_json_object(raw)
                    data = task.validate(obj, text)  # ground against the FULL original text
                except (ValueError, json.JSONDecodeError) as exc:
                    METRICS.inc(f"llm.{p.name}.invalid_output")
                    errors.append(f"[{p.name}] invalid output: {exc}")
                    break
                br.success()
                p.tpm_bucket.reward()
                METRICS.inc(f"llm.{p.name}.success")
                if truncated:
                    METRICS.inc("llm.truncated_payloads")
                return ExtractionResult(data, p.label, total_attempts, truncated)
            METRICS.inc(f"llm.{p.name}.failover")
        METRICS.inc("llm.all_failed")
        raise AllProvidersFailed("; ".join(errors[-4:]) or "no providers configured")
