"""LLM provider adapters with normalised error taxonomy.

Each adapter maps its vendor's error shapes onto LLMError.kind:
  rate_limit         429 (per-minute RPM/TPM)          -> backoff + jitter, same tier
  quota              429 with daily/quota exhaustion   -> open circuit, next tier
  payload_too_large  413 (Groq returns this when one request > TPM limit)
  context_length     400 'maximum context length' etc. -> shrink payload, same tier
  auth               401/403                           -> disable tier for the run
  server             5xx / network                     -> backoff, then next tier
  bad_response       unparsable / empty / safety block -> next tier
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

import aiohttp

from ..config import Settings
from ..http.backoff import TokenBucket, parse_retry_after


class LLMError(Exception):
    def __init__(self, provider: str, kind: str, status: int | None = None, retry_after: float | None = None,
                 message: str = "") -> None:
        super().__init__(f"[{provider}] {kind} ({status}): {message[:200]}")
        self.provider, self.kind, self.status, self.retry_after, self.message = provider, kind, status, retry_after, message


_CTX = re.compile(r"(context.length|maximum context|too many tokens|exceeds the maximum|input is too long|"
                  r"reduce the length|prompt is too long|request too large)", re.I)
_QUOTA = re.compile(r"(per.day|daily|quota|RESOURCE_EXHAUSTED.*(day|quota)|tokens per day|TPD|RPD|insufficient.balance|"
                    r"billing)", re.I)


def classify_http_error(provider: str, status: int, body: str, headers: dict) -> LLMError:
    ra = parse_retry_after(headers.get("Retry-After") or headers.get("retry-after"), body)
    if status == 413:
        return LLMError(provider, "payload_too_large", status, ra, body)
    if status == 429:
        if _QUOTA.search(body) and (ra is None or ra > 120):
            return LLMError(provider, "quota", status, ra, body)
        return LLMError(provider, "rate_limit", status, ra, body)
    if status == 402:
        return LLMError(provider, "quota", status, None, body)
    if status in (400, 422) and _CTX.search(body):
        return LLMError(provider, "context_length", status, None, body)
    if status in (401, 403):
        return LLMError(provider, "auth", status, None, body)
    if status >= 500:
        return LLMError(provider, "server", status, ra, body)
    return LLMError(provider, "bad_request", status, None, body)


@dataclass
class Provider:
    name: str
    model: str
    api_key: str
    max_input_tokens: int      # our *budget*, kept well under the real window / TPM
    rpm: int
    tpm: int
    max_output_tokens: int = 1024
    rpm_bucket: TokenBucket = field(init=False)
    tpm_bucket: TokenBucket = field(init=False)

    def __post_init__(self) -> None:
        self.rpm_bucket = TokenBucket(self.rpm / 60.0, capacity=max(1, self.rpm // 6))
        self.tpm_bucket = TokenBucket(self.tpm / 60.0, capacity=self.tpm)

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}"

    async def complete(self, session: aiohttp.ClientSession, system: str, user: str) -> str:
        raise NotImplementedError


@dataclass
class OpenAICompatProvider(Provider):
    base_url: str = ""
    json_mode: bool = True

    async def complete(self, session: aiohttp.ClientSession, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        if "gpt-oss" in self.model and self.name == "groq":
            # reasoning models otherwise spend max_tokens on hidden reasoning and emit empty JSON
            body["reasoning_effort"] = "low"
        try:
            async with session.post(
                f"{self.base_url}/chat/completions", json=body,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=90),
            ) as r:
                text = await r.text()
                if r.status != 200:
                    raise classify_http_error(self.name, r.status, text, dict(r.headers))
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise LLMError(self.name, "server", None, None, type(exc).__name__) from exc
        try:
            data = json.loads(text)
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            raise LLMError(self.name, "bad_response", 200, None, text[:300]) from exc
        if not content:
            raise LLMError(self.name, "bad_response", 200, None, "empty content")
        return content


@dataclass
class GeminiProvider(Provider):
    async def complete(self, session: aiohttp.ClientSession, system: str, user: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": self.max_output_tokens,
                                 "responseMimeType": "application/json"},
        }
        try:
            async with session.post(url, json=body, headers={"x-goog-api-key": self.api_key},
                                    timeout=aiohttp.ClientTimeout(total=90)) as r:
                text = await r.text()
                if r.status != 200:
                    raise classify_http_error(self.name, r.status, text, dict(r.headers))
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise LLMError(self.name, "server", None, None, type(exc).__name__) from exc
        try:
            data = json.loads(text)
            cand = data["candidates"][0]
            if cand.get("finishReason") in {"SAFETY", "RECITATION", "PROHIBITED_CONTENT"}:
                raise LLMError(self.name, "bad_response", 200, None, f"finishReason={cand.get('finishReason')}")
            return "".join(p.get("text", "") for p in cand["content"]["parts"])
        except (KeyError, IndexError, json.JSONDecodeError, TypeError) as exc:
            raise LLMError(self.name, "bad_response", 200, None, text[:300]) from exc


def build_providers(s: Settings) -> list[Provider]:
    """Tier order = cost/latency order. Budgets sized to free-tier limits:
    Groq's free tier rejects any single request above its TPM with HTTP 413, so
    its budget is the tightest; the chunker sizes payloads per tier."""
    tiers: list[Provider] = []
    if s.gemini_api_key:
        tiers.append(GeminiProvider("gemini", s.gemini_model, s.gemini_api_key,
                                    max_input_tokens=24000, rpm=10, tpm=250_000))
    if s.groq_api_key:
        tiers.append(OpenAICompatProvider("groq", s.groq_model, s.groq_api_key,
                                          max_input_tokens=4500, rpm=30, tpm=12_000,
                                          base_url="https://api.groq.com/openai/v1"))
    if s.deepseek_api_key:
        tiers.append(OpenAICompatProvider("deepseek", s.deepseek_model, s.deepseek_api_key,
                                          max_input_tokens=24000, rpm=60, tpm=1_000_000,
                                          base_url="https://api.deepseek.com/v1"))
    if s.openrouter_api_key:
        tiers.append(OpenAICompatProvider("openrouter", s.openrouter_model, s.openrouter_api_key,
                                          max_input_tokens=8000, rpm=20, tpm=200_000,
                                          base_url="https://openrouter.ai/api/v1"))
    return tiers
