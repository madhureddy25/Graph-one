"""Central, environment-driven configuration.

Everything that changes between a laptop run and a 50-node cluster is an
environment variable, so scaling is an infrastructure change, not a code change.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    return int(v) if v else default


def _float(name: str, default: float) -> float:
    v = os.getenv(name, "").strip()
    return float(v) if v else default


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name, "").strip().lower()
    if not v:
        return default
    return v in {"1", "true", "yes", "on"}


def _str(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, "").strip()
    return v or default


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    output_dir: Path
    state_path: Path

    state_backend: str
    redis_url: str | None

    http_concurrency: int
    per_host_concurrency: int
    per_host_rps: float
    request_timeout: float
    max_retries: int
    proxy_url: str | None
    browser_fallback: bool
    browser_concurrency: int
    respect_robots: bool
    user_agent: str

    llm_concurrency: int
    gemini_api_key: str | None
    gemini_model: str
    groq_api_key: str | None
    groq_model: str
    deepseek_api_key: str | None
    deepseek_model: str
    openrouter_api_key: str | None
    openrouter_model: str

    github_token: str | None
    product_hunt_token: str | None

    target_startups: int
    target_products: int
    target_papers: int
    freshness_hours: int

    google_sa_file: str | None
    google_sheet_id: str | None

    def sources(self) -> dict:
        """Source registry (config/sources.toml) — add a source without code changes."""
        with open(self.root / "config" / "sources.toml", "rb") as fh:
            return tomllib.load(fh)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    out = Path(_str("OUTPUT_DIR", str(ROOT / "output")))
    out.mkdir(parents=True, exist_ok=True)
    return Settings(
        root=ROOT,
        data_dir=ROOT / "data",
        output_dir=out,
        state_path=Path(_str("STATE_PATH", str(out / "state.sqlite3"))),
        state_backend=_str("STATE_BACKEND", "sqlite"),
        redis_url=_str("REDIS_URL"),
        http_concurrency=_int("HTTP_CONCURRENCY", 64),
        per_host_concurrency=_int("PER_HOST_CONCURRENCY", 4),
        per_host_rps=_float("PER_HOST_RPS", 2.0),
        request_timeout=_float("REQUEST_TIMEOUT", 25.0),
        max_retries=_int("MAX_RETRIES", 4),
        proxy_url=_str("PROXY_URL"),
        browser_fallback=_bool("BROWSER_FALLBACK", True),
        browser_concurrency=_int("BROWSER_CONCURRENCY", 4),
        respect_robots=_bool("RESPECT_ROBOTS", True),
        user_agent=_str(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
        ),
        llm_concurrency=_int("LLM_CONCURRENCY", 8),
        gemini_api_key=_str("GEMINI_API_KEY"),
        gemini_model=_str("GEMINI_MODEL", "gemini-2.5-flash"),
        groq_api_key=_str("GROQ_API_KEY"),
        groq_model=_str("GROQ_MODEL", "llama-3.3-70b-versatile"),
        deepseek_api_key=_str("DEEPSEEK_API_KEY"),
        deepseek_model=_str("DEEPSEEK_MODEL", "deepseek-chat"),
        openrouter_api_key=_str("OPENROUTER_API_KEY"),
        openrouter_model=_str("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        github_token=_str("GITHUB_TOKEN"),
        product_hunt_token=_str("PRODUCT_HUNT_TOKEN"),
        target_startups=_int("TARGET_STARTUPS", 1000),
        target_products=_int("TARGET_PRODUCTS", 1000),
        target_papers=_int("TARGET_PAPERS", 1000),
        freshness_hours=_int("FRESHNESS_HOURS", 24),
        google_sa_file=_str("GOOGLE_SERVICE_ACCOUNT_FILE"),
        google_sheet_id=_str("GOOGLE_SHEET_ID"),
    )
