"""Phase I — startups from the Y Combinator directory.

1. Bulk-load the directory index (one request, ~5k companies) -> AI subset.
2. Concurrently verify each company against its live ycombinator.com page and
   read the current team size from the page's embedded JSON (Inertia data-page)
   — the record's source.url is the page we actually fetched.
"""
from __future__ import annotations

import html as htmllib
import json
import logging
import re

from bs4 import BeautifulSoup

from ..http.fetcher import FetchError, NotFound
from ..logging_setup import METRICS, log
from ..models import Source, StartupContent, StartupData, StartupRecord
from .base import Context, bounded_map

logger = logging.getLogger("graphone.startups")

_AI_TAGS = {"artificial intelligence", "generative ai", "ai", "machine learning", "deep learning", "nlp",
            "computer vision", "aiops", "ai assistant", "conversational ai", "ai-enhanced learning", "llm",
            "ai agents", "agents", "robotics", "data science", "mlops"}
_AI_TEXT = re.compile(r"\b(AI|A\.I\.|LLMs?|GPT|machine learning|agents?|agentic|generative|neural|copilot)\b", re.I)


def is_ai_company(c: dict) -> bool:
    tags = {t.lower() for t in (c.get("tags") or []) + (c.get("industries") or [])}
    if tags & _AI_TAGS:
        return True
    return bool(_AI_TEXT.search(f"{c.get('one_liner') or ''} {c.get('long_description') or ''}"[:600]))


def _find_key(obj, key: str):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def parse_yc_page(html: str) -> dict:
    """Extract name/team_size/website from a live YC company page."""
    out: dict = {}
    soup = BeautifulSoup(html, "lxml")
    node = soup.find(attrs={"data-page": True})
    if node:
        try:
            data = json.loads(htmllib.unescape(node["data-page"]))
            company = _find_key(data, "company") or {}
            out["team_size"] = company.get("team_size") if isinstance(company, dict) else None
            out["name"] = company.get("name") if isinstance(company, dict) else None
            out["website"] = company.get("website") if isinstance(company, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass
    if out.get("team_size") is None:
        m = re.search(r"Team Size:\s*</?[^>]*>?\s*(?:<[^>]+>\s*)*([\d,]+)", html)
        if m:
            out["team_size"] = int(m.group(1).replace(",", ""))
    return out


async def collect_startups(ctx: Context, target: int) -> list[StartupRecord]:
    cfg = ctx.settings.sources()["startups"]["yc"]
    index = await ctx.fetcher.get_json(cfg["index_url"])
    METRICS.inc("startups.index_size", len(index))
    ai = [c for c in index if is_ai_company(c) and c.get("slug") and ctx.shard.owns(c["slug"])]
    ai.sort(key=lambda c: (-(c.get("launched_at") or 0), c["slug"]))  # newest first, deterministic
    log(logger, logging.INFO, "YC index loaded", total=len(index), ai_companies=len(ai))

    verify = cfg.get("verify_live_pages", True)
    candidates = ai[: int(target * 1.15) + 20]
    raw_rows: list[dict] = []

    async def enrich(c: dict) -> dict | None:
        page_url = c.get("url") or f"https://www.ycombinator.com/companies/{c['slug']}"
        row = {"c": c, "url": page_url, "live": None}
        if verify:
            try:
                res = await ctx.fetcher.get(page_url)
                row["live"] = parse_yc_page(res.text)
                METRICS.inc("startups.verified_live")
            except NotFound:
                METRICS.inc("startups.dropped_404")
                return None  # record would not trace to a valid URL
            except FetchError as exc:
                METRICS.inc("startups.verify_failed")
                log(logger, logging.DEBUG, "live verify failed; using index values", url=page_url, err=str(exc))
        return row

    raw_rows = await bounded_map(candidates, enrich, concurrency=24)

    # Resolve names sequentially in a sorted order -> deterministic canonical names.
    raw_rows.sort(key=lambda r: r["c"]["slug"])
    records: list[StartupRecord] = []
    seen: set[str] = set()
    for r in raw_rows:
        c, live = r["c"], r["live"] or {}
        raw_name = live.get("name") or c["name"]
        website = live.get("website") or c.get("website")
        res = ctx.resolver.resolve(raw_name, website=website, record_type="STARTUP", source_url=r["url"])
        if res.canonical in seen:
            METRICS.inc("startups.duplicate_entity")
            continue
        seen.add(res.canonical)
        team = live.get("team_size") if live.get("team_size") is not None else c.get("team_size")
        records.append(StartupRecord(
            source=Source(name=cfg["name"], url=r["url"]),
            content=StartupContent(entityName=res.canonical, data=StartupData(
                employeeCount=int(team) if isinstance(team, (int, float)) and team > 0 else None,
                website=website, description=c.get("one_liner"), batch=c.get("batch"),
                location=c.get("all_locations"), tags=c.get("tags") or [], rawName=raw_name,
                employeeCountSource=("live_page" if live.get("team_size") is not None else
                                     "directory_index" if team else None),
            )),
        ))
    records = records[:target]
    METRICS.inc("startups.records", len(records))
    return records
