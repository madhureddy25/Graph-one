"""Backfill LLM extraction for news records stored while every LLM tier was
out of quota (extractionMethod == "none").

Dedup claims mean the normal Phase II run never revisits an article, so this
is the only path that upgrades those records. The stored full_text is reused:
no page is re-fetched and freshness/dates are untouched.
"""
from __future__ import annotations

import logging

from ..llm.orchestrator import AllProvidersFailed
from ..llm.tasks import NEWS_TASK
from ..logging_setup import METRICS, log
from .base import Context

logger = logging.getLogger("graphone.reextract")


def _pending(data: dict) -> bool:
    return (data.get("content") or {}).get("extractionMethod") == "none"


def _resolve_orgs(ctx: Context, orgs: list[str], url: str) -> list[str]:
    canon: list[str] = []
    for o in orgs:
        name = ctx.resolver.resolve(o, record_type="NEWS", source_url=url).canonical
        if name not in canon:
            canon.append(name)
    return canon


async def reextract_news(ctx: Context) -> tuple[int, int]:
    """Return (upgraded, pending) counts."""
    if not ctx.llm.enabled:
        log(logger, logging.ERROR, "no LLM tier configured; nothing to do")
        return 0, 0
    rows = [(k, ts, d) for k, ts, d in await ctx.store.get_keyed_records("news") if _pending(d)]
    upgraded = 0
    for key, ts, data in rows:
        c = data["content"]
        try:
            res = await ctx.llm.extract(NEWS_TASK, f"{c['title']}\n\n{c['full_text']}")
        except AllProvidersFailed as exc:
            log(logger, logging.WARNING, "still no LLM capacity; stopping", err=str(exc)[:160])
            break
        new_content = {
            **c,
            "organizations": _resolve_orgs(ctx, res.data["organizations"], data["source"]["url"]),
            "event_type": res.data["event_type"],
            "extractionMethod": f"llm:{res.provider}",
        }
        await ctx.store.put_record("news", key, ts or None, {**data, "content": new_content})
        METRICS.inc("news.reextracted")
        upgraded += 1
    return upgraded, len(rows)
