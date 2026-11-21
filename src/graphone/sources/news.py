"""Phase II — AI news with full text and guaranteed 24h freshness.

Per source: feed -> (cheap pre-filter on feed date) -> distributed claim on the
canonical URL -> fetch article (browser tier on challenge) -> full-text
extraction -> multi-signal date extraction -> strict freshness gate ->
near-duplicate (SimHash/LSH) gate -> LLM entity extraction -> resolver -> store.

Undated content: an item is accepted only if (a) it sits above a fresh dated
item in the reverse-chronological feed, or (b) this source was crawled within
the freshness window and the URL was not present then (first-seen heuristic).
"""
from __future__ import annotations

import logging
from datetime import timedelta

import feedparser

from ..dates import DateResult, extract_published_date, infer_from_listing_order, is_fresh, parse_datetime
from ..http.fetcher import FetchError
from ..llm.orchestrator import AllProvidersFailed
from ..llm.tasks import NEWS_TASK
from ..logging_setup import METRICS, log
from ..models import NewsContent, NewsRecord, Source, to_iso
from ..text import canonical_url, hamming, html_to_text, sha1, simhash, simhash_bands
from .base import Context, bounded_map

logger = logging.getLogger("graphone.news")


async def _is_near_duplicate(ctx: Context, h: int) -> bool:
    bands = simhash_bands(h)
    for b in bands:
        prev = await ctx.store.kv_get(f"simband:{b}")
        if prev is not None and hamming(int(prev), h) <= 3:
            return True
    for b in bands:
        await ctx.store.kv_set(f"simband:{b}", str(h))
    return False


async def _process_entry(ctx: Context, src: dict, e: dict) -> NewsRecord | None:
    hours = ctx.settings.freshness_hours
    key = sha1(e["curl"])
    if await ctx.store.has_record("news", key):
        METRICS.inc("news.already_processed")
        return None
    if not await ctx.store.claim("news_inflight", key, ttl=3600):
        METRICS.inc("news.claimed_by_other_node")
        return None
    try:
        page = await ctx.fetcher.get(e["link"])
    except FetchError as exc:
        METRICS.inc("news.fetch_failed")
        log(logger, logging.WARNING, "article fetch failed", url=e["link"], err=str(exc)[:120])
        await ctx.store.release("news_inflight", key)  # let a later run retry
        return None

    dr = extract_published_date(page.text, e["link"], now=ctx.now, feed_date=e["feed_dt"])
    if e["feed_dt"] and dr.dt and dr.source != "feed" and abs((dr.dt - e["feed_dt"]).total_seconds()) > 36 * 3600:
        METRICS.inc("news.date_conflict")  # page wins; conflict logged for audit
        log(logger, logging.INFO, "feed/page date conflict", url=e["link"], page=dr.dt, feed=e["feed_dt"])

    fresh = is_fresh(dr, ctx.now, hours)
    if not fresh and dr.dt is None:
        if e["order_inferred"]:
            dr, fresh = DateResult(None, "listing_order", "inferred"), True
        elif e["first_seen"]:
            dr, fresh = DateResult(None, "first_seen_since_last_run", "inferred"), True
    if not fresh:
        METRICS.inc("news.stale")
        await ctx.store.release("news_inflight", key)
        return None

    text = html_to_text(page.text, e["link"])
    if len(text) < 200:
        METRICS.inc("news.no_fulltext")
        await ctx.store.release("news_inflight", key)
        return None
    if await _is_near_duplicate(ctx, simhash(text)):
        METRICS.inc("news.near_duplicate")
        return None

    orgs, event, method = [], None, "none"
    if ctx.llm.enabled:
        try:
            res = await ctx.llm.extract(NEWS_TASK, f"{e['title']}\n\n{text}")
            orgs, event, method = res.data["organizations"], res.data["event_type"], f"llm:{res.provider}"
        except AllProvidersFailed as exc:
            log(logger, logging.WARNING, "news LLM extraction failed", url=e["link"], err=str(exc)[:160])
    canon = []
    for o in orgs:
        r = ctx.resolver.resolve(o, record_type="NEWS", source_url=e["link"])
        if r.canonical not in canon:
            canon.append(r.canonical)

    rec = NewsRecord(
        source=Source(name=src["name"], url=e["link"]),
        content=NewsContent(
            title=e["title"], url=page.final_url or e["link"], published_date=to_iso(dr.dt), author=e.get("author"),
            full_text=text, word_count=len(text.split()), organizations=canon, event_type=event,
            dateSource=dr.source, dateConfidence=dr.confidence, extractionMethod=method,
        ),
    )
    await ctx.store.put_record("news", key, rec.content.published_date or rec.collectedAt, rec.model_dump(mode="json"))
    METRICS.inc("news.records")
    return rec


async def _collect_source(ctx: Context, src: dict) -> list[NewsRecord]:
    hours = ctx.settings.freshness_hours
    try:
        feed_res = await ctx.fetcher.get(src["feed"], accept="application/rss+xml, application/atom+xml, */*")
    except FetchError as exc:
        log(logger, logging.ERROR, "feed failed", source=src["name"], err=str(exc)[:160])
        METRICS.inc("news.feed_failed")
        return []
    parsed = feedparser.parse(feed_res.text)
    last_run = await ctx.store.kv_get(f"last_run:news:{src['name']}")
    recent_prev_run = bool(last_run) and (ctx.now - parse_datetime(last_run)) <= timedelta(hours=hours)

    entries = []
    for it in parsed.entries:
        link = it.get("link")
        if not link:
            continue
        feed_dt = parse_datetime(it.get("published") or it.get("created"), ctx.now)
        entries.append({"link": link, "curl": canonical_url(link), "title": " ".join((it.get("title") or "").split()),
                        "author": it.get("author"), "feed_dt": feed_dt})
    order = infer_from_listing_order(
        [DateResult(e["feed_dt"], "feed", "medium") if e["feed_dt"] else None for e in entries], ctx.now, hours)

    todo = []
    for e, inferred in zip(entries, order):
        # first-seen bookkeeping (independent of freshness)
        newly_seen = await ctx.store.claim(f"newsseen:{src['name']}", sha1(e["curl"]))
        e["first_seen"] = newly_seen and recent_prev_run
        e["order_inferred"] = inferred
        # cheap pre-filter: clearly-old feed dates never cost a page fetch (6h slack for bad feed TZs)
        if e["feed_dt"] and ctx.now - e["feed_dt"] > timedelta(hours=hours + 6):
            METRICS.inc("news.prefiltered_stale")
            continue
        if ctx.shard.owns(e["curl"]):
            todo.append(e)
    recs = await bounded_map(todo, lambda e: _process_entry(ctx, src, e), concurrency=8)
    await ctx.store.kv_set(f"last_run:news:{src['name']}", ctx.now.isoformat())
    log(logger, logging.INFO, "news source done", source=src["name"], feed_items=len(entries), candidates=len(todo),
        fresh_new=len(recs))
    return recs


async def collect_news(ctx: Context) -> list[NewsRecord]:
    out: list[NewsRecord] = []
    for batch in await bounded_map(ctx.settings.sources()["news"], lambda s: _collect_source(ctx, s), concurrency=5):
        out.extend(batch)
    return out
