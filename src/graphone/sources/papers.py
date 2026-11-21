"""Phase I — AI research papers with live GitHub metrics.

1. Hugging Face Papers (successor of Papers with Code) daily feeds, fetched
   concurrently day-by-day backwards, keeping papers that link a code repo.
2. arXiv API (batched id_list, 100 ids/request, polite 1 req/3 s) for the exact
   first-version publication timestamp.
3. GitHub GraphQL for *current* star counts (see github.py).
Checkpointing: processed days are recorded in the state store so an
interrupted run resumes instead of re-fetching.
"""
from __future__ import annotations

import logging
import re
from datetime import timedelta
from xml.etree import ElementTree as ET

from ..dates import parse_datetime
from ..http.fetcher import FetchError
from ..logging_setup import METRICS, log
from ..models import PaperContent, PaperRecord, Source, to_iso, utcnow_iso
from .base import Context, bounded_map
from .github import fetch_stars, parse_repo

logger = logging.getLogger("graphone.papers")
_ATOM = {"a": "http://www.w3.org/2005/Atom"}
_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")


async def _fetch_day(ctx: Context, api: str, day: str) -> list[dict]:
    try:
        data = await ctx.fetcher.get_json(api, params={"date": day, "limit": 100})
    except FetchError as exc:
        log(logger, logging.WARNING, "HF day failed", day=day, err=str(exc)[:120])
        return []
    METRICS.inc("papers.days_fetched")
    return data if isinstance(data, list) else []


async def arxiv_published(ctx: Context, ids: list[str]) -> dict[str, str]:
    """Exact <published> timestamps from the arXiv API (v1 submission time)."""
    out: dict[str, str] = {}
    ids = [i for i in ids if _ARXIV_ID.match(i)]
    for i in range(0, len(ids), 100):
        chunk = ids[i:i + 100]
        try:
            res = await ctx.fetcher.get("https://export.arxiv.org/api/query",
                                        params={"id_list": ",".join(chunk), "max_results": len(chunk)},
                                        accept="application/atom+xml", allow_browser=False, check_robots=False)
            root = ET.fromstring(res.text.encode())
        except (FetchError, ET.ParseError) as exc:
            log(logger, logging.WARNING, "arXiv batch failed; keeping HF dates", err=str(exc)[:120])
            continue
        for entry in root.findall("a:entry", _ATOM):
            eid = (entry.findtext("a:id", default="", namespaces=_ATOM) or "").rsplit("/", 1)[-1]
            eid = re.sub(r"v\d+$", "", eid)
            pub = entry.findtext("a:published", default=None, namespaces=_ATOM)
            if eid and pub:
                out[eid] = pub
                METRICS.inc("papers.arxiv_dates")
    return out


async def collect_papers(ctx: Context, target: int) -> list[PaperRecord]:
    cfg = ctx.settings.sources()["papers"]["huggingface"]
    api = cfg["api"]
    papers: dict[str, dict] = {}
    day = ctx.now.date()
    days_back, step = 0, 14
    while len(papers) < target * 1.05 and days_back < cfg.get("max_days_back", 365):
        days = [(day - timedelta(days=days_back + k)).isoformat() for k in range(step)]
        days = [d for d in days if ctx.shard.owns(d)]
        batches = await bounded_map(days, lambda d: _fetch_day(ctx, api, d), concurrency=6)
        for items in batches:
            for it in items:
                p = it.get("paper") or {}
                pid = p.get("id")
                if not pid or pid in papers or not parse_repo(p.get("githubRepo")):
                    continue
                papers[pid] = p
        days_back += step
        log(logger, logging.INFO, "HF papers progress", days_back=days_back, with_code=len(papers))

    ids = sorted(papers)
    arxiv_dates = await arxiv_published(ctx, ids)
    repos = {pid: parse_repo(papers[pid].get("githubRepo")) for pid in ids}
    stars = await fetch_stars(ctx, [r for r in repos.values() if r])
    checked = utcnow_iso()

    records: list[PaperRecord] = []
    for pid in ids:
        p = papers[pid]
        o, n = repos[pid]
        live = stars.get(f"{o}/{n}".lower())
        if live and live.get("stars") is not None:
            star_count, star_src, gh_url = live["stars"], live["source"], live["url"]
        elif p.get("githubStars") is not None:
            star_count, star_src, gh_url = p["githubStars"], "huggingface_snapshot", f"https://github.com/{o}/{n}"
        else:
            METRICS.inc("papers.no_star_data")
            continue
        published = arxiv_dates.get(pid) or p.get("publishedAt")
        records.append(PaperRecord(
            source=Source(name=cfg["name"], url=f"https://huggingface.co/papers/{pid}"),
            content=PaperContent(
                title=" ".join((p.get("title") or "").split()),
                authors=[a["name"] for a in p.get("authors", []) if a.get("name")],
                paper_url=f"https://arxiv.org/abs/{pid}" if _ARXIV_ID.match(pid) else f"https://huggingface.co/papers/{pid}",
                github_url=gh_url, github_stars=star_count, published_date=to_iso(parse_datetime(published)),
                arxiv_id=pid, github_stars_source=star_src, github_stars_checked_at=checked,
            ),
        ))
    records.sort(key=lambda r: r.content.published_date or "", reverse=True)
    records = records[:target]
    METRICS.inc("papers.records", len(records))
    return records
