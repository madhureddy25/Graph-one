"""Re-run entity resolution over every saved record, without re-crawling.

Used after a resolver fix: canonical names are baked into the exported
records and the mapping log, so they must be recomputed from the raw names.
A timestamped backup of the output files and state DB is written first.

Order is fixed (startups, products, jobs, news; each sorted by source URL),
so the result is deterministic.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..export.exporter import write_jsonl
from ..logging_setup import log
from .base import Context

logger = logging.getLogger("graphone.reresolve")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _backup(out: Path) -> Path:
    dest = out / f"backup_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"
    dest.mkdir()
    for f in [*out.glob("*.jsonl"), out / "state.sqlite3"]:
        if f.exists():
            shutil.copy2(f, dest / f.name)
    return dest


def _resolve_startups(ctx: Context, rows: list[dict]) -> list[dict]:
    out = []
    for r in sorted(rows, key=lambda r: r["source"]["url"]):
        data = r["content"]["data"]
        raw = data.get("rawName") or r["content"]["entityName"]
        res = ctx.resolver.resolve(raw, website=data.get("website"), record_type="STARTUP", source_url=r["source"]["url"])
        out.append({**r, "content": {**r["content"], "entityName": res.canonical}})
    return out


def _resolve_products(ctx: Context, rows: list[dict]) -> list[dict]:
    out = []
    for r in sorted(rows, key=lambda r: r["source"]["url"]):
        c = r["content"]
        res = ctx.resolver.resolve(c["productName"], website=c.get("website"), record_type="PRODUCT",
                                   source_url=r["source"]["url"])
        out.append({**r, "content": {**c, "startupName": res.canonical}})
    return out


async def _resolve_jobs(ctx: Context) -> int:
    rows = sorted(await ctx.store.get_keyed_records("jobs"), key=lambda x: x[2]["source"]["url"])
    for key, ts, r in rows:
        c = r["content"]
        canon = ctx.resolver.resolve(c.get("rawCompany") or c["company"], record_type="JOB",
                                     source_url=r["source"]["url"]).canonical
        await ctx.store.put_record("jobs", key, ts or None, {**r, "content": {**c, "company": canon}})
    return len(rows)


async def _resolve_news(ctx: Context, raw_by_url: dict[str, list[dict]]) -> int:
    rows = sorted(await ctx.store.get_keyed_records("news"), key=lambda x: x[2]["source"]["url"])
    for key, ts, r in rows:
        url = r["source"]["url"]
        renamed = {}
        for m in raw_by_url.get(url, []):
            renamed[m["canonical"]] = ctx.resolver.resolve(m["raw"], record_type="NEWS", source_url=url).canonical
        orgs: list[str] = []
        for o in r["content"].get("organizations") or []:
            name = renamed.get(o, o)
            if name not in orgs:
                orgs.append(name)
        await ctx.store.put_record("news", key, ts or None, {**r, "content": {**r["content"], "organizations": orgs}})
    return len(rows)


async def reresolve_all(ctx: Context) -> dict[str, int]:
    out = ctx.settings.output_dir
    backup = _backup(out)
    log(logger, logging.INFO, "backup written", path=str(backup))

    raw_by_url: dict[str, list[dict]] = {}
    for m in await ctx.store.get_raw_mappings("NEWS"):
        raw_by_url.setdefault(m["sourceUrl"], []).append(m)
    await ctx.store.clear_mappings()

    counts: dict[str, int] = {}
    for name, fn in (("startups", _resolve_startups), ("products", _resolve_products)):
        rows = _read_jsonl(out / f"{name}.jsonl")
        if rows:
            write_jsonl(out / f"{name}.jsonl", fn(ctx, rows))
        counts[name] = len(rows)
    counts["jobs"] = await _resolve_jobs(ctx)
    counts["news"] = await _resolve_news(ctx, raw_by_url)
    await ctx.store.add_mappings(ctx.resolver.mapping_rows())
    counts["mappings"] = len(ctx.resolver.mapping_rows())
    return counts
