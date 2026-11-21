"""Command-line entry point.

  python -m graphone phase1 [--startups N --products N --papers N] [--only papers,...] [--shard k/N]
  python -m graphone phase2                # news + jobs (run it on a schedule, e.g. hourly)
  python -m graphone export [--sheets]     # CSV / JSONL / XLSX (+ Google Sheets)
  python -m graphone all [--sheets]
  python -m graphone llm-check             # ping every configured LLM tier
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from contextlib import AsyncExitStack

from .config import get_settings
from .export.exporter import gather_tables, upload_sheets, write_files, write_jsonl
from .http.fetcher import Fetcher
from .llm.orchestrator import LLMOrchestrator
from .llm.providers import build_providers
from .llm.tasks import PRICING_TASK
from .logging_setup import METRICS, log, setup_logging
from .resolver.resolver import EntityResolver
from .sources.base import Context, Shard
from .state.store import make_store

logger = logging.getLogger("graphone")

HOST_RPS = {  # per-host politeness overrides (requests/second)
    "export.arxiv.org": 0.34,            # arXiv asks for 1 request / 3 s
    "www.ycombinator.com": 4,
    "hn.algolia.com": 4,
    "huggingface.co": 5,
    "api.github.com": 5,
    "remoteok.com": 0.5,
    "remotive.com": 0.5,
}


async def _with_context(fn, args):
    s = get_settings()
    store = make_store(s)
    resolver = EntityResolver(s.data_dir / "seed_entities.json")
    async with AsyncExitStack() as stack:
        fetcher = await stack.enter_async_context(Fetcher(s, HOST_RPS))
        llm = await stack.enter_async_context(LLMOrchestrator(build_providers(s), store, s.llm_concurrency))
        ctx = Context(s, fetcher, store, llm, resolver, Shard.parse(getattr(args, "shard", None)))
        log(logger, logging.INFO, "run start", llm_tiers=[p.label for p in llm.providers] or "none (rules only)",
            state=s.state_backend, shard=f"{ctx.shard.index}/{ctx.shard.total}")
        try:
            return await fn(ctx, args)
        finally:
            await store.add_mappings(resolver.mapping_rows())
            summary = METRICS.snapshot()
            (s.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
            log(logger, logging.INFO, "run summary written", path=str(s.output_dir / "run_summary.json"))
            await store.close()


async def _phase1(ctx: Context, args) -> None:
    from .sources.papers import collect_papers
    from .sources.products import collect_products
    from .sources.startups import collect_startups

    s, out = ctx.settings, ctx.settings.output_dir
    only = set((args.only or "startups,products,papers").split(","))
    suffix = f".shard{ctx.shard.index}" if ctx.shard.total > 1 else ""
    jobs = []
    if "papers" in only:
        jobs.append(("papers", collect_papers(ctx, args.papers or s.target_papers)))
    if "startups" in only:
        jobs.append(("startups", collect_startups(ctx, args.startups or s.target_startups)))
    if "products" in only:
        jobs.append(("products", collect_products(ctx, args.products or s.target_products)))
    async def run_and_write(name, coro) -> None:
        # Write each vertical the moment it finishes, so a slow/failed vertical
        # never throws away the others' results.
        try:
            res = await coro
        except Exception as exc:
            log(logger, logging.ERROR, f"{name} collection failed", error=repr(exc)[:300])
            return
        write_jsonl(out / f"{name}{suffix}.jsonl", [r.model_dump(mode="json") for r in res])
        log(logger, logging.INFO, f"{name} written", count=len(res))

    await asyncio.gather(*(run_and_write(name, coro) for name, coro in jobs))


async def _phase2(ctx: Context, args) -> None:
    from .sources.jobs import collect_jobs
    from .sources.news import collect_news

    news, jobs = await asyncio.gather(collect_news(ctx), collect_jobs(ctx), return_exceptions=True)
    for name, res in (("news", news), ("jobs", jobs)):
        if isinstance(res, Exception):
            log(logger, logging.ERROR, f"{name} failed", error=repr(res)[:300])
        else:
            log(logger, logging.INFO, f"{name}: new fresh records this run", count=len(res))


async def _export(args) -> None:
    s = get_settings()
    store = make_store(s)
    tables = await gather_tables(store, s.output_dir, s.freshness_hours)
    write_files(tables, s.output_dir)
    if args.sheets:
        if not s.google_sa_file:
            log(logger, logging.ERROR, "GOOGLE_SERVICE_ACCOUNT_FILE not set; skipping Sheets upload")
        else:
            url = upload_sheets(tables, s.google_sa_file, s.google_sheet_id)
            print(f"\nPublic Google Sheet: {url}\n")
    await store.close()


async def _llm_check(ctx: Context, args) -> None:
    sample = "Acme Cloud pricing. Hobby: Free forever, 1 project. Pro: $20 per month. Enterprise: contact sales."
    for p in ctx.llm.providers:
        single = LLMOrchestrator([p], None, 1)
        single._session = ctx.llm._session
        try:
            res = await single.extract(PRICING_TASK, sample)
            print(f"  OK   {p.label:45} -> {res.data}")
        except Exception as exc:
            print(f"  FAIL {p.label:45} -> {exc}")
    if not ctx.llm.providers:
        print("  no LLM keys configured (.env) — pipeline will run in rules-only mode")


async def _reextract(ctx: Context, args) -> None:
    from .sources.reextract import reextract_news
    n, total = await reextract_news(ctx)
    print(f"  news re-extracted with LLM: {n}/{total} pending records")


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    ap = argparse.ArgumentParser(prog="graphone")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("phase1", help="bulk startups/products/papers")
    for a in ("startups", "products", "papers"):
        p1.add_argument(f"--{a}", type=int)
    p1.add_argument("--only", help="comma list: startups,products,papers")
    p1.add_argument("--shard", help="k/N partition for multi-node runs")
    p2 = sub.add_parser("phase2", help="fresh news + jobs")
    p2.add_argument("--shard")
    ex = sub.add_parser("export")
    ex.add_argument("--sheets", action="store_true")
    al = sub.add_parser("all")
    al.add_argument("--sheets", action="store_true")
    al.add_argument("--only")
    al.add_argument("--shard")
    for a in ("startups", "products", "papers"):
        al.add_argument(f"--{a}", type=int)
    sub.add_parser("llm-check")
    sub.add_parser("reextract", help="re-run LLM extraction on records saved while all tiers were out of quota")
    args = ap.parse_args(argv)

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    if args.cmd == "phase1":
        asyncio.run(_with_context(_phase1, args))
    elif args.cmd == "phase2":
        asyncio.run(_with_context(_phase2, args))
    elif args.cmd == "export":
        asyncio.run(_export(args))
    elif args.cmd == "reextract":
        asyncio.run(_with_context(_reextract, args))
    elif args.cmd == "llm-check":
        asyncio.run(_with_context(_llm_check, args))
    elif args.cmd == "all":
        asyncio.run(_with_context(_phase1, args))
        asyncio.run(_with_context(_phase2, args))
        asyncio.run(_export(args))


if __name__ == "__main__":
    main()
