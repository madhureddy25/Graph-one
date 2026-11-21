"""End-to-end news ingestion against a local fake site (feed + articles)."""
from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime, timedelta, timezone
from pathlib import Path

from aiohttp import web

from graphone.config import get_settings
from graphone.http.fetcher import Fetcher
from graphone.llm.orchestrator import LLMOrchestrator
from graphone.resolver.resolver import EntityResolver
from graphone.sources.base import Context
from graphone.sources.news import collect_news
from graphone.state.store import SqliteStateStore

ROOT = Path(__file__).resolve().parents[1]
BODY = " ".join(f"Sentence {i} about Mistral AI and its new open model release for enterprises." for i in range(40))


def _article(jsonld_date: str | None, modified: str | None = None, body: str = BODY) -> str:
    ld = f'<script type="application/ld+json">{{"datePublished": "{jsonld_date}"}}</script>' if jsonld_date else ""
    mod = f'<meta property="article:modified_time" content="{modified}">' if modified else ""
    return f"<html><head>{ld}{mod}<title>t</title></head><body><article><p>{body}</p></article></body></html>"


def test_news_pipeline(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    iso = lambda d: d.strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    rfc = lambda d: d.strftime("%a, %d %b %Y %H:%M:%S GMT")  # noqa: E731

    async def main():
        app = web.Application()
        port_box = {}

        async def feed(_):
            b = f"http://127.0.0.1:{port_box['p']}"
            items = [
                ("undated", None),                                   # above a fresh item -> inferred fresh
                ("fresh", rfc(now - timedelta(hours=2))),
                ("old-edited", rfc(now - timedelta(hours=3))),       # feed lies; JSON-LD says 9 days old
                ("ancient", rfc(now - timedelta(days=5))),           # pre-filtered, never fetched
            ]
            xml = "".join(f"<item><title>{n}</title><link>{b}/a/{n}</link>"
                          + (f"<pubDate>{d}</pubDate>" if d else "") + "</item>" for n, d in items)
            return web.Response(text=f"<rss><channel>{xml}</channel></rss>", content_type="application/rss+xml")

        pages = {
            "undated": _article(None, body=BODY.replace("Mistral", "Cohere")),
            "fresh": _article(iso(now - timedelta(hours=2))),
            "old-edited": _article(iso(now - timedelta(days=9)), modified=iso(now), body=BODY.replace("open", "closed")),
            "ancient": _article(iso(now - timedelta(days=5))),
        }
        fetched = []

        async def art(req):
            fetched.append(req.match_info["n"])
            return web.Response(text=pages[req.match_info["n"]], content_type="text/html")

        app.router.add_get("/feed", feed)
        app.router.add_get("/a/{n}", art)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port_box["p"] = site._server.sockets[0].getsockname()[1]

        cfg_root = tmp_path / "root"
        (cfg_root / "config").mkdir(parents=True)
        (cfg_root / "config" / "sources.toml").write_text(
            f'[[news]]\nname = "Local"\nfeed = "http://127.0.0.1:{port_box["p"]}/feed"\n')
        s = dataclasses.replace(get_settings(), root=cfg_root, browser_fallback=False, respect_robots=False)
        store = SqliteStateStore(tmp_path / "state.db")
        try:
            async with Fetcher(s) as f:
                ctx = Context(s, f, store, LLMOrchestrator([], store), EntityResolver(ROOT / "data/seed_entities.json"))
                recs = await collect_news(ctx)
                titles = sorted(r.content.title for r in recs)
                assert titles == ["fresh", "undated"], titles
                by = {r.content.title: r for r in recs}
                assert by["fresh"].content.dateSource == "jsonld"
                assert by["undated"].content.dateSource == "listing_order"
                assert by["fresh"].content.word_count > 100
                assert "ancient" not in fetched                     # cheap pre-filter worked
                # second run: nothing is processed twice
                again = await collect_news(ctx)
                assert again == []
                assert len(await store.get_records("news")) == 2
        finally:
            await store.close()
            await runner.cleanup()

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    asyncio.run(main())
