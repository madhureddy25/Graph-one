"""Phase I — AI products.

Discovery:  Hacker News 'Show HN' launches via the public Algolia API (time-window
            pagination walks past Algolia's 1,000-hit cap), plus Product Hunt's
            GraphQL API when PRODUCT_HUNT_TOKEN is set.
Enrichment: crawl each product's own website + /pricing page, classify the
            pricing model with deterministic rules, fall back to the LLM chain,
            and store the verbatim evidence quote + the URL it came from.
"""
from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from ..http.fetcher import FetchError
from ..llm.orchestrator import AllProvidersFailed
from ..llm.tasks import PRICING_TASK
from ..logging_setup import METRICS, log
from ..models import PricingModel, ProductContent, ProductRecord, Source, to_iso
from ..dates import parse_datetime
from ..text import domain_of, page_text
from .base import Context, bounded_map

logger = logging.getLogger("graphone.products")

_ENRICH_TIMEOUT_S = 120  # hard ceiling per product (home + pricing page + LLM)

_SKIP_DOMAINS = ("github.com", "gitlab.com", "huggingface.co", "arxiv.org", "youtube.com", "youtu.be",
                 "twitter.com", "x.com", "medium.com", "substack.com", "news.ycombinator.com", "apps.apple.com",
                 "play.google.com", "chromewebstore.google.com", "chrome.google.com", "docs.google.com",
                 "reddit.com", "linkedin.com", "loom.com", "producthunt.com", "npmjs.com", "pypi.org",
                 "marketplace.visualstudio.com", "addons.mozilla.org", "discord.gg", "colab.research.google.com")


def product_name_from_title(title: str) -> str | None:
    t = re.sub(r"^\s*(show|launch)\s+hn\s*:\s*", "", title, flags=re.I).strip()
    name = re.split(r"\s+[–—-]\s+|:\s+|,\s+|\s+\|\s+|\s+\(", t, maxsplit=1)[0].strip(" .!\"'")
    if not name or len(name) > 40 or len(name.split()) > 4:
        return None
    if re.match(r"^(i|we|my|our|a|an|the|this|how|why|what|built|made)\b", name, re.I):
        return None
    return name


def tagline_from_title(title: str) -> str:
    t = re.sub(r"^\s*(show|launch)\s+hn\s*:\s*", "", title, flags=re.I).strip()
    parts = re.split(r"\s+[–—-]\s+|:\s+|,\s+|\s+\|\s+", t, maxsplit=1)
    return (parts[1] if len(parts) > 1 else t)[:200]


def _skip(url: str | None) -> bool:
    d = domain_of(url)
    return not d or any(d == s or d.endswith("." + s) for s in _SKIP_DOMAINS)


# ------------------------------------------------------------------ discovery
async def discover_show_hn(ctx: Context, want: int) -> list[dict]:
    cfg = ctx.settings.sources()["products"]["show_hn"]
    found: dict[str, dict] = {}
    for q in cfg["queries"]:
        before = int(ctx.now.timestamp())
        for _ in range(15):  # time-window pagination
            if len(found) >= want:
                break
            data = await ctx.fetcher.get_json(cfg["api"], params={
                "query": q, "tags": "show_hn", "hitsPerPage": 1000,
                "numericFilters": f"points>={cfg.get('min_points', 3)},created_at_i<{before}",
            })
            hits = data.get("hits", [])
            if not hits:
                break
            for h in hits:
                url, title = h.get("url"), h.get("title") or ""
                name = product_name_from_title(title)
                d = domain_of(url)
                if not name or _skip(url) or not d or d in found or not ctx.shard.owns(d):
                    continue
                found[d] = {"name": name, "website": url, "tagline": tagline_from_title(title),
                            "source_name": cfg["name"],
                            "source_url": f"https://news.ycombinator.com/item?id={h['objectID']}",
                            "launched": h.get("created_at")}
            before = min(h["created_at_i"] for h in hits)
        if len(found) >= want:
            break
    METRICS.inc("products.discovered.show_hn", len(found))
    return list(found.values())


_PH_QUERY = """query($after:String,$topic:String){posts(topic:$topic,order:VOTES,first:20,after:$after){
 pageInfo{endCursor hasNextPage} edges{node{name tagline url website createdAt}}}}"""


async def discover_product_hunt(ctx: Context, want: int) -> list[dict]:
    token = ctx.settings.product_hunt_token
    if not token:
        return []
    cfg = ctx.settings.sources()["products"]["product_hunt"]
    out, after = [], None
    while len(out) < want:
        try:
            res = await ctx.fetcher.get("https://api.producthunt.com/v2/api/graphql", method="POST",
                                        json_body={"query": _PH_QUERY, "variables": {"after": after, "topic": cfg["topic"]}},
                                        headers={"Authorization": f"Bearer {token}"}, accept="application/json",
                                        allow_browser=False, check_robots=False)
        except FetchError as exc:
            log(logger, logging.WARNING, "product hunt stopped", err=str(exc))
            break
        posts = (res.json().get("data") or {}).get("posts") or {}
        for e in posts.get("edges", []):
            n = e["node"]
            out.append({"name": n["name"], "website": n.get("website"), "tagline": n.get("tagline"),
                        "source_name": cfg["name"], "source_url": n["url"], "launched": n.get("createdAt")})
        if not posts.get("pageInfo", {}).get("hasNextPage"):
            break
        after = posts["pageInfo"]["endCursor"]

    async def unwrap(p: dict) -> dict | None:
        # PH 'website' is a producthunt.com/r/... redirect: follow it to the real product domain
        if p.get("website") and domain_of(p["website"]) == "producthunt.com":
            try:
                p["website"] = (await ctx.fetcher.get(p["website"], check_robots=False)).final_url
            except FetchError:
                return None
        return p
    out = await bounded_map(out, unwrap, concurrency=8)
    METRICS.inc("products.discovered.product_hunt", len(out))
    return out


# ------------------------------------------------------------------ pricing
_PRICE = re.compile(r"(?:[$€£₹]\s?(?!0\b)\d+(?:[.,]\d+)?(?:\s?/\s?(?:mo|month|yr|year|user|seat))?|"
                    r"\b\d+(?:[.,]\d+)?\s?(?:USD|EUR)\s?/\s?(?:mo|month))", re.I)
_FREE_PLAN = re.compile(r"(free plan|free tier|free forever|forever free|[$€£]0(?:\.00)?\b|free\s*/\s*mo|"
                        r"\bfree\b\s*\n\s*[$€£]0)", re.I)
_ALL_FREE = re.compile(r"(completely free|100% free|totally free|free and open[- ]source|free & open[- ]source|"
                       r"open[- ]source and free|free to use|always free)", re.I)
_CONTACT = re.compile(r"(contact sales|talk to sales|custom pricing|contact us for pricing|get a quote|"
                      r"request (?:a )?quote|pricing on request)", re.I)


def _snippet(text: str, m: re.Match, width: int = 70) -> str:
    a, b = max(0, m.start() - width), min(len(text), m.end() + width)
    return " ".join(text[a:b].split())


def classify_pricing_rules(text: str, is_pricing_page: bool) -> tuple[PricingModel | None, str | None]:
    price = _PRICE.search(text)
    free_plan = _FREE_PLAN.search(text)
    if price and free_plan:
        return PricingModel.FREEMIUM, _snippet(text, free_plan)
    if price:
        return PricingModel.PAID, _snippet(text, price)
    all_free = _ALL_FREE.search(text)
    if all_free:
        return PricingModel.FREE, _snippet(text, all_free)
    contact = _CONTACT.search(text)
    if contact and is_pricing_page:
        return PricingModel.ENTERPRISE, _snippet(text, contact)
    return None, None


def find_pricing_link(html: str, base: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        label = (a.get_text(" ", strip=True) or "").lower()
        href = a["href"]
        if label in {"pricing", "plans", "plans & pricing", "pricing & plans", "price"} or re.search(r"/(pricing|plans)\b", href):
            url = urljoin(base, href)
            if domain_of(url) == domain_of(base):
                return url
    return None


async def enrich_pricing(ctx: Context, p: dict) -> dict:
    p.update(pricing=None, evidence=None, evidence_url=None, method="none")
    try:
        home = await ctx.fetcher.get(p["website"])
    except FetchError:
        METRICS.inc("products.site_unreachable")
        return p
    pages = []
    link = find_pricing_link(home.text, home.final_url)
    if link:
        try:
            pr = await ctx.fetcher.get(link)
            pages.append((pr.final_url, page_text(pr.text), True))
        except FetchError:
            pass
    pages.append((home.final_url, page_text(home.text), False))
    for url, text, is_pricing in pages:
        model, ev = classify_pricing_rules(text, is_pricing)
        if model:
            p.update(pricing=model, evidence=ev, evidence_url=url, method="rules")
            METRICS.inc("products.pricing.rules")
            return p
    if ctx.llm.enabled:
        url, text, _ = pages[0]
        try:
            res = await ctx.llm.extract(PRICING_TASK, text[:60000])
            if res.data["pricing_model"]:
                p.update(pricing=PricingModel(res.data["pricing_model"]), evidence=res.data["evidence"],
                         evidence_url=url, method=f"llm:{res.provider}")
                METRICS.inc("products.pricing.llm")
                return p
        except AllProvidersFailed as exc:
            log(logger, logging.DEBUG, "pricing llm failed", err=str(exc)[:200])
    METRICS.inc("products.pricing.unknown")
    return p


async def collect_products(ctx: Context, target: int) -> list[ProductRecord]:
    want = int(target * 1.3) + 50  # headroom: some sites are dead/unreachable
    discovered = await discover_product_hunt(ctx, want // 2) + await discover_show_hn(ctx, want)
    uniq: dict[str, dict] = {}
    for p in discovered:
        d = domain_of(p.get("website"))
        if d and d not in uniq and not _skip(p.get("website")):
            uniq[d] = p
    log(logger, logging.INFO, "products discovered", unique_domains=len(uniq))
    async def enrich_with_deadline(p: dict) -> dict:
        # One hung site (slow browser render, tarpit) must not stall the whole vertical.
        try:
            return await asyncio.wait_for(enrich_pricing(ctx, p), timeout=_ENRICH_TIMEOUT_S)
        except asyncio.TimeoutError:
            METRICS.inc("products.enrich_timeout")
            p.update(pricing=None, evidence=None, evidence_url=None, method="none")
            return p

    enriched = await bounded_map(list(uniq.values()), enrich_with_deadline, concurrency=32)
    log(logger, logging.INFO, "products enriched", count=len(enriched))

    enriched.sort(key=lambda p: p["source_url"])
    records: list[ProductRecord] = []
    for p in enriched:
        res = ctx.resolver.resolve(p["name"], website=p["website"], record_type="PRODUCT", source_url=p["source_url"])
        records.append(ProductRecord(
            source=Source(name=p["source_name"], url=p["source_url"]),
            content=ProductContent(
                productName=p["name"], startupName=res.canonical, pricingModel=p["pricing"],
                website=p["website"], tagline=p.get("tagline"), launchedAt=to_iso(parse_datetime(p.get("launched"))),
                pricingEvidence=p["evidence"], pricingEvidenceUrl=p["evidence_url"], pricingMethod=p["method"],
            ),
        ))
    # prefer records whose site we could actually read, then keep target count
    records.sort(key=lambda r: (r.content.pricingModel is None, r.source.url))
    records = records[:target]
    METRICS.inc("products.records", len(records))
    return records
