"""Offline tests for the components the assessment scores (no network needed)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from aiohttp import web

from graphone.dates import DateResult, extract_published_date, infer_from_listing_order, is_fresh, parse_datetime
from graphone.http.antibot import detect_challenge
from graphone.http.backoff import backoff_delay, parse_retry_after
from graphone.llm.chunking import estimate_tokens, fit_to_budget
from graphone.llm.grounding import is_grounded
from graphone.llm.orchestrator import AllProvidersFailed, LLMOrchestrator
from graphone.llm.providers import LLMError, Provider, classify_http_error
from graphone.llm.tasks import NEWS_TASK, PRICING_TASK
from graphone.resolver.resolver import EntityResolver
from graphone.sources.jobs import classify_remote, classify_role
from graphone.sources.products import classify_pricing_rules, product_name_from_title
from graphone.sources.startups import parse_yc_page
from graphone.state.store import SqliteStateStore
from graphone.text import canonical_url, hamming, simhash

NOW = datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ dates
@pytest.mark.parametrize("raw,expected", [
    ("2 hours ago", NOW - timedelta(hours=2)),
    ("an hour ago", NOW - timedelta(hours=1)),
    ("3h", NOW - timedelta(hours=3)),
    ("Posted 5d", NOW - timedelta(days=5)),
    ("45 mins ago", NOW - timedelta(minutes=45)),
    ("just now", NOW),
    ("2026-11-20T09:30:00Z", datetime(2026, 11, 20, 9, 30, tzinfo=timezone.utc)),
    ("2026-11-20T15:00:00+05:30", datetime(2026, 11, 20, 9, 30, tzinfo=timezone.utc)),
    ("Thu, 20 Nov 2026 08:00:00 GMT", datetime(2026, 11, 20, 8, 0, tzinfo=timezone.utc)),
    (1763625600, datetime.fromtimestamp(1763625600, timezone.utc)),
    (1763625600000, datetime.fromtimestamp(1763625600, timezone.utc)),
    ("Nov 19, 2026", datetime(2026, 11, 19, tzinfo=timezone.utc)),
])
def test_parse_datetime(raw, expected):
    assert parse_datetime(raw, NOW) == expected


def test_parse_rejects_garbage_and_future():
    assert parse_datetime("not a date", NOW) is None
    assert parse_datetime("2031-01-01T00:00:00Z", NOW) is None


def test_extract_prefers_jsonld_over_modified_and_feed():
    html = """<html><head>
      <meta property="article:modified_time" content="2026-11-20T11:00:00Z">
      <script type="application/ld+json">{"@type":"NewsArticle","datePublished":"2026-11-10T08:00:00Z",
        "dateModified":"2026-11-20T11:00:00Z"}</script></head><body></body></html>"""
    r = extract_published_date(html, "https://x.com/a", now=NOW, feed_date=NOW)
    assert r.source == "jsonld" and r.dt.day == 10
    assert not is_fresh(r, NOW)  # edited-today old article must NOT be fresh


def test_extract_relative_byline_when_meta_missing():
    html = '<html><body><div class="byline">By Jane · 3 hours ago</div><p>Body</p></body></html>'
    r = extract_published_date(html, "", now=NOW)
    assert r.source == "byline_relative" and r.dt == NOW - timedelta(hours=3) and is_fresh(r, NOW)


def test_extract_url_date_is_day_precision():
    r = extract_published_date("<html></html>", "https://site.com/2026/11/19/story", now=NOW)
    assert r.precision == "day" and not is_fresh(r, NOW)  # yesterday-by-date is not *guaranteed* < 24h
    r2 = extract_published_date("<html></html>", "https://site.com/2026/11/20/story", now=NOW)
    assert is_fresh(r2, NOW)


def test_listing_order_heuristic():
    dated = [None, DateResult(NOW - timedelta(hours=5), "feed", "medium"), None,
             DateResult(NOW - timedelta(days=3), "feed", "medium")]
    assert infer_from_listing_order(dated, NOW) == [True, False, False, False]


# ------------------------------------------------------------------ resolver
@pytest.fixture(scope="module")
def resolver():
    return EntityResolver(ROOT / "data" / "seed_entities.json")


@pytest.mark.parametrize("raw,canon", [
    ("OpenAI", "OpenAI"), ("OpenAI, Inc.", "OpenAI"), ("Open AI", "OpenAI"), ("OPENAI LLC", "OpenAI"),
    ("Hugging Face, Inc.", "Hugging Face"), ("huggingface", "Hugging Face"), ("Antrhopic", "Anthropic"),
    ("Weights and Biases", "Weights & Biases"), ("Mistral", "Mistral AI"), ("The Writer", "Writer"),
])
def test_resolver_canonicalises(resolver, raw, canon):
    assert resolver.resolve(raw).canonical == canon


def test_resolver_guards(resolver):
    assert resolver.resolve("OpenAir").canonical == "OpenAir"           # near-miss is NOT merged
    assert resolver.resolve("Cursor", website="https://cursor.com").canonical == "Anysphere"
    r = resolver.resolve("Modal", website="https://modal-forms.xyz")     # same name, different company
    assert r.canonical != "Modal" and r.method == "dynamic_new_domain_conflict"
    a = resolver.resolve("Zeta Robotics, Inc.").canonical
    assert resolver.resolve("ZETA ROBOTICS LLC").canonical == a == "Zeta Robotics"


# ------------------------------------------------------------------ chunking / grounding
def test_fit_to_budget_keeps_dense_content_and_respects_budget():
    filler = "\n\n".join(f"This is generic filler paragraph number {i} with nothing notable." for i in range(400))
    key = "Acme raised $40 million in a Series B led by Sequoia on 12 November 2026."
    text = "Lead paragraph about Acme.\n\n" + filler + "\n\n" + key + "\n\n" + filler
    out, truncated = fit_to_budget(text, 400)
    assert truncated and estimate_tokens(out) <= 400
    assert key in out and out.startswith("Lead paragraph")


def test_grounding():
    src = "Anthropic announced Claude for Chrome today."
    assert is_grounded("Anthropic", src)
    assert not is_grounded("OpenAI", src)


# ------------------------------------------------------------------ backoff / errors
def test_retry_after_parsing():
    assert parse_retry_after("7") == 7
    assert parse_retry_after(None, "Rate limit reached. Please try again in 7.66s.") == pytest.approx(7.66)
    assert parse_retry_after(None, "Please try again in 1m30s") == 90
    assert parse_retry_after(None, '{"retryDelay": "13s"}') == 13
    assert parse_retry_after(None, "try again in 20m") == 1200
    assert parse_retry_after(None, "try again in 500ms") == 0.5


def test_backoff_jitter_bounds():
    for a in range(6):
        d = backoff_delay(a, base=1, cap=30)
        assert min(30, 2 ** a) / 2 <= d <= min(30, 2 ** a)


def test_error_classification():
    assert classify_http_error("groq", 413, "Request too large for model", {}).kind == "payload_too_large"
    assert classify_http_error("x", 400, "This model's maximum context length is 8192", {}).kind == "context_length"
    assert classify_http_error("x", 429, "try again in 2s", {}).kind == "rate_limit"
    assert classify_http_error("x", 429, "Rate limit reached for tokens per day (TPD). try again in 20m", {}).kind == "quota"


# ------------------------------------------------------------------ orchestrator fallback chain
class FakeProvider(Provider):
    def __init__(self, name, script, max_input_tokens=4000):
        super().__init__(name, "fake", "k", max_input_tokens=max_input_tokens, rpm=6000, tpm=10**8)
        self.script, self.calls, self.prompt_sizes = list(script), 0, []

    async def complete(self, session, system, user):
        self.calls += 1
        self.prompt_sizes.append(len(user))
        step = self.script.pop(0) if self.script else self.script_default
        if isinstance(step, LLMError):
            raise step
        return step

    script_default = '{"pricing_model": null, "evidence": ""}'


def _run(coro):
    return asyncio.run(coro)


def test_fallback_429_then_413_then_success():
    src = "Pro plan $20 per month. Free forever plan available."
    good = '{"pricing_model": "FREEMIUM", "evidence": "Free forever plan available."}'
    p1 = FakeProvider("gemini", [LLMError("gemini", "rate_limit", 429, 1.0), LLMError("gemini", "quota", 429)])
    p2 = FakeProvider("groq", [LLMError("groq", "payload_too_large", 413), good])
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    orch = LLMOrchestrator([p1, p2], None, sleep=fake_sleep)
    res = _run(orch.extract(PRICING_TASK, src))
    assert res.data["pricing_model"] == "FREEMIUM" and res.provider.startswith("groq")
    assert p1.calls == 2 and p2.calls == 2 and sleeps and sleeps[0] >= 1.0
    assert orch.breakers["gemini"].is_open   # quota -> tier disabled


def test_hallucinated_evidence_is_rejected():
    src = "Our product helps teams write docs."
    p = FakeProvider("gemini", ['{"pricing_model": "PAID", "evidence": "Only $99/month"}'])
    res = _run(LLMOrchestrator([p], None).extract(PRICING_TASK, src))
    assert res.data["pricing_model"] is None


def test_news_orgs_grounded():
    src = "Mistral AI partnered with Nvidia to launch a new model."
    p = FakeProvider("g", ['{"organizations": ["Mistral AI", "Nvidia", "Google"], "event_type": "PARTNERSHIP",'
                           ' "event_evidence": "Mistral AI partnered with Nvidia"}'])
    res = _run(LLMOrchestrator([p], None).extract(NEWS_TASK, src))
    assert res.data["organizations"] == ["Mistral AI", "Nvidia"] and res.data["event_type"] == "PARTNERSHIP"


def test_all_providers_fail():
    p = FakeProvider("g", [LLMError("g", "auth", 401)])
    with pytest.raises(AllProvidersFailed):
        _run(LLMOrchestrator([p], None).extract(PRICING_TASK, "x"))


def test_413_shrinks_payload():
    big = "\n\n".join(f"Paragraph {i} pricing $ {i} per month free plan details." for i in range(3000))
    p = FakeProvider("groq", [LLMError("groq", "payload_too_large", 413), LLMError("groq", "payload_too_large", 413),
                              '{"pricing_model": null, "evidence": ""}'], max_input_tokens=4500)
    _run(LLMOrchestrator([p], None).extract(PRICING_TASK, big))
    assert p.prompt_sizes[0] > p.prompt_sizes[1] > p.prompt_sizes[2]


# ------------------------------------------------------------------ fetcher against a local server
def test_fetcher_retries_429_and_detects_challenge(tmp_path, monkeypatch):
    from graphone.config import get_settings
    from graphone.http.fetcher import FetchError, Fetcher

    hits = {"n": 0}

    async def flaky(request):
        hits["n"] += 1
        if hits["n"] < 3:
            return web.Response(status=429, headers={"Retry-After": "0"})
        return web.json_response({"ok": True})

    async def cf(request):
        return web.Response(status=403, text="<title>Just a moment...</title>", headers={"Server": "cloudflare"})

    async def main():
        app = web.Application()
        app.router.add_get("/flaky", flaky)
        app.router.add_get("/cf", cf)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        s = get_settings()
        import dataclasses
        s = dataclasses.replace(s, browser_fallback=False, respect_robots=False, max_retries=4)
        try:
            async with Fetcher(s) as f:
                assert (await f.get_json(f"http://127.0.0.1:{port}/flaky"))["ok"] is True
                with pytest.raises(FetchError, match="cloudflare"):
                    await f.get(f"http://127.0.0.1:{port}/cf")
        finally:
            await runner.cleanup()

    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    _run(main())
    assert hits["n"] == 3


def test_detect_challenge_vendors():
    assert detect_challenge(403, {"Server": "cloudflare"}, "") == "cloudflare"
    assert detect_challenge(403, {"x-datadome": "protected"}, "") == "datadome"
    assert detect_challenge(200, {}, "<html>normal</html>") is None


# ------------------------------------------------------------------ dedup store
def test_claim_is_atomic_once(tmp_path):
    async def main():
        st = SqliteStateStore(tmp_path / "s.db")
        results = await asyncio.gather(*(st.claim("news", "k1") for _ in range(20)))
        assert sum(results) == 1
        await st.close()
    _run(main())


def test_url_canonicalisation_and_simhash():
    a = canonical_url("https://www.TechCrunch.com/2026/11/20/story/?utm_source=x&b=2&a=1#top")
    assert a == "https://techcrunch.com/2026/11/20/story?a=1&b=2"
    import random
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(3000)]
    t1 = " ".join(rng.choice(vocab) for _ in range(600))          # article-length text
    t2 = t1.replace(t1.split()[100], "syndicated", 1) + " Originally published by Reuters."
    t3 = " ".join(rng.choice(vocab) for _ in range(600))
    assert hamming(simhash(t1), simhash(t2)) <= 3                    # near-duplicate
    assert hamming(simhash(t1), simhash(t3)) > 15                    # different article


# ------------------------------------------------------------------ source parsers
def test_pricing_rules():
    assert classify_pricing_rules("Free plan forever. Pro $19/mo", True)[0].value == "FREEMIUM"
    assert classify_pricing_rules("Starter $49 per month. 14-day free trial", True)[0].value == "PAID"
    assert classify_pricing_rules("Free and open-source, MIT licensed", False)[0].value == "FREE"
    assert classify_pricing_rules("Custom plans. Contact sales", True)[0].value == "ENTERPRISE"
    assert classify_pricing_rules("Book a demo. Contact sales", False)[0] is None  # homepage: ambiguous


def test_show_hn_title_parsing():
    assert product_name_from_title("Show HN: Lumen – an AI copilot for spreadsheets") == "Lumen"
    assert product_name_from_title("Show HN: DocsGPT: chat with your docs") == "DocsGPT"
    assert product_name_from_title("Show HN: I built an AI that writes my emails") is None


def test_job_classifiers():
    assert classify_role("Senior Software Engineer, Inference") == "Engineering"
    assert classify_role("Research Scientist, Alignment") == "Research"
    assert classify_role("Enterprise Account Executive") == "Sales"
    assert classify_remote("Remote - US", None, None) == (True, "location_field")
    assert classify_remote("San Francisco, CA", "Hybrid", None)[0] is False


def test_yc_page_parser():
    html = ('<div id="app" data-page="{&quot;props&quot;:{&quot;company&quot;:{&quot;name&quot;:&quot;Acme&quot;,'
            '&quot;team_size&quot;:42,&quot;website&quot;:&quot;https://acme.ai&quot;}}}"></div>')
    assert parse_yc_page(html) == {"team_size": 42, "name": "Acme", "website": "https://acme.ai"}
