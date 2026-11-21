"""Phase II — AI jobs from five boards, strictly published within 24h.

Boards: Greenhouse, Lever, Ashby (public ATS JSON APIs polled per AI company),
RemoteOK and Remotive (remote job boards, AI-filtered).

Date semantics per board (only *publication* timestamps prove freshness):
  Greenhouse first_published -> published   (updated_at is only an upper bound)
  Lever createdAt (ms)       -> published
  Ashby publishedAt          -> published
  RemoteOK epoch/date        -> published
  Remotive publication_date  -> published (naive, documented as UTC)
If only an upper bound exists, the job is accepted only via the first-seen
heuristic (board crawled < 24h ago and this job id was not present then).
"""
from __future__ import annotations

import html as htmllib
import logging
import re
from datetime import timedelta

from ..dates import DateResult, is_fresh, parse_datetime
from ..http.fetcher import FetchError, NotFound
from ..llm.orchestrator import AllProvidersFailed
from ..llm.tasks import JOB_TASK
from ..logging_setup import METRICS, log
from ..models import JobContent, JobRecord, Source, to_iso
from ..text import canonical_url, html_to_text, sha1
from .base import Context, bounded_map

logger = logging.getLogger("graphone.jobs")

_AI = re.compile(r"\b(ai|a\.i\.|ml|llms?|machine learning|deep learning|nlp|computer vision|genai|generative|"
                 r"data scien\w+|mlops|pytorch|tensorflow|rag|prompt|agents?)\b", re.I)

ROLE_RULES = [
    ("Research", r"research (scientist|engineer)|\bresearcher\b|scientist|phd|post-?doc"),
    ("Data", r"data (scientist|engineer|analyst)|analytics|\banalyst\b|bi engineer"),
    ("Engineering", r"engineer|developer|\bsre\b|devops|architect|programmer|swe\b|technical staff|infrastructure|"
                    r"security|firmware|\bqa\b|tech lead"),
    ("Product", r"product manager|\bpm\b|product lead|head of product|product owner|program manager"),
    ("Design", r"designer|\bux\b|\bui\b|design lead|creative"),
    ("Sales", r"account executive|\bsales\b|\bbdr\b|\bsdr\b|business development|account manager|solutions consultant|"
              r"partnerships?"),
    ("Marketing", r"marketing|growth|content|communications|\bpr\b|brand|community|developer relations|devrel"),
    ("Customer Success", r"customer (success|support|experience)|support (engineer|specialist)|implementation"),
    ("People", r"recruit|talent|people|\bhr\b|human resources"),
    ("Legal & Policy", r"counsel|legal|policy|compliance|attorney|paralegal|regulatory"),
    ("Finance", r"finance|accountant|accounting|controller|tax|treasury|\bfp&a\b"),
    ("Operations", r"operations|\bops\b|chief of staff|executive assistant|office|workplace|facilities|procurement"),
]


def classify_role(title: str) -> str | None:
    t = title.lower()
    for fam, pat in ROLE_RULES:
        if re.search(pat, t):
            return fam
    return None


_REMOTE_TXT = re.compile(r"\b(fully remote|100% remote|remote[- ]first|remote[- ]friendly|work from anywhere|"
                         r"remote \(|remote,|remote - |remote$|remote position|remote role)", re.I)


def classify_remote(location: str | None, workplace: str | None, desc: str | None) -> tuple[bool | None, str]:
    w = (workplace or "").lower()
    if w in {"remote", "fully_remote"}:
        return True, "workplace_field"
    if w in {"onsite", "on-site", "hybrid", "in_office"}:
        return False, "workplace_field"
    loc = (location or "").strip()
    if re.search(r"\bremote\b|anywhere|worldwide", loc, re.I):
        return True, "location_field"
    if desc and _REMOTE_TXT.search(desc[:5000]):
        return True, "description_rule"
    if loc:
        return False, "location_field_onsite"
    return None, "unknown"


def _job(board, title, company, url, date_raw, date_field, kind, location=None, workplace=None, desc=None,
         remote=None, job_id=None):
    return {"board": board, "title": " ".join((title or "").split()), "company": company, "url": url,
            "date_raw": date_raw, "date_field": date_field, "date_kind": kind, "location": location,
            "workplace": workplace, "desc": desc, "remote": remote, "id": job_id or canonical_url(url)}


# ---------------------------------------------------------------- board adapters
async def greenhouse(ctx: Context, cfg: dict) -> list[dict]:
    async def one(slug):
        base = f"https://boards-api.greenhouse.io/v1/boards/{slug}"
        try:
            meta = await ctx.fetcher.get_json(base)
            data = await ctx.fetcher.get_json(f"{base}/jobs")
        except NotFound:
            METRICS.inc("jobs.greenhouse.board_missing")
            return []
        name = meta.get("name") or slug
        out = []
        for j in data.get("jobs", []):
            if j.get("first_published"):
                raw, field, kind = j["first_published"], "api:first_published", "published"
            else:
                raw, field, kind = j.get("updated_at"), "api:updated_at", "upper_bound"
            out.append(_job(cfg["name"], j.get("title"), j.get("company_name") or name, j.get("absolute_url"), raw, field,
                            kind, location=(j.get("location") or {}).get("name"),
                            job_id=f"gh:{slug}:{j.get('id')}") | {"detail": f"{base}/jobs/{j.get('id')}"})
        return out
    res = await bounded_map(cfg["companies"], one, concurrency=6)
    return [j for lst in res for j in lst]


async def lever(ctx: Context, cfg: dict) -> list[dict]:
    async def one(slug):
        try:
            data = await ctx.fetcher.get_json(f"https://api.lever.co/v0/postings/{slug}", params={"mode": "json"})
        except NotFound:
            METRICS.inc("jobs.lever.board_missing")
            return []
        out = []
        for j in data if isinstance(data, list) else []:
            cat = j.get("categories") or {}
            desc = " ".join(filter(None, [j.get("descriptionPlain"), j.get("additionalPlain")]))
            out.append(_job(cfg["name"], j.get("text"), slug.replace("-", " ").title(), j.get("hostedUrl"),
                            j.get("createdAt"), "api:createdAt", "published", location=cat.get("location"),
                            workplace=j.get("workplaceType"), desc=desc, job_id=f"lever:{j.get('id')}"))
        return out
    res = await bounded_map(cfg["companies"], one, concurrency=6)
    return [j for lst in res for j in lst]


async def ashby(ctx: Context, cfg: dict) -> list[dict]:
    async def one(slug):
        try:
            data = await ctx.fetcher.get_json(f"https://api.ashbyhq.com/posting-api/job-board/{slug}")
        except NotFound:
            METRICS.inc("jobs.ashby.board_missing")
            return []
        out = []
        for j in data.get("jobs", []):
            if j.get("isListed") is False:
                continue
            remote = True if j.get("isRemote") else None
            out.append(_job(cfg["name"], j.get("title"), slug.replace("-", " ").title(), j.get("jobUrl"),
                            j.get("publishedAt"), "api:publishedAt", "published", location=j.get("location"),
                            workplace=j.get("workplaceType"), desc=j.get("descriptionPlain"), remote=remote,
                            job_id=f"ashby:{j.get('id')}"))
        return out
    res = await bounded_map(cfg["companies"], one, concurrency=6)
    return [j for lst in res for j in lst]


async def remoteok(ctx: Context, cfg: dict) -> list[dict]:
    try:
        data = await ctx.fetcher.get_json(cfg["api"])
    except FetchError as exc:
        log(logger, logging.ERROR, "remoteok failed", err=str(exc)[:160])
        return []
    out = []
    for j in data[1:] if isinstance(data, list) else []:  # [0] is the legal notice
        blob = f"{j.get('position')} {' '.join(j.get('tags') or [])}"
        if not _AI.search(blob):
            continue
        out.append(_job(cfg["name"], j.get("position"), j.get("company"), j.get("url"),
                        j.get("epoch") or j.get("date"), "api:epoch", "published", location=j.get("location"),
                        desc=html_to_text(j.get("description") or ""), remote=True, job_id=f"rok:{j.get('id')}"))
    return out


async def remotive(ctx: Context, cfg: dict) -> list[dict]:
    out: dict[str, dict] = {}
    for q in cfg.get("searches", ["ai"]):
        try:
            data = await ctx.fetcher.get_json(cfg["api"], params={"search": q})
        except FetchError as exc:
            log(logger, logging.ERROR, "remotive failed", err=str(exc)[:160])
            continue
        for j in data.get("jobs", []):
            out[str(j.get("id"))] = _job(
                cfg["name"], j.get("title"), j.get("company_name"), j.get("url"), j.get("publication_date"),
                "api:publication_date", "published", location=j.get("candidate_required_location"),
                desc=html_to_text(j.get("description") or ""), remote=True, job_id=f"remotive:{j.get('id')}")
    return list(out.values())


BOARDS = {"greenhouse": greenhouse, "lever": lever, "ashby": ashby, "remoteok": remoteok, "remotive": remotive}


# ---------------------------------------------------------------- pipeline
async def _finalize(ctx: Context, j: dict, recent_prev_run: bool) -> JobRecord | None:
    hours = ctx.settings.freshness_hours
    key = sha1(j["id"])
    first_seen = await ctx.store.claim(f"jobseen:{j['board']}", key)
    if await ctx.store.has_record("jobs", key):
        METRICS.inc("jobs.already_processed")
        return None
    dt = parse_datetime(j["date_raw"], ctx.now)
    if j["date_kind"] == "published":
        dr = DateResult(dt, j["date_field"], "high")
        if not is_fresh(dr, ctx.now, hours):
            METRICS.inc("jobs.stale")
            return None
    else:  # upper bound only
        if dt is None or ctx.now - dt > timedelta(hours=hours):
            METRICS.inc("jobs.stale")
            return None
        if not (first_seen and recent_prev_run):
            METRICS.inc("jobs.undetermined_cold_start")
            return None
        dr = DateResult(None, "first_seen_since_last_run", "inferred")
    if not await ctx.store.claim("jobs_inflight", key, ttl=3600):
        return None

    desc = j.get("desc")
    if desc is None and j.get("detail"):
        try:
            d = await ctx.fetcher.get_json(j["detail"])
            desc = html_to_text(htmllib.unescape(d.get("content") or ""))
        except FetchError:
            desc = None
    role, method = classify_role(j["title"]), "rules"
    if j.get("remote") is not None:
        remote, rmethod = j["remote"], "board_field"
    else:
        remote, rmethod = classify_remote(j.get("location"), j.get("workplace"), desc)
    if (role is None or remote is None) and ctx.llm.enabled:
        try:
            text = f"Title: {j['title']}\nLocation: {j.get('location') or ''}\n\n{desc or ''}"
            res = await ctx.llm.extract(JOB_TASK, text)
            role = role or res.data["role_family"]
            if remote is None:
                remote, rmethod = res.data["is_remote"], f"llm:{res.provider}"
            method = f"rules+llm:{res.provider}"
        except AllProvidersFailed:
            pass
    company = ctx.resolver.resolve(j["company"] or "", record_type="JOB", source_url=j["url"]).canonical
    rec = JobRecord(
        source=Source(name=j["board"], url=j["url"]),
        content=JobContent(title=j["title"], company=company, date=to_iso(dr.dt) or to_iso(ctx.now),
                           is_remote=remote, role_family=role or "Other", location=j.get("location"), url=j["url"],
                           rawCompany=j["company"], dateSource=dr.source, dateConfidence=dr.confidence,
                           classificationMethod=f"role:{method}; remote:{rmethod}"),
    )
    await ctx.store.put_record("jobs", key, rec.content.date, rec.model_dump(mode="json"))
    METRICS.inc("jobs.records")
    return rec


async def collect_jobs(ctx: Context) -> list[JobRecord]:
    cfg = ctx.settings.sources()["jobs"]
    hours = ctx.settings.freshness_hours
    out: list[JobRecord] = []
    for board, fn in BOARDS.items():
        if board not in cfg:
            continue
        bcfg = cfg[board]
        last = await ctx.store.kv_get(f"last_run:jobs:{bcfg['name']}")
        recent = bool(last) and ctx.now - parse_datetime(last) <= timedelta(hours=hours)
        raw = [j for j in await fn(ctx, bcfg) if j.get("url") and ctx.shard.owns(j["id"])]
        recs = await bounded_map(raw, lambda j: _finalize(ctx, j, recent), concurrency=16)
        await ctx.store.kv_set(f"last_run:jobs:{bcfg['name']}", ctx.now.isoformat())
        log(logger, logging.INFO, "job board done", board=bcfg["name"], listed=len(raw), fresh_new=len(recs))
        out.extend(recs)
    return out
