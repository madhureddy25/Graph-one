"""Publication-date extraction and normalisation (Phase II 'Freshness Challenge').

Every returned datetime is timezone-aware UTC. Every result carries *where* the
date came from and how much we trust it, which is exported with each record.

Priority for an HTML article (highest first):
  1. JSON-LD  datePublished                         (high)
  2. <meta> article:published_time & friends        (high)
  3. <time datetime> / itemprop=datePublished       (medium)
  4. visible byline text, absolute or relative      (medium)  "2 hours ago"
  5. feed <pubDate> passed in by the caller         (medium)
  6. URL path /2026/11/20/                          (low, day precision)
dateModified / article:modified_time are deliberately ignored: an old article
that was edited today must NOT be considered fresh.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from bs4 import BeautifulSoup
from dateutil import parser as du_parser

UTC = timezone.utc

# -------------------------------------------------------------------- primitives
_UNITS = {
    "s": "seconds", "sec": "seconds", "secs": "seconds", "second": "seconds", "seconds": "seconds",
    "m": "minutes", "min": "minutes", "mins": "minutes", "minute": "minutes", "minutes": "minutes",
    "h": "hours", "hr": "hours", "hrs": "hours", "hour": "hours", "hours": "hours",
    "d": "days", "day": "days", "days": "days",
    "w": "weeks", "wk": "weeks", "wks": "weeks", "week": "weeks", "weeks": "weeks",
    "mo": "months", "month": "months", "months": "months",
    "y": "years", "yr": "years", "yrs": "years", "year": "years", "years": "years",
}
_WORD_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9, "ten": 10, "few": 3, "couple of": 2, "several": 3}

_REL = re.compile(
    r"\b(?P<num>\d+|an?|one|two|three|four|five|six|seven|eight|nine|ten|few|couple of|several)\+?\s*"
    r"(?P<unit>seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d|weeks?|wks?|w|months?|mo|years?|yrs?|y)"
    r"\.?\s+ago\b",
    re.I,
)
# Compact forms used by job boards: "3h", "5d", "Posted 2d" (no 'ago')
_REL_COMPACT = re.compile(r"^(?:posted\s+|updated\s+)?(?P<num>\d+)\s*(?P<unit>m|h|d|w|mo)$", re.I)


def ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_relative(text: str, now: datetime) -> datetime | None:
    """'2 hours ago', 'an hour ago', '3h', 'yesterday', 'just now', 'today'."""
    t = " ".join(text.strip().lower().split())
    if not t:
        return None
    if t in {"just now", "now", "moments ago", "a moment ago", "few seconds ago", "seconds ago"}:
        return now
    if re.search(r"\b(today|just posted|new)\b", t) and not _REL.search(t):
        return now.replace(hour=0, minute=0, second=0, microsecond=0)  # day precision
    if re.search(r"\byesterday\b", t):
        return (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    m = _REL.search(t) or _REL_COMPACT.match(t)
    if not m:
        return None
    raw_num = m.group("num").lower()
    n = int(raw_num) if raw_num.isdigit() else _WORD_NUM.get(raw_num, 1)
    unit = _UNITS.get(m.group("unit").lower().rstrip("."))
    if unit is None:
        return None
    if unit == "months":
        return now - timedelta(days=30 * n)
    if unit == "years":
        return now - timedelta(days=365 * n)
    return now - timedelta(**{unit: n})


def parse_datetime(value, now: datetime | None = None) -> datetime | None:
    """Best-effort parser for anything date-like. Returns aware UTC or None.

    Accepts epoch seconds/milliseconds, ISO-8601, RFC-2822, free-form absolute
    dates and relative expressions. Rejects implausible values.
    """
    now = ensure_utc(now or datetime.now(UTC))
    if value is None or value == "":
        return None
    dt: datetime | None = None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=UTC)
    elif isinstance(value, (int, float)) or (isinstance(value, str) and re.fullmatch(r"\d{10}(\d{3})?(\.\d+)?", value.strip())):
        x = float(value)
        if x > 1e12:  # milliseconds
            x /= 1000
        dt = datetime.fromtimestamp(x, UTC)
    else:
        s = " ".join(str(value).split())
        rel = parse_relative(s, now)
        if rel is not None:
            dt = rel
        else:
            try:
                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                try:
                    dt = parsedate_to_datetime(s)
                except (TypeError, ValueError, IndexError):
                    try:
                        cleaned = re.sub(r"^(published|posted|updated|on)\s*:?\s*", "", s, flags=re.I)
                        dt = du_parser.parse(cleaned, fuzzy=False, default=datetime(now.year, 1, 1))
                    except (ValueError, OverflowError, TypeError):
                        return None
    dt = ensure_utc(dt)
    if dt.year < 1990 or dt > now + timedelta(days=2):
        return None
    return dt


def is_day_precision(raw: str | None) -> bool:
    return bool(raw) and bool(re.fullmatch(r"\s*\d{4}-\d{2}-\d{2}\s*", str(raw)))


# -------------------------------------------------------------------- freshness
@dataclass
class DateResult:
    dt: datetime | None
    source: str          # jsonld | meta:<name> | time_tag | byline_text | feed | url_path | api:<field> | first_seen | none
    confidence: str      # high | medium | low | inferred | none
    precision: str = "second"  # second | day


def is_fresh(result: DateResult, now: datetime, hours: int = 24, future_tolerance_min: int = 15) -> bool:
    """Strict freshness. Day-precision dates are only accepted if they are *today*
    (UTC): that is the only day for which 'within 24h' is guaranteed."""
    if result.dt is None:
        return False
    now = ensure_utc(now)
    if result.dt > now + timedelta(minutes=future_tolerance_min):
        return False
    if result.precision == "day":
        return result.dt.date() == now.date()
    return now - result.dt <= timedelta(hours=hours)


# -------------------------------------------------------------------- HTML
_META_PUBLISHED = [
    ("property", "article:published_time"), ("property", "og:article:published_time"),
    ("property", "og:published_time"), ("name", "article:published_time"),
    ("itemprop", "datePublished"), ("name", "datePublished"), ("name", "parsely-pub-date"),
    ("name", "sailthru.date"), ("name", "publish-date"), ("name", "publish_date"),
    ("name", "pubdate"), ("name", "date"), ("name", "dc.date"), ("name", "DC.date.issued"),
    ("name", "dcterms.created"), ("name", "cXenseParse:recs:publishtime"), ("name", "article.published"),
]
_URL_DATE = re.compile(r"/(20\d{2})[/-](0[1-9]|1[0-2])[/-](0[1-9]|[12]\d|3[01])(?:/|-|$)")
_BYLINE_ABS = re.compile(
    r"(?:published|posted|updated)?\s*(?:on\s*)?"
    r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+20\d{2}"
    r"(?:\s*(?:at\s*)?\d{1,2}:\d{2}\s*(?:am|pm)?(?:\s*[a-z]{2,4})?)?)",
    re.I,
)


def _walk_jsonld(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_jsonld(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_jsonld(v)


def extract_published_date(
    html: str,
    url: str = "",
    *,
    now: datetime | None = None,
    feed_date: datetime | None = None,
) -> DateResult:
    now = ensure_utc(now or datetime.now(UTC))
    soup = BeautifulSoup(html or "", "lxml")

    # 1. JSON-LD
    for tag in soup.find_all("script", type=lambda t: t and "ld+json" in t):
        try:
            data = json.loads(tag.string or tag.get_text() or "")
        except (json.JSONDecodeError, TypeError):
            continue
        for node in _walk_jsonld(data):
            raw = node.get("datePublished") or node.get("dateCreated") or node.get("uploadDate")
            if isinstance(raw, str):
                dt = parse_datetime(raw, now)
                if dt:
                    return DateResult(dt, "jsonld", "high", "day" if is_day_precision(raw) else "second")

    # 2. meta tags
    for attr, name in _META_PUBLISHED:
        tag = soup.find("meta", attrs={attr: re.compile(f"^{re.escape(name)}$", re.I)})
        if tag and tag.get("content"):
            raw = tag["content"]
            dt = parse_datetime(raw, now)
            if dt:
                return DateResult(dt, f"meta:{name}", "high", "day" if is_day_precision(raw) else "second")

    # 3. <time> tags (prefer explicit pubdate / itemprop)
    times = soup.find_all("time")
    times.sort(key=lambda t: 0 if (t.has_attr("pubdate") or t.get("itemprop") == "datePublished") else 1)
    for t in times:
        raw = t.get("datetime") or t.get("title")
        if raw:
            dt = parse_datetime(raw, now)
            if dt:
                return DateResult(dt, "time_tag", "medium", "day" if is_day_precision(raw) else "second")
        txt = t.get_text(" ", strip=True)
        dt = parse_relative(txt, now) if txt else None
        if dt:
            return DateResult(dt, "time_tag_relative", "medium")

    # 4. visible byline text (relative first, then absolute) — restricted to
    #    byline-ish containers to avoid matching dates mentioned in the body.
    candidates = soup.select(
        "[class*=byline], [class*=date], [class*=time], [class*=meta], [class*=posted], header, .article-header"
    )[:40]
    for el in candidates:
        txt = el.get_text(" ", strip=True)[:200]
        if not txt:
            continue
        dt = parse_relative(txt, now)
        if dt:
            return DateResult(dt, "byline_relative", "medium",
                              "day" if re.search(r"today|yesterday", txt, re.I) else "second")
        m = _BYLINE_ABS.search(txt)
        if m:
            dt = parse_datetime(m.group(1), now)
            if dt:
                has_time = bool(re.search(r"\d{1,2}:\d{2}", m.group(1)))
                return DateResult(dt, "byline_absolute", "medium", "second" if has_time else "day")

    # 5. feed date
    if feed_date:
        return DateResult(ensure_utc(feed_date), "feed", "medium")

    # 6. URL path
    m = _URL_DATE.search(url or "")
    if m:
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=UTC)
        return DateResult(dt, "url_path", "low", "day")

    return DateResult(None, "none", "none")


def infer_from_listing_order(dated: list[DateResult | None], now: datetime, hours: int = 24) -> list[bool]:
    """Positional heuristic for undated items in a reverse-chronological listing.

    If an undated item sits *above* (newer than) an item whose date is known to be
    fresh, it must itself be newer than that item -> fresh. Returns, per index,
    whether an undated item can be inferred fresh.
    """
    out = [False] * len(dated)
    for i, r in enumerate(dated):
        if r is not None and r.dt is not None:
            continue
        for j in range(i + 1, len(dated)):
            rj = dated[j]
            if rj is not None and rj.dt is not None:
                out[i] = is_fresh(rj, now, hours)
                break
    return out
