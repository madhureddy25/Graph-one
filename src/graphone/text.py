"""Text utilities: full-text extraction, URL canonicalisation, hashing, SimHash."""
from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup

_TRACKING = re.compile(r"^(utm_.*|fbclid|gclid|mc_cid|mc_eid|ref|ref_src|source|cmpid|guccounter|_ga|igshid|via)$", re.I)


def canonical_url(url: str) -> str:
    """Stable identity for a URL: lower-case host, no tracking params, no fragment,
    sorted query, no trailing slash. Used as the dedup key across crawler nodes."""
    p = urlsplit(url.strip())
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not _TRACKING.match(k)))
    path = re.sub(r"/+$", "", p.path) or "/"
    return urlunsplit(("https", host, path, query, ""))


def domain_of(url: str | None) -> str | None:
    if not url:
        return None
    host = urlsplit(url if "://" in url else "https://" + url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host or None


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def normalize_ws(s: str) -> str:
    return re.sub(r"[ \t ]+", " ", re.sub(r"\n\s*\n+", "\n\n", s)).strip()


def html_to_text(html: str, url: str | None = None) -> str:
    """Main-content extraction (trafilatura) with a BeautifulSoup fallback."""
    try:
        import trafilatura

        txt = trafilatura.extract(html, url=url, include_comments=False, include_tables=False,
                                  favor_precision=True, deduplicate=True)
        if txt and len(txt) > 200:
            return normalize_ws(txt)
    except Exception:
        pass
    soup = BeautifulSoup(html or "", "lxml")
    for t in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg"]):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    paras = [p.get_text(" ", strip=True) for p in root.find_all(["p", "h1", "h2", "h3", "li"])]
    return normalize_ws("\n\n".join(p for p in paras if len(p) > 30))


def page_text(html: str) -> str:
    """Whole-page visible text (for pricing pages where 'main content' extraction
    would drop the pricing tables)."""
    soup = BeautifulSoup(html or "", "lxml")
    for t in soup(["script", "style", "noscript", "svg"]):
        t.decompose()
    return normalize_ws(soup.get_text("\n", strip=True))


def simhash(text: str, bits: int = 64) -> int:
    """64-bit SimHash over word 3-shingles: near-duplicate (syndicated / lightly
    edited) articles land within a small Hamming distance."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 3:
        words = words + ["_"] * (3 - len(words))
    v = [0] * bits
    for i in range(len(words) - 2):
        h = int(hashlib.md5(" ".join(words[i:i + 3]).encode()).hexdigest()[:16], 16)
        for b in range(bits):
            v[b] += 1 if (h >> b) & 1 else -1
    return sum(1 << b for b in range(bits) if v[b] > 0)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def simhash_bands(h: int, bands: int = 4, bits: int = 64) -> list[str]:
    """LSH bands: two hashes within Hamming distance < bands share >= 1 band, so
    near-dup lookup is a handful of exact key lookups in Redis/SQLite."""
    width = bits // bands
    return [f"{i}:{(h >> (i * width)) & ((1 << width) - 1):x}" for i in range(bands)]
