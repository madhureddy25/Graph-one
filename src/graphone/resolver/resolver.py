"""Deterministic entity resolution for startups / products / companies.

Resolution cascade (first hit wins, every decision is logged with method+score):
  1. seed_exact        raw string equals a seed canonical or alias (case-insensitive)
  2. seed_domain       record's website domain belongs to a seed entity
  3. seed_normalized   normalised key equals a seed key
                       ("OpenAI, Inc." / "Open AI" / "openai" -> "openai")
  4. seed_weak         key with generic trailing tokens removed matches a seed
                       ("Mistral AI Labs" -> "mistral") — only if unambiguous
  5. seed_fuzzy        OSA (Damerau) similarity >= 0.88 on keys of length >= 8
                       (typos / transpositions: 'Antrhopic' -> Anthropic)
  6. dynamic_*         unseen entity: joins an existing dynamic cluster by key /
                       weak key / fuzzy, or founds a new cluster.
Guard: if both sides have a known domain and the domains differ, a name-only
match is rejected (e.g. an unrelated product also called "Modal").
Determinism: callers resolve in a sorted order, normalisation is pure, ties are
broken lexicographically — the same input always yields the same output.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import process
from rapidfuzz.distance import OSA

from ..text import domain_of

_LEGAL = {
    "inc", "incorporated", "llc", "l.l.c", "ltd", "limited", "corp", "corporation", "co", "company", "gmbh",
    "sas", "sa", "ag", "bv", "plc", "pbc", "pte", "pty", "oy", "ab", "srl", "spa", "kk", "lp", "llp", "holdings",
}
# Words that are part of a business name in practice ("The Token Company"), so the
# display name keeps them even though matching keys still ignore them.
_DISPLAY_KEEP = {"co", "company", "holdings"}
_JOINERS = {"and", "the", "a", "an", "of"}
_WEAK = {"ai", "labs", "lab", "technologies", "technology", "tech", "hq", "app", "io", "systems", "software",
         "inc", "group", "the", "platform", "studio", "studios"}


def _ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def tokens(name: str) -> list[str]:
    s = _ascii_fold(name).casefold().replace("&", " and ").replace("+", " plus ")
    s = re.sub(r"\(.*?\)", " ", s)                 # "Foo (YC W24)"
    toks = re.findall(r"[a-z0-9]+", s.replace(".", " ").replace("'", ""))
    while len(toks) > 1 and toks[-1] in _LEGAL and toks[-2] not in _JOINERS:  # strip trailing legal suffixes
        toks.pop()
    if toks and toks[0] == "the" and len(toks) > 1:
        toks = toks[1:]
    return toks


def norm_key(name: str) -> str:
    return "".join(tokens(name))


def weak_key(name: str) -> str:
    toks = tokens(name)
    while len(toks) > 1 and toks[-1] in _WEAK:
        toks.pop()
    return "".join(toks)


def clean_display(name: str) -> str:
    s = " ".join(name.replace(" ", " ").split()).strip(" ,.-–—|")
    s = re.sub(r"\s*\((?:yc|s|w|f)\s*\d{2,4}\)\s*$", "", s, flags=re.I)
    while True:
        m = re.search(r",?\s+([A-Za-z.]+)\.?$", s)
        suffix = m.group(1).lower().strip(".") if m else ""
        rest = s[: m.start()].rstrip(" ,") if m else ""
        last = rest.split()[-1].lower() if rest.split() else ""
        if m and suffix in _LEGAL and suffix not in _DISPLAY_KEEP and len(rest) > 1                 and last not in _JOINERS and last != "&":
            s = rest
        else:
            return s


@dataclass
class Resolution:
    raw: str
    canonical: str
    method: str
    score: float


@dataclass
class _Entity:
    canonical: str
    domains: set[str] = field(default_factory=set)
    seed: bool = False


class EntityResolver:
    FUZZY_MIN_LEN = 8
    FUZZY_THRESHOLD = 0.88  # OSA similarity: 1 typo/transposition per ~9 chars

    def __init__(self, seed_path: Path | None = None) -> None:
        self.entities: list[_Entity] = []
        self.by_alias: dict[str, int] = {}
        self.by_key: dict[str, int] = {}
        self.by_weak: dict[str, set[int]] = {}
        self.by_domain: dict[str, int] = {}
        self.log: list[dict] = []
        if seed_path:
            for row in json.loads(Path(seed_path).read_text(encoding="utf-8")):
                self._add(row["canonical"], row.get("aliases", []), row.get("domains", []), seed=True)

    # ------------------------------------------------------------- registry
    def _add(self, canonical: str, aliases: list[str], domains: list[str], seed: bool) -> int:
        idx = len(self.entities)
        self.entities.append(_Entity(canonical, {d.lower() for d in domains}, seed))
        for name in [canonical, *aliases]:
            self.by_alias.setdefault(name.casefold().strip(), idx)
            k = norm_key(name)
            if k:
                self.by_key.setdefault(k, idx)
                self.by_weak.setdefault(weak_key(name), set()).add(idx)
        for d in domains:
            self.by_domain.setdefault(d.lower(), idx)
        return idx

    def _domain_conflict(self, idx: int, domain: str | None) -> bool:
        ent = self.entities[idx]
        return bool(domain and ent.domains and domain not in ent.domains
                    and not any(domain.endswith("." + d) for d in ent.domains))

    def _hit(self, idx: int, raw: str, method: str, score: float, domain: str | None) -> Resolution:
        if domain:
            self.entities[idx].domains.add(domain)
            self.by_domain.setdefault(domain, idx)
        return Resolution(raw, self.entities[idx].canonical, method, score)

    # ------------------------------------------------------------- resolve
    def resolve(self, raw: str, *, website: str | None = None, record_type: str = "",
                source_url: str | None = None) -> Resolution:
        res = self._resolve(raw, self._identity_domain(website))
        self.log.append({"raw": raw, "canonical": res.canonical, "method": res.method, "score": round(res.score, 3),
                         "recordType": record_type, "sourceUrl": source_url})
        return res

    _GENERIC_DOMAINS = ("github.com", "github.io", "vercel.app", "netlify.app", "herokuapp.com", "notion.site",
                        "producthunt.com", "ycombinator.com", "google.com", "apple.com", "chrome.google.com",
                        "huggingface.co", "medium.com", "substack.com", "linkedin.com", "twitter.com", "x.com")

    # Paths where a platform hosts things its *users* made: a product at
    # claude.ai/artifact/... is by its author, not by Anthropic.
    _USER_CONTENT = re.compile(
        r"^https?://(?:www\.)?(?:claude\.ai/(?:artifact|public|share)|chatgpt\.com/(?:g|share|canvas)|"
        r"chat\.openai\.com/(?:g|share)|gemini\.google\.com/(?:share|gem)|g\.co/gemini|"
        r"poe\.com/|copilot\.microsoft\.com/shares)", re.I)
    _USER_CONTENT_HOSTS = ("claude.site", "lovable.app", "replit.app", "repl.co", "bolt.host", "v0.dev",
                           "glitch.me", "streamlit.app", "gradio.live", "pages.dev", "web.app", "firebaseapp.com")

    def _identity_domain(self, website: str | None) -> str | None:
        if not website or self._USER_CONTENT.match(website.strip()):
            return None
        d = domain_of(website)
        if d and any(d == h or d.endswith("." + h) for h in self._USER_CONTENT_HOSTS):
            return None
        return d

    def _resolve(self, raw: str, domain: str | None) -> Resolution:
        raw = (raw or "").strip()
        if domain and any(domain == g or domain.endswith("." + g) for g in self._GENERIC_DOMAINS):
            domain = None  # hosting/platform domains say nothing about identity
        key, wkey = norm_key(raw), weak_key(raw)
        if not key:
            return Resolution(raw, raw, "unresolvable", 0.0)

        # 1. exact alias
        idx = self.by_alias.get(raw.casefold())
        if idx is not None and self.entities[idx].seed and not self._domain_conflict(idx, domain):
            return self._hit(idx, raw, "seed_exact", 1.0, domain)
        # 2. domain
        if domain:
            idx = self.by_domain.get(domain)
            if idx is not None:
                return self._hit(idx, raw, "seed_domain" if self.entities[idx].seed else "dynamic_domain", 0.99, domain)
        # 3. normalised key
        idx = self.by_key.get(key)
        if idx is not None and not self._domain_conflict(idx, domain):
            return self._hit(idx, raw, "seed_normalized" if self.entities[idx].seed else "dynamic_merge", 0.97, domain)
        # 4. weak key (unique only)
        cands = sorted(self.by_weak.get(wkey, set()))
        cands = [c for c in cands if not self._domain_conflict(c, domain)]
        if len(cands) == 1 and len(wkey) >= 3:
            idx = cands[0]
            return self._hit(idx, raw, "seed_weak" if self.entities[idx].seed else "dynamic_merge_weak", 0.9, domain)
        # 5. fuzzy on keys (typos), blocked by first char for scale
        if len(key) >= self.FUZZY_MIN_LEN:
            block = [k for k in self.by_key if k[:1] == key[:1] and abs(len(k) - len(key)) <= 2]
            if block:
                match = process.extractOne(key, block, scorer=OSA.normalized_similarity,
                                           score_cutoff=self.FUZZY_THRESHOLD)
                if match:
                    idx = self.by_key[match[0]]
                    if not self._domain_conflict(idx, domain):
                        m = "seed_fuzzy" if self.entities[idx].seed else "dynamic_fuzzy"
                        return self._hit(idx, raw, m, match[1], domain)
        # 6. new dynamic entity
        display = clean_display(raw) or raw
        method = "dynamic_new"
        if display.casefold() in self.by_alias or key in self.by_key:  # name taken by a different (domain) entity
            display = f"{display} ({domain})" if domain else display
            method = "dynamic_new_domain_conflict"
        idx = self._add(display, [raw] if raw != display else [], [domain] if domain else [], seed=False)
        return Resolution(raw, display, method, 1.0)

    def mapping_rows(self) -> list[dict]:
        seen, out = set(), []
        for r in self.log:
            k = (r["raw"], r["recordType"])
            if k not in seen:
                seen.add(k)
                out.append(r)
        return out
