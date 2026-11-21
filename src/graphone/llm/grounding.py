"""Anti-hallucination guard: every LLM-asserted string must be traceable to the
source text it was extracted from. Ungrounded values are dropped, not 'fixed'."""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

_TRANS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "})


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").translate(_TRANS).lower()
    return re.sub(r"\s+", " ", s).strip()


def is_grounded(value: str, source: str, *, fuzzy_threshold: int = 92, normalized_source: str | None = None) -> bool:
    v = norm(value)
    if not v:
        return False
    src = normalized_source if normalized_source is not None else norm(source)
    if v in src:
        return True
    if len(v) < 12:  # short strings (names) must match exactly
        return False
    # longer quotes may differ by whitespace/punctuation from trafilatura output
    return fuzz.partial_ratio(v, src) >= fuzzy_threshold


def ground_list(values, source: str) -> tuple[list[str], list[str]]:
    src = norm(source)
    kept, dropped = [], []
    seen = set()
    for v in values or []:
        if not isinstance(v, str):
            continue
        key = norm(v)
        if key in seen:
            continue
        seen.add(key)
        (kept if is_grounded(v, source, normalized_source=src) else dropped).append(v.strip())
    return kept, dropped
