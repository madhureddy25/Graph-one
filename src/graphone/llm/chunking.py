"""Token budgeting + semantic-density truncation (prevents 413 / context errors).

We never send a payload whose *estimated* size exceeds the provider's budget.
The estimate is deliberately conservative (~3.2 chars/token for English; CJK
counted per char), so the real tokenizer count is lower than our estimate.
When a document is too big we do not cut the tail blindly; we keep the lead,
then the paragraphs with the highest information density (numbers, money,
dates, proper nouns, task keywords), and re-assemble them in original order.
"""
from __future__ import annotations

import math
import re

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
_BOILERPLATE = re.compile(
    r"(cookie|subscribe|newsletter|sign up for|all rights reserved|privacy policy|terms of (use|service)|"
    r"advertisement|share this|follow us|related articles|read more|click here|log in|create account)",
    re.I,
)
_DENSE = re.compile(
    r"(\$\s?\d|\d+(?:\.\d+)?\s?(?:%|million|billion|m\b|bn\b|k\b)|\b(?:19|20)\d{2}\b|"
    r"\b(?:raised|funding|series [a-e]|seed|acquir\w+|launch\w*|partner\w*|valuation|revenue|"
    r"per month|/mo|pricing|free|enterprise|remote|hybrid|salary)\b)",
    re.I,
)
_PROPER = re.compile(r"\b[A-Z][a-zA-Z0-9&.-]+(?:\s+[A-Z][a-zA-Z0-9&.-]+)*")

SEPARATOR = "\n[…]\n"


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    return math.ceil((len(text) - cjk) / 3.2) + cjk


def _paragraphs(text: str, max_chars: int = 1200) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[-•*] )", text) if p.strip()]
    out: list[str] = []
    for p in paras:  # split giant paragraphs on sentence boundaries
        while len(p) > max_chars:
            cut = p.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if p:
            out.append(p)
    return out


def _score(p: str, idx: int, terms: list[str]) -> float:
    if _BOILERPLATE.search(p) and len(p) < 300:
        return -1.0
    n = max(len(p), 1)
    dense = len(_DENSE.findall(p)) * 3 + len(_PROPER.findall(p))
    term_hits = sum(p.lower().count(t) for t in terms) * 4
    lead_bonus = 6.0 / (1 + idx)  # inverted-pyramid: early paragraphs matter
    return (dense + term_hits) / (n / 100) + lead_bonus


def fit_to_budget(text: str, max_tokens: int, query_terms: list[str] | None = None) -> tuple[str, bool]:
    """Return (text_within_budget, was_truncated)."""
    if estimate_tokens(text) <= max_tokens:
        return text, False
    terms = [t.lower() for t in (query_terms or [])]
    paras = _paragraphs(text)
    seen: set[str] = set()
    scored = []
    for i, p in enumerate(paras):
        key = re.sub(r"\W+", "", p.lower())[:200]
        if key in seen:
            continue
        seen.add(key)
        scored.append((_score(p, i, terms), i, p))
    budget = max_tokens - 8
    chosen: list[tuple[int, str]] = []
    used = 0
    for s, i, p in sorted(scored, key=lambda x: (-x[0], x[1])):
        if s < 0:
            continue
        cost = estimate_tokens(p) + 3
        if used + cost > budget:
            continue
        chosen.append((i, p))
        used += cost
    if not chosen:  # single gigantic paragraph: hard cut on a char budget
        return text[: int(budget * 3.2)], True
    chosen.sort()
    out, prev = [], None
    for i, p in chosen:
        if prev is not None and i != prev + 1:
            out.append(SEPARATOR.strip())
        out.append(p)
        prev = i
    return "\n\n".join(out), True


def chunk_text(text: str, max_tokens: int, overlap_paragraphs: int = 1) -> list[str]:
    """Paragraph-aligned chunks for map-reduce extraction over very long docs."""
    paras = _paragraphs(text)
    chunks, cur, cur_tok = [], [], 0
    for p in paras:
        t = estimate_tokens(p) + 2
        if cur and cur_tok + t > max_tokens:
            chunks.append("\n\n".join(cur))
            cur = cur[-overlap_paragraphs:] if overlap_paragraphs else []
            cur_tok = sum(estimate_tokens(x) + 2 for x in cur)
        cur.append(p if t <= max_tokens else fit_to_budget(p, max_tokens)[0])
        cur_tok += min(t, max_tokens)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks
