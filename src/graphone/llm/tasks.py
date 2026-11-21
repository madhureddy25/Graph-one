"""Extraction tasks: prompt + strict validator (schema + enum + grounding).

Design rule: the LLM never supplies URLs, dates or IDs (those come from the
crawler), and every factual value must come with a verbatim evidence quote
that we verify against the source. Classification labels (enums) are allowed
because they are judgments over grounded text, not new facts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..models import ROLE_FAMILIES, PricingModel
from .grounding import ground_list, is_grounded


@dataclass(frozen=True)
class ExtractionTask:
    name: str
    version: str
    system: str
    user_template: str
    validate: Callable[[dict, str], dict]
    query_terms: tuple[str, ...] = ()

    def render(self, text: str, **kw) -> str:
        return self.user_template.format(text=text, **kw)


_SYS = (
    "You are a precise information-extraction engine. Use ONLY the provided text. "
    "Never guess, never use outside knowledge. If a value is not explicitly supported "
    "by the text, return null (or an empty list). Evidence fields must be exact, "
    "verbatim quotes copied from the text. Respond with a single JSON object only."
)

# ----------------------------------------------------------------- news
EVENT_TYPES = ["FUNDING", "ACQUISITION", "PRODUCT_LAUNCH", "PARTNERSHIP", "RESEARCH", "POLICY_REGULATION",
               "HIRING_LAYOFFS", "LEGAL", "EARNINGS", "OTHER"]


def _validate_news(d: dict, source: str) -> dict:
    orgs, dropped = ground_list(d.get("organizations") or [], source)
    ev = d.get("event_type")
    ev = ev if ev in EVENT_TYPES else None
    if ev and ev != "OTHER" and not is_grounded(str(d.get("event_evidence") or ""), source):
        ev = None
    return {"organizations": orgs[:25], "event_type": ev, "_dropped": dropped}


NEWS_TASK = ExtractionTask(
    name="news_entities", version="2", system=_SYS, validate=_validate_news,
    query_terms=("raised", "launch", "acquire", "announced", "funding", "model"),
    user_template=(
        "Extract from this AI news article:\n"
        '- "organizations": companies/labs/institutions named in the text, spelled exactly as in the text\n'
        f'- "event_type": one of {EVENT_TYPES}\n'
        '- "event_evidence": the exact sentence from the text that shows the event type\n'
        'Return JSON: {{"organizations": [...], "event_type": "...", "event_evidence": "..."}}\n\n'
        "ARTICLE:\n{text}"
    ),
)

# ----------------------------------------------------------------- jobs


def _validate_job(d: dict, source: str) -> dict:
    rf = d.get("role_family")
    rf = rf if rf in ROLE_FAMILIES else None
    remote = d.get("is_remote")
    if not isinstance(remote, bool) or not is_grounded(str(d.get("remote_evidence") or ""), source):
        remote = None
    return {"role_family": rf, "is_remote": remote}


JOB_TASK = ExtractionTask(
    name="job_classify", version="2", system=_SYS, validate=_validate_job,
    query_terms=("remote", "hybrid", "on-site", "onsite", "office", "location", "responsibilities"),
    user_template=(
        f'Classify this job posting.\n- "role_family": one of {ROLE_FAMILIES}\n'
        '- "is_remote": true if the text says the role can be done remotely, false if it says on-site/hybrid '
        "only, null if not stated\n"
        '- "remote_evidence": exact quote from the text supporting is_remote (empty if null)\n'
        'Return JSON: {{"role_family": "...", "is_remote": true|false|null, "remote_evidence": "..."}}\n\n'
        "JOB POSTING:\n{text}"
    ),
)

# ----------------------------------------------------------------- pricing
_PRICING = [p.value for p in PricingModel]


def _validate_pricing(d: dict, source: str) -> dict:
    pm = d.get("pricing_model")
    ev = str(d.get("evidence") or "")
    if pm not in _PRICING or not is_grounded(ev, source):
        return {"pricing_model": None, "evidence": None}
    return {"pricing_model": pm, "evidence": ev.strip()[:300]}


PRICING_TASK = ExtractionTask(
    name="pricing_model", version="2", system=_SYS, validate=_validate_pricing,
    query_terms=("pricing", "free", "plan", "month", "/mo", "enterprise", "contact sales", "trial", "$"),
    user_template=(
        "Determine the pricing model of the product described on this website text.\n"
        "FREE = entirely free / open source with no paid plan; FREEMIUM = a free plan or tier plus paid plans; "
        "PAID = only paid plans (a free *trial* is still PAID); ENTERPRISE = only custom/contact-sales pricing.\n"
        'Return JSON: {{"pricing_model": "FREE|FREEMIUM|PAID|ENTERPRISE" or null, '
        '"evidence": "exact quote from the text that proves it"}}\n\nWEBSITE TEXT:\n{text}'
    ),
)
