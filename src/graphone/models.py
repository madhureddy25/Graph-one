"""Canonical output schemas (spec: 'Expected Schemas').

Required spec fields are kept verbatim (including their camelCase / snake_case
mix). Extra provenance fields (e.g. dateSource, pricingEvidence) are additive so
that every value can be audited back to its source URL.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "1.0"


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def to_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class PricingModel(str, Enum):
    FREE = "FREE"
    FREEMIUM = "FREEMIUM"
    PAID = "PAID"
    ENTERPRISE = "ENTERPRISE"


ROLE_FAMILIES = [
    "Engineering", "Research", "Data", "Product", "Design", "Sales", "Marketing",
    "Customer Success", "Operations", "People", "Legal & Policy", "Finance", "Other",
]


class Source(BaseModel):
    name: str
    url: str


class _Base(BaseModel):
    schemaVersion: str = SCHEMA_VERSION
    collectedAt: str = Field(default_factory=utcnow_iso)


# ---------------------------------------------------------------- startup
class StartupData(BaseModel):
    employeeCount: int | None = None
    website: str | None = None
    description: str | None = None
    batch: str | None = None
    location: str | None = None
    tags: list[str] = []
    rawName: str | None = None
    employeeCountSource: str | None = None  # "live_page" | "directory_index"


class StartupContent(BaseModel):
    entityName: str
    data: StartupData


class StartupRecord(_Base):
    recordType: Literal["STARTUP"] = "STARTUP"
    source: Source
    content: StartupContent


# ---------------------------------------------------------------- product
class ProductContent(BaseModel):
    productName: str
    startupName: str
    pricingModel: PricingModel | None = None
    website: str | None = None
    tagline: str | None = None
    launchedAt: str | None = None
    pricingEvidence: str | None = None   # verbatim quote from the product's own site
    pricingEvidenceUrl: str | None = None
    pricingMethod: str | None = None     # rules | llm:<provider> | none


class ProductRecord(_Base):
    recordType: Literal["PRODUCT"] = "PRODUCT"
    source: Source
    content: ProductContent


# ---------------------------------------------------------------- research paper
class PaperContent(BaseModel):
    title: str
    authors: list[str]
    paper_url: str
    github_url: str | None = None
    github_stars: int | None = None
    published_date: str | None = None
    arxiv_id: str | None = None
    github_stars_source: str | None = None   # github_graphql | github_rest | huggingface_snapshot
    github_stars_checked_at: str | None = None


class PaperRecord(_Base):
    recordType: Literal["RESEARCH_PAPER"] = "RESEARCH_PAPER"
    source: Source
    content: PaperContent


# ---------------------------------------------------------------- job
class JobContent(BaseModel):
    title: str
    company: str
    date: str
    is_remote: bool | None = None
    role_family: str | None = None
    location: str | None = None
    url: str
    rawCompany: str | None = None
    dateSource: str | None = None
    dateConfidence: str | None = None
    classificationMethod: str | None = None


class JobRecord(_Base):
    recordType: Literal["JOB"] = "JOB"
    source: Source
    content: JobContent


# ---------------------------------------------------------------- news
class NewsContent(BaseModel):
    title: str
    url: str
    published_date: str | None
    author: str | None = None
    full_text: str
    word_count: int
    organizations: list[str] = []
    event_type: str | None = None
    dateSource: str | None = None
    dateConfidence: str | None = None
    extractionMethod: str | None = None


class NewsRecord(_Base):
    recordType: Literal["NEWS"] = "NEWS"
    source: Source
    content: NewsContent


# ---------------------------------------------------------------- entity mapping
class EntityMapping(BaseModel):
    raw: str
    canonical: str
    method: str          # seed_exact | seed_normalized | seed_domain | seed_fuzzy | dynamic_new | dynamic_merge
    score: float
    recordType: str
    sourceUrl: str | None = None


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested dict -> dot-keyed flat dict (for CSV / Sheets)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        elif isinstance(v, list):
            out[key] = "; ".join(str(x) for x in v)
        elif isinstance(v, Enum):
            out[key] = v.value
        else:
            out[key] = v
    return out
