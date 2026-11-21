"""Live GitHub repository metrics.

GraphQL lets us batch 50 repositories per request with aliases, so 500k papers
need ~10k requests (2 hours inside the 5,000 points/h budget of one token; N
tokens / N nodes scale linearly). Falls back to REST per repo, then to the
snapshot value supplied by the paper source (flagged as such).
"""
from __future__ import annotations

import logging
import re

from ..http.fetcher import FetchError
from ..logging_setup import METRICS, log
from .base import Context

logger = logging.getLogger("graphone.github")
_REPO = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.I)


def parse_repo(url: str | None) -> tuple[str, str] | None:
    if not url:
        return None
    m = _REPO.search(url)
    if not m:
        return None
    owner, name = m.group(1), re.sub(r"\.git$", "", m.group(2))
    if owner.lower() in {"orgs", "topics", "features", "sponsors", "apps", "marketplace"}:
        return None
    return owner, name


def _gql_str(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


async def fetch_stars(ctx: Context, repos: list[tuple[str, str]], batch: int = 50) -> dict[str, dict]:
    """Return {'owner/name' (lower): {'stars': int|None, 'url': str, 'source': str}}."""
    out: dict[str, dict] = {}
    token = ctx.settings.github_token
    uniq = sorted({(o, n) for o, n in repos}, key=lambda x: (x[0].lower(), x[1].lower()))
    if token:
        for i in range(0, len(uniq), batch):
            chunk = uniq[i:i + batch]
            fields = " ".join(
                f'r{j}: repository(owner:"{_gql_str(o)}", name:"{_gql_str(n)}"){{nameWithOwner stargazerCount url}}'
                for j, (o, n) in enumerate(chunk)
            )
            try:
                res = await ctx.fetcher.get("https://api.github.com/graphql", method="POST",
                                            json_body={"query": f"query{{ {fields} rateLimit{{remaining}} }}"},
                                            headers={"Authorization": f"bearer {token}"}, accept="application/json",
                                            allow_browser=False, check_robots=False)
                data = (res.json().get("data") or {})
            except FetchError as exc:
                log(logger, logging.WARNING, "graphql batch failed; will try REST", err=str(exc)[:200])
                data = {}
            for j, (o, n) in enumerate(chunk):
                node = data.get(f"r{j}")
                if node:  # null => repo deleted / private / renamed-away
                    out[f"{o}/{n}".lower()] = {"stars": node["stargazerCount"], "url": node["url"],
                                               "source": "github_graphql"}
                    METRICS.inc("github.graphql_ok")
    # REST fallback for anything still missing (also used without a token: 60 req/h)
    missing = [(o, n) for o, n in uniq if f"{o}/{n}".lower() not in out]
    budget = len(missing) if token else min(len(missing), 50)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for o, n in missing[:budget]:
        try:
            d = await ctx.fetcher.get_json(f"https://api.github.com/repos/{o}/{n}", headers=headers)
            out[f"{o}/{n}".lower()] = {"stars": d.get("stargazers_count"), "url": d.get("html_url"),
                                       "source": "github_rest"}
            METRICS.inc("github.rest_ok")
        except FetchError as exc:
            METRICS.inc("github.rest_failed")
            if exc.status in (401, 403):
                log(logger, logging.WARNING, "GitHub API rate-limited/forbidden; set GITHUB_TOKEN")
                break
    return out
