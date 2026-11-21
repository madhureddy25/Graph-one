# GraphOne Ingestion Pipeline

Asynchronous, fault-tolerant ingestion pipeline for the AI ecosystem:

* **Phase I:** bulk startups, products and research papers, with live GitHub stars
* **Phase II:** 24-hour-fresh news and jobs
* **Phase III:** multi-tier LLM extraction with 413/429 handling
* **Phase IV:** deterministic entity resolution
* **Phase V:** anti-bot escalation
* **Phase VI:** architecture design. Full design: [`architecture.pdf`](architecture.pdf) (source: [`docs/architecture.md`](docs/architecture.md)).

> **No hallucinated data.** Every record links to the page or API it came from. The LLM never produces URLs, dates or IDs. Every LLM-asserted value must come with a verbatim evidence quote, which is checked against the source text; if the check fails, the value is dropped.

## Quick start (VS Code, Windows / macOS / Linux)

Requires **Python 3.11+**.

```bash
git clone https://github.com/madhureddy25/Graph-one.git && cd Graph-one
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium            # browser tier for JS / Cloudflare pages
copy .env.example .env                 # macOS/Linux: cp .env.example .env
```

Fill in `.env`. Every key is free:

| Key | Where | Used for |
|---|---|---|
| `GITHUB_TOKEN` | github.com → Settings → Developer settings → *Fine-grained token*, no scopes | live star counts (GraphQL, 50 repos/request) |
| `GEMINI_API_KEY` | aistudio.google.com/apikey | LLM tier 1 |
| `GROQ_API_KEY` | console.groq.com/keys | LLM tier 2 (Llama 3.3 70B) |
| `DEEPSEEK_API_KEY` | platform.deepseek.com | LLM tier 3 |
| `OPENROUTER_API_KEY` *(optional)* | openrouter.ai/keys | LLM tier 4 |
| `PRODUCT_HUNT_TOKEN` *(optional)* | producthunt.com/v2/oauth/applications | second product source |

Every LLM key is optional. Tiers without a key are skipped. With no keys, the pipeline runs in rules-only mode and marks records as such.

```bash
python -m pytest -q                    # 56 offline tests (dates, resolver, chunking, fallback chain, fetcher, news e2e)
python -m graphone llm-check           # pings each configured LLM tier
python -m graphone phase1              # 1,000 startups + 1,000 products + 1,000 papers  (~15–30 min)
python -m graphone phase2              # fresh news + jobs (run again an hour later: see "freshness")
python -m graphone reextract           # backfill LLM fields on news saved while every LLM tier was out of quota
python -m graphone reresolve           # re-run entity resolution on saved records after a resolver change (backs up first)
python -m graphone export              # output/*.csv, *.jsonl and output/graphone_output.xlsx (6 tabs)
```

(If `python -m graphone` is not found, run `pip install -e .` once, or set `PYTHONPATH=src`.)

Useful flags:

* `phase1 --only papers --papers 200` runs a single vertical with a smaller target.
* `--shard 0/4` partitions the work across 4 nodes.

### Getting the public Google Sheet

* **Option A (simplest):** upload `output/graphone_output.xlsx` to Google Drive. Choose *Open with Google Sheets*, then *File → Save as Google Sheets*. Under *Share*, set *Anyone with the link → Viewer*.
* **Option B (automated):** create a Google Cloud service account with the Sheets and Drive APIs enabled, and save its JSON as `service_account.json`. Then run `python -m graphone export --sheets`. It prints the public URL. Set `GOOGLE_SHEET_ID` to update an existing sheet that you shared with the service account.

## Sources

| Vertical | Source (all public APIs / feeds) | Notes |
|---|---|---|
| Startups | Y Combinator directory index, each company verified against its live `ycombinator.com/companies/<slug>` page | `employeeCount` read from the live page (`employeeCountSource`) |
| Products | Hacker News *Show HN* (Algolia API, time-window pagination), plus Product Hunt GraphQL if a token is set | Pricing is classified from the product's own site and `/pricing` page. Rules run first, then the LLM; the verbatim `pricingEvidence` and its URL are stored |
| Papers | Hugging Face Papers (the successor to Papers with Code, which redirects there), arXiv API for exact timestamps, GitHub GraphQL for current stars | Only papers that have a code repo. `github_stars_source` records where each star count came from |
| News (5) | TechCrunch AI, VentureBeat AI, The Verge AI, MIT Tech Review AI, Ars Technica AI | RSS discovery, then full-text crawl of each article, plus the date engine |
| Jobs (5) | Greenhouse, Lever and Ashby (public ATS APIs, about 60 AI companies), RemoteOK, Remotive | Only *publication* timestamps count as proof of freshness |

Add or remove feeds, boards and companies in [`config/sources.toml`](config/sources.toml). No code changes are needed.

## Architecture

```
src/graphone/
  http/fetcher.py      async aiohttp crawler: pooled, per-host semaphores + AIMD token buckets,
                       retries w/ exponential backoff + jitter, Retry-After, robots.txt, proxy
  http/antibot.py      Cloudflare / DataDome / PerimeterX detection -> Playwright Chromium tier
  http/backoff.py      backoff, Retry-After parsing (incl. "try again in 7.66s"), async token bucket
  dates.py             date engine: JSON-LD > meta > <time> > byline (relative "2 hours ago") > feed > URL;
                       strict 24h gate; listing-order + first-seen heuristics for undated items
  text.py              full-text extraction, URL canonicalisation, SimHash + LSH bands (near-dup)
  llm/providers.py     Gemini / Groq / DeepSeek / OpenRouter adapters + normalised error taxonomy
  llm/chunking.py      token estimation + semantic-density truncation (prevents 413s)
  llm/orchestrator.py  fallback chain, RPM/TPM buckets, circuit breakers, 413 shrink-and-retry, cache
  llm/grounding.py     evidence verification (anti-hallucination)
  llm/tasks.py         news / job / pricing extraction prompts + strict validators
  resolver/resolver.py deterministic entity resolution (seed of 50 + dynamic clusters), mapping log
  state/store.py       atomic dedup claims, checkpoints, records — SQLite or Redis (STATE_BACKEND)
  sources/*.py         startups, products, papers (+github), news, jobs
  export/exporter.py   JSONL / CSV / XLSX (6 tabs) / Google Sheets
  cli.py               `python -m graphone ...`
data/seed_entities.json  50 canonical AI companies with aliases + domains
config/sources.toml      source registry
```

Key decisions (details in `architecture.pdf`):

* **413s are prevented, then handled.**
  * Each LLM tier has its own token budget. Groq's is the smallest, because Groq's free tier returns 413 for any request above its per-minute token limit.
  * Payloads are cut down by information density, so the most data-rich paragraphs are kept.
  * If a 413 still happens, the budget is halved and the call is retried on the same tier.
* **429s:**
  * RPM and TPM token buckets are shared across all coroutines.
  * The pipeline waits for `Retry-After`, or uses backoff with equal jitter.
  * Long waits, daily quotas and auth errors open a circuit breaker, and traffic moves to the next tier.
* **Never processing an item twice:**
  * Canonical URL and board job ID are claimed atomically: `INSERT OR IGNORE` locally, `SET NX` in Redis for multiple nodes.
  * SimHash/LSH catches syndicated near-duplicates.
* **Freshness:**
  * `dateModified` is ignored.
  * Day-precision dates count only if they are today.
  * Greenhouse `updated_at` is treated only as an upper bound.
  * Undated items are admitted only through the listing-order or first-seen proofs.
  * On a cold start, items without proof are excluded rather than guessed.

### Freshness and the second run

The *first-seen* heuristic needs a previous crawl of the source less than 24 hours earlier. Run `phase2` twice, about an hour apart, before exporting. The second run admits newly appeared undated items. Items that were already processed are never re-processed.

## Outputs

`output/` contains:

* `startups/products/papers/jobs/news/entity_mapping` `.csv` and `.jsonl`
* `graphone_output.xlsx`, with tabs *Startups, Products, Research Papers, Jobs, News, Entity Mapping Log*
* `run_summary.json`, with counters for requests, retries, 429s/413s per LLM tier, failovers, browser escalations, stale/duplicate drops and more
* `state.sqlite3`

## Limitations

Known gaps and deliberate trade-offs, such as precision over recall on freshness and pricing, are listed in [`LIMITATIONS.md`](LIMITATIONS.md).

## Scaling to 500k

* Start N workers with `--shard k/N` and set `STATE_BACKEND=redis`.
* Raise `HTTP_CONCURRENCY` and add GitHub tokens or LLM keys.

The code doesn't change. See section 1 of `architecture.pdf`.
