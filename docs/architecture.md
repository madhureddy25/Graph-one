# GraphOne Ingestion Pipeline — Architecture

**Goal:** a fault-tolerant, asynchronous pipeline that bulk-acquires startups, products and research papers (scaling to 500k+), continuously ingests 24-hour-fresh AI news and jobs, structures them with a multi-tier LLM chain, and canonicalises entities. Every record carries its source URL plus provenance fields (`dateSource`, `pricingEvidence`, `github_stars_source`, …).

```
            ┌──────────── config/sources.toml (registry) ────────────┐
 Discovery  │ YC index · Show HN/Algolia · Product Hunt · HF Papers   │  Phase II: RSS feeds · Greenhouse/Lever/Ashby
            └───────────────┬────────────────────────────────────────┘           RemoteOK · Remotive APIs
                            ▼
   Work units (slug / domain / day / URL) ──hash % N──► shard k of N  (--shard k/N, one process/container each)
                            ▼
   Fetcher (asyncio + aiohttp): global pool · per-host semaphore + AIMD token bucket · retry/backoff/Retry-After
        │ challenge detected (Cloudflare / DataDome / PerimeterX)
        └──► Playwright Chromium tier (persistent per-host context) ──► still CAPTCHA ⇒ host marked blocked, use alt. source
                            ▼
   Parse → full text (trafilatura) → date engine → freshness gate → dedup gate (URL claim + SimHash/LSH)
                            ▼
   LLM orchestrator: budget-fit → Gemini Flash → Groq Llama 3.3 → DeepSeek (→ OpenRouter) → schema + grounding check
                            ▼
   Entity resolver (seed of 50 + dynamic clusters) → records → state store / sinks → CSV · JSONL · XLSX · Google Sheets
```

## 1. Scale strategy — 500,000 records without manual intervention

* **API-first discovery.** Each vertical starts from an index that enumerates entities cheaply: YC's directory index (~5k companies in one request), Algolia's HN index walked by *time windows* (bypasses the 1,000-hit pagination cap), Hugging Face daily-paper feeds (one request per day, 100 papers), Product Hunt GraphQL cursors. At 500k we add more indexes of the same shape (Crunchbase/Dealroom exports, sitemaps of directories, arXiv OAI-PMH for papers); each is a config entry plus a small adapter.
* **Bounded worker pools, constant memory.** `bounded_map` streams work units through an `asyncio.Queue` with N workers, so 1k and 500k items use the same memory; one failing item is logged and never kills a run.
* **Horizontal sharding with no code changes.** Every work unit has a stable key (startup slug, product domain, paper day, canonical URL). `--shard k/N` makes a node process only keys with `md5(key) % N == k`. Scaling = launching N containers (K8s Job with `completions=N`, or N ECS tasks) and setting `STATE_BACKEND=redis`.
* **Batching expensive enrichment.** GitHub stars use GraphQL aliasing (50 repos/request ⇒ 500k papers ≈ 10k requests; with a token pool of 5 tokens × 5,000 points/h this is ~25 min). arXiv metadata uses `id_list` batches of 100 at arXiv's requested 1 req/3 s.
* **Politeness is a first-class budget.** Per-host token buckets (`HOST_RPS` overrides: arXiv 0.34 rps, YC 4 rps) with AIMD: a 429/503 halves that host's rate, successes recover it. This is what lets a 64-connection crawler run for hours without bans.
* **Resumability.** All dedup claims, checkpoints and processed records live in the state store; a crashed node restarts and skips everything already claimed or stored.

## 2. Handling 413s and 429s across thousands of concurrent extractions

**413 / context overflow — prevented, then handled.**
1. *Budget per tier.* Each provider has an input budget below its real limit; Groq's is tightest (4.5k tokens) because its free tier rejects any single request larger than the TPM cap with **HTTP 413**.
2. *Conservative token estimate* (3.2 chars/token; CJK = 1 token/char) so the real count is always below our estimate.
3. *Semantic truncation* (`fit_to_budget`): split into paragraphs, drop boilerplate (cookie/newsletter/nav text), score each paragraph by information density (money, numbers, dates, proper nouns, task keywords) plus a lead-paragraph prior, greedily keep the best paragraphs that fit, re-assemble in original order with `[…]` markers. Long docs can also use paragraph-aligned map-reduce chunks (`chunk_text`).
4. *If a 413/context error still occurs,* the orchestrator halves the budget and retries the **same** tier immediately (no sleep); below 10 % it fails over.

**429 — shared budgets, backoff, failover.**
1. Each tier has two async token buckets shared by all coroutines: **RPM** (requests) and **TPM** (estimated tokens). Calls wait locally *before* the provider would reject them; with N nodes, budgets are divided by N (or moved to Redis with a Lua token bucket).
2. A global semaphore (`LLM_CONCURRENCY`) caps in-flight calls.
3. On 429: wait `Retry-After` / provider hint ("try again in 7.66s", Gemini `retryDelay`) plus jitter, else exponential backoff with equal jitter (`base·2^n/2 + U(0, base·2^n/2)`), which de-synchronises thousands of workers. The TPM bucket is also multiplicatively reduced (AIMD).
4. If the wait exceeds 45 s, or the error is a *daily quota* (TPD/RPD/RESOURCE_EXHAUSTED), the tier's **circuit breaker** opens and traffic flows to the next tier. Auth errors disable a tier for the run; 5xx trip the breaker after 5 consecutive failures.
5. Results are cached by `(task, version, sha1(text))`, so re-runs and duplicate pages cost zero tokens.

**Hallucination control.** The LLM never produces URLs, dates or IDs (the crawler does). Every LLM fact must be accompanied by a verbatim evidence quote, which is verified against the source text (normalised substring, fuzzy only for long quotes). Ungrounded values are dropped to `null`, never "repaired".

## 3. Freshness tracking — never processing the same article/job twice

* **Identity.** Articles: canonical URL (lower-case host, no `www`, tracking params stripped, sorted query, no fragment) → SHA-1. Jobs: board-native IDs (`gh:anthropic:123`, `lever:<uuid>`).
* **Atomic claims.** Before work, a node calls `claim(ns, key)`: `INSERT OR IGNORE` on SQLite, or `SET key NX EX ttl` on Redis when nodes are distributed. Only one node ever wins. In-flight claims carry a TTL and are released on transient failure, so a crash never loses an item permanently. Completed records are persisted; subsequent runs skip them before any fetch.
* **Near-duplicates.** Syndicated or lightly edited copies are caught with a 64-bit SimHash over word 3-shingles, indexed by 4 LSH bands (16 bits each) so lookup is 4 exact key gets; Hamming distance ≤ 3 ⇒ duplicate.
* **Strict 24h guarantee.** Only *publication* timestamps prove freshness. Date engine priority: JSON-LD `datePublished` → `article:published_time`-family meta → `<time>` → byline text (absolute or relative "2 hours ago") → feed date → URL path. `dateModified` is deliberately ignored (an old article edited today is not fresh); feed/page conflicts > 36 h are logged and the page wins. Day-precision dates are accepted only if they are *today* (the only day guaranteed < 24 h). Greenhouse's `updated_at` is treated as an upper bound, not a publication date.
* **Undated content heuristics.** (a) *Listing-order inference:* in a reverse-chronological feed, an undated item above a fresh dated item is newer than it. (b) *First-seen:* every run records all observed IDs per source; if the previous crawl of that source was < 24 h ago and an ID was absent then, it appeared within the window. On a cold start neither proof exists, so the item is excluded (counted as `undetermined_cold_start`), favouring precision over recall.
* **Export-time window.** The Jobs/News tabs are re-filtered at export time, so a sheet never shows an item older than 24 h even if the crawl ran earlier.

## 4. Storage strategy

| Concern | Choice | Why |
|---|---|---|
| Primary store | **PostgreSQL** (JSONB + typed columns) | Records are semi-structured but must be queryable and constrained (unique `(record_type, canonical_url)`, foreign keys to canonical entities); JSONB keeps `schemaVersion` evolution cheap; mature partitioning by `collected_at` handles 10M+ rows. The SQLite store here has the same tables, so the swap is a backend change. |
| Coordination | **Redis** | Atomic `SET NX` claims, SimHash band index, token buckets shared across nodes, all with TTLs; sub-ms latency in the hot path. |
| Vectors | **pgvector** (HNSW) in the same Postgres | Embeddings of entity descriptions/news for semantic search and resolver candidate blocking, joinable with relational filters in one query; no second consistency domain until > ~50M vectors (then Qdrant/Weaviate). |
| Graph | **Neo4j** (or Apache AGE on Postgres) | The intelligence graph is many-hop (founder → startup → product → paper → repo → job/news mention). Postgres stays the source of truth; a CDC stream (Debezium → Kafka) projects canonical entities and edges into the graph, so graph queries never block ingestion. |
| Raw archive | **S3 / object storage** (gzip HTML + JSON) | Re-parse history without re-crawling; audit trail for every extracted value. |
| Queue at scale | **Kafka / SQS** | Decouples discovery from fetch from LLM extraction, so each stage scales and back-pressures independently. |

## 5. Anti-bot strategy (Cloudflare, DataDome, heavy JS)

1. **Don't fight the WAF if you can avoid it.** Prefer official APIs, RSS/Atom, sitemaps and JSON embedded in the HTML (`__NEXT_DATA__`, Inertia `data-page`, JSON-LD). Every source in this repo is collected this way; this is the cheapest and most stable path.
2. **Look like one consistent, slow browser.** A coherent header set (UA, Accept, Accept-Language), cookies kept per session, per-host concurrency ≤ 4 and adaptive rate (AIMD on 429/503), `Retry-After` honoured, robots.txt respected (`RESPECT_ROBOTS`).
3. **Detect, then escalate.** `detect_challenge` recognises Cloudflare ("Just a moment…", `cf-chl`, `cf-mitigated`), DataDome (`x-datadome`, captcha-delivery) and PerimeterX. A challenged URL is re-fetched in **Playwright Chromium**, which runs the JS check like a visitor's browser. The per-host persistent context keeps the clearance cookie for the rest of the crawl; images/fonts/media are blocked to save bandwidth.
4. **Network reputation.** `PROXY_URL` routes HTTP and browser traffic through a residential/ISP gateway with sticky sessions per host when datacenter IPs are flagged; or a commercial unblocker/rendering API for the few highest-value domains.
5. **Stop at interactive CAPTCHAs.** If a challenge persists after rendering, the host is marked blocked for the run and the entity is sourced elsewhere. CAPTCHAs are never solved or outsourced; for high-value sources the long-term answer is a data partnership or licensed feed.

## 6. Entity resolution

The cascade runs exact alias → website-domain → normalised key → weak key (generic suffixes like *AI/Labs* removed, unique only) → OSA/Damerau typo match (≥ 0.88 on keys ≥ 8 chars, blocked by first letter) → new dynamic cluster. Normalisation is NFKD-folded, case-folded, `&`→`and`, punctuation removed and legal suffixes stripped (*Inc, LLC, Ltd, PBC, GmbH…*), so "OpenAI", "OpenAI, Inc." and "Open AI" all key to `openai`. A **domain-conflict guard** rejects name-only matches when both sides have different known domains. Hosting domains (github.io, vercel.app…) are ignored as identity signals. Names are resolved in sorted order, so output is deterministic. Every decision (raw, canonical, method, score, source URL) is written to the Entity Mapping Log.
