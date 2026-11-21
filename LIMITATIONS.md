# Known Limitations & Trade-offs

Deliberate choices and known gaps, stated plainly. Where precision and recall conflicted, I chose precision: a `null` or an excluded item is better than a wrong one.

## Freshness
- **Cold start excludes undated items.** On the very first Phase II run, there is no previous crawl to prove when an undated item appeared. Items without a publication date are therefore excluded, rather than guessed. A second run within 24 hours admits newly appeared items through the first-seen heuristic.
- **Greenhouse `updated_at` is not a publication date.** When a Greenhouse job has no `first_published` field, it needs the first-seen proof described above.
- **Day-precision dates** (for example, a date only in the URL path) count as fresh only if they are *today* (UTC). A date of "yesterday" might be more than 24 hours old.
- **The export window moves.** `export` keeps jobs and news from the 24 hours before the export runs, so exporting later drops items that have aged out. The Sheet is a snapshot from 21 November 2026.
- **Remotive** returns timezone-less timestamps. They are interpreted as UTC.

## Data coverage
- **Products from Show HN** are often early-stage. The "startup" behind a product is often the product's own name, which the resolver creates as a dynamic entity.
- **Pricing can be `null`** when:
  - the product's site was unreachable
  - the site sits behind a login
  - the site never states its pricing

  The pipeline never guesses pricing. Every non-null value has a verbatim `pricingEvidence` quote and its source URL.
- **`employeeCount`** is YC's self-reported team size. It is read from the live company page when that page is reachable; otherwise it comes from the directory index (see `employeeCountSource`).
- **Papers with Code** has been sunset and now redirects to Hugging Face Papers, which this pipeline uses. Only papers that link to a GitHub repository are included, as the task requires.
- **GitHub stars** are a snapshot at collection time (`github_stars_checked_at`). Without a `GITHUB_TOKEN`, the pipeline falls back to Hugging Face's cached star count and labels it `huggingface_snapshot`.

## LLM
- **Free-tier rate limits** make full LLM enrichment slow at scale. This is why deterministic rules run first and the LLM is a fallback. At production scale, use paid tiers or batch APIs, and move token buckets to Redis.
- **Free quotas ran out during this run.** Gemini, Groq (`gpt-oss-120b`) and OpenRouter all hit their daily limits. The circuit breakers failed over as designed, but some news was saved before any tier had capacity, so it had no LLM fields. `python -m graphone reextract` backfilled them later from the stored text. Groq has also retired Llama 3.3, so the Groq tier now uses `openai/gpt-oss-20b` with `reasoning_effort: low`.
- **Grounding is lexical.** Evidence quotes must appear in the source text, so paraphrased-but-correct answers are rejected. This trades some recall for precision.

## Crawling
- **No CAPTCHA solving.** Hosts that still show an interactive CAPTCHA after browser rendering are skipped for that run. For high-value sources, the long-term answer is a data partnership or licensed feed.
- **Ars Technica blocked every article request in this run.** It answers bots with HTTP 405. The pipeline retried each article once in headless Chromium, was blocked again, and skipped it with a log line instead of using the RSS summary as "full text". The other four news sources worked.
- **Some ATS board slugs in `config/sources.toml` may not exist.** Companies change job platforms. Missing boards are logged (`board_missing`) and skipped; edit the list as needed.

## Infrastructure (prototype vs production)
- The prototype stores state in SQLite (single node). The Redis backend covers dedup claims and checkpoints for multiple nodes.
- PostgreSQL, pgvector, Neo4j and Kafka are part of the production design in `architecture.pdf`. They are not deployed in this repo.
