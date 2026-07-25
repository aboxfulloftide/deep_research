# Deep-Research Work Handoff

Last verified against the running system and `main`: **July 25, 2026**.

## Current State

- The durable processing queue is **active** and had no queued or running jobs
  at the time of this update.
- The primary llama.cpp service is running normally with Qwen3-14B.
- The nightly verification timer is active and runs at 11:00 PM. Counter-view
  checks for supported claims are queued after cron-triggered verification.
- The deletion of `PLAN_GPU_COORDINATOR.md` was intentionally committed and
  pushed in `5d76e4b`; it is no longer an outstanding worktree concern.
- Interactive Extra Research is available. Batch 1 (evidence correctness and
  observability), the domain-neutrality authority-gate fix, all of Batch 2
  (RRF fusion, typed provider-failure classification, adaptive waterfall,
  result caching, concurrent-request coalescing), and Batch 3 Round A (a
  shared SSRF-safe fetch contract plus the PDF-ingestion routing fix) are
  committed (`e5c7db1`, `6cc32ab`, `1ab7dd3`, `d8c9866`, `83ad2b1`).
  Capability routing is still cosmetic (no `SourceAdapter` registry), and
  nothing Extra Research collects is persisted to the KB yet -- it should
  still not be treated as a source-native, KB-integrated research system.

## What We Were Working On

Improve the general deep-research source-collection workflow before comparing
local models on analysis/synthesis quality.

The test question was:

> Research deeply the best local LLMs that can run with usable context windows
> in less than 180 GB of memory. Coding is the main use, but also general use
> would be used.

This question is only a benchmark. The design must work for arbitrary research
questions and must not hard-code model cards, benchmarks, or LLM-specific
source requirements.

## Implemented Work

### General Evidence-Facet Collection

Commit `5432c12` introduced a generic research plan:

- question ambiguities;
- evidence facets;
- facet coverage and source attempts;
- raw source bundle preserved before synthesis.

### Routed Collection and Source Fitness

Commit `b5b51ee` added:

- generic capability labels: `web`, `primary`, `scholarly`,
  `official_documentation`, `repository`, `news`, `local_knowledge`;
- adapter-routing records and bounded collection budgets;
- deterministic source-fitness fields for authority, lexical directness,
  extractability, and acceptance;
- coverage-driven recovery searches.

Important limitation: capability routing is currently cosmetic. Each
capability adds query terms and calls the same layered `web_search` function.
There is no `SourceAdapter` interface or registry yet. Wikipedia and Wikidata
now use direct APIs underneath `web_search`, but they are invoked as part of
the shared search path rather than selected as facet-specific adapters.

### Batch 1 — Evidence Correctness and Observability (July 25, 2026)

Commit `e5c7db1` implemented the first batch from "Updated Implementation
Sequence" below:

- Mojeek removed from `SEARXNG_BASE_ENGINES` (near-zero yield per the
  provider-yield table); kept reachable only via `check_providers_now()`'s
  health probe (`SEARXNG_HEALTH_PROBE_ENGINES`).
- `SearchResult` gained `canonical_url` and `observations: list[ProviderObservation]`
  (provider, rank, query). `search.py`'s `_merge()` now keys on canonical URL
  and merges observation lists instead of dropping the losing duplicate's
  signal; `kb/canonical.py`'s `normalize_url()` gained `www.` stripping and
  arXiv abs/html folding so both the cross-provider merge and (indirectly)
  `extra_research.py`'s dedup share one canonicalizer. AMP/print-page
  variants remain unhandled -- deliberately deferred to Batch 2.
- `web_search()` takes optional `capability`/`run_id`/`plan_id`/`facet_id`/
  `attempt_id` kwargs, logged into new nullable columns on `search_usage.py`'s
  `search_calls` SQLite table (added via `PRAGMA table_info`, since SQLite --
  unlike the KB's Postgres schema -- has no `ADD COLUMN IF NOT EXISTS`).
  `kb/verification.py`'s web-fallback call tags `capability="claim_verification"`,
  `attempt_id=claim_id`; a real `run_id` for verification is not yet threaded
  through (three call levels up, flagged as a Batch 2 follow-up).
- `extra_research.py`'s `collect_sources()` now walks the full ranked
  candidate list per query (bounded by `MAX_FETCH_ATTEMPTS_PER_QUERY = 4`),
  backfilling to the next candidate on a duplicate/low-quality/non-HTML/failed
  fetch/unusable scrape instead of stopping at one preselected candidate. A
  failed fetch **never** falls back to the search snippet as page content.
  Every candidate considered gets a terminal `CandidateOutcome` (accepted or
  a specific rejection reason), collected on `ResearchBundle.candidate_outcomes`
  -- in-memory only today, but field names are chosen to map onto a future KB
  candidate-ledger row without a rename (see "Persist the common source
  contract" in Recommended Next Steps).
- `_source_assessment()`'s `"independent": True` placeholder is now
  `"independent": "unknown"` -- see "Independence Is a Placeholder" below,
  which this makes honest but does not resolve.

### Domain-Neutral Authority Signals (July 25, 2026)

Commit `6cc32ab` addressed the "Domain-Neutrality Is Not Yet Achieved" gap
below for the authority/duplication/prompt-wording layer specifically (not
adapter routing, which is a separate, still-open gap):

- `classify_source()` no longer special-cases Hugging Face organizations,
  model-vendor domains (`qwenlm.github.io`, `mistral.ai`, `llama.com`,
  `ai.meta.com`), or coding-benchmark sites. It now tiers by structural
  signals: `.gov`/`.mil` and recognized scholarly-publishing infrastructure
  (arXiv, OpenReview, PubMed, DOI, SSRN, JSTOR, bioRxiv) at the top tier;
  `.edu` and `docs.`/`developer.` subdomains at the next tier; a small
  generic repository-host set (GitHub, GitLab, Bitbucket, SourceForge,
  Hugging Face) at the same tier; a generic `/discussions/`, `/community/`,
  `/forum/`, `/comments/` path-marker check (any host, not just Hugging
  Face) demotes forum content to the lowest tier.
- `OFFICIAL_HF_ORGANIZATIONS` and `_model_family_key()` (FP8/AWQ/GGUF
  quantization-repack collapsing) are deleted outright, not generalized --
  there is no domain-neutral way to know which organizations are "official"
  for an arbitrary entity.
- `has_authoritative_source()` no longer requires `source_kind in
  {"primary", "paper"}` (documented as needing "a model card or paper"). It
  now accepts quality tier ≥ 4, which includes official documentation,
  `.edu`, and recognized repositories -- so a documentation-only or
  government-only bundle can still reach synthesis.
- `_adapter_query()`'s `"scholarly"` suffix no longer hard-restricts to
  `site:arxiv.org OR site:openreview.net`; `"repository"` no longer
  hard-restricts to `site:github.com`. Query-planning prompts and fallbacks
  in `derive_starting_queries()`/`derive_follow_up_queries()`/
  `_fallback_research_plan()` no longer hard-code "model card," "hardware
  requirements," "benchmark owner," or "benchmark evidence."
- Not touched: `web/app.py`'s synthesis prompt (`"primary or paper"` there
  names the generic tier values, not LLM specifics) and
  `kb/model_experiments.py`'s "model card" error messages (that module is
  legitimately about comparing local LLM configurations).

Still open: the `SourceAdapter` protocol/registry does not exist, so
capability routing is still cosmetic exactly as described above -- this fix
made the *authority signals* domain-neutral, not the *routing mechanism*.

### Batch 2a — RRF Fusion, Typed Provider Failures, Adaptive Waterfall (July 25, 2026)

Commit `1ab7dd3` implemented the first three of Batch 2's five items (see
"Updated Implementation Sequence" below); result caching and concurrent-
request coalescing were deliberately deferred -- neither has any existing
pattern in the codebase to build on and both need their own backend/TTL
design pass.

- `search.py`'s `_rank_results()` now ranks by reciprocal-rank-fusion score
  (`_rrf_score`, `RRF_K=60`, same formula as `kb_search.py`'s own RRF but not
  imported from it -- kept as a small local constant to avoid coupling web
  search to KB search) summed across every provider's observed rank for a
  canonical URL, falling back to the existing lexical `_relevance_score` only
  to break ties. Cross-provider agreement now outranks whichever single
  provider happened to rank one result first.
- New `classify_http_error()` categorizes provider failures as
  `rate_limited`/`auth_error`/`not_found`/`server_error`/`network_error`/
  `unknown`. Every error-logging call site in `search.py` (`web_search()` and
  `check_providers_now()`) now records this via a new `error_category` column
  on `search_calls`, and `provider_monthly_quota_exhausted()` checks that
  typed column instead of `error_message LIKE '%429%'` string-matching.
- Adaptive waterfall: Wikidata now only fires when a caller passes
  `capability="grounding"` (confirmed with the user -- Wikipedia stays
  unconditional, since its yield doesn't have Wikidata's ~67%-empty waste
  problem). Serper and Brave are now gated behind the same
  sufficient-relevant-results check Tavily already used (renamed
  `MIN_SUFFICIENT_RELEVANT_RESULTS`, still 3), instead of firing
  unconditionally, and Serper now runs before Brave to match this section's
  intended order. `extra_research.py` gained a `"grounding"` capability the
  planner can request for facets needing basic entity/definition lookups.
  `kb/verification.py` deliberately keeps `capability="claim_verification"`,
  not `"grounding"` -- its LLM-generated natural-language queries were a poor
  match for Wikidata's entity-label-shaped `wbsearchentities` endpoint anyway.

### Batch 2b — Result Caching and Concurrent-Request Coalescing (July 25, 2026)

Commit `d8c9866` closed out the two items Batch 2a deferred, completing
Batch 2 in full.

- New `deep_research/tools/search_cache.py`: a self-contained SQLite file
  (`search_cache.db`), the same pattern as `search_usage.py` and for the
  same reason -- confirmed directly in code that `web_search()` and every
  caller above it (`agent.py`'s `ResearchAgent`, `extra_research.py`'s
  `collect_sources()`/`collect_research_bundle()`, `web/app.py`'s
  `_extra_research_answer()`) only ever receive `Config`, never a
  `KBDatabase` handle (`cli/main.py`'s `_init_kb_db()` returns `None` on any
  connection failure by design, and the web server checks
  `kb_routes.kb_db` for truthiness everywhere). Putting the cache in the KB
  Postgres DB would have meant threading `kb_db` through all of those call
  sites first.
- Cache key: normalized query (casefold + collapsed whitespace) + capability
  + `include_alternate_query_engines` + freshness tier. Deliberately
  excludes `run_id`/`plan_id`/`facet_id`/`attempt_id` -- those identify the
  *run*, not what was searched, and including them would make every call a
  unique key.
- Three freshness tiers via a new `freshness` parameter on `web_search()`
  (`"volatile"` 10min / `"default"` 2h / `"stable"` 7d) -- no caller passes
  anything but the default yet; the plumbing exists for a future caller
  (e.g. a "current events" capability) to opt into a shorter/longer TTL.
- Concurrent identical searches are coalesced in-process (module-level
  `dict[str, asyncio.Future]` + lock in `search.py`, same locality as the
  existing SearXNG throttle) instead of each spending their own provider
  calls -- no cross-process coordination, which would need a much heavier
  mechanism this doesn't need yet.
- New `config.search_cache.enabled` (default `True`, env-overridable) is a
  real on/off switch. A cache hit logs a `provider="cache"` row via the
  existing `log_search_call` so `/api/search-usage` can show cache
  effectiveness alongside real provider calls.
- Test-isolation fix: bare `Config()` resolves to a fixed real on-disk path,
  not a per-test tmp_path, and several `tests/test_search.py` tests reuse
  the same query text -- one autouse fixture neutralizes caching for that
  whole file (mirrors how those tests already neutralize
  `search_usage.py` logging), with real cache/coalescing behavior covered
  separately in the new `tests/test_search_cache.py`.

### Batch 3, Round A — Shared Safe Fetch Contract + PDF Routing Fix (July 25, 2026)

Commit `83ad2b1` covers the first piece of Batch 3 ("Reusable acquisition
and passage retrieval"): the shared SSRF-safe/byte-bounded fetch contract
and the PDF-ingestion routing fix it enables. Persisting Extra Research's
sources into the KB, facet-relevant passage retrieval, structured
`kb_search()` records, and OpenAlex/arXiv adapters are deliberately
deferred to later rounds.

- New `deep_research/tools/fetch.py`'s `safe_fetch()` is now the only fetch
  path for `scrape_page()` (Extra Research) and `ingest_web_page()` (KB
  ingestion) -- confirmed by grepping every `httpx.AsyncClient` call site in
  the repo that these are the only two that fetch attacker-influenced
  (search-result or user-supplied) URLs; every other call site hits a
  fixed/deployment-configured/hardcoded trusted host.
- SSRF protection resolves the hostname via `socket.getaddrinfo` and rejects
  private/loopback/link-local/reserved/multicast addresses -- catching DNS
  rebinding (a normal-looking hostname that resolves to an internal
  address), not just literal-IP string matching -- and re-validates every
  redirect hop, not just the initial URL.
- The response body is streamed with a hard byte cap
  (`config.scraping.max_response_bytes`, default 10 MB) enforced *during*
  download, and redirects are capped
  (`config.scraping.max_redirects`, default 5).
- PDF routing fix: `kb/artifacts.py`'s `build_artifact_for_version()` now
  checks the source *version's* own stored `mime_type` for
  `application/pdf` when `source_type_code == "web"` and routes to the
  existing `pypdf` extractor -- previously a PDF fetched at an ordinary URL
  was always run through the HTML clean-text extractor because
  `ingest_web_page()` registers every URL-based source as `"web"` before
  the MIME type is even known. Turned out simpler than expected: no
  reordering of fetch-before-source-creation was needed, since
  `build_artifact_for_version()` already receives the `version` dict with
  the correct `mime_type` on it -- `artifacts.py` just wasn't reading it.

### Existing Foundations Not Yet Integrated Into Routed Collection

- `deep_research/tools/kb_search.py` already provides hybrid full-text and
  semantic retrieval over current KB chunks using reciprocal-rank fusion.
  The missing work is returning structured provenance and exposing it through
  Extra Research's adapter boundary.
- The KB already stores versioned sources, immutable artifacts, chunks,
  embeddings, transcript timestamps, and claim evidence. Extend this storage
  rather than creating a parallel research-source database.
- Direct Wikipedia and Wikidata API clients, provider-level usage logging,
  quota controls, cooldowns, and circuit breakers already exist in the shared
  search layer.
- The Research page exposes Standard and Extra modes. Extra mode currently
  plans facets, collects a bounded source bundle, builds a quote-validated
  claim ledger, synthesizes a draft, and performs a final ledger-based check.

## Revised Data-Gathering Direction

Live search-engine access is the scarcest, least reliable resource. The
collection design should therefore prefer programmatic, cacheable, and
source-native routes before attempting a SERP query. The goal is not more raw
text; it is fewer, better passages for a smaller local model.

### Required Routing Order

For each evidence facet, try these routes in order and record every fallback:

1. **Local corpus first.** Query the existing KB's accepted sources, snapshots,
   artifacts, transcripts, and user documents. Return provenance, retrieval
   date, source version, and freshness status. For time-sensitive facets, the
   planner must still request live corroboration.
2. **Domain-native APIs.** Add adapters for sources designed for programmatic
   use rather than search-engine scraping. Initial priorities:
   - scholarly: OpenAlex, Crossref, arXiv, Semantic Scholar, PubMed;
   - grounding/reference: Wikipedia and Wikidata;
   - technical: GitHub and Stack Exchange;
   - current events: RSS/Atom feeds and GDELT;
   - authoritative domains: direct sitemap/RSS enumeration and polite fetch.
3. **Link-following expansion.** Once an accepted seed source is fetched,
   extract references, citations, canonical links, related pages, feeds, and
   authoritative outbound links. Rank candidates against the uncovered facet
   and fetch a bounded number per domain. This is "search less, crawl more."
   Only permit HTTP(S), reject private/link-local destinations, cap redirects,
   response bytes, depth, and per-domain requests, and retain the parent link
   that led to every candidate.
4. **Archives/crawl indexes.** Add optional Common Crawl index lookup and
   Wayback fallback for historical, blocked, or disappeared pages. These must
   retain capture dates and archive provenance.
5. **Live search as last resort.** Use the existing layered search providers
   only when the earlier routes leave a documented gap. Search calls need a
   per-plan budget and a recorded reason. Any future browser-assisted search
   must be explicit, user-initiated, and compliant with the provider's terms;
   it must not attempt to evade CAPTCHAs or access controls.

### Fetch, Clean, and Distill Contract

Every adapter should feed the same pipeline:

1. Fetch with a polite HTTP client, robots/terms-aware per-domain rate limits,
   and conditional/cache-aware requests where possible.
2. Use a readability/trafilatura-class main-content extractor before any LLM
   sees a page. Preserve raw snapshots separately for audit.
3. Chunk cleaned text, embed it, and retrieve only the best passages for the
   facet. Add an optional small reranker before LLM work.
4. Use a small local model to produce short sourced notes from selected
   passages. The larger model should receive notes and the compact research
   state, not full pages or an ever-growing chat transcript.
5. Store cleaned text, passages, notes, retrieval metadata, and source-version
   links in the local corpus. Reuse it on later research instead of fetching
   again.

Every accepted or rejected candidate should retain: adapter, requested
capability, query or parent seed, canonical URL, retrieval and publication
dates when available, content hash, source/version IDs, freshness decision,
assessment outcome, and rejection/fallback reason. Passage evidence must
retain page, section, character, or timestamp locators.

The research loop should carry a compact research-state artifact: approved
plan, covered facets, distilled findings, unresolved questions, budget use,
and next actions. Raw material remains stored but is not repeatedly injected
into model context.

### YouTube: Existing Capability and Required Expansion

Already implemented:

- caption-track ingestion through `youtube-transcript-api`;
- oEmbed title/author lookup when available;
- versioned raw transcript snapshots in the KB;
- timestamp-preserving transcript chunks;
- playlist tracking and keyless `yt-dlp --flat-playlist` enumeration;
- bounded durable playlist-ingest jobs with visible per-video state.

Required next work:

1. **Transcript-first intake.** Treat captions as the primary artifact and
   cache them permanently with language, caption type, and retrieval metadata.
2. **Fallback transcription.** When captions are absent, enqueue a background
   `yt-dlp` audio-only download plus local Whisper/whisper.cpp or
   faster-whisper transcription. Record that the transcript is generated and
   keep segment timestamps.
3. **Metadata triage.** Before full transcription, collect title, channel,
   description, chapters, duration, and publication date. A small model or
   deterministic policy decides whether the video merits processing and which
   chapter/time range is relevant.
4. **Low-cost discovery.** Add channel RSS feeds, upload/playlist enumeration,
   and video links discovered from already accepted web pages. Reserve YouTube
   search/API search for documented uncovered gaps; enumeration should not use
   the expensive search endpoint.
5. **Citation links.** Preserve each note/claim's timestamp range and render
   citations as `watch?v=<id>&t=<seconds>s` links, rather than only a video
   URL.
6. **Background accumulation.** Allow trusted channel/playlist subscriptions
   to poll on a bounded schedule, triage new videos, and ingest approved
   transcripts while the system is otherwise idle.

### Storage and Caching Rules

- The existing KB/snapshot store is the foundation of the local corpus; extend
  it rather than adding a competing source database.
- Make cleaned text, passage artifacts, distilled notes, source assessments,
  link graphs, and adapter provenance first-class versioned artifacts.
- Use content hashes and source-version links for immutable provenance.
- Apply TTL/revalidation only to facets marked time-sensitive; do not
  needlessly re-fetch stable papers, documents, or transcripts.
- Idle-time accumulation must have explicit budgets, trusted-source allowlists,
  and queue visibility so it cannot silently consume search or GPU resources.

### Plan-Only Preview

Commits `2ace3c1` and `e2d7e96` added:

- `POST /api/research-plan` with `{ "query": "..." }`;
- no search or scrape calls (`searches_performed: 0`);
- JSON planning followed by a simple line-format repair attempt;
- rejection of a search query that exactly repeats the user's raw question.

The endpoint uses the currently loaded local model. Qwen3-14B failed to
produce valid structured plans in both formats for the benchmark question, so
the endpoint used the deterministic keyword fallback. The fallback avoids the
exact raw question but is still not a high-quality search plan.

The preview is not yet connected to execution. It returns an ephemeral plan,
while starting Extra Research generates a new plan. There is no persisted plan
ID, edit/approve state, or guarantee that a reviewed plan is the plan that
collection executes.

## Current Implementation Gaps

### Domain-Neutrality — Authority Signals Fixed, Adapter Routing Still Cosmetic

**Resolved (commit `6cc32ab`, see "Domain-Neutral Authority Signals" above):**
`extra_research.py` no longer contains the Hugging Face organization
allowlist, model-family/quantization normalization, model-vendor domains, or
coding-benchmark domains that used to drive `classify_source()`, and
`has_authoritative_source()` no longer requires a `primary` source or
`paper` specifically. Authority now comes from structural signals
(gov/mil/edu TLDs, recognized scholarly-publishing infrastructure, generic
documentation/repository/forum patterns) that apply the same way to legal,
government, financial, historical, medical, and ordinary product questions.

**Still open:** capability routing itself remains cosmetic (see "Routed
Collection and Source Fitness" above) -- each capability still only adds
query terms to the same `web_search` call. There is still no `SourceAdapter`
interface or registry, so authority/completion signals come from a shared
generic function rather than from adapter-specific provenance the way the
routing-order design ultimately intends.

### Independence Is a Placeholder

**Partially resolved:** every source assessment used to record
`independent: true` unconditionally; it now records `independent: "unknown"`
(commit `e5c7db1`), so the handoff no longer misrepresents unassessed
independence as an established fact. No publisher, syndication, citation,
mirror, common-origin, or derivative-content analysis is actually performed
yet -- the placeholder is honest now, not resolved. Do not count independence
assessment as complete until `"unknown"` is replaced with a real
deterministic result.

### Source Processing Is Shallow

- `ResearchSource` gained `canonical_url` (commit `e5c7db1`) but still lacks
  adapter provenance, source/version IDs, retrieval and publication dates,
  content hash, freshness, language, license, and parent seed.
- PDF results are currently discarded before fetch in `collect_sources()`
  (`_is_html_result()`), even though scholarly and official evidence is
  frequently PDF-only -- still open, deliberately deferred past Batch 3
  Round A. **Partially resolved (commit `83ad2b1`):** a downloaded PDF now
  reaches the existing `pypdf` artifact path correctly -- `kb/ingest.py`
  still registers every URL-based source as `web` before the MIME type is
  known, but `kb/artifacts.py`'s `build_artifact_for_version()` now checks
  the version's own stored `mime_type` and routes PDFs to the PDF extractor
  regardless. What remains: `collect_sources()`'s `_is_html_result()` filter
  itself still skips PDF candidates before ever fetching them (recording a
  `rejected_non_html` outcome rather than silently dropping them) -- lifting
  that filter, so Extra Research can actually collect PDF evidence, is the
  remaining piece.
- **Resolved (commit `e5c7db1`):** a failed page fetch no longer substitutes
  the search-result snippet for the page body. `collect_sources()` now
  backfills to the next ranked candidate on any duplicate/low-quality/
  non-HTML/failed-fetch/unusable-scrape outcome, bounded by
  `MAX_FETCH_ATTEMPTS_PER_QUERY`, and records a terminal `CandidateOutcome`
  for every candidate considered -- in-memory on `ResearchBundle.candidate_outcomes`
  only; not yet persisted to the KB (see "Persist the common source contract"
  below).
- Claim-ledger extraction sees only the first 2,500 characters of each
  accepted source instead of facet-relevant passages from the complete
  document.
- Extra Research saves fetched pages to session storage, not to the versioned
  KB corpus, so later runs cannot reliably reuse or revalidate them.
- Generic page scraping always attempts product-card extraction before normal
  main-content extraction. On pages with product-like grids this can replace
  useful research text with a synthetic product list. Structured product
  extraction should be request-specific and the generic cleaned body should
  always remain available.
- The same loaded model performs planning, source analysis, claim extraction,
  synthesis, and fact checking in the interactive flow. Model-role separation
  currently exists only in the experiment harness.

### Budgets Need Exact Accounting

The current routed defaults are four facets, two requested capabilities per
facet, ten accepted sources, one source per adapter attempt, and one
gap-closing round. A single `web_search` call can fan out to multiple SearXNG
engines and direct providers, so the research budget must separately count:

- adapter attempts;
- provider/API calls;
- candidate URLs;
- fetches and bytes;
- accepted sources;
- local-model calls and tokens;
- elapsed time.

Partial progress: `collect_sources()` now bounds and counts fetch attempts
per query (`MAX_FETCH_ATTEMPTS_PER_QUERY`) and `search_calls` now carries
`run_id`/`plan_id`/`facet_id`/`attempt_id`/`capability` columns so provider
calls can in principle be joined back to the bundle/attempt that made them.
None of the other dimensions above (candidate URLs considered, fetch bytes,
local-model tokens, elapsed time) are tracked yet, and nothing surfaces this
accounting to the UI/API.

## Historical Test Runs and Findings — July 16, 2026

### Baseline Facet Collection

Six source-only runs were performed for Qwen3-14B, Qwen3-30B-A3B, and
Qwen3-32B, with reasoning on and off. These were preserved as model-experiment
records before the routed comparison.

Finding: collectors typically returned one source and covered only one facet.
The sources were often generic Hugging Face community articles or context
papers, not complete candidate/specification/benchmark evidence.

### Routed `routed_v1` Collection

Six more source-only runs were performed with the routed workflow and local KB
explicitly disabled. They are labeled `collection_workflow: routed_v1` in the
job payload.

Results:

| Model | Reasoning | Sources | Covered facets |
|---|---|---:|---:|
| Qwen3-14B | Off | 2 | 1 |
| Qwen3-14B | On | 2 | 2 |
| Qwen3-30B-A3B | Off | 2 | 2 |
| Qwen3-30B-A3B | On | 2 | 2 |
| Qwen3-32B | Off | 3 | 2 |
| Qwen3-32B | On | 3 | 2 |

Useful retained seed sources:

- LongCodeBench and other long-context/coding papers: useful for methodology
  and context-performance evidence.
- Curated GitHub lists: useful only for discovering model/benchmark names.

Not usable as final evidence:

- Hugging Face community “best models” articles;
- curated lists as specification or benchmark evidence;
- any bundle lacking official specifications, hardware/quantization fit, and
  independent coding/general-use evaluation.

No current bundle should be used for final synthesis.

## Web Search Status and Improvement Plan

Commit `6137138` fixed the Search Usage page. Its monthly counts were capped
at 500 because aggregation fetched only the latest 500 rows per provider.
Counts now aggregate directly in SQLite for the whole current month.

### Observed Provider Yield

The following trailing-30-day totals were read from the local search-usage log
on July 24, 2026. They include calls from before recent provider-policy changes,
so they are directional rather than a clean post-change benchmark.

| Provider | Calls | Successful | Empty | Errors | Average results |
|---|---:|---:|---:|---:|---:|
| Wikipedia API | 2,699 | 2,105 | 590 | 4 | 6.7 |
| Bing | 2,688 | 2,677 | 0 | 11 | 9.6 |
| DuckDuckGo | 2,267 | 1,465 | 1 | 801 | 6.4 |
| Wikidata API | 2,135 | 688 | 1,443 | 4 | 2.8 |
| Brave primary | 1,962 | 1,586 | 2 | 374 | 8.1 |
| Mojeek | 1,628 | 55 | 172 | 1,401 | 0.3 |
| Brave fallback | 1,109 | 1,101 | 0 | 8 | 9.9 |
| Serper | 790 | 774 | 8 | 8 | 9.0 |
| Google CSE | 231 | 98 | 0 | 133 | 8.5 |
| Startpage | 231 | 76 | 0 | 155 | 3.3 |
| Tavily | 10 | 6 | 1 | 3 | 6.0 |

The main operational conclusions are:

- Bing and Serper are the most consistently productive general providers.
- Mojeek is currently not worth its latency: most failures are access-denied
  responses and its average result yield is nearly zero. Disable it until a
  deliberate health probe shows sustained recovery.
- DuckDuckGo remains useful when healthy, but CAPTCHA failures justify its
  three-hour circuit breaker.
- Wikidata should be an entity/reference adapter, not an unconditional search:
  approximately two thirds of its calls returned no result.
- Google CSE and Startpage should remain limited to genuinely alternate,
  non-literal queries under the existing 48-hour backoff and attempt cap.

### Required Web Search Changes

**Status update, July 25, 2026 (commits `1ab7dd3`, `d8c9866`):** items 1, 2,
3, 7, and 8 are partially resolved: item 1's Serper/Brave/Wikidata gating
exists, but the richer "unique publishers, source diversity, or required
authority" signals for advancing to Brave are not implemented (gating still
reuses the same relevant-results count for every tier); item 2's canonical
URL and provider/rank/query observations exist (Batch 1), but result type,
publication date, retrieval date, and freshness metadata do not; item 7's
whole-call caching and in-process coalescing exist, but not per-provider/
per-language-region caching or cross-process coalescing; item 8 is partially
resolved (typed failure categories exist; the cross-process durable lease,
`retry_after` storage, and half-open health probes do not). Items 4-6, 9,
and 10 remain open exactly as described.

1. **[PARTIAL] Use an adaptive provider waterfall.** Start with Bing, then use Serper
   when candidate sufficiency is not met or cross-provider ranking evidence is
   requested. Use Brave when those routes still lack relevant results, unique
   publishers, source diversity, or required authority. Invoke
   Wikipedia/Wikidata only for grounding facets and Tavily only as the final
   thin-results fallback. Evaluate usable coverage rather than raw result
   count before advancing.
2. **[PARTIAL] Preserve provider provenance and rank.** Extend `SearchResult` with
   a canonical URL, provider observations containing provider rank/score and
   query variant, result type, publication date, retrieval date, and freshness
   metadata. Also carry research-run, plan, facet, and adapter-attempt IDs.
   When multiple providers return the same source, merge their observations
   rather than discarding the duplicate signal.
3. **[PARTIAL] Fuse rankings instead of only concatenating them.** Use
   reciprocal-rank fusion across provider result lists, then apply
   deterministic entity, number, date, phrase, and source-quality signals.
   Provider observations must be retained before implementing RRF; the
   current `SearchResult` shape discards the per-provider lists and ranks
   required to calculate it. Locally embed and rerank the bounded candidate
   set against the facet evidence need before fetching.
   `_rank_results()` now fuses via RRF (`_rrf_score`) with lexical relevance
   as a tie-break only -- the deterministic entity/number/date/phrase signals
   and local embedding-based rerank described here are not yet implemented.
4. **Canonicalize before deduplication.** Reuse the KB URL normalizer for what
   it currently supports: host/port normalization, tracking-parameter and
   fragment removal, stable query ordering, and trailing-slash cleanup. Extend
   canonicalization separately for scheme aliases, `rel=canonical`, AMP,
   print, and other alternate renderings. The Extra Research canonicalizer
   currently handles only arXiv abstract/HTML pairs; it is not yet a general
   URL normalizer. Add title/content similarity for syndicated or mirrored
   pages and retain publisher/common-origin information for independence
   assessment.
5. **Backfill failed fetches.** Keep a ranked candidate queue per query. If the
   selected page fails, is blocked, is duplicate, or produces unusable text,
   try the next candidate until one is accepted or the query's fetch-attempt
   budget is exhausted. Never substitute a search-result snippet as accepted
   page evidence. The current one-preselected-result behavior can turn a
   healthy search into zero evidence or, through snippet fallback, apparent
   evidence that was never fetched.
6. **Improve the first query.** Build a short deterministic query from the
   claim's subject, predicate, distinctive numbers/units, date or period, and
   optional short quotation. Keep a literal-claim query as one possible
   variant, not the universal first attempt. Use the local model for a
   meaningfully different follow-up query when the structured query is thin.
7. **[PARTIAL] Cache and coalesce.** Cache provider results by normalized
   query, provider, language/region, and freshness window. Give current-event
   queries short TTLs and stable historical queries longer TTLs. Coalesce
   identical concurrent requests so verification workers do not independently
   spend the same call. `search_cache.py` (commit `d8c9866`) caches by
   normalized query/capability/engine-set/freshness with three TTL tiers, and
   coalescing is in-process via shared `asyncio.Future`s -- caching by
   provider and by language/region individually is not implemented (the
   cache is keyed at the whole-`web_search()`-call level, not per provider),
   and coalescing does not cross process boundaries (a durable cross-process
   lease, per item 8, would be needed for that).
8. **[PARTIAL] Use shared, typed rate controls.** Replace the process-local
   SearXNG throttle with a durable cross-process lease. Distinguish short
   rate limits, monthly quota exhaustion, authentication errors, DNS/network
   failures, CAPTCHAs, and upstream server failures. Store `retry_after` and
   use bounded half-open health probes rather than treating every error
   alike. `classify_http_error()` now distinguishes
   rate_limited/auth_error/not_found/server_error/network_error/unknown and
   `provider_monthly_quota_exhausted()` reads the typed category -- the
   cross-process durable lease, `retry_after` storage, and half-open probes
   are still open.
9. **Connect search logs to evidence outcomes.** Add research-run, plan, facet,
   adapter-attempt, and query-variant IDs to each provider call and result.
   Record whether each candidate was selected, fetched, extracted, accepted,
   rejected, and cited. Optimize for accepted evidence per provider call,
   time-to-first-usable-source, domain diversity, and facet coverage—not HTTP
   success alone.
10. **Create an offline search-quality suite.** Maintain representative
    technical/product, current-event, academic, documentation,
    government/legal, historical, and entity questions with expected useful
    sources or domains. Measure recall at K, reciprocal rank, duplicate rate,
    accepted-evidence yield, latency, and external calls before changing
    provider order or ranking weights.

The provider waterfall above is the policy *inside* `WebSearchAdapter`. It does
not replace the broader local-corpus and domain-native routing order. General
web search remains the final discovery route after those adapters report a
visible gap.

The first implementation batch should disable Mojeek; introduce a structured
search request and provider-observation result contract; canonicalize results;
forbid snippets as evidence; record candidate rejection reasons; and backfill
failed fetches from the ranked candidate queue. Adaptive provider tiers and RRF
should follow once the observations and terminal candidate outcomes needed to
measure them are preserved.

### Code-Verified Gaps Against This Plan — July 24, 2026

A direct read of `deep_research/tools/search.py` and
`deep_research/tools/extra_research.py` against the plan above found that
several items are still open in code, plus a few concrete, low-risk fixes the
plan does not call out individually. Ranked by leverage:

**Status update, July 25, 2026:** items 1, 3, and 4 below are resolved by
commit `e5c7db1` (see "Batch 1 — Evidence Correctness and Observability"
above). Items 2, 5, 6, 7, and 8 remain open exactly as described.

1. **[RESOLVED] Mojeek is still an active SearXNG engine.** `search.py:49` sets
   `SEARXNG_BASE_ENGINES = ("duckduckgo", "bing", "mojeek")`, so Mojeek is
   still fired on every search call despite the provider-yield table above
   showing 1,401 errors out of 1,628 calls and a 0.3 average result count.
   Zero-design-risk, one-line removal; do this first.
2. **There is no adaptive waterfall yet.** `web_search()`
   (`search.py:399-544`) unconditionally calls SearXNG (multiple engines),
   the Wikipedia API, and the Wikidata API, then Brave and Serper whenever
   their keys are configured. Only Tavily is gated, behind a relevance-count
   threshold (`MIN_RESULTS_BEFORE_TAVILY_FALLBACK`, `search.py:526`). One
   client request to SearXNG fans out to several engines and is followed by
   multiple direct API requests, so ordinary searches spend several provider
   attempts regardless of whether Bing already answered the query. After the
   result/outcome contract exists, short-circuit when the current tier has
   sufficient candidates and select Wikipedia/Wikidata only for an explicit
   grounding capability.
3. **[RESOLVED] Cross-provider dedup is exact-URL-only.** `_merge()`
   (`search.py:322-328`) compares raw `SearchResult.url` values, so the same
   article returned by two providers with different tracking parameters or
   an AMP/print variant is kept as two "independent" results.
   The reusable KB `normalize_url()` handles tracking parameters, fragments,
   host/port normalization, query ordering, and trailing slashes, but it does
   not merge scheme, AMP, print, or arbitrary alternate renderings.
   `extra_research.py`'s `_canonical_source_key()` only recognizes arXiv
   abstract and HTML URLs. First centralize the existing safe normalization,
   then add explicit alternate-rendering rules and preserve all provider
   observations on the merged candidate.
   Resolution: `normalize_url()` now strips `www.` and folds arXiv abs/html,
   `SearchResult` carries `canonical_url` + `observations`, `_merge()` keys
   on canonical URL and merges observations, and `_canonical_source_key()`
   is deleted. AMP/print/scheme-alias variants are still unhandled --
   deliberately deferred to Batch 2 to avoid a rushed false-merge heuristic.
4. **[RESOLVED] A failed fetch can become snippet-based evidence, and there is no
   candidate backfill.**
   `collect_sources()` (`extra_research.py:194-228`) selects its top
   `per_query_limit` candidates from the ranked results up front and then
   scrapes them (`extra_research.py:230-250`). If fetching raises, `read()`
   falls back to `result.snippet`; if that snippet is at least 240 normalized
   characters it can be accepted and later quoted as if it were page content.
   If the selected result is unusable, nothing pulls the next-ranked candidate
   to replace it. Fix: discovery snippets are never evidence, every fetch gets
   a terminal outcome/reason, and collection walks `ranked_results` until
   `per_query_limit` *retrieved and usable* sources are found or the explicit
   fetch-attempt budget is exhausted.
5. **[PARTIAL] PDFs need MIME-aware web ingestion, not just removal of a
   filter.** `_is_html_result()` (`extra_research.py:120-121`) is used as a
   hard candidate filter in `collect_sources` (`extra_research.py:216`), so
   PDF search results are dropped before they are ever fetched. Because
   primary evidence — papers, specs, whitepapers, official documentation —
   is disproportionately PDF-only, this likely affects evidence quality more
   than adding another general provider. The existing KB artifact builder
   already parses sources typed as `pdf`, but `ingest_web_page()` creates a
   `web` source before it sees the response MIME type. The common fetch/
   ingest contract must sniff and validate MIME type, persist the raw PDF as
   a PDF source version, extract page-located text, and return relevant
   passages.
   Resolved (commit `83ad2b1`): `build_artifact_for_version()` now reads the
   source version's stored `mime_type` and correctly routes a web-fetched
   PDF to the `pypdf` extractor — a downloaded PDF now reaches page-located
   text extraction. Still open: `extra_research.py`'s `_is_html_result()`
   filter still drops PDF candidates in `collect_sources()` before they're
   ever fetched, so Extra Research itself still can't collect PDF evidence
   even though the KB-side ingestion/artifact path now handles it correctly.
6. **Ranking is pure lexical overlap, and RRF first needs provider-ranked
   observations.** `_rank_results`/`_relevance_score` (`search.py:340-364`)
   concatenate every provider's results and sort by title/snippet term
   overlap alone — no rank fusion across providers. `kb_search.py:36-46`
   already contains a clean RRF implementation for FTS-plus-semantic results,
   but web search first needs to retain each provider's ranked list rather than
   flattening results into the three-field `SearchResult`. Reuse the RRF
   calculation after adding that provenance, then keep lexical/entity/date
   signals as transparent reranking features.
7. **Generic scraping can discard the useful body in favor of a product
   listing.** `scrape_page()` always runs `_extract_products()` before normal
   text extraction. Pages with product-like cards can therefore return
   `Found N products` instead of their main research content. The current bad
   scrape markers cover only a few observed strings, not the underlying
   failure mode. Structured product extraction should be opt-in for a product
   data task and stored alongside, not instead of, generic cleaned text.
8. **[PARTIAL] The current fetch paths are not safe or reusable enough for
   expanded collection.** `scrape_page()` and `ingest_web_page()`
   independently follow redirects and read complete responses before
   applying content-length truncation. They do not share redirect-target
   SSRF validation, a hard byte limit, MIME policy, conditional request
   handling, or one fetched artifact. Introduce one bounded
   `FetchedDocument` contract that validates every redirect, rejects
   private/link-local destinations, streams to a byte cap, records headers/
   final URL/hash, and feeds both KB snapshots and extraction.
   Resolved (commit `83ad2b1`): `tools/fetch.py`'s `safe_fetch()` is now the
   one shared, bounded `FetchedDocument` contract both `scrape_page()` and
   `ingest_web_page()` use — redirect-target SSRF validation (via DNS
   resolution, not just literal-IP matching) and a streamed byte cap are
   both done. Still open: no MIME allowlist/policy decision (any MIME type
   is currently returned as-is to the caller), no conditional request
   support (ETag/If-Modified-Since), and `FetchedDocument` doesn't carry a
   content hash or the full response headers itself — callers that need a
   hash (e.g. `ingest_web_page()`) still compute it themselves from
   `.content` after the fetch returns.

Removing Mojeek can land immediately. The highest-leverage correctness change
is item 4, followed by the common result/fetch contracts required for items 2,
3, 5, 6, and 8. The adaptive waterfall should then be tuned against accepted
evidence per provider call rather than implemented on top of result-count-only
signals.

Items 1, 3, and 4 shipped in commit `e5c7db1` (July 25, 2026). Remaining:
2 (adaptive waterfall), 5 (PDF MIME-aware ingestion), 6 (RRF fusion --
provider observations now exist as the prerequisite, but fusion itself is
not implemented), 7 (product-card scraping bug), 8 (shared SSRF-safe fetch
contract).

### Updated Implementation Sequence — July 24, 2026

**Batch 1 — Evidence correctness and observability — DONE (commit `e5c7db1`, July 25, 2026)**

- remove Mojeek from active search while keeping it available only for a
  deliberate health probe; ✅
- add a structured search request carrying intent/capability, language/region,
  freshness need, and run/plan/facet/attempt IDs; ✅ (capability + run/plan/
  facet/attempt IDs land as optional kwargs on `web_search()`; language/region/
  freshness fields do not exist yet -- no caller needs them until Batch 2's
  waterfall)
- preserve per-provider observations and canonical candidate identity; ✅
- forbid snippets as evidence, record a terminal outcome for every candidate,
  and backfill until the usable-source or fetch-attempt budget is reached; ✅
- replace `independent: true` with `unknown` until common-origin analysis
  exists. ✅

A follow-on fix, not originally scoped to Batch 1 but landed the same day
(commit `6cc32ab`): removed the LLM-specific authority/duplication rules from
`extra_research.py`'s `classify_source()`/`has_authoritative_source()`/
`_adapter_query()`/query-planning prompts (see "Domain-Neutral Authority
Signals" above). This is "Remove domain-specific gates" from Recommended Next
Steps below, pulled forward because it was small, contained, and directly
unblocked by nothing else in Batch 1.

**Batch 2 — Fewer, better provider calls — DONE except metrics (commits `1ab7dd3`, `d8c9866`, July 25, 2026)**

- try Bing first, then call Serper when candidate sufficiency is not met or
  cross-provider ranking evidence is requested; ✅ (sufficiency-only; "cross-
  provider ranking evidence is requested" as a distinct trigger is not
  implemented -- there's one sufficiency check, not two triggers)
- add Brave only for remaining relevance, diversity, authority, or source-type
  gaps, and keep Tavily as the final thin-results fallback; 🟡 Brave is gated
  by the same sufficiency check as Serper, not the richer diversity/
  authority/source-type-gap signal this bullet describes; Tavily's gating is
  unchanged and already matched this
- select Wikipedia/Wikidata through an explicit grounding capability; ✅
  (Wikipedia was kept unconditional by deliberate decision -- see Batch 2a
  above -- so read this as "select Wikidata," not both)
- add canonical merge, RRF, deterministic reranking, typed provider failures,
  result caching, and concurrent-request coalescing; 🟡 canonical merge
  (Batch 1), RRF, typed provider failures, result caching, and (in-process)
  concurrent-request coalescing ✅ done (`d8c9866`); deterministic reranking
  beyond existing lexical relevance remains open, and coalescing doesn't
  cross process boundaries;
- measure accepted evidence per provider call, time to first usable source,
  unique publishers, cited-claim yield, and facet coverage. ⬜ not started --
  blocked on `candidate_outcomes` actually being persisted (Batch 3) so a
  real join against `search_calls` is possible.

**Batch 3 — Reusable acquisition and passage retrieval — Round A DONE (commit `83ad2b1`, July 25, 2026)**

- introduce the shared SSRF-safe, byte-bounded, MIME-aware fetch contract; ✅
  `tools/fetch.py`'s `safe_fetch()`/`FetchedDocument`. "MIME-aware" here
  means the PDF-routing consequence below; the contract itself doesn't
  enforce a MIME allowlist/policy.
  Also lands the matching PDF-artifact routing fix in `kb/artifacts.py` (see
  "Batch 3, Round A" above) — the actual PDF result was blocked on both the
  fetch contract's byte/redirect safety *and* this routing fix together.
- persist one raw snapshot and derived cleaned/PDF artifact in the existing
  KB; ⬜ still open — this bullet is about Extra Research's own collected
  sources, not KB URL ingestion (which already persists snapshots/artifacts
  today, and now correctly routes PDFs per the fix above).
- return complete structured records from local hybrid retrieval instead of
  formatting them immediately as a string; ⬜ still open (`kb_search.py`).
- retrieve facet-relevant chunks/passages with page, section, character, or
  timestamp locators before claim extraction; ⬜ still open
  (`extra_research.py`'s claim-ledger extraction still reads the first 2,500
  characters, not facet-relevant passages).
- add OpenAlex discovery and arXiv retrieval only after the common source
  contract can store and replay their output. ⬜ still open, still gated on
  the item above.

## Recommended Next Steps

1. **DONE (`e5c7db1`). Land evidence correctness and observability.**
   Implement Batch 1 above so failed retrieval can never masquerade as
   evidence and provider/search outcomes become measurable.
2. **DONE (`6cc32ab`). Remove domain-specific gates.** Replace model-vendor
   authority lists and the universal paper/model-card requirement with
   generic provenance signals and facet-specific completion criteria.
   Record independence as `unknown` until it is actually assessed. (Facet-
   specific completion criteria -- as opposed to domain-neutral but still
   universal authority tiers -- remain future work; see "Domain-Neutrality"
   above.)
3. **Introduce the source and fetch boundaries.** Define a `SourceAdapter`
   protocol with capability declarations, discovery, fetch, health,
   cost/rate-limit metadata, and structured diagnostics, plus the shared
   bounded `FetchedDocument` contract. Wrap the existing layered search as
   `WebSearchAdapter`.
4. **DONE except metrics (`1ab7dd3`, `d8c9866`). Apply the adaptive web
   waterfall.** Implement Batch 2 using preserved provider observations and
   terminal evidence outcomes. Do not optimize provider order from HTTP
   success or raw result counts alone. Serper/Brave/Wikidata gating, RRF
   fusion, result caching, and in-process concurrent-request coalescing are
   all done; evidence-outcome metrics (item 9 in "Required Web Search
   Changes") remain open, blocked on `candidate_outcomes` actually being
   persisted (Batch 3).
5. **Integrate local retrieval first.** Build `LocalKBAdapter` on the existing
   full-text/semantic chunk search, returning structured source/version,
   retrieval date, passage locator, freshness, and trust metadata. Keep it
   disabled in collection-model benchmarks where reuse would bias results.
6. **Persist and approve plans.** Give each plan a stable ID and
   `draft -> approved -> executing -> completed` lifecycle. Add an editable
   review screen and make collection execute the exact approved plan.
7. **Separate reference adapters.** Route Wikipedia and Wikidata deliberately
   for grounding/reference facets rather than calling both for every general
   search.
8. **Add one scholarly slice.** Implement OpenAlex discovery plus arXiv
   metadata/content retrieval before attempting the entire adapter list.
   Preserve DOI/arXiv identifiers and deduplicate alternate renderings.
9. **Persist the common source contract.** Persist accepted and rejected
   candidates, raw snapshots, cleaned text, adapter/routing provenance,
   assessments, fallback reasons, and immutable bundle membership in the KB.
10. **[PARTIAL] (`83ad2b1`). Process complete documents.** Add MIME-aware web
    PDF ingestion, clean and chunk complete content, retrieve facet-relevant
    passages, optionally rerank them, and build the claim ledger from those
    passages rather than document openings. MIME-aware web PDF ingestion is
    done (KB URL ingestion now correctly routes a fetched PDF to the `pypdf`
    extractor); facet-relevant passage retrieval and building the claim
    ledger from passages rather than document openings remain open.
11. **Make coverage and budgets auditable.** Distinguish `covered`, `partial`,
   and `uncovered`; expose provider calls, fetches, accepted sources, model
   calls, elapsed time, and remaining gaps before synthesis.
12. **Add safe bounded expansion.** Follow citations/links only after the core
    adapter and storage contracts work, with SSRF protections, canonical
    deduplication, depth/byte/domain limits, and explicit provenance.
13. **Run a multi-domain acceptance suite.** Use technical/product, current
    event, academic, software-documentation, government/legal, and local-KB
    questions. Freeze approved evidence bundles before comparing extractor,
    synthesizer, and fact-checker models.
14. **Continue YouTube as a parallel source track.** Add metadata triage,
    caption provenance, Whisper fallback, channel RSS, scheduled trusted
    subscriptions, and timestamp-deep-link citations after the common source
    contract is available.

## Immediate Milestone

The next shippable milestone should be **evidence-correct approved-plan
execution with real local and web adapters**, not a larger search budget or a
plan-review UI sitting on top of the current shallow collection path.

It is complete when (status as of July 25, 2026):

- ⬜ a user can preview, edit, approve, and execute one persisted plan;
- ⬜ execution uses that exact plan and records every routing/fallback
  decision (`collection_attempts`/`candidate_outcomes` record decisions
  in-memory per run, but there is no persisted plan for execution to bind to);
- ⬜ a covered local-KB facet causes zero live-search calls (no `LocalKBAdapter`
  exists; Extra Research never consults the KB at all yet);
- ⬜ web search is used only after a visible local/adapter gap;
- ✅ a failed fetch cannot become snippet-based evidence and the next ranked
  candidate is attempted within budget;
- 🟡 every discovered candidate has provider provenance, recorded lifecycle
  events, and a terminal accepted or rejected outcome; accepted evidence also
  records whether it was cited. Terminal outcomes exist
  (`CandidateOutcome`/`candidate_outcomes`), and `SearchResult` itself now
  carries provider observations -- but `CandidateOutcome` does not yet copy
  provider identity onto the outcome record, and nothing tracks whether
  accepted evidence was actually cited in the final synthesis;
- 🟡 a web-hosted PDF can be stored, parsed, retrieved by relevant passage, and
  cited with a page locator. Stored and parsed with page-located chunks ✅
  (`83ad2b1`, KB URL ingestion only); retrieval by relevant passage and
  citation with a page locator are not implemented yet, and Extra
  Research's own `_is_html_result()` filter still refuses to fetch a PDF
  candidate in the first place;
- 🟡 the web adapter uses canonical multi-provider fusion, an adaptive
  provider waterfall, and candidate backfill within explicit call/fetch
  budgets. Canonical fusion ✅, candidate backfill ✅, a first adaptive
  waterfall (Serper/Brave/Wikidata gating, RRF fusion) ✅, result caching ✅,
  and in-process concurrent-request coalescing ✅ are all done (`1ab7dd3`,
  `d8c9866`); only the richer diversity/authority-aware gating signals and
  cross-process coalescing are not;
- ✅ authority and completion work for at least three non-LLM domains
  (`classify_source`'s gov/edu/scholarly/docs tiers are domain-neutral;
  completion criteria are still universal-generic rather than
  facet-specific, which is a lighter remaining gap than the original
  LLM-specific gates);
- 🟡 rejected candidates and reasons remain inspectable -- true in-memory
  (`candidate_outcomes`) for the duration of one run; not persisted, not
  exposed in any UI/API yet;
- ⬜ the resulting evidence bundle can be replayed without network access
  (nothing Extra Research collects is persisted to the KB; fetched pages
  still only live in the session-scoped `scraped_pages` table).

## Relevant Documents

- `PLAN_RESEARCH_SOURCE_ROUTING.md` — longer architecture and benchmark plan.
- `RESEARCH_WORK_HANDOFF.md` — this operational handoff.
