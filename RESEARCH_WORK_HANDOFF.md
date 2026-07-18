# Deep-Research Work Handoff

## Current State

- The Deep Research processing queue is **paused**. Do not resume it until a
  reviewed collection plan is ready; this prevents additional search-provider
  usage.
- The primary llama.cpp service is running normally with Qwen3-14B.
- The current worktree includes a user-owned deletion of
  `PLAN_GPU_COORDINATOR.md`; do not restore or commit that deletion unless the
  user explicitly asks.

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
- deterministic source fitness: authority, directness, independence,
  extractability, and acceptance;
- coverage-driven recovery searches.

Important limitation: all capabilities currently route through constrained
general web search. There are **no real specialized adapters yet** for
academic databases, official documentation, repositories, news, or the local
KB.

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
- playlist tracking and keyless `yt-dlp --flat-playlist` enumeration.

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

## Test Runs and Findings

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

## Search Usage Fix

Commit `6137138` fixed the Search Usage page. Its monthly counts were capped
at 500 because aggregation fetched only the latest 500 rows per provider.
Counts now aggregate directly in SQLite for the whole current month.

## Recommended Next Steps

1. Keep the queue paused and use only `/api/research-plan` while improving
   planning; do not spend searches on another collection run yet.
2. Decide the planner role:
   - test Qwen3-30B-A3B and Qwen3-32B on plan-only output; or
   - build a small editable plan-review UI/API so a user can correct a plan
     before any search is issued.
3. Implement the local-KB adapter first, with source-version, retrieval-date,
   freshness, and provenance rules. Keep it disabled for model-collection
   benchmarks to avoid bias.
4. Add domain-native adapters (OpenAlex/arXiv/Crossref, GitHub, RSS/Atom,
   Wikipedia/Wikidata) and make general web search a recorded last resort.
5. Add bounded link-following/citation expansion from accepted seed sources.
6. Add the common fetch-clean-chunk-embed-rerank-distill pipeline before
   increasing source budgets.
7. Upgrade source fitness with semantic directness assessment plus deterministic
   URL/duplicate/extractability checks.
8. Expand YouTube to metadata triage, caption caching, Whisper fallback,
   channel RSS, and timestamp-deep-link citations.
9. Run a plan-only comparison across models first. Score valid facets, useful
   search queries, capability selection, and no raw-question queries.
10. After approving a plan, run source-only collection again. Compare raw
   bundles before running extraction or synthesis.
11. Freeze one approved evidence bundle, then evaluate extractor, synthesizer,
   and fact-checker models separately.

## Relevant Documents

- `PLAN_RESEARCH_SOURCE_ROUTING.md` — longer architecture and benchmark plan.
- `RESEARCH_WORK_HANDOFF.md` — this operational handoff.
