# Knowledge Base / Local Research Tool Plan

## Current Goal

Evolve this repo from a session-oriented local research tool into a knowledge-base-driven system that can:

- collect data from the web when given only a topic
- ingest YouTube transcripts
- ingest PDFs, Markdown, and other documents
- extract specific claims, events, entities, and metrics
- verify claims against other sources
- store results for later reporting and comparison
- report using the latest data

This document captures the planning discussion so work can resume later even if chat context is lost.

## Current Repo Constraints

The current codebase is a lightweight local agent with:

- SQLite storage for chat sessions, messages, and scraped pages
- an Ollama/OpenAI-compatible local model client
- web search via SearXNG
- page scraping and summarization

It is not yet structured as a reusable knowledge base.

Important current limitation:

- scraped page text is truncated before reaching the model
- the app mostly works as "fetch data, put it in prompt, summarize"
- this means large context windows are not the main value driver right now

## Decision Register

Single source of truth for decisions made during planning. Spec sections cite these
by number; when a decision changes, update it here first so the rest of the document
has one place to disagree with.

From the planning discussion:

1. Database choice:
- MySQL is not a hard requirement
- PostgreSQL is acceptable and preferred if it is the better fit

2. Model hosting:
- prefer fully local models
- avoid relying on OpenAI/Anthropic API usage due to limited paid plans and token conservation

3. YouTube support:
- transcript-only is enough initially

4. Main research domains:
- mostly news
- events
- history
- some products/spec research

5. Verification:
- unverified claims are acceptable
- they must be clearly marked and tracked as unverified

6. Data volume:
- unknown for now

7. Users:
- single user for now

8. Reporting mode:
- prefer latest data rather than reproducible snapshot reports

9. Raw source storage:
- store raw sources and snapshots as files on disk
- keep database references and metadata in PostgreSQL

10. Refresh model:
- mostly on-demand refresh
- background monitoring would be a nice later capability

11. Comparison priority:
- timeline view should be designed first
- claim side-by-side comparison can come later

12. Data scale expectations:
- medium to large

13. Retention/versioning policy:
- keep selected source versions rather than everything
- keep the first version
- keep the newest two versions

14. Local model goal:
- use the best local capability practical for the hardware available

15. Update freshness:
- hybrid model
- manual refresh by default unless otherwise specified

16. Version retention exceptions:
- no *policy* exceptions — the "first version + newest two versions" rule applies
  universally as the default retention behavior
- one *integrity* carve-out is mandatory and not negotiable: a version that is cited by
  any claim's evidence must never be pruned (see "Retention vs. Evidence Integrity" in
  the schema draft). This is not a discretionary exception to the policy; it is a
  precondition for the policy not to corrupt the knowledge base.

17. Monitoring scope:
- monitoring should only run for explicitly selected topics

18. Focused source analysis:
- when analyzing a source, the user should be able to specify what to focus on
- the system should also note other potentially important topics present in the source
- sources should remain re-analyzable later for different questions
- it is acceptable not to store every possible detail if later re-analysis can recover it

19. Topic-independent ingestion:
- the system should support adding sources without attaching them to a specific topic
- those sources can be re-used later if a relevant topic is created
- the system should support broad domain extraction such as "historical and economic related data"
- part of the goal is to build a general-purpose local knowledge database, not only topic-bound research projects

20. v1 product scope (from Batch 1):
- the personal local knowledge base and the deep-research agent are equal priorities
- v1 workflow set:
  - ingest a YouTube video, playlist, or list of documents and extract general data by
    broad categories such as history/economic unless a specific topic is given
  - start from a topic plus starting sources (YouTube, websites, PDFs), then continue
    researching with additional web search
  - KB querying and timeline/report outputs are extensions of those two workflows,
    not separate workflow categories
- out of scope for v1: multi-user support
- monitoring: manual refresh by default; monitoring only when explicitly enabled for
  selected topics
- some review interface is required in v1
- OCR may later be reused from an existing separate OCR project; not required for v1

21. Source and trust policy (from Batch 2):
- sources have trust tiers
- source types are not hard-deprioritized by default; ranking is driven by
  trust/ranking logic instead
- conflicting claims are shown side by side, indicating which source is currently
  preferred based on trust/ranking

22. Extraction policy (from Batch 3):
- default always-extract set: source metadata, broad topic tags, entities,
  dates/times, claims, key metrics/numbers when present
- the default set stays configurable so the baseline can be edited later
- explicit-only extraction (examples): exhaustive relationship mapping, detailed
  sentiment/opinion analysis, fine-grained economic breakdowns, full legal/regulatory
  extraction, product-spec normalization across every attribute, quote-level
  extraction everywhere
- extraction modes exist from the start: general, historical, economic, legal,
  product/spec

23. Retrieval and freshness (from Batch 4):
- hybrid retrieval: prefer stored knowledge first unless stale or incomplete
- reanalyze a stored source when the question/focus changes; re-fetch only if stale,
  missing, or explicitly refreshed
- after model/schema/prompt improvements, mark affected sources outdated; re-extract
  on demand or via optional batch re-extraction for selected topics/sources

  **Implemented (post-step-7 follow-up):** the original chat research agent
  (`deep_research/agent.py`, predates the KB by several build-order steps) had never
  actually been connected to it — every query did a fresh web search/scrape regardless
  of what was already known locally. Added a `kb_search` tool (thin wrapper over the
  step-3 FTS5 `search_chunks`) plus a user-facing toggle — "prioritize local knowledge
  base" vs. "start with web search" — a checkbox in the web UI (persisted like dark
  mode), `--prioritize-kb` on the CLI. Two system prompt variants
  (`SYSTEM_PROMPT_KB_FIRST`/`SYSTEM_PROMPT_WEB_FIRST`) share one critical-rules block
  and only differ in tool-priority instructions; `kb_search` is only ever offered as a
  tool when a `KBDatabase` connection actually succeeded (both `cli/main.py` and
  `web/kb_routes.py`'s KB init are now best-effort — the base research agent must keep
  working standalone if Postgres isn't running, since it predates the KB entirely and
  many users may never touch it). The toggle had to reach the **text-mode** path too
  (`_run_text_mode`/`_text_mode_answer`), not just the tool-calling loop — text mode is
  the common fallback for models without tool support, so a KB-first option that only
  worked for tool-calling models would miss most real usage.

  Verified end-to-end, including the GPU-contention lesson learned along the way:
  Ollama and llama.cpp fighting over the same 16GB card for two separate 14B model
  loads makes the Ollama-backed agent path unusably slow — tests were pointed at the
  already-loaded llama.cpp instance instead. With the toggle on, the agent called
  `kb_search` first (confirmed via CLI step log and browser network/status stream),
  answered directly from KB chunks when sufficient, and fell back to `web_search` +
  `scrape_webpage` when it wasn't (correctly re-finding the same Fortune article from
  step 6's verification). With the toggle off, `kb_search` was never called — default
  behavior unchanged. Browser-driven: checkbox renders, persists across reload, and a
  live query completed with zero console errors.

24. Operations and review workflow (from Batch 5):
- jobs are hybrid: manual by default, background allowed for explicit monitoring or
  larger queued work
- optional manual review exists for important claims before they are treated as
  trusted
- on partial failure: keep partial results, mark the run incomplete, allow retry later

25. Resolution strategy (from build-order step 1, informed by the step-0 spike):
- entities: auto-merge only on exact normalized-name match (lowercased,
  punctuation-stripped); any fuzzy/substring match additionally requires a minimum
  normalized-name length before it is even considered, and always lands in
  `resolution_candidates` for review — it is never auto-merged. The spike found naive
  substring matching without a length guard unusably noisy (e.g. "AI" flagged as a
  duplicate of "Britain" and "Mark Twain" because both strings happen to contain "ai")
- claims: no auto-merge tier at all in v1. Candidate generation runs on embedding
  cosine similarity (`nomic-embed-text`, "clustering:" instruction prefix) between
  claims from different sources/extraction runs; pairs above a similarity threshold
  are written to `resolution_candidates` as `claim_duplicate` for review. Lexical/
  trigram similarity is not used for cross-source claim dedup — the spike measured it
  catching zero real duplicates. Embedding similarity does catch real duplicates the
  lexical pass misses, but precision degrades fast past the top few pairs per claim
  (measured: only 50% precision even at a 0.85 cosine cutoff), so this is a candidate
  filter feeding review, not a merge trigger
- events: not directly exercised by the spike (chunking didn't surface multiple
  independent mentions of the same event). Use the same pattern as claims —
  normalized-title exact match as the only auto-merge tier, everything else queued —
  until real event volume justifies revisiting
- consequence for Role C (embeddings) in `MODELS.md`: no longer purely optional or
  deferrable. An embeddings model needs to be resident starting at the resolution
  step (build order step 1), not just as a later retrieval improvement
- see [spike/FINDINGS.md](spike/FINDINGS.md) for the full validation this decision is
  based on, including the specific true/false-positive examples at each threshold

26. Interfaces and outputs (from Batch 6):
- web UI is the primary interface first
- v1 exports for other local tools: JSON, SQL views, local API (CSV not required)
- v1 review UI: source list/detail, topic timelines, claim list with
  status/confidence, contradiction/conflict view, job/run status view

27. Topic scope (from build-order step 7 pre-work):
- topic attachment is hybrid: explicit attachment (topic_source_links,
  claim_topics) is the core mechanism; the system also suggests likely-relevant
  unattached claims/sources for the user to accept, using the same
  suggest-then-review pattern as `resolution_candidates` rather than a fully
  automatic dynamic query
- topic suggestion is retroactive AND forward: creating a topic backfills
  suggestions against all existing claims/entities once, and every subsequent
  extraction is also checked against existing topics going forward. This is
  the same class of gap already found and fixed once in claim-duplicate
  candidate generation (step 4/6, which only checks new-vs-all-existing, never
  old-vs-new) — deciding it deliberately here instead of repeating it
- timeline entries are strict: only claims with an `event_id` and a start
  date qualify, parsed best-effort at query time (e.g. a "2008–2013" range
  takes the start). No loose inclusion of claims with a date-like phrase in
  their text but no formal event — that risks pulling in claims that aren't
  actually timeline-worthy. This requires extending the extraction prompt to
  have the model self-report `date_precision` (exact/day/month/year/
  approximate) rather than us inferring precision from free-text formatting
  after the fact — that column already exists in the schema but nothing
  populates it yet
- `preferred_source_id` on conflicting claims is set automatically from the
  highest `trust_tiers.rank_weight` among the claim's evidence sources, with a
  manual override path (mirroring the existing `reviewed_by`/`reviewed_at`
  pattern on claims) — consistent with every other automation-plus-review
  pair already built (exact-match auto-merge + resolution_candidates,
  embedding candidates + review, LLM-judged contradictions + recorded conflict)
- reports keep a persisted row per generation (cheap, incidental audit trail),
  but the product-facing behavior only ever shows the latest one for a topic —
  decision 8 ("prefer latest data rather than reproducible snapshot reports")
  already settled this; there is no report-history browsing feature in v1
- both the CLI and the existing web UI (FastAPI + Vue) get the timeline/report
  output in this step, not CLI-first-then-web-later like steps 2-6 — a
  timeline is the first output in this project genuinely more useful rendered
  visually than as a terminal table, and decision 26 already calls the web UI
  the primary interface
- topic membership is many-to-many (`claim_topics`, `topic_source_links`),
  not one-topic-per-claim — decision 19 already implies claims/sources should
  be reusable across topics created later, which is exactly what these join
  tables (rather than a foreign key on `claims`/`sources`) are for

## Recommended Direction

Build this as a small local data platform with an agent on top, not just as an agent that saves outputs.

The platform should support both:

- topic-driven research workflows
- topic-independent source ingestion for building a reusable local knowledge base

Three-layer model:

1. Ingestion
- web pages
- search results
- YouTube transcripts
- PDFs
- Markdown
- office documents and spreadsheets later

2. Extraction and verification
- extract entities, events, claims, metrics, dates, and relationships
- verify claims against additional sources
- store confidence and provenance

3. Querying and reporting
- answer questions from stored knowledge first
- collect fresh data when coverage is weak or stale
- compare events/topics across time and source sets

## Why PostgreSQL Instead of MySQL

PostgreSQL is the current recommendation because it is a better fit for this workload:

- stronger JSON/query ergonomics for semi-structured extracted data
- better full-text and analytical query patterns
- better long-term fit if vector search is added later
- cleaner balance between normalized tables and flexible metadata

MySQL would still work, but PostgreSQL is the better default choice for this system.

## Target Capabilities

### Topic-only research

Given only a topic:

- create or refresh a topic record
- search for candidate sources
- ingest and snapshot selected sources
- extract entities, claims, events, and metrics
- run verification
- generate a report from stored facts plus any new live collection

### Topic-independent knowledge ingestion

The system should also support ingesting a source without attaching it to a specific topic.

Examples:

- add a source now and let future topics discover and use it later
- ingest a source and extract broad classes of information like historical or economic facts
- build up a reusable local fact base that other local tools can query later

In this mode, the source still goes through:

- canonical source registration
- snapshot/versioning
- chunking
- extraction
- evidence linkage

The difference is that topic association can be deferred until later.

### YouTube research

Given a YouTube video:

- ingest metadata
- ingest transcript
- chunk transcript by timestamp
- extract claims, events, metrics, and named entities
- verify important claims against external sources
- store evidence links back to transcript ranges

### File/document research

Given a PDF, Markdown file, or other document:

- ingest source metadata
- parse to normalized text while preserving structure where possible
- chunk content
- extract claims, events, entities, and metrics
- verify if requested or configured
- connect facts to exact source chunks/pages/sections

### Focus-guided source analysis

When analyzing a source, the system should support both:

- explicit user focus instructions
- lightweight detection of other potentially relevant topics in the same source

Examples:

- "focus on pricing changes and launch dates"
- "focus on leadership changes, but also note anything about legal disputes"

The goal is to preserve future value from a source even when the first pass had a narrow question.
This does not require storing every possible extracted detail. The system can:

- store the source and chunked content
- store the requested focus for the analysis run
- store the facts extracted for that focus
- store lightweight hints about other notable topics for later re-analysis

### Comparison/reporting

Be able to answer:

- what changed over time
- compare two events
- compare what multiple sources say
- show contradictions
- show latest known view of a topic

## Core Design Rules

1. Facts must be first-class records
- do not only store big JSON blobs

Topic linkage should be optional at ingest time.
Facts, entities, events, and metrics should be able to exist before they are attached to a specific topic.

2. Every derived fact must point to evidence
- exact source chunk, transcript segment, or document section when possible

3. Sources must be versioned
- web pages change
- "latest data" should not destroy history

4. Unverified claims must remain visible
- but must be labeled with status and confidence

5. Reports are outputs, not truth storage
- the primary truth store is claims + evidence + sources

6. Source analysis should be re-runnable
- preserve enough source material and chunk metadata to revisit a source later with different questions
- do not assume the first analysis pass captured everything worth knowing

## Recommended Schema Direction

This is the current recommended table set for v1.

### Core tables

- `topics`
- `topic_runs`
- `topic_source_links`
- `sources`
- `source_versions`
- `source_fetch_attempts`
- `artifacts`
- `artifact_chunks`
- `analysis_focuses`
- `extraction_runs`
- `extracted_observations`
- `entities`
- `entity_mentions`
- `events`
- `event_mentions`
- `claims`
- `claim_topics`
- `claim_evidence`
- `claim_links`
- `resolution_candidates`
- `metrics`
- `reports`
- `jobs`
- `job_dependencies`
- `sessions`
- `messages`

### Table intent

`topics`
- broad research areas like a company, historical event, product family, or ongoing issue

`topic_runs`
- refresh attempts or collection/reporting runs for a topic

`topic_source_links`
- optional link table connecting sources to one or more topics
- allows sources to exist independently before they are attached to a topic

`sources`
- canonical identity for a source such as URL, YouTube video ID, or file hash

`source_versions`
- snapshots of a source over time, especially for changing web pages

`source_fetch_attempts`
- history of fetch/parse attempts, redirects, failures, blocked pages, and transcript or
  parser errors

`artifacts`
- normalized extracted assets such as cleaned text, transcript text, parsed PDF text

`artifact_chunks`
- chunked segments used for retrieval, extraction, and evidence references

`analysis_focuses`
- records user-requested focus instructions for a source or topic run
- can also store lightweight notes about other notable topics detected for later re-analysis

This table should support both:

- topic-bound analysis
- source-only analysis without a topic

`extraction_runs`
- provenance for a specific extraction pass, including model, prompt version, extraction
  modes, parameters, scope, status, and output summary

`extracted_observations`
- raw model outputs from a source chunk before they are resolved into canonical claims,
  entities, events, or metrics
- protects the curated knowledge base from first-pass extraction noise

`entities`
- people, organizations, products, places, documents, etc.

`entity_mentions`
- where entities were mentioned in a source chunk

`events`
- dated or time-bounded happenings that can be compared or grouped

`event_mentions`
- where events are mentioned in sources

`claims`
- atomic statements such as "X happened on date Y" or "product Z has 32 GB RAM"

`claim_topics`
- optional many-to-many link between claims and topics, mirroring `topic_source_links`
- claims can exist with no topic and be attached to several topics later (decision 19)

`claim_evidence`
- links from claims to supporting, contradicting, or contextual evidence in source chunks

`claim_links`
- relationships between claims such as duplicate, derived-from, contradicts, supersedes

`resolution_candidates`
- possible entity/event/claim duplicates, contradictions, or merges awaiting automatic or
  manual resolution

`metrics`
- structured numeric/time/currency/spec values for comparison queries

`reports`
- generated topic summaries and responses built from the knowledge base

`jobs`
- background ingestion, extraction, verification, refresh, and retry state

`job_dependencies`
- dependency edges between jobs so multi-step workflows can retry or resume safely

`sessions` / `messages`
- keep chat history separate from the knowledge store

## Claim Status Model

Recommended statuses:

- `unverified`
- `supported`
- `contradicted`
- `mixed`
- `deprecated`

Each claim should also carry:

- confidence score
- extraction method
- model used
- prompt/version metadata
- timestamps

Important distinction:

- raw extracted model output should be stored separately from canonical claims
- canonical `claims` are the resolved KB view
- `extracted_observations` are the audit trail of what a model said before resolution,
  review, deduplication, or promotion

## Evidence Model

Each claim should be traceable to one or more evidence records including:

- source ID
- source version ID
- artifact ID
- chunk ID
- evidence type: support / contradict / mention / derived
- excerpt text
- exact character offsets where available
- quote hash or normalized excerpt hash where useful
- transcript time offsets where available
- extraction timestamp

This is necessary for trust, comparison, and debugging.

## Verification Policy and Budget

Verification is the one workflow whose cost can explode: every claim can fan out into
searches, scrapes, and extra extraction passes. Build order step 6 gates it with an
explicit per-claim budget so cost stays bounded.

Triggers:

- manual request (verify a claim, a topic, or the claims behind a report)
- claims whose `importance_score` exceeds a configurable threshold after extraction
- report generation may queue verification for high-importance unverified claims it
  wants to cite, rather than blocking on them

Per-claim budget (defaults, configurable):

- at most 2 web searches
- at most 3 additional sources examined
- one extraction/comparison pass per examined source

Stop conditions:

- budget exhausted, or
- 2 independent supporting sources found, or
- a contradiction found — at which point the goal switches to recording the conflict
  (`claim_links`, `resolution_candidates`), not resolving it inside the budget

On budget exhaustion the claim stays `unverified`, with
`claims.verification_attempted_at` set so reports can distinguish "never checked"
from "checked, inconclusive". Verification work runs through `jobs`, so topic-wide
verification is many small budgeted jobs — resumable, retryable, and individually
cheap.

## Ingestion Pipeline Plan

### Phase 1: Foundation

- replace SQLite-only persistence with PostgreSQL-backed domain storage
- keep existing session/message history but separate it from KB tables
- add migrations
- define canonical source identity and versioning rules

### Phase 2: Source ingestion

- web page ingestion
- search result capture
- YouTube transcript ingestion
- PDF / Markdown / file ingestion
- source dedupe via canonical URL or content hash
- snapshot/version storage
- allow ingestion with or without a topic association

### Phase 3: Chunking and retrieval foundation

- chunk long artifacts
- store chunk offsets and metadata
- support retrieval for extraction and reporting

### Phase 4: Extraction

- entity extraction
- event extraction
- claim extraction
- metric/spec extraction
- relationship extraction
- detection of other notable topics in the source for later revisit
- support broad taxonomy-style extraction passes such as historical, economic, legal, product, etc.

### Phase 5: Verification

- search for corroborating evidence
- search for contradicting evidence
- update claim status
- preserve disagreements rather than overwriting them

### Phase 6: Reporting

- use stored facts first
- trigger fresh collection when topic coverage is stale or thin
- answer with latest data
- include confidence and citation trail

### Phase 7: Improvements

- embeddings / vector retrieval if needed
- reranking
- stronger comparison tooling
- automated topic monitoring

## Recommended Build Order

### Guiding principle: prove the hard core before building the foundation

The earlier version of this plan sequenced the work horizontally — database first,
then ingestion, then chunking, then extraction. That front-loads the well-understood
plumbing and defers the two things that actually determine whether this system works:

1. whether a local `14B`–`30B` model can reliably turn messy source text into atomic,
   dedup-able claims with usable evidence links
2. whether entities, events, and claims can be **resolved** across sources (so the same
   thing mentioned in five sources becomes one record, not five)

Both are unknowns. Designing ~20 tables around an extraction output shape that has not
been observed on real content is the main risk in this plan. So the build order below
starts with a throwaway spike that de-risks extraction and resolution, lets the schema
fall out of what is actually observed, and defers the PostgreSQL migration until the
claim/evidence schema has stabilized.

### Order

0. **Extraction + resolution spike (throwaway).** Take one real YouTube transcript and
   one real article. Chunk them, run claim/entity/event extraction with the intended
   local model, and hand-inspect ~100 claims. The goal is to answer, on real content:
   - is first-pass claim quality good enough to build on?
   - what fields do claims/entities/metrics actually need?
   - how bad is the duplication problem, and what resolution strategy is required?

   Prototype this on **SQLite** — do not gate it on a database migration. Throw the code
   away; keep the learnings and a schema shaped by real output. During the spike, model
   outputs should be treated as raw `extracted_observations`, not as final canonical
   `claims`.

1. **Done.** Lock the **entity/claim/event resolution strategy** explicitly. Resolution
   is not a second-wave nicety — it is what makes the KB queryable instead of a pile of
   disconnected extractions. Locked as decision 25 in the Decision Register: exact
   normalized-name match auto-merges entities, everything else (fuzzy entity matches,
   all claim matches via embedding similarity, event matches) lands in
   `resolution_candidates` for review — nothing else auto-merges. This is not the
   "lexical + trigram" default this section originally floated: the step-0 spike
   measured lexical/trigram claim matching catching zero real cross-source duplicates,
   so claim candidate generation runs on embedding similarity instead (see
   [spike/FINDINGS.md](spike/FINDINGS.md)). In v1, resolution reads mention locations
   from `extracted_observations` (chunk IDs + offsets) — `entity_mentions` and
   `event_mentions` are second-wave tables, not resolution prerequisites.

2. **Done.** Add the **source registry plus versioned ingestion** for web, files, and
   YouTube transcripts, still on SQLite. Wire in the retention invariant (see retention
   section: never prune a version that has evidence pointing at it). Implemented in
   `deep_research/kb/` (`db.py`, `canonical.py`, `storage.py`, `ingest.py`) with a CLI
   at `cli/kb.py` (`deep-research-kb ingest-url|ingest-youtube|ingest-file|list-sources|
   show-source`), using a SQLite db separate from chat sessions (`kb.db` next to
   `research.db`) plus a `kb_snapshots/` directory of raw files on disk. Verified
   end-to-end: dedup on unchanged content, new versions on real content changes,
   first+newest-two pruning with files deleted from disk, the `retention_locked`
   evidence-integrity carve-out surviving a prune pass, and all three ingestion
   failure paths (bad URL, missing file, bad YouTube ID) logging to
   `source_fetch_attempts` instead of crashing. One design correction made during
   verification: file source identity is the file path, not the content hash — content
   hash as identity would make every edit a brand-new unrelated source instead of a new
   version of the same one, which would have defeated file versioning entirely.

3. **Done.** Add **chunk storage and retrieval** (FTS5 baseline). Implemented as
   `artifacts` + `artifact_chunks` tables plus a manually-synced FTS5 virtual table
   in `deep_research/kb/db.py`, with per-source-type text extraction in
   `deep_research/kb/artifacts.py` (web/html via BeautifulSoup, markdown/text via
   direct decode, PDF via `pypdf` with per-page chunking and `page_number` metadata,
   DOCX via `python-docx`, YouTube transcripts via time-based chunking preserving
   `time_start_seconds`/`time_end_seconds`). Re-chunking with unchanged parameters is
   a no-op; re-chunking with different parameters creates a new artifact generation
   and leaves old chunks untouched, satisfying the immutability requirement in
   "Retention vs. Evidence Integrity" ahead of `claim_evidence` existing. CLI:
   `deep-research-kb chunk-source <id>` and `search <query>`. Verified against all
   five ingested source types, plus idempotency, new-generation-on-param-change, and
   correct FTS results (including the cross-source Furman "92% of GDP growth" claim
   from the spike surfacing from both the article and transcript). One bug found and
   fixed during verification: raw user search queries containing punctuation (e.g.
   `92% GDP growth`) crashed FTS5's query parser — fixed by quoting each token as an
   FTS5 string literal before matching.

4. **Done.** Add the **extraction pipeline** for entities, events, claims, and metrics,
   with an **`extraction_runs` provenance record** so re-extraction is idempotent and
   claims can be traced to the model + prompt + params that produced them. Implemented
   in `deep_research/kb/extraction.py` (chunk-by-chunk extraction reusing the spike's
   validated prompt, extended with a `metrics[]` field per decision 22; idempotent via
   a model+prompt+schema signature hash on `extraction_runs`) and
   `deep_research/kb/resolution.py` (promotion + resolution, implementing decision 25
   exactly: entities/events auto-merge only on exact normalized-name/title match via DB
   `UNIQUE` constraints; claims also get an exact-text-match tier — a safe addition
   consistent with decision 25's intent, not a fuzzy/lexical merge; fuzzy entity
   matches require a minimum name length before consideration, per the spike's finding;
   claim near-duplicates are generated via embedding cosine similarity into
   `resolution_candidates`, never auto-merged). `claim_evidence` creation is the first
   real caller of `lock_version_retention` (stubbed in step 2) — verified the flag
   actually flips on real data. CLI: `extract-source`, `list-claims`, `show-claim`,
   `list-resolution-candidates`, `review-candidate` (review only changes candidate
   status; merge execution is explicitly deferred, not implemented). Verified against
   the real article and YouTube transcript: 44 and 81 observations respectively, exact
   dedup and idempotent re-extraction both confirmed, a metrics table populated with
   56 real values, entity fuzzy-candidate gating correctly avoided spike-style
   nonsense matches, and claim resolution candidates reproduced the spike's precision
   pattern — genuine same-fact duplicates ranked highest (0.94-0.95 cosine), degrading
   into topically-related-but-distinct claims further down, all correctly queued for
   review rather than merged. One specific cross-source duplicate the spike found
   (the Jason Furman "92% of GDP growth" claim) didn't reappear verbatim this run —
   traced to LLM extraction non-determinism (the source sentence is intact in its
   chunk; the model simply phrased/omitted it differently on this pass), not a
   pipeline defect.

5. **Done.** **Migrate to PostgreSQL** once the claim/evidence schema has stabilized.
   Migrating once, against a validated schema, is far cheaper than designing Postgres
   tables twice. This is also where jsonb ergonomics, real FTS, and concurrent workers
   start to pay off. Postgres runs via `docker-compose.yml` (a new `postgres` service
   alongside `searxng`), on the same machine as the LLM and snapshot storage — decided
   explicitly rather than splitting across machines, since decision 9 (raw snapshots on
   disk, DB stores only paths/hashes) means separating the DB host from the file host
   would require a shared network filesystem for no real benefit at single-user scale.
   The frontend is the one piece that *can* run elsewhere, since it already just talks
   to the backend over HTTP.

   `deep_research/kb/db.py` was rewritten in place (same class, same public method
   signatures, so `ingest.py`/`artifacts.py`/`extraction.py`/`resolution.py`/`cli/kb.py`
   needed no changes beyond timestamp/type handling — see below) against `asyncpg`.
   Upgrades that came free with the migration: `JSONB` instead of `TEXT`+
   `json.dumps`/`loads` for metadata/payload columns (a registered codec means callers
   pass/receive Python dicts directly), real `TIMESTAMPTZ` instead of ISO8601 strings,
   and native full-text search (`tsvector` generated column + GIN index +
   `websearch_to_tsquery`) replacing the SQLite FTS5 virtual table and its manual sync
   code — `websearch_to_tsquery` also fixes the punctuation-crash bug from the SQLite
   version for free, verified directly (`92% GDP growth`, unbalanced quotes, and `&`/`|`
   all resolve to sane tsqueries with zero errors, unlike FTS5's MATCH operator).

   Clean cutover, not a data migration: existing KB content was verification/spike data
   only, so the plan was to recreate the schema and re-run the same ingest → chunk →
   extract → resolve pipeline against Postgres rather than write a SQLite→Postgres
   migrator for throwaway rows. Two real bugs were found and fixed during that
   re-verification, both now-obvious in hindsight once real data hit them:
   - `ingest.py` had its own local `_now()` returning an ISO string (predating the
     Postgres-wide `datetime` convention in `db.py`) — `TIMESTAMPTZ` columns rejected
     it; fixed to return a `datetime`.
   - Postgres `REAL` is 4-byte single precision, unlike SQLite's `REAL` which is always
     8-byte — this showed up as `confidence: 0.9000000238418579` instead of `0.9`.
     Fixed by widening every float column (`confidence`, `importance_score`, `score`,
     `value_numeric`, `rank_weight`, `trust_score`, `time_start_seconds`,
     `time_end_seconds`) to `DOUBLE PRECISION`.

   Re-verified the full step 2-4 test matrix against Postgres after both fixes: all
   five source types ingest/chunk/extract identically, dedup and idempotent
   re-extraction hold, retention pruning and the `retention_locked` evidence-integrity
   lock both still fire correctly, and FTS search (including the previously-crashing
   punctuation query) returns correct, better-stemmed results.

   Known minor gap, not fixed: CLI commands never explicitly close the `asyncpg`
   pool they open (no observed warnings in practice since each CLI invocation is a
   short-lived process that exits immediately after, but worth cleaning up if this
   code is ever driven from a long-running process instead of one-shot CLI calls).

6. **Done.** Add the **verification workflow**, gated by a per-claim verification budget
   tied to `importance_score` (see "Verification Policy and Budget") so search cost
   cannot explode. Implemented in `deep_research/kb/verification.py`: search the KB's
   own data first via a live embedding-similarity pass against claims independent of the
   target claim's own source(s) (not the possibly-stale `resolution_candidates` rows
   from promotion time, since those never get retroactively recomputed for claims
   created later); fall back to a real web search (reusing the existing SearXNG tool)
   plus the full ingest → chunk → extract → promote pipeline only if internal coverage
   is thin and budget remains. The budget counts distinct *sources* examined (not
   candidate claims — several matching claims from one source count once), capped at
   `verification_max_sources_examined` (3) and `verification_max_web_searches` (2).

   One new piece of infrastructure this needed: embedding similarity alone can't tell
   "same fact, reworded" apart from "same topic, conflicting numbers." Added an LLM
   comparison pass (supports/contradicts/unrelated) and validated it directly against 5
   known pairs before building the pipeline around it — including a same-fact-different-
   units case requiring real arithmetic (`$23,000/sec` vs `$83M/hour`) and a synthetic
   contradiction — 5/5 correct. Contradictions are recorded via
   `resolution_candidates(candidate_type='claim_contradiction')`, reusing the same v1
   review queue as entity/claim duplicates rather than requiring the deferred
   `claim_links` table — the original schema draft already anticipated this
   `candidate_type` value. Added `claims.verification_notes` (jsonb) for a lightweight
   audit trail (support/contradict counts, sources examined, searches used) alongside
   `verification_attempted_at`.

   Verified end-to-end on real claims: `verify-claim` on a claim with a known
   transcript-side match found that internal support, then genuinely fell back to a
   live web search, ingested a real, previously-unseen Fortune article, and found
   independent corroboration there too — correctly reaching `supported`. A vague,
   generic claim correctly exhausted its budget (1 support, 3 sources, 1 search) and
   stayed `unverified` rather than guessing. `verify-source` (batch, importance-gated)
   correctly rejected a known same-article "duplicate" as non-independent before
   falling to web search, on both eligible claims. `--force` re-verification and the
   default skip-if-already-attempted behavior both confirmed.

   New CLI: `verify-claim <id> [--force]`, `verify-source <id> [--threshold] [--force]`.

7. **Done.** Add **topic reports and event comparison** (timeline first). Implemented
   per decision 27's scope resolution:
   - Schema: `topics`, `topic_source_links`, `claim_topics` (many-to-many, decision 19),
     `reports`. `link_status` on the two link tables carries both explicit attachment
     and the suggest-then-review workflow in one column ('attached'/'suggested'/
     'rejected') rather than a parallel suggestions table.
   - `deep_research/kb/topics.py`: entity-overlap suggestion generation, both backfill
     (`generate_topic_suggestions`, run at topic creation) and forward-check
     (`check_claims_against_topics`, run after every `resolve_and_promote` via
     `extract-source` — verified this actually re-surfaces a claim when simulated as
     newly-created, closing the same "only new-vs-existing, never old-vs-new" gap
     already found once in step 4/6's embedding candidates).
   - `deep_research/kb/timeline.py`: strict timeline (event + parseable date only),
     best-effort date parsing tested against every real messy format in the KB
     ("2008–2013", "1999 and a part of 2000", month/year, year-only, full ISO).
     Extraction prompt extended with `date_precision` (bumped to
     `v3-with-date-precision`) so the model self-reports certainty instead of us
     inferring it.
   - Preferred-source: `recompute_preferred_source` runs automatically inside
     `add_claim_evidence`, picks the highest `trust_tiers.rank_weight`, never
     overwrites a manually-reviewed claim (`set_preferred_source_manual`).
   - `deep_research/kb/reports.py`: LLM-synthesized markdown report, new `reports` row
     per generation (decision 27) but only the latest is ever surfaced. Found and fixed
     a real bug during first test: the input (130 claims) exceeded the local llama.cpp
     server's configured 4096-token context, so report input is now built against a
     configurable character budget (`kb.report_max_input_chars`), timeline content
     first, undated claims by importance after — not a fixed claim count, since the
     server's actual context window isn't something this app controls.
   - CLI: `create-topic`, `list-topics`, `show-topic`, `attach-source`, `attach-claim`,
     `backfill-topic-suggestions`, `review-topic-suggestion`, `generate-report`,
     `set-preferred-source`.
   - Web: `web/kb_routes.py` (new FastAPI router, separate from the chat-session API)
     plus `TopicsView.vue`/`TopicDetailView.vue`, wired into the existing router/nav —
     both CLI and web UI landed in this step per decision 27, not CLI-first.
   - Verified end-to-end on the real AI-bubble topic: 130 claims auto-attached from two
     seed sources, 22-24 chronologically-correct timeline entries (including an 1850
     railway-bubble claim), 34 claims / 5 sources correctly suggested via entity
     overlap (including sources discovered during step 6's verification web-fallback),
     a coherent cited report, and the `SUPPORTED` status badge from step 6 correctly
     flowing through to the timeline UI. Browser-driven with Playwright (no project
     skill existed for this yet — recommend `/run-skill-generator` next time): topics
     list, timeline, suggestions, and report all render correctly with zero console
     errors, and a live accept/reject click was confirmed to actually persist (35 to 34
     pending). One pre-existing gap noticed, not fixed (out of scope for this step):
     the SPA has no server-side fallback route, so direct navigation/refresh on any
     client-side route (`/history`, `/topics`, ...) 404s — only navigation via in-app
     links works. Confirmed this predates step 7 (`/history` has the same issue).

   **Follow-up: report generation is now map-reduce, not truncation.** The first
   version bounded report input to a static guessed character budget and silently
   dropped whatever didn't fit — correctness that degrades as topics grow, and the
   budget was never the server's real limit anyway. Replaced with:
   - `detect_context_size` (in `extraction.py`) queries llama.cpp's native `/slots`
     endpoint (server root, not `/v1`) for the actual configured per-slot context,
     instead of guessing. Falls back to `kb.report_context_fallback_tokens` only if
     that endpoint is unavailable.
   - If a topic's content doesn't fit in one pass, it's batched and each batch is
     summarized (map, explicitly instructed to preserve every date/number/citation/
     status flag), then the summaries are recursively re-batched and re-summarized
     (reduce) until they fit one final synthesis call. Nothing is ever dropped — an
     arbitrarily large topic just costs more LLM calls.
   - If map-reduce had to run, the result carries a human-readable suggestion (detected
     context size + a concrete `llama-server` flag recommendation), surfaced via CLI and
     web UI. The pipeline never restarts the inference server itself — that stays the
     user's call, since this app doesn't know the server's full original launch flags,
     the server may be shared with other work, and more context means more VRAM for
     the KV cache on a card that's already partly committed to the loaded model.

   Found and fixed a second real bug while verifying this on the actual 130-claim
   topic: even after map-reduce correctly shrank the *input*, the final synthesis
   call's own *response* got cut off mid-sentence, because the token reserve used to
   size the input budget (1200 tokens) assumed the same short output as a batch
   summary — but the final call writes a full multi-section report, which runs much
   longer. Fixed by splitting the reserve in two: `BATCH_RESPONSE_TOKEN_RESERVE` (700,
   for the map/reduce steps' compact bullet-point output) and
   `FINAL_RESPONSE_TOKEN_RESERVE` (2200, for the final report). The reduce loop now
   converges toward the stricter final budget, not the looser batch budget, so there's
   always headroom left for the model to finish its sentence. Re-verified on the same
   topic afterward: report ends cleanly, no truncation, same coherent multi-section
   structure.

8. Add **embeddings / vector retrieval only after** the source/claim/evidence model is
   solid.

Important notes:

- do not start with embeddings first — for news/events/history, the
  source/claim/evidence model matters more
- do not start with the PostgreSQL migration first — it de-risks nothing; SQLite (FTS5 +
  JSON1) is enough to carry the spike and early phases
- treat step 0 as mandatory, not optional — it is the cheapest way to avoid designing the
  whole schema around extraction output you have not seen

## Extraction + Resolution Spike (Step 0)

Status: **done**. Harness lives in `spike/` (throwaway, per the plan below); results
and the answers to both exit questions are in [spike/FINDINGS.md](spike/FINDINGS.md).
Headline results: 158 claims extracted from one article + one YouTube transcript on
the same topic using `Qwen3-14B` via local `llama.cpp`; extraction quality cleared all
minimum quality gates. Key adjustment for step 1: entity resolution should auto-merge
only on exact normalized-name match (naive substring/fuzzy matching on short names was
unusably noisy — e.g. "AI" spuriously matched "Britain"), and claim resolution needs an
embedding-similarity pass since lexical matching found zero cross-source claim
duplicates even where sources overlapped in substance.

This section below captures the original intent so the spike's design rationale
stays visible next to the results.

### Why this comes first

The plan commits to roughly 20 tables whose columns are shaped by one unproven
assumption: that a local `14B`–`30B` model can turn messy source text into clean, atomic,
dedup-able claims with usable evidence links. Almost everything downstream depends on
that assumption:

- `claims.canonical_text` vs `normalized_text`
- `claims.confidence` and `claims.importance_score`
- the `claim_evidence` provenance model
- `claim_links.duplicate_of` and claim dedup
- `entities.normalized_name` and cross-source entity resolution

No one has yet looked at what the model actually produces on this project's content with
this project's models. Designing the schema against imagined output is the largest risk
in this plan. The spike replaces imagination with ~100 real extracted rows that can be
read by hand.

Treat it as a load-bearing experiment, not a feature. The harness code is thrown away.
What is kept is the learnings and a schema shaped by observed output.

### The two questions it must answer

1. Is first-pass extraction quality good enough to build on?
- are claims actually atomic (one fact per claim, not blobs)?
- are they faithful to the chunk, or does the model hallucinate specifics?
- can the model reliably point a claim back to the chunk it came from (evidence linking),
  or does it invent citations?
- is the model's self-reported confidence/importance meaningful or noise?

2. How bad is the duplication / resolution problem?
- feed two sources about the same topic
- does one real-world entity collapse to one record or split into several
  (for example "Nvidia" / "NVIDIA Corp" / "the company")?
- does the same event become one `events` row or several near-duplicates?
- do near-identical claims collapse or pile up as reworded duplicates?

The second question is the one this plan most hand-waves, and its answer drives one of the
biggest v1 scope decisions: what resolution strategy is actually required
(lexical/trigram match, an embedding pass, a manual merge queue, or a mix).

### Concrete shape

A small standalone script. No production database migration, no web UI, no final schema:

- load 1 real article + 1 real YouTube transcript covering the same story or topic —
  required, since the resolution question cannot be tested unless the sources overlap
  (reuse the existing scraper)
- chunk them (naive fixed-size chunking is fine for the spike)
- for each chunk, call the local model with a claim-extraction prompt returning JSON such
  as `[{claim_text, entities[], event?, confidence, source_chunk_id}]`
- write everything to a JSONL file or throwaway SQLite tables
- print summary counts (claims per source, roughly-unique entities, obvious duplicate
  clusters)

Then read the JSONL or SQLite output by hand. That reading is the actual deliverable.

For SQLite-backed spike tables, keep the shape intentionally small:

- `spike_chunks`
- `spike_extraction_runs`
- `spike_extracted_observations`
- `spike_resolution_candidates`

Do not treat those names or columns as final. Their job is to expose what the model
actually returns and what the later PostgreSQL schema must support.

### Why no database for the spike

The revised build order deliberately does not start with the SQLite to PostgreSQL
migration, because migration de-risks none of the questions above — it is pure plumbing.
The spike wants the fastest path to seeing model output, so a JSONL file (or at most a
throwaway SQLite table) is enough. The PostgreSQL migration happens later, once the claim
rows have a known, validated shape.

### What the spike produces

- a `claims` / `entities` / `claim_evidence` schema reverse-engineered from real output
  instead of guessed
- an `extracted_observations` shape that separates raw model output from canonical KB
  records
- a first pass at `resolution_candidates` for duplicate entities, events, and claims
- a read on whether the intended fast model (for example `Qwen3-14B`) is good enough for
  first-pass extraction, or whether the heavy model is needed from the start
- an honest sizing of the resolution problem, feeding the step-1 resolution decision
- reusable prompt drafts (claim extraction, entity tagging) even though the harness code
  is discarded

### Effort and exit criteria

- roughly one day of work
- done when ~100 extracted claims across two sources have been inspected and the two
  questions above have concrete answers
- output feeds directly into the schema draft and the step-1 resolution strategy decision

Minimum quality gates before moving past the spike:

- most extracted claims must be atomic enough to dedupe and verify
- evidence links must point to the correct chunk or transcript range often enough to be
  trusted as provenance
- hallucinated specifics must be rare enough that review and verification can contain the
  risk
- duplicate entity/event/claim clusters must be measurable, with a proposed v1 resolution
  strategy
- self-reported confidence and importance scores must be judged useful or explicitly
  treated as weak metadata

If these gates fail, do not proceed directly to schema implementation. Adjust prompt
shape, model choice, chunking, or extraction mode design first.

## Hardware & Resource Planning

Hardware sizing, GPU/RAM/storage tiers, and purchase logic have moved to a
dedicated doc so they can evolve independently of the data model and pipeline:

- [HARDWARE.md](HARDWARE.md) — hardware recommendation, machine tiers, and purchase logic

## Initial Feature Boundaries

For v1, keep scope limited:

- single user
- local-only deployment
- transcript-only YouTube support
- latest-data reporting
- support unverified claims with proper labeling

Defer until later unless needed:

- OCR/frame extraction from videos
- multi-user support
- hosted API dependency
- snapshot/reproducible report mode as the primary behavior

Note:

- OCR may later be integrated from an existing separate OCR project, but it is not required for this project's v1 scope

## Decision Backlog For Project Definition

To avoid a large unfocused design pass, the remaining decisions should be handled in small batches.
As each answer is provided, this file should be updated so the project can resume cleanly after a pause.

### Batch 1: v1 product scope

Answer these first:

1. What are the top 3 workflows that must work in v1?
2. What is explicitly out of scope for v1?
3. Is v1 primarily:
- a personal local knowledge base
- a deep-research agent backed by a knowledge base
- both, with equal priority

Current answer status:

- resolved — recorded as decision 20 in the Decision Register

Why this batch matters:

- it sets the build order
- it prevents accidental scope explosion
- it decides whether ingestion or reporting gets priority first

### Batch 2: source and trust policy

After Batch 1:

1. Should sources have trust tiers
- for example: official, reputable reporting, secondary analysis, user-generated
2. Should some source types be deprioritized by default?
3. How should contradictions be shown:
- simply as conflicting claims
- or with a preferred source ranking

Why this batch matters:

- it affects claim verification logic
- it affects ranking and report generation

Current answer status:

- resolved — recorded as decision 21 in the Decision Register

### Batch 3: extraction policy

After Batch 2:

1. What should always be extracted from every source?
2. What should only be extracted when explicitly requested?
3. Should extraction modes exist from the start:
- historical
- economic
- legal
- product/spec
- general

Why this batch matters:

- it determines prompt design
- it controls storage growth and compute cost

Current answer status:

- resolved — recorded as decision 22 in the Decision Register

### Batch 4: retrieval and freshness behavior

After Batch 3:

1. When answering a question, should the system prefer:
- stored knowledge first
- fresh collection first
- hybrid with stored-first unless stale
2. When should a source be reanalyzed instead of re-fetched?
3. What should trigger re-extraction after:
- model upgrades
- schema changes
- improved prompts

Why this batch matters:

- it determines runtime behavior and cost
- it affects how much value you get from stored sources

Current answer status:

- resolved — recorded as decision 23 in the Decision Register

### Batch 5: operations and review workflow

After Batch 4:

1. Should long ingestion/extraction jobs run automatically in the background, or only when started manually?
2. Do you want a review step for claims before they are treated as trusted?
3. What should happen on partial failure:
- keep partial results
- mark incomplete and retry later
- fail the whole run

Why this batch matters:

- it affects the job system
- it affects usability and trust

Current answer status:

- resolved — recorded as decision 24 in the Decision Register

### Batch 6: interfaces and outputs

After Batch 5:

1. What should be the primary interface first:
- CLI
- web UI
- both equally
2. Do you want export/output support for other local tools:
- JSON
- CSV
- SQL views
- local API
3. Do you want a review UI in v1 for:
- sources
- timelines
- claims
- contradictions

Why this batch matters:

- it affects how much frontend work is needed early
- it defines how this project will integrate with other local tools

Current answer status:

- resolved — recorded as decision 26 in the Decision Register

## Summary of Current Recommendation

- Use PostgreSQL, not MySQL
- Build a knowledge-base-first system rather than extending the current prompt-only summarizer shape
- Support both topic-driven workflows and topic-independent local knowledge accumulation
- Make claims, evidence, sources, and source versions first-class entities
- Support unverified claims explicitly
- Prefer latest-data reporting
- Plan for local ingestion/extraction/verification workflows
- If this roadmap is the real target, 32 GB VRAM is meaningfully more attractive than it was for the current repo alone

## PostgreSQL Schema Draft

This is a first-pass schema draft intended to make the architecture concrete before coding begins.
It is not final SQL, but it should be close enough to drive migrations, service boundaries, and API design.

### Design goals

- support both topic-bound and topic-independent ingestion
- preserve provenance and re-analysis capability
- keep chat/session data separate from the knowledge base
- support trust/ranking, contradictions, timelines, and review UI needs
- support partial extraction now and richer extraction later

### Conventions

- use `uuid` primary keys for most domain tables
- use `created_at` and `updated_at` on nearly all mutable tables
- use `jsonb` only where flexibility is useful; do not replace core normalized tables with blobs
- store raw files and snapshots on disk; store paths, hashes, and metadata in PostgreSQL
- use enums or constrained text values for status/type fields depending on migration preference

### Source identity and lifecycle rules

- define canonicalization rules before bulk ingestion, including URL normalization,
  YouTube video/playlist identity, file hash behavior, and redirect handling
- store fetch failures and parse failures; they are useful for retry logic and freshness
  decisions
- support source deletion as an explicit lifecycle operation, not an ad hoc row delete
- when deleting a source, decide whether to keep reports, remove generated facts, or
  tombstone affected claims
- never delete evidence-referenced source versions unless the dependent evidence and
  derived records are intentionally removed as part of a controlled cleanup

### Suggested extensions

- `pgcrypto` for UUID generation if needed
- `pg_trgm` for fuzzy matching and search helpers
- full vector support can be deferred until later

### Core reference tables

#### `source_types`

Purpose:
- normalized source kinds for ingestion/routing

Suggested fields:
- `id`
- `code` such as `web`, `youtube_video`, `youtube_playlist`, `pdf`, `markdown`, `html_file`, `docx`, `text`
- `label`

#### `trust_tiers`

Purpose:
- source trust/ranking categories used in verification and reporting

Suggested fields:
- `id`
- `code` such as `official`, `reputable_reporting`, `secondary_analysis`, `user_generated`
- `label`
- `rank_weight`
- `description`

### Topic and run tables

#### `topics`

Purpose:
- canonical research topics

Suggested fields:
- `id uuid primary key`
- `slug text unique`
- `name text not null`
- `description text`
- `status text`
- `default_extraction_mode text`
- `monitoring_enabled boolean default false`
- `monitoring_interval_minutes integer null`
- `stale_after_minutes integer null`
- `created_at timestamptz`
- `updated_at timestamptz`

Indexes:
- unique index on `slug`
- index on `monitoring_enabled`

#### `topic_aliases`

Purpose:
- alternate names for a topic

Suggested fields:
- `id uuid primary key`
- `topic_id uuid not null`
- `alias text not null`
- `created_at timestamptz`

Indexes:
- unique index on `(topic_id, alias)`
- trigram or lowercased lookup index later if needed

#### `topic_runs`

Purpose:
- track topic research/refresh/report runs

Suggested fields:
- `id uuid primary key`
- `topic_id uuid null`
- `run_type text` such as `ingest`, `research`, `refresh`, `report`, `verification`, `reextract`
- `trigger_type text` such as `manual`, `monitoring`, `api`, `ui`
- `status text` such as `queued`, `running`, `completed`, `partial`, `failed`, `cancelled`
- `requested_focus text`
- `requested_modes jsonb`
- `started_at timestamptz null`
- `completed_at timestamptz null`
- `created_at timestamptz`
- `updated_at timestamptz`

Indexes:
- index on `(topic_id, created_at desc)`
- index on `status`

#### `topic_source_links`

Purpose:
- optional many-to-many link between topics and sources

Suggested fields:
- `id uuid primary key`
- `topic_id uuid not null`
- `source_id uuid not null`
- `link_reason text` such as `seed_source`, `discovered`, `manual_attach`, `verification`
- `is_primary boolean default false`
- `created_at timestamptz`

Indexes:
- unique index on `(topic_id, source_id)`

### Source and artifact tables

#### `sources`

Purpose:
- canonical source identity regardless of version changes

Suggested fields:
- `id uuid primary key`
- `source_type_id uuid not null`
- `canonical_uri text not null`
- `canonical_key text not null`
- `title text`
- `author text`
- `publisher text`
- `published_at timestamptz null`
- `trust_tier_id uuid null`
- `trust_score numeric(5,2) null`
- `language_code text null`
- `is_active boolean default true`
- `created_at timestamptz`
- `updated_at timestamptz`

Notes:
- `canonical_key` should be a stable dedupe identity such as normalized URL, YouTube video ID, playlist ID, or file hash

Indexes:
- unique index on `canonical_key`
- index on `source_type_id`
- index on `trust_tier_id`

#### `source_versions`

Purpose:
- retained snapshots for a source over time

Suggested fields:
- `id uuid primary key`
- `source_id uuid not null`
- `version_number integer not null`
- `snapshot_path text not null`
- `content_hash text not null`
- `http_status integer null`
- `mime_type text null`
- `captured_at timestamptz not null`
- `is_first_version boolean default false`
- `is_latest boolean default false`
- `retention_locked boolean default false`
- `metadata jsonb`
- `created_at timestamptz`

Notes:
- retention policy should preserve first version plus newest two versions
- `is_first_version` and `is_latest` are denormalized flags owned by the ingest and
  prune jobs; update them in the same transaction that inserts or removes versions,
  or they will drift
- `retention_locked` allows manual exceptions later if policy ever changes
- **evidence integrity overrides the prune rule.** See "Retention vs. Evidence
  Integrity" below — a version that has `claim_evidence` referencing it must never be
  pruned, even if the "first + newest two" rule would otherwise drop it.

Indexes:
- unique index on `(source_id, version_number)`
- index on `(source_id, captured_at desc)`
- index on `is_latest`

##### Retention vs. Evidence Integrity

The universal "first version + newest two versions" rule (see decisions 13 and 16)
directly conflicts with `claim_evidence.source_version_id`. If a claim's evidence points
at version 4, and version 4 is a "middle" version, the naive prune rule would delete the
exact snapshot the provenance depends on — orphaning the evidence and destroying the one
thing this whole design exists to protect.

This is a latent data-corruption bug, so the invariant must be explicit:

- **A `source_version` that is referenced by any `claim_evidence` row must never be
  pruned.** The moment evidence links to a version, set `retention_locked = true` on it.
- The prune job must exclude `retention_locked` (and therefore evidence-referenced)
  versions when applying "first + newest two."
- Enforce it at two layers:
  - `claim_evidence.source_version_id -> source_versions.id` with `ON DELETE RESTRICT`
    (a stray delete fails loudly instead of silently orphaning evidence)
  - the prune query filters out `retention_locked = true` before selecting deletion
    candidates
- Net effect: the retained set for a source is "first + newest two + every version any
  claim cites." This can exceed three versions for heavily-cited sources — that is
  correct and intended.

The same invariant applies one level down. Evidence actually anchors to
`artifact_chunks`, not just versions, and chunks have a second death path that the
prune rule alone does not cover: **re-chunking**. Re-running chunking with different
parameters (new chunk size, changed text normalization, prompt/model upgrades per
Batch 4) would orphan every `claim_evidence.artifact_chunk_id` pointing at the
replaced rows — the same corruption, with no pruning involved. So:

- **chunks are immutable once any `claim_evidence` row references them** — re-chunking
  creates a new chunk set (a new `artifacts` row or a new chunk generation), never an
  in-place update or delete of referenced chunk rows
- enforce with `claim_evidence.artifact_chunk_id -> artifact_chunks.id`
  `ON DELETE RESTRICT`, exactly like the version FK
- `excerpt_hash` is the detection net for normalization drift, not a license to
  rewrite referenced chunks

#### `source_fetch_attempts`

Purpose:
- operational history for attempts to fetch, parse, or refresh a source

Suggested fields:
- `id uuid primary key`
- `source_id uuid null`
- `source_version_id uuid null`
- `attempt_type text` such as `fetch`, `parse`, `transcript_fetch`, `metadata_fetch`
- `status text` such as `succeeded`, `partial`, `failed`, `blocked`, `not_found`
- `requested_uri text null`
- `final_uri text null`
- `http_status integer null`
- `error_code text null`
- `error_message text null`
- `started_at timestamptz`
- `completed_at timestamptz null`
- `metadata jsonb`
- `created_at timestamptz`

Notes:
- failed fetches are useful data, especially for web pages, transcripts, redirects,
  paywalls, unsupported MIME types, and parser failures
- this table should not replace `jobs`; it records source-specific access outcomes

Indexes:
- index on `(source_id, created_at desc)`
- index on `status`

#### `artifacts`

Purpose:
- normalized extracted representation of a specific source version

Suggested fields:
- `id uuid primary key`
- `source_version_id uuid not null`
- `artifact_type text` such as `clean_text`, `transcript`, `parsed_markdown`, `parsed_pdf`, `search_result_set`
- `storage_path text not null`
- `content_hash text not null`
- `title text null`
- `summary text null`
- `metadata jsonb`
- `created_at timestamptz`
- `updated_at timestamptz`

Indexes:
- unique index on `(source_version_id, artifact_type)`
- index on `artifact_type`

#### `artifact_chunks`

Purpose:
- retrieval, evidence, and re-analysis units

Suggested fields:
- `id uuid primary key`
- `artifact_id uuid not null`
- `chunk_index integer not null`
- `chunk_text text not null`
- `chunk_hash text not null`
- `char_start integer null`
- `char_end integer null`
- `token_estimate integer null`
- `section_label text null`
- `page_number integer null`
- `time_start_seconds numeric(12,3) null`
- `time_end_seconds numeric(12,3) null`
- `metadata jsonb`
- `created_at timestamptz`

Notes:
- chunks referenced by `claim_evidence` are immutable; re-chunking produces a new
  chunk set instead of replacing rows (see "Retention vs. Evidence Integrity")

Indexes:
- unique index on `(artifact_id, chunk_index)`
- full-text index on `chunk_text`
- index on `(artifact_id, page_number)`
- index on `(artifact_id, time_start_seconds)`

### Analysis configuration and job tables

#### `analysis_focuses`

Purpose:
- record what the system or user wanted to focus on during a run

Suggested fields:
- `id uuid primary key`
- `topic_run_id uuid null`
- `source_id uuid null`
- `focus_text text not null`
- `extraction_modes jsonb`
- `is_default boolean default false`
- `other_notable_topics jsonb`
- `created_by text` such as `user`, `system`
- `created_at timestamptz`

Notes:
- supports source-only analysis and topic-bound analysis
- `other_notable_topics` can store lightweight leads for future re-analysis

Indexes:
- index on `topic_run_id`
- index on `source_id`

#### `extraction_runs`

Purpose:
- provenance for a specific extraction pass over one or more chunks, artifacts, sources,
  or topic runs

Suggested fields:
- `id uuid primary key`
- `topic_run_id uuid null`
- `analysis_focus_id uuid null`
- `source_id uuid null`
- `source_version_id uuid null`
- `artifact_id uuid null`
- `run_scope text` such as `chunk`, `artifact`, `source_version`, `source`, `topic_run`
- `extraction_modes jsonb`
- `model_id text not null`
- `runtime text null`
- `prompt_name text null`
- `prompt_version text null`
- `extraction_schema_version text null`
- `parameters jsonb`
- `status text` such as `queued`, `running`, `completed`, `partial`, `failed`, `stale`
- `output_hash text null`
- `result_summary jsonb`
- `started_at timestamptz null`
- `completed_at timestamptz null`
- `created_at timestamptz`
- `updated_at timestamptz`

Notes:
- every extracted observation should point back to an `extraction_run`
- model, prompt, and schema versions are required so old outputs can be marked stale
  after prompt/model/schema improvements

Indexes:
- index on `(source_id, created_at desc)`
- index on `(artifact_id, created_at desc)`
- index on `status`

#### `jobs`

Purpose:
- background and queued work execution state

Suggested fields:
- `id uuid primary key`
- `topic_run_id uuid null`
- `job_type text` such as `ingest`, `chunk`, `extract`, `verify`, `report`, `reextract`
- `status text`
- `priority integer default 100`
- `attempt_count integer default 0`
- `max_attempts integer default 3`
- `scheduled_at timestamptz null`
- `started_at timestamptz null`
- `completed_at timestamptz null`
- `last_error text null`
- `payload jsonb`
- `result_summary jsonb`
- `created_at timestamptz`
- `updated_at timestamptz`

Indexes:
- index on `(status, priority, scheduled_at)`
- index on `topic_run_id`

#### `job_dependencies`

Purpose:
- dependency edges between jobs in multi-step workflows

Suggested fields:
- `id uuid primary key`
- `job_id uuid not null`
- `depends_on_job_id uuid not null`
- `dependency_type text` such as `blocks_until_success`, `blocks_until_finished`
- `created_at timestamptz`

Notes:
- ingestion pipelines are naturally ordered: fetch, parse, chunk, extract, verify, report
- dependency edges make partial retry and resume behavior explicit

Indexes:
- unique index on `(job_id, depends_on_job_id)`
- index on `depends_on_job_id`

### Knowledge tables

#### `extracted_observations`

Purpose:
- raw model outputs before resolution into canonical entities, events, claims, or metrics

Suggested fields:
- `id uuid primary key`
- `extraction_run_id uuid not null`
- `artifact_chunk_id uuid not null`
- `observation_type text` such as `claim`, `entity`, `event`, `metric`, `relationship`
- `raw_text text not null`
- `normalized_text text null`
- `raw_payload jsonb`
- `candidate_claim_id uuid null`
- `candidate_entity_id uuid null`
- `candidate_event_id uuid null`
- `candidate_metric_id uuid null`
- `confidence numeric(5,2) null`
- `importance_score numeric(5,2) null`
- `char_start integer null`
- `char_end integer null`
- `time_start_seconds numeric(12,3) null`
- `time_end_seconds numeric(12,3) null`
- `status text` such as `new`, `promoted`, `rejected`, `duplicate`, `needs_review`, `stale`
- `created_at timestamptz`
- `updated_at timestamptz`

Notes:
- this is the buffer between noisy model output and the curated knowledge base
- canonical tables should be populated by promotion/resolution logic, not by blindly
  trusting first-pass extraction

Indexes:
- index on `extraction_run_id`
- index on `artifact_chunk_id`
- index on `(observation_type, status)`
- full-text or trigram index on `normalized_text`

#### `entities`

Purpose:
- canonical named things

Suggested fields:
- `id uuid primary key`
- `entity_type text` such as `person`, `organization`, `product`, `location`, `document`, `concept`
- `name text not null`
- `normalized_name text not null`
- `description text null`
- `metadata jsonb`
- `created_at timestamptz`
- `updated_at timestamptz`

Indexes:
- index on `(entity_type, normalized_name)`

#### `entity_aliases`

Purpose:
- alternate names for entities

Suggested fields:
- `id uuid primary key`
- `entity_id uuid not null`
- `alias text not null`
- `normalized_alias text not null`
- `created_at timestamptz`

Indexes:
- unique index on `(entity_id, normalized_alias)`

#### `entity_mentions`

Purpose:
- where an entity appeared in a chunk

Suggested fields:
- `id uuid primary key`
- `entity_id uuid not null`
- `artifact_chunk_id uuid not null`
- `mention_text text not null`
- `confidence numeric(5,2) null`
- `metadata jsonb`
- `created_at timestamptz`

Indexes:
- index on `entity_id`
- index on `artifact_chunk_id`

#### `events`

Purpose:
- canonical events for timelines and comparisons

Suggested fields:
- `id uuid primary key`
- `title text not null`
- `normalized_title text not null`
- `description text null`
- `event_type text null`
- `start_at timestamptz null`
- `end_at timestamptz null`
- `date_precision text` such as `exact`, `day`, `month`, `year`, `approximate`
- `location_entity_id uuid null`
- `metadata jsonb`
- `created_at timestamptz`
- `updated_at timestamptz`

Indexes:
- index on `(start_at, end_at)`
- index on `normalized_title`

#### `event_mentions`

Purpose:
- where an event was mentioned in a chunk

Suggested fields:
- `id uuid primary key`
- `event_id uuid not null`
- `artifact_chunk_id uuid not null`
- `mention_text text null`
- `confidence numeric(5,2) null`
- `created_at timestamptz`

Indexes:
- index on `event_id`
- index on `artifact_chunk_id`

#### `claims`

Purpose:
- atomic extracted statements

Suggested fields:
- `id uuid primary key`
- `claim_type text` such as `fact`, `event_fact`, `economic`, `historical`, `product_spec`, `quote`
- `subject_entity_id uuid null`
- `object_entity_id uuid null`
- `event_id uuid null`
- `canonical_text text not null`
- `normalized_text text not null`
- `status text` such as `unverified`, `supported`, `contradicted`, `mixed`, `deprecated`
- `preferred_source_id uuid null`
- `confidence numeric(5,2) null`
- `importance_score numeric(5,2) null`
- `verification_attempted_at timestamptz null`
- `is_user_reviewed boolean default false`
- `reviewed_at timestamptz null`
- `reviewed_by text null`
- `created_at timestamptz`
- `updated_at timestamptz`

Notes:
- topic linkage lives in `claim_topics`, not on the claim row — a claim can belong to
  zero topics at ingest time and several topics later (decision 19)

Indexes:
- index on `event_id`
- index on `status`
- full-text or trigram index on `normalized_text`

#### `claim_topics`

Purpose:
- optional many-to-many link between claims and topics, mirroring `topic_source_links`

Suggested fields:
- `id uuid primary key`
- `claim_id uuid not null`
- `topic_id uuid not null`
- `link_reason text` such as `extracted_in_run`, `manual_attach`, `derived`
- `created_at timestamptz`

Indexes:
- unique index on `(claim_id, topic_id)`
- index on `topic_id`

#### `claim_evidence`

Purpose:
- provenance records linking claims to source chunks

Suggested fields:
- `id uuid primary key`
- `claim_id uuid not null`
- `extraction_run_id uuid null`
- `extracted_observation_id uuid null`
- `artifact_chunk_id uuid not null`
- `source_id uuid not null`
- `source_version_id uuid not null`
- `evidence_type text` such as `support`, `contradict`, `mention`, `derived`
- `excerpt_text text null`
- `excerpt_hash text null`
- `char_start integer null`
- `char_end integer null`
- `time_start_seconds numeric(12,3) null`
- `time_end_seconds numeric(12,3) null`
- `confidence numeric(5,2) null`
- `created_at timestamptz`

Notes:
- offsets should refer to the normalized artifact/chunk text where possible
- transcript evidence should preserve timestamps where available
- `excerpt_hash` helps detect when a chunk was regenerated or text normalization changed

Indexes:
- index on `claim_id`
- index on `artifact_chunk_id`
- index on `(claim_id, evidence_type)`

#### `claim_links`

Purpose:
- relationships between claims

Suggested fields:
- `id uuid primary key`
- `from_claim_id uuid not null`
- `to_claim_id uuid not null`
- `link_type text` such as `duplicate_of`, `contradicts`, `derived_from`, `supersedes`, `related`
- `confidence numeric(5,2) null`
- `created_at timestamptz`

Indexes:
- unique index on `(from_claim_id, to_claim_id, link_type)`

#### `resolution_candidates`

Purpose:
- possible duplicate, merge, contradiction, or supersession candidates for entities,
  events, claims, and metrics

Suggested fields:
- `id uuid primary key`
- `candidate_type text` such as `entity_duplicate`, `event_duplicate`, `claim_duplicate`, `claim_contradiction`, `claim_supersession`
- `left_entity_id uuid null`
- `right_entity_id uuid null`
- `left_event_id uuid null`
- `right_event_id uuid null`
- `left_claim_id uuid null`
- `right_claim_id uuid null`
- `left_observation_id uuid null`
- `right_observation_id uuid null`
- `score numeric(5,2) null`
- `reason text null`
- `method text` such as `normalized_text`, `trigram`, `embedding`, `model`, `manual`
- `status text` such as `open`, `accepted`, `rejected`, `deferred`, `auto_applied`
- `reviewed_by text null`
- `reviewed_at timestamptz null`
- `created_at timestamptz`
- `updated_at timestamptz`

Notes:
- this table is the v1 merge/review queue
- uncertain matches should become candidates instead of being silently merged

Indexes:
- index on `(candidate_type, status)`
- index on `left_claim_id`
- index on `right_claim_id`

#### `metrics`

Purpose:
- structured numerical values for comparison/reporting

Suggested fields:
- `id uuid primary key`
- `claim_id uuid null`
- `event_id uuid null`
- `entity_id uuid null`
- `metric_name text not null`
- `metric_group text null`
- `value_numeric numeric null`
- `value_text text null`
- `unit text null`
- `currency_code text null`
- `effective_at timestamptz null`
- `metadata jsonb`
- `created_at timestamptz`

Notes:
- topic scoping is derived through `claim_id -> claim_topics` when needed

Indexes:
- index on `(metric_name, effective_at)`
- index on `entity_id`

### Reporting and review tables

#### `reports`

Purpose:
- generated outputs for topics or source collections

Suggested fields:
- `id uuid primary key`
- `topic_id uuid null`
- `topic_run_id uuid null`
- `report_type text` such as `timeline`, `summary`, `comparison`, `kb_answer`
- `title text`
- `content_markdown text not null`
- `status text`
- `generated_from_scope jsonb`
- `created_at timestamptz`
- `updated_at timestamptz`

Indexes:
- index on `(topic_id, created_at desc)`
- index on `report_type`

#### `report_sources`

Purpose:
- sources used by a report

Suggested fields:
- `id uuid primary key`
- `report_id uuid not null`
- `source_id uuid not null`
- `source_version_id uuid null`
- `created_at timestamptz`

Indexes:
- unique index on `(report_id, source_id, source_version_id)`

#### `claim_reviews`

Purpose:
- optional manual review trail for important claims

Suggested fields:
- `id uuid primary key`
- `claim_id uuid not null`
- `review_decision text` such as `accepted`, `rejected`, `needs_followup`
- `review_notes text null`
- `reviewed_by text`
- `created_at timestamptz`

Indexes:
- index on `claim_id`

### Session/UI tables

#### `sessions`

Purpose:
- UI or agent conversation sessions kept separate from KB data

Suggested fields:
- `id uuid primary key`
- `title text`
- `created_at timestamptz`
- `updated_at timestamptz`

#### `messages`

Purpose:
- session conversation and tool-call history

Suggested fields:
- `id bigserial primary key`
- `session_id uuid not null`
- `role text not null`
- `content text null`
- `tool_calls jsonb null`
- `tool_call_id text null`
- `tool_name text null`
- `created_at timestamptz`

Indexes:
- index on `(session_id, id)`

### High-priority foreign-key relationships

Important relationships that should exist from the start:

- `topic_aliases.topic_id -> topics.id`
- `topic_runs.topic_id -> topics.id`
- `topic_source_links.topic_id -> topics.id`
- `topic_source_links.source_id -> sources.id`
- `sources.source_type_id -> source_types.id`
- `sources.trust_tier_id -> trust_tiers.id`
- `source_versions.source_id -> sources.id`
- `source_fetch_attempts.source_id -> sources.id`
- `source_fetch_attempts.source_version_id -> source_versions.id`
- `artifacts.source_version_id -> source_versions.id`
- `artifact_chunks.artifact_id -> artifacts.id`
- `analysis_focuses.topic_run_id -> topic_runs.id`
- `analysis_focuses.source_id -> sources.id`
- `extraction_runs.topic_run_id -> topic_runs.id`
- `extraction_runs.analysis_focus_id -> analysis_focuses.id`
- `extraction_runs.source_id -> sources.id`
- `extraction_runs.source_version_id -> source_versions.id`
- `extraction_runs.artifact_id -> artifacts.id`
- `job_dependencies.job_id -> jobs.id`
- `job_dependencies.depends_on_job_id -> jobs.id`
- `extracted_observations.extraction_run_id -> extraction_runs.id`
- `extracted_observations.artifact_chunk_id -> artifact_chunks.id`
- `extracted_observations.candidate_claim_id -> claims.id`
- `extracted_observations.candidate_entity_id -> entities.id`
- `extracted_observations.candidate_event_id -> events.id`
- `extracted_observations.candidate_metric_id -> metrics.id`
- `entity_aliases.entity_id -> entities.id`
- `entity_mentions.entity_id -> entities.id`
- `entity_mentions.artifact_chunk_id -> artifact_chunks.id`
- `event_mentions.event_id -> events.id`
- `event_mentions.artifact_chunk_id -> artifact_chunks.id`
- `claims.subject_entity_id -> entities.id`
- `claims.object_entity_id -> entities.id`
- `claims.event_id -> events.id`
- `claims.preferred_source_id -> sources.id`
- `claim_topics.claim_id -> claims.id`
- `claim_topics.topic_id -> topics.id`
- `claim_evidence.claim_id -> claims.id`
- `claim_evidence.extraction_run_id -> extraction_runs.id`
- `claim_evidence.extracted_observation_id -> extracted_observations.id`
- `claim_evidence.artifact_chunk_id -> artifact_chunks.id`
- `claim_evidence.source_id -> sources.id`
- `claim_evidence.source_version_id -> source_versions.id`
- `claim_links.from_claim_id -> claims.id`
- `claim_links.to_claim_id -> claims.id`
- `resolution_candidates.left_entity_id -> entities.id`
- `resolution_candidates.right_entity_id -> entities.id`
- `resolution_candidates.left_event_id -> events.id`
- `resolution_candidates.right_event_id -> events.id`
- `resolution_candidates.left_claim_id -> claims.id`
- `resolution_candidates.right_claim_id -> claims.id`
- `resolution_candidates.left_observation_id -> extracted_observations.id`
- `resolution_candidates.right_observation_id -> extracted_observations.id`
- `metrics.claim_id -> claims.id`
- `metrics.event_id -> events.id`
- `metrics.entity_id -> entities.id`
- `reports.topic_id -> topics.id`
- `reports.topic_run_id -> topic_runs.id`
- `report_sources.report_id -> reports.id`
- `report_sources.source_id -> sources.id`
- `report_sources.source_version_id -> source_versions.id`
- `claim_reviews.claim_id -> claims.id`
- `messages.session_id -> sessions.id`

### Likely v1 indexes beyond primary keys

- full-text search index on `artifact_chunks.chunk_text`
- lookup indexes on `sources.canonical_key`, `entities.normalized_name`, `events.start_at`
- status indexes on `claims.status`, `jobs.status`, `topic_runs.status`,
  `extraction_runs.status`, `extracted_observations.status`
- recent-first indexes on `source_versions(source_id, captured_at desc)` and `reports(topic_id, created_at desc)`

### What can be deferred from v1 if needed

- advanced entity resolution tables beyond the basic `resolution_candidates` queue
- graph-style relationship expansion
- vector embeddings tables
- automated taxonomy-management tables
- multi-user ownership/permissions tables

### Recommended v1 implementation subset

If the first build needs to stay disciplined, prioritize these tables first:

- `topics`
- `topic_runs`
- `topic_source_links`
- `source_types`
- `trust_tiers`
- `sources`
- `source_versions`
- `source_fetch_attempts`
- `artifacts`
- `artifact_chunks`
- `analysis_focuses`
- `extraction_runs`
- `extracted_observations`
- `entities`
- `events`
- `claims`
- `claim_topics`
- `claim_evidence`
- `resolution_candidates`
- `metrics`
- `reports`
- `jobs`
- `job_dependencies`
- `sessions`
- `messages`

Then add second-wave tables after the pipeline works:

- `topic_aliases`
- `entity_aliases`
- `entity_mentions`
- `event_mentions`
- `claim_links`
- `report_sources`
- `claim_reviews`

Deferring `entity_mentions` / `event_mentions` does not block resolution: v1
resolution reads mention locations from `extracted_observations`.

## Local Model & Runtime Plan

Model roles, runtime (`llama.cpp`), task routing, and the recommended v1 model
stack have moved to a dedicated doc so model/runtime churn stays out of the
architecture spec:

- [MODELS.md](MODELS.md) — local model plan and runtime guidance
