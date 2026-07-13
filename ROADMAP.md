# Roadmap

Last updated: 2026-07-13. Extracted from HANDOFF.md (which stays as the
qwen3-14b-vs-30b eval record) when the planning content outgrew it. Where
this document and `PLAN_KB_ARCHITECTURE.md` disagree, this document wins —
the plan doc's "fuzzy/LLM matches never auto-merge", "web/CLI have full
parity", and "manual by default" statements are all stale (see
Documentation Debt below).

## Product principles

Stated directly by the user; every item below is in service of these:

1. **Hands-off by default.** The user keeps final control to correct
   classifications/claims, but every manual step upstream of that final
   judgment call is a tax on whether the system actually gets used. Target
   workflow: create a topic, add 3-4 sources, receive useful data with no
   further required actions.
2. **Every automatic action is visible and correctable.** Anywhere the
   system decides something on its own there must be a trail explaining why
   (decision log), and either a safe undo or an explicit
   `reversible=false` marker until one exists.
3. **Human keeps genuine judgment calls.** Claim-contradiction adjudication
   ("which of these conflicting claims is true") and topic naming/framing
   stay human decisions — assisted, never auto-resolved.

## Definition: "idle time"

Several items below run as background/speculative work "during idle time."
Definition (user's, 2026-07-13): **the GPUs are not doing LLM work.** This
box is the user's main LLM server — today it effectively only serves this
project, but it will pick up other workloads over time, so "our job queue is
empty" is *not* a sufficient idleness signal. Background work must check
actual GPU/LLM activity before starting each job:

- llama-server `/slots` (are any slots busy) and/or Ollama's running-model
  state for the servers this project knows about;
- an `nvidia-smi` utilization check as the backstop for LLM work this
  project *didn't* start (the "other things" this server will do later).

Concretely: the background scheduler ranks speculative work (counter-claim
search, playlist ingestion, retroactive ad sweep, topic discovery, report
refresh) strictly below user-requested work in the job queue, and
additionally refuses to *start* a speculative job while the GPUs are busy
with anything else. No preemption of a job once started — check-before-start
is enough at this scale.

## P0: the golden-path gap

The extraction/verification core is substantial, but the main desired user
workflow is not complete. Today the ordinary web workflow is still:

1. Create a topic (only inserts the topic row).
2. Navigate separately to Sources and ingest each source.
3. Open every source and click Chunk, Extract, then Verify.
4. Attach the source to the topic separately.
5. Manually generate a report.

The backend has `POST /topics/{topic_id}/sources`, but the frontend has no
corresponding user control. A new empty topic also cannot bootstrap the
entity-overlap suggestion engine because it has no attached claims/entities
to match against. There is an additional ordering trap: attaching a source
before extraction attaches zero claims; later extraction does not sweep the
new claims into that already-attached topic. The conversation-paste flow
works around this by attaching the source again after promotion, but normal
URL/YouTube/file ingestion does not.

**Required P0 feature: topic intake + automatic processing.** The topic
creation/detail UI should accept multiple URLs, YouTube links, and file
uploads directly. Adding them should enqueue one shared, idempotent pipeline:

```text
ingest -> chunk -> extract -> ad check -> attach source+claims
       -> verify eligible claims -> generate/refresh topic overview
```

All stages default on. Advanced per-topic/source toggles may disable or defer
individual stages, but a normal user should not need to understand pipeline
internals. Independent source ingestion should use the same pipeline, just
without the topic-attachment step. Empty-topic bootstrapping: until a topic
has attached claims, fall back to matching on the topic's name/description
(e.g. embedding search) so suggestions can start before the first extraction
lands.

### Durable jobs, recovery, and progress

Only pasted conversations currently auto-chain the pipeline, and their
background state is held in process-local Python sets. A web-server restart
loses job/progress state; an exception simply makes `processing` become
false, which the UI cannot distinguish from success. Normal extraction/
verification requests are also long synchronous HTTP calls.

Add a durable `processing_jobs` table/worker with at least: subject type/id,
pipeline stage, queued/running/partial/failed/completed/cancelled status,
progress counts, attempt count, error message, idempotency key, timestamps,
and cancellation request. Jobs should resume or become explicitly retryable
after restart. The UI must show the actual stage and offer Retry, Cancel, and
where safe Undo; it should never require reading logs to learn why a source
has no output.

### The job worker is the single choke point (added 2026-07-13 review)

Three existing gaps are all symptoms of not having one worker own the GPU
and the search quota. Build these in from day one rather than retrofitting:

- **Global GPU/verification mutual exclusion.** Today only the KB-wide
  sweep is single-flight (`run_verification_sweep`'s in-progress check).
  `verify-source`, `/topics/{id}/verify`, `/claims/{id}/verify`, and the
  nightly cron can all run simultaneously — from web and CLI as separate
  processes — stacking LLM calls on one GPU and multiplying search volume.
  Requirement: **all** verification (and extraction) paths route through
  the job queue, which holds one GPU lock.
- **Cross-process search throttling.** `_throttle_searxng` is module-level
  state (`deep_research/tools/search.py`) — it paces one process only. The
  nightly cron (CLI process) and the web server bypass each other's pacing,
  exactly the burst pattern that caused Round 2's quota burn. Once all
  search-driving work runs in one worker process this is correct for free;
  until then it's a known hole.
- **Per-run search budget + sweep prioritization.** `run_verification_sweep`
  processes `list_claims(limit=10000)` in DB-default order with no cap on
  total external searches per run. Add: (a) importance-descending (and
  topic-attached-first) ordering so the most valuable claims verify before
  any quota/time runs out, and (b) a per-run external-search budget — the
  `/search-usage` SQLite log already counts calls, so a "stop external
  fallback after N calls tonight" circuit breaker is cheap and directly
  prevents a Round-2-style silent quota exhaustion in production.
- **Cache `detect_model` per run.** Every `verify_claim` builds its own
  `LLMClient` and hits the server's model-detect endpoint; a 100-claim
  sweep makes 100 of those calls. Detect once per job-worker run.

### Make the default result actually useful

The topic page currently exposes raw claims/timeline and a pull-only report.
Reports are generated only after the user clicks a button, have no freshness
indicator, and `_format_claim_line` labels only `contradicted`/`mixed`
claims; an `unverified` assertion is passed to report generation without an
explicit unverified marker. This can make a source assertion read like an
established fact.

Automatically generate a preliminary topic overview after extraction, then
refresh it when verification materially changes the topic. The default view
should contain:

- a concise synthesis;
- points supported by multiple user-provided sources;
- important but unverified assertions, clearly separated;
- contradictions/competing accounts and the evidence on each side;
- coverage gaps and verification still pending;
- attached user sources with clickable citations; and
- a visible "generated from" timestamp/claim-set freshness state.

This raises enhancement 8 (report auto-refresh) from an optional future item
to part of the P0/P1 golden path.

### Source-state and source-purpose hygiene

A read-only audit of the current production KB found 119 source rows; 70 were
failed-ingest shells with no latest version and no claims, yet the Sources
page lists them beside useful sources. The UI's current "no surviving claims"
label conflates failed ingestion, not-yet-processed content, intentionally
partial verification evidence, and a successfully-processed source that
genuinely yielded nothing. One of the three current topics was also empty,
which is consistent with the topic bootstrapping problem above.

Add an explicit/computed source lifecycle (`failed`, `queued`, `ingested`,
`chunked`, `extracted`, `ready`, `partial`) and a `source_purpose` such as
`user_added`, `playlist_discovered`, or `verification_evidence`. The default
Sources view should be "My sources"; verification evidence and failed shells
belong in separate filtered views. Failed shells should be automatically
cleaned up or archived, with the error and Retry action preserved.

### Partially wired features

- Ad/sponsor classification is currently called by CLI `extract-source`, but
  not by the web extraction route or the conversation background pipeline.
  Conversation ingestion also skips automatic trust classification.
  Centralize every post-promotion action in the shared pipeline so CLI/web/
  background entry points cannot silently diverge again.
- Most frontend API helpers do not check non-2xx responses (~8 of 45 fetch
  calls in `useApi.js` check `resp.ok`). Failed operations can look like
  malformed success rather than showing an actionable error.

## Cross-cutting: decision/action journal

User's directive: anywhere the system makes a decision on its own, there
should be a log/trail so the user can figure out why something happened,
after the fact. Every currently-automated decision path throws the "why"
away today:

- `trust.py`'s and `ad_check.py`'s LLM prompts already ask for a
  `"reasoning"` string and the model returns one — but only the boolean/tier
  outcome gets written. The confidence score and reasoning sentence are
  computed, then discarded.
- Entity/claim auto-merges (`resolution.py`) are worse: a confident LLM
  verdict calls `merge_entities`/`merge_claims` directly, **skipping
  `resolution_candidates` entirely** — for auto-resolved pairs (the
  majority, by design) there is no row, no confidence, no reasoning beyond
  the tombstoned `merged_into_*_id` pointer.
- `recompute_preferred_source` is a deterministic documented-in-code rule
  (re-derivable after the fact), but still writes `preferred_source_id`
  with no record of which candidates were considered.

**Plan**: a single, generic, append-only `decision_log` table, built once
and used by every automated decision path — including every enhancement
below, from day one:

```
decision_log(
  id, decision_type,       -- 'entity_merge' | 'claim_merge' | 'trust_tier'
                            -- | 'ad_check_exclude' | 'preferred_source'
                            -- | 'topic_auto_attach' | 'playlist_video_ingested'
                            -- | 'topic_auto_discovered' | ... (open set, not an enum,
                            -- so a new automation never needs a migration to log)
  subject_type, subject_id, -- polymorphic: 'claim' | 'entity' | 'source' | 'topic'
  related_ids JSONB,        -- e.g. the losing entity/claim id in a merge
  decision TEXT,            -- one-line outcome, e.g. "merged into <id>", "tier: official"
  confidence REAL NULL,     -- NULL for deterministic/non-LLM decisions
  reasoning TEXT NULL,      -- the LLM's own explanation, verbatim, when there is one
  model TEXT NULL,          -- which model made the call
  previous_state JSONB,     -- enough state to explain/reverse the action
  resulting_state JSONB,    -- state immediately after the action
  reversible BOOLEAN,       -- whether the current implementation can undo it safely
  undo_of_decision_id TEXT NULL, -- reversal rows point at the original action;
                                 -- keeps the journal append-only
  decided_at TIMESTAMPTZ
)
```

- One shared writer (`deep_research/kb/decision_log.py::record_decision`),
  retrofitted into the existing automated paths (entity merge, claim merge,
  trust tier, ad check, preferred-source), then a hard requirement for every
  new automation.
- A read surface: `list-decisions`/`show-decision` (CLI + web), filterable
  by type/subject/date, analogous to `list-resolution-candidates`.
- This is an **automation action journal**, not just an explanation feed:
  each automated path must either expose a safe undo or explicitly record
  `reversible=false` until one exists. Confident LLM entity/claim merges
  currently execute immediately with no unmerge path; until merge reversal
  exists, consider returning them to the review queue rather than allowing
  an irreversible automatic action. Other required reset/detach controls:
  remove a source or claim from a topic, reset trust tier to automatic,
  reset preferred-source to automatic, restore an ad-excluded claim, and
  archive/delete a failed source or empty topic.
- Fold in the eval finding: log a JSON **parse-success flag** (and ideally
  the raw response) for every LLM classification call, so "model judged no
  relationship" is distinguishable from "response didn't parse" —
  `_parse_json_object`'s silent unrelated-default is a known confound from
  the cross-verify experiment.

## Schema/data enhancements from the 2026-07-13 review (approved)

- **First-class support evidence.** A contradiction gets a durable
  `resolution_candidates` row, but "supported by claim X" lives only as IDs
  inside `verification_notes` JSONB — dangling silently if the supporting
  claim is later merged or deleted. Make supports first-class rows in the
  same schema pass that adds `counter_evidence` (enhancement 2 below);
  reports and the topic overview ("supported by multiple sources") should
  query rows, not parse notes JSON.
- **Trust-tier-weighted verification.** Trust tiers are computed at ingest
  but never consulted by verification — two `user_generated` blogs settle a
  claim "supported" exactly like two `official` sources (only the hard
  social-media exclusion exists). Weight `_Budget`'s stop conditions by
  tier (e.g. one `official` corroboration suffices; `user_generated` counts
  half), and surface tier in the overview's support summary.
- **Query-path cleanups (N+1s).** `cmd_verify_source` fetches evidence for
  every claim in the KB to find one source's claims (the web route already
  uses `list_claims_for_source` — consolidate on that); `cmd_show_claim`/
  `cmd_verify_claim` load 5000 claims to prefix-match one ID (add a DB-side
  prefix lookup); report generation calls `_claim_source_title` once per
  claim (bulk-fetch). Do these as part of the shared-pipeline refactor.

## Enhancements (designed, statuses as of 2026-07-13)

**1. Auto-attach high-confidence topic matches.** Today, `topics.py`'s
entity-overlap matching always lands in a human-review queue, even when the
LLM relevance check is very confident — its confidence score is computed
but currently discarded. Plan: persist it (new `relevance_confidence` column
on `claim_topics`/`topic_source_links`), and when relevance confidence
clears a new, deliberately-high `TOPIC_AUTO_ATTACH_THRESHOLD` (proposed 0.9,
above the existing 0.85 suppress-threshold), attach directly instead of
queuing a suggestion. Auto-attached items stay removable — confirmed
`review_topic_claim_link`/`review_topic_source_link` already do a plain
status update to `'rejected'` regardless of prior status, so detaching an
already-attached item needs **zero backend changes**, only a missing "Remove
from topic" button in `TopicDetailView.vue` (today only *suggested* items
have accept/reject; attached items have no action at all). **Gated on
decision logging + detach/undo controls.**

**2. Strongest counter-claim search, for balance.** `_Budget.should_stop()`
in `verification.py` halts on the first contradiction or 2nd support, so a
"supported" claim never gets a deliberate look for the best opposing
argument. Plan: a new, separate `find_strongest_counter_claim` function
(not a change to `_Budget`) that only runs for already-`supported` claims,
gathers *every* contradicting candidate within its own small budget instead
of stopping at the first, and keeps only the single highest-confidence one.
User's explicit decisions: (a) **this must not change the original claim's
status** — stored as a new, distinct `resolution_candidates.candidate_type
= "counter_evidence"` (not `claim_contradiction`, which flips status on
accept), displayed as a separate "Counter-view (for balance)" annotation;
(b) **scope/priority**: not automatic on every verify — runs as idle-time
fill-in work (see the idle definition above) only after the primary
eligible-claims queue is exhausted, plus a standalone on-demand CLI/web
command for one claim or one topic (e.g. right before generating a report).
Needs a new `claims.counter_claim_checked_at` column (cooldown, proposed 7
days — longer than verification's own 72h retry cooldown, since this is
explicitly lower-priority). **Build the first-class support-evidence rows
(above) in the same schema pass.**

**3. Split production pipeline: 30B extracts, 14B verifies.** Motivated
directly by the Cross-Verify Experiment result (14B verifying 30B's own
claims resolves ~2x more of them than 30B verifying itself — see
HANDOFF.md) plus 30B's ~2x faster/higher-recall extraction (Round 1). Today
a single `config.kb.extraction_llm_base_url` serves both `extraction.py`
and `verification.py`. Plan:

- **Config split**: add `config.kb.verification_llm_base_url` (new field,
  defaults to `extraction_llm_base_url`'s value if unset, so today's
  single-model setups keep working unchanged). `verification.py` reads the
  new field instead of reusing `extraction_llm_base_url`.
- **Decouple batch extraction from ingestion for the nightly path.**
  Ingestion is already a separate step from `extract-source` (only the
  paste-a-conversation feature auto-chains), so a new `extract-pending`
  command (mirrors `verify-unverified`'s shape) can batch-process every
  source that has a chunked version but no completed extraction run,
  sequentially (single llama-server, no concurrent GPU contention),
  continuing past any single source's failure.
- **Shared server-lifecycle helper**: pull the generic
  subprocess-launch/health-check logic (`build_launch_command`,
  `is_healthy`, `wait_ready`) out of `deep_research/evals/server.py` into a
  registry-agnostic module (e.g. `deep_research/tools/llama_server.py`), so
  both the eval CLI and the nightly-swap script share one implementation.
- **New nightly cron wrapper** (replacing today's single-line
  `verify-unverified` cron entry) sequencing, in order: start 30B -> run
  `extract-pending` -> stop 30B (wait for full VRAM release, not just
  process-exit) -> start 14B -> run `verify-unverified` (this naturally
  verifies tonight's freshly-extracted claims) -> **leave 14B loaded**
  (user's explicit choice: daytime interactive verification defaults to the
  better verifier; daytime interactive *extraction* runs on whatever's
  loaded, unoptimized, per the user's own "manual starts may not be
  optimized" carve-out). Exactly one swap per night (30B -> 14B).
- **Out of scope**: auto-detecting/auto-swapping the loaded model for
  interactive (non-cron) requests — explicitly deferred.

**4. YouTube playlist tracking + idle-time ingestion.** Give the KB a
playlist URL once; from then on it checks periodically for new videos and
works through them using otherwise-idle capacity. **Design principle,
stated directly by the user: the playlist is not itself a piece of content —
it's purely a discovery mechanism.** Each video that comes out of it becomes
a completely normal, independent `youtube_video` source, identical in every
respect to one the user pasted in by hand. The playlist has no claims, no
transcript, no trust tier of its own. This means the existing
seeded-but-unused `source_types` row `youtube_playlist` (`db.py:500`) is
**not** the right fit and should stay unused rather than pressed into this.

- **Enumeration method — DECIDED (2026-07-13): `yt-dlp --flat-playlist`.**
  No API key, no quota; a scraper (same fragility class as SearXNG's
  scraping path) but the right cost tradeoff for a single-user local
  system, matching the general free-scraping-first/paid-API-fallback
  preference. YouTube Data API v3 remains the documented fallback if yt-dlp
  enumeration proves unreliable.
- **New schema**: a lightweight, separate pair of tables, not part of
  `sources` at all — `tracked_playlists(id, platform, playlist_id, url,
  title, default_trust_tier_code, active, created_at, last_checked_at)`
  plus `playlist_videos(tracked_playlist_id, video_id, discovered_at,
  source_id NULL, ingested_at NULL)`. `source_id` stays NULL until the
  video is actually ingested, so "known about but not yet processed" and
  "ingested" are distinct, queryable states.
- **Trust tier for unattended ingestion**: `default_trust_tier_code` is set
  once when the user starts tracking the playlist and applied to every
  video pulled from it (nobody is in the loop overnight).
- **Tracking already-grabbed vs new**: on each poll, diff returned video
  IDs against existing `playlist_videos` rows, insert new rows for anything
  unseen. Deliberately separate from `sources.canonical_key` dedup — a
  video could have been manually ingested standalone before appearing in a
  playlist scan, so the diff must check both "does a `playlist_videos` row
  exist" and "does a `sources` row already exist for this `canonical_key`"
  (link rather than re-fetch) before deciding a video is genuinely new.
- **Idle-time processing**: ranks below any user-requested pending
  extraction; slots into the nightly pipeline after `extract-pending`
  finishes, up to a configurable per-playlist-per-night cap (proposed 3-5
  videos). The cap prevents a large playlist monopolizing a night and paces
  transcript fetching against YouTube-side rate limiting (same risk class
  the SearXNG throttle addresses).
- **Removed/private videos**: no special handling — leave the
  `playlist_videos` row and any ingested source as historical record.
- **UI**: a "Track playlist" option alongside the existing ingest forms
  (`SourcesView.vue`), plus a per-playlist video list broken down by
  discovered/ingested/extracted/pending state.
- **Generalizes past YouTube**: the poll/diff/ingest-what's-new mechanism
  applies to RSS/news feeds or a publisher page later — don't name/shape
  the schema YouTube-only.

**5. Retroactive ad-check sweep (APPROVED, next up).** `ad_check.py` was
deliberately scoped to new extractions only. The classifier's safety
posture (only acts above a high confidence threshold, never guesses, skips
anything with an existing override) makes it safe to also run as a standing
idle-time sweep over existing older claims, cleaning up the ad/sponsor
noise documented in eval Rounds 1/2/4 without manual claim-by-claim
exclusion.

**6. LLM-assisted contradiction triage, not auto-resolution (APPROVED,
next up).** claim_contradiction stays human-decided — but today the user
has to read both claims' full evidence cold. A same-shape-as-
counter-claim-search LLM pass reads both sides and writes a one-line
recommendation + reasoning onto the `resolution_candidates` row (e.g.
"evidence favors claim A: cites a primary SEC filing vs. claim B's
secondary blog summary") — busywork removed, judgment call kept.

**7. Automatic topic discovery (held, not scheduled).** A brand new topic
still only comes into being when the user notices a pattern themselves and
manually runs `create-topic` — the one real manual bottleneck upstream of
everything else in the topic pipeline. Plan: an idle-time sweep that
clusters claims/entities not currently attached to any topic (shared
entities + embedding proximity, building blocks `topics.py` already has),
and where a cluster is large/cohesive enough, proposes a *new* topic
candidate — entirely a suggestion in a review queue, never auto-created,
since naming and framing a topic is an editorial choice that stays with the
user.

**8. Auto-refresh topic reports (promoted into the golden path).**
`generate_topic_report` is pull-only today. During idle capacity,
regenerate and cache the latest report for any topic whose claim set
changed materially since its last report (new source attached, a claim's
verification status flipped) — a UI hit becomes "show the cached one, mark
its freshness."

## Already automated (audited 2026-07-13 — no new work needed)

- Entity-duplicate resolution: LLM auto-merges above
  `ENTITY_LLM_CONFIDENCE_THRESHOLD`; only ambiguous cases reach review.
- Claim-duplicate resolution: identical pattern, own threshold.
- Trust-tier classification: auto-classified from URL/title at ingest when
  the user doesn't set one; leaves it unset under low confidence.
- Preferred-source ranking: `recompute_preferred_source` runs after every
  merge; the CLI/web `set-preferred-source` is a manual override path only.
- Ad/sponsor claim detection: confidence-gated auto-exclude (CLI extraction
  path; web/conversation wiring is P0 work above).
- Nightly verification sweep: fully unattended.
- Claim-contradiction candidates are, correctly, **never** auto-resolved —
  the right place to keep a human in the loop, not a gap.

## Documentation debt

- `PLAN_KB_ARCHITECTURE.md` is stale: fuzzy/LLM matches *do* auto-merge
  now, web/CLI parity has drifted, and its "manual by default" operations
  decision is superseded by the hands-off principle above.
- No root README / one-command setup path. `config.example.yaml` is behind
  the current provider, throttling, and verification settings; Docker
  Compose starts PostgreSQL/SearXNG but not the app or model services.
- `tests/test_extraction.py` has 2 tests that reach a live llama-server
  when one is running (auto-detection isn't mocked) — they pass only when
  no server is up. Mock the detection so the suite is
  environment-independent.

## Recommended implementation order

1. Durable `processing_jobs` + the decision/action journal (including
   previous/resulting state, undo metadata, and the LLM parse-success
   flag). Build the job worker as the single choke point: global GPU lock,
   cross-process search throttling, per-run search budget, prioritized
   sweep ordering, per-run `detect_model` caching.
2. One shared source pipeline used by CLI, web, conversation import, later
   playlist ingestion, and recovery/retry workers — fixing the ad-check/
   trust wiring gaps and the N+1 query paths as part of the consolidation.
3. Topic intake (topic + multiple sources in one flow), automatically
   attaching both sources and their post-extraction claims; empty-topic
   bootstrap.
4. Progress/error/retry/cancel UI, source lifecycle/purpose filtering,
   attached-source management on the topic page, frontend non-2xx error
   handling.
5. Automatic, freshness-aware topic overview/report (enhancement 8),
   explicitly separating unverified assertions; first-class support
   evidence + trust-tier-weighted verdict display.
6. Enhancements 5 and 6 (retroactive ad sweep, contradiction triage).
7. Topic auto-attach (1) — only once detach/undo and decision logging are
   present.
8. The 30B-extract/14B-verify split + `extract-pending` (3), then playlist/
   feed tracking (4) on top of the same job system — build 3 before 4.
9. Counter-claim search (2) and automatic topic discovery (7) behind the
   golden path.
