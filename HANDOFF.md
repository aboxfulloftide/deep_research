# Handoff: Qwen3 Source Eval

Date: 2026-07-11

> **Scope note (2026-07-13):** this document is the *eval record* — what was
> run, what was measured, what went wrong, and what the numbers mean. It had
> grown to also carry the whole product roadmap; that content now lives in
> [ROADMAP.md](ROADMAP.md). If you're picking up build work, start there.
> If you're re-running or extending the model comparison, start here.

Goal: compare claim extraction and later cross-source claim comparison quality between the original Qwen3-14B model and Qwen3-30B-A3B. Quality is more important than speed.

## Do Not Delete

Keep these eval databases and files. They are the baseline for importing the next source.

- Postgres DB: `deep_research_eval_qwen3_14b`
- Postgres DB: `deep_research_eval_qwen3_30b`
- Config: `evals/configs/qwen3-14b.yaml` (generated/tracked by the eval registry)
- Config: `evals/configs/qwen3-30b.yaml` (generated/tracked by the eval registry)
- Snapshot dir: `/home/matheau/.local/share/deep_research/kb_eval_qwen3_14b`
- Snapshot dir: `/home/matheau/.local/share/deep_research/kb_eval_qwen3_30b`
- Result dir: `/home/matheau/code/deep_research/evals/results/source-eval-20260711`
- Backup: `evals/backups/qwen3-30b-round4-baseline.dump` (Round 4 "30B verifies
  itself" baseline, restorable — see Cross-Verify section)

> **Config cleanup (2026-07-13):** the original hand-copied root configs
> (`tmp-qwen3-14b-eval.yaml`, `tmp-qwen3-30b-eval.yaml`,
> `tmp-14b-verifies-30b-data-eval.yaml`) were verified byte-equivalent to the
> registry-generated `evals/configs/*.yaml` files (which the registry DB also
> points at) and deleted. The cross-verify config had no distinct content —
> it was identical to the 30B config; the "cross" part was only which model
> the llama-server on port 18080 happened to be serving. To reproduce a
> cross-verify run: `deep-research-eval start-server qwen3-14b`, then run
> `deep-research-kb --config evals/configs/qwen3-30b.yaml verify-source ... --force`.

## Completed Source

URL:

```text
https://www.youtube.com/watch?v=VdAmhumeoLQ&list=PLSlu7-EATw54
```

Title:

```text
130 Years Of The Same Dumb Excuse
```

Qwen3-14B:

- Source ID: `f12787bf-9930-4b03-b85e-69b8c0e8cedd`
- Source version ID: `7e447345-2ff1-43af-9cb0-85e2e03a7293`
- Chunks: 18
- Promoted observations: 117
- Final source claims: 108
- Eligible claims at importance >= 0.8: 32
- Extraction elapsed: `9:13.65`

Qwen3-30B-A3B:

- Source ID: `0fca92b8-9a0b-467f-8a62-da781bf30d28`
- Source version ID: `8c8a0601-3847-463d-ad2f-a59c2217f21d`
- Chunks: 18
- Promoted observations: 149
- Final source claims: 142
- Eligible claims at importance >= 0.8: 36
- Extraction elapsed: `4:30.22`

## Current Evidence

30B was faster and extracted more claims, but it is not proven better.

Observed concern: both models extracted sponsor/ad claims, but 30B extracted more of them and assigned high confidence to several `The Book`/Kickstarter/product claims. This is noise for source-to-source claim comparison.

Claim overlap between the two model outputs:

- Exact normalized overlap: 25
- High fuzzy matches >= 0.86: 48
- Medium fuzzy matches 0.72-0.86: 27
- Low fuzzy matches < 0.72: 33

Saved exports:

- `qwen3-14b.claims.jsonl`
- `qwen3-30b.claims.jsonl`
- `claim-overlap-summary.json`
- `claim-overlap-sample.jsonl`

## What Was Not Completed

Full `verify-source` was intentionally not completed. One Qwen3-14B verification took about 7.5 minutes, so verifying all eligible claims for both models would take hours.

One Qwen3-14B claim completed as `supported`; 30B verification was not run.

Temporary llama-server process was stopped after the run.

## Round 2 (2026-07-12): Third Source + Pre-Existing Leftover Source

Both eval DBs turned out to already contain leftover content cloned from a shared
production KB before this eval began (many `web` sources at ~1 claim each, plus
one fully-processed `youtube_video` source — see "Leftover Source" below). This
was not previously documented. Also, this session ran with uncommitted
working-tree changes present (`resolution.py`, `topics.py`, `verification.py`,
new `trust.py`) — proceeded as instructed, but the pipeline code is not
byte-identical to what produced the original "130 Years" baseline.

### New Source (clean, apples-to-apples)

URL: `https://www.youtube.com/watch?v=a8zoVGyfr6Q`
Title: `Does Inequality Only Ever Get Worse?`

- Qwen3-14B source ID: `0d07a2fa-29c3-43ed-893b-51110d8a2b8c` (in `deep_research_eval_qwen3_14b`)
- Qwen3-30B-A3B source ID: `3348dc18-4da8-4fb2-9cf1-64c4fd852665` (in `deep_research_eval_qwen3_30b`)
- Chunks: 25 (both)
- Qwen3-14B: 112 observations, 112 promoted, 108 final source claims, 43 eligible >=0.8, extraction elapsed 9:48.6
- Qwen3-30B-A3B: 176 observations, 176 promoted, 163 final source claims, 38 eligible >=0.8, extraction elapsed 5:11.9
- Claim overlap (canonical_text, same method as round 1): exact 18, high fuzzy 26, medium fuzzy 14, low/no match 155
- Ad/sponsor pollution: none found in either model's output
- Exports: `qwen3-14b.inequality.claims.jsonl`, `qwen3-30b.inequality.claims.jsonl`, `claim-overlap-summary.inequality.json`, `claim-overlap-sample.inequality.jsonl`

### Leftover Source (contaminated comparison — do not treat as clean)

Both DBs already contained `9db079c2-5d88-42ac-8134-3f93baf15dd7`, "The AI
Bubble… We Need to Talk" (`https://www.youtube.com/watch?v=2J2Fb1bBufA`),
ingested/chunked/extracted with Qwen3-14B on 2026-07-08 — before this eval
existed. 86 claims, 41 eligible >=0.8, 16/86 already had a non-"unverified"
status from prior production use.

This session ran `extract-source 9db079c2 --force` against the 30B eval
config to add a Qwen3-30B-A3B pass. Because the source
already had 86 claims in that DB, the promotion step's own dedup logic merged
many of 30B's new observations into the pre-existing 14B claim rows instead of
creating independent claims:

- 156 total claims after the 30B pass
- 73 are genuinely 30B-only (new)
- 25 remain 14B-only (30B didn't re-surface them)
- **58 are merged rows** carrying evidence from both extraction runs under one
  canonical claim — these are not independent agreement, they're the
  pipeline's live dedup collapsing near-duplicates in real time. Don't count
  them toward "how often do the two models agree" the way round 1's ad-hoc
  fuzzy-match overlap did for "130 Years".
- Ad/sponsor pollution: 1 low-importance (0.3) "sponsors of the video are
  Printful" claim, shared by both models via the merge above — not a
  meaningful new pollution signal either way.

If a clean 14B-vs-30B comparison for this specific video is wanted later, it
needs a fresh source row per model (this app's `get_or_create_source` dedups
by canonical URL, so re-running `ingest-youtube` on the same URL returns the
existing source — a clean re-run would need to bypass that, e.g. a temporary
canonical_uri override or a throwaway third eval DB).

### Cross-Source (video-to-video) Duplicate/Contradiction Candidates

Queried each DB's own `resolution_candidates` table for `claim_duplicate` /
`claim_contradiction` rows linking claims between two of the three test videos
(130 Years / AI Bubble / Inequality), excluding matches against the other
pre-existing leftover web sources:

- Qwen3-14B: 0 cross-video candidates of any kind among the three videos
- Qwen3-30B-A3B: 2 `claim_duplicate` candidates between AI Bubble and
  Inequality, 0 contradictions; still 0 involving "130 Years"

Read cautiously: these three videos cover fairly distinct topics (1890s-1940s
labor history, AI investment bubble, wealth inequality), so sparse cross-video
signal may reflect genuinely low content overlap rather than a model
capability gap. Not yet strong evidence either way for "does 30B find better
cross-source matches."

### Verification Run (2026-07-11 22:01 - 2026-07-12 02:01) — RESULT IS INVALID, DO NOT USE FOR MODEL COMPARISON

Ran `verify-source` for all three sources, 14B first then 30B (cron disabled
for the duration, restored afterward). All 6 runs completed with exit 0, but
**the 30B half of this run is not a valid signal** — root-caused, not
speculation:

The local SearXNG instance (`deep-research-searxng` container, port 8888)
backs web-search fallback with Brave / DuckDuckGo / Google CSE / Startpage.
By the time the 30B pass started (~02:00), all four were rate-limited or
blocked (confirmed by querying SearXNG directly post-run:
`unresponsive_engines: brave "too many requests", duckduckgo "access denied",
google cse "too many requests", startpage "Suspended: CAPTCHA"` — still true
as of this writing, queries return 0 results for even generic terms). The
14B pass's own search volume over ~4 hours burned the quota.

Evidence this wrecked the 30B results specifically: across all 101 claims
verified under 30B, timing breakdowns show `web_search` called 98 times but
**`scrape_ingest`/`chunk`/`llm_extract`/`resolve_promote` never appear even
once** — every web-search fallback returned zero results, so no page was ever
actually fetched or read. Outcome: 0 "Supported" (vs 14B's 13), only 3
"Contradicted" out of 101 (vs 14B's 6 out of 74), avg 1.6-2.2s/claim (vs
14B's 244-500s/claim). This reads exactly like a model failing at tool use,
but it is not — it's 30B verifying against an empty search backend.

The 14B pass is *partially* affected too, and by degree, not a hard cutoff:
14B/Inequality (run second, after 130Years had already used the search
budget for ~2.5h) shows a lower avg sources-examined (3.05 vs 3.72) and much
shorter avg time (244s vs 500s) than 14B/130Years (run first) — quota
depletion was already underway before the 30B pass even started. Only the
first ~1-2 hours of this run (14B/130Years) should be treated as
representative of real verification depth.

Raw counts are preserved in the DB (`claims.status`) and in
`verify-round2.log` in the result directory for reference, but **do not
report "14B found more supported/contradicted claims than 30B" as a quality
finding** — Round 4 below is the valid redo.

## Round 4 (2026-07-12/13): Trustworthy Verification Comparison — DONE

Root-caused and fixed two real blockers before this run:

1. **`cmd_verify_source`'s `--force` flag was silently broken** —
   `cli/kb.py` built its eligibility list via
   `is_claim_eligible_for_verification(c, threshold)`, never passing
   `force=args.force` through (unlike `run_verification_sweep`, which did
   this correctly). So `--force` had zero effect on already-attempted
   claims — exactly why round 3's forced re-run kept reporting "No
   unverified claims." Fixed: now passes `force=args.force`.
2. **Search infrastructure was actually dead** — round 2's 30B pass ran
   after Brave/Google CSE/Startpage got rate-limited by the 14B pass's own
   search volume; 98/101 web_search calls returned zero results (confirmed
   via timing breakdowns never reaching `scrape_ingest`). Rebuilt the whole
   stack before round 4 — see "Search Infrastructure" below.

Full `--force` redo, all 3 sources, both models, ran cleanly end to end
(~13.5h: 14B ~4h46m, 30B ~8h39m). No errors, servers stopped cleanly.

### Result: importance>=0.8 claim status breakdown

| Model / Source | Total | Supported | Contradicted | Mixed | Unverified | Resolved % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14B / 130 Years | 35 | 6 | 5 | 0 | 24 | 31.4% |
| 14B / AI Bubble* | 39 | 7 | 7 | 0 | 25 | 35.9% |
| 14B / Inequality | 44 | 6 | 5 | 3 | 30 | 31.8% |
| 30B / 130 Years | 41 | 5 | 4 | 0 | 32 | 22.0% |
| 30B / AI Bubble* | 101 | 20 | 2 | 0 | 79 | 21.8% |
| 30B / Inequality | 45 | 2 | 0 | 0 | 43 | 4.4% |

\* AI Bubble counts are still affected by the claim-row merging contamination
documented in Round 2 — treat its numbers as directional only.

**Clean comparison (130 Years + Inequality only, excluding contaminated AI Bubble):**

- Qwen3-14B: 25 resolved / 79 total = **31.6%** resolution rate
- Qwen3-30B-A3B: 11 resolved / 86 total = **12.8%** resolution rate

Spot-checked a sample of supported/contradicted verdicts from both models —
verdicts look legitimate (real claims, plausible resolutions), not noise.

### Conclusion

With working search infrastructure, **Qwen3-14B resolves claims to a
supported/contradicted verdict roughly 2.5x more often than Qwen3-30B-A3B**,
despite 30B extracting more claims and running faster during extraction.
This directly answers the eval's original framing ("before deciding to use
30B, require a quality win, not just higher count or faster extraction") —
on verification specifically, 30B does not show a quality win; if anything
14B does meaningfully better at reaching a real verdict instead of
"budget exhausted, unverified."

## Search Infrastructure (2026-07-12/13) — what actually made Round 4 trustworthy

Round 4's verification numbers above are only valid because of a full rebuild
of `web_search()` done between Round 2 and Round 4 (committed 2026-07-13,
"Rebuild web search: layered API fallbacks, throttling, and usage logging").
Without this, verification still runs and produces plausible-looking output,
but silently against zero real search results (exactly Round 2's failure
mode). Anyone re-running this eval on a new machine or after a long gap needs
this context:

- **Root cause of Round 2's failure**: SearXNG's own scraped engines
  (duckduckgo, brave, google cse, startpage, mojeek, bing) all get
  rate-limited/CAPTCHA'd under sustained query volume — undocumented
  thresholds, no fixed quota, self-resetting cooldowns that just re-trip
  under continued load. A multi-hour verification run reliably exhausts them.
- **Fix — layered fallback in `deep_research/tools/search.py`**: SearXNG
  (duckduckgo + whichever of bing/mojeek/wikipedia are enabled, all free) +
  Wikipedia's own REST API (free, no key, but requires a policy-compliant
  `User-Agent` with real contact info — foundation.wikimedia.org/wiki/Policy:User-Agent_policy,
  confirmed empirically: identical request, only the header changed, 403 →
  200) + Brave's real API (paid, $5/1000, always called) → Tavily (paid,
  $8/1000, called when combined results are thin) → Serper (paid, cheapest
  at volume but its free tier is a **one-time** 2500-query trial, not
  monthly — kept as last-resort only). Brave/Google CSE/Startpage disabled
  in `searxng/settings.yml` where they kept causing problems; Bing and
  Mojeek added there (both `disabled: false`) — Bing has proven solid
  (15/15 in testing), Mojeek 403s quickly under load but costs nothing when
  it fails, so it's left enabled as a free bonus.
- **A min-interval throttle** (`SearXNGConfig.min_interval_seconds`, default
  1.5s) serializes SearXNG calls even under `verification_concurrency=2`, to
  reduce how often the free engines get rate-limited in the first place.
  (Known limitation: the throttle is process-local — see ROADMAP.md's job
  system requirements for the cross-process fix.)
- **API keys** (Brave/Tavily/Serper/Wikipedia-contact) are env vars only —
  `~/.bashrc` (above the interactive-shell guard, so non-interactive Bash
  tool calls pick them up too) and the crontab's env block (for the nightly
  `verify-unverified` job) — never in any tracked config file.
- **`/search-usage` page** (web UI) + a SQLite call log
  (`deep_research/tools/search_usage.py`) shows live per-provider status
  (responding/rate-limited), call counts, and mode (scrape vs. api) — check
  this before trusting a new verification run, the same way `unresponsive_engines`
  had to be manually curled all through Round 2's postmortem.

## Ad/Sponsor Claim Detection (2026-07-13)

`deep_research/kb/ad_check.py` — LLM classifier (same pattern as
`trust.py`'s source-trust classifier) that sets
`verification_override='exclude'` on claims confidently identified as
ad/sponsor/promotional content (the "The Book", Brilliant.org-style claims
documented in Round 2/4 above). Conservative posture: only acts above 0.7
confidence, since wrongly excluding a real claim is worse than leaving an
actual ad claim to harmlessly resolve as "unverified" like any other
unverifiable claim. Not applied retroactively to any existing claims from
Rounds 1-4 — only affects extractions from here forward.

> **Wiring correction (2026-07-13):** it runs automatically after promotion
> in the **CLI `extract-source` path only**. The web extraction route and
> the paste-a-conversation background pipeline do *not* call it yet — fixing
> that is part of the shared-pipeline work in ROADMAP.md.

## Cross-Verify Experiment (2026-07-13, DONE)

Testing whether 14B's higher verification resolution rate (Round 4) is
really about the *verifying* model's judgment, or an artifact of 30B's own
claim phrasing being harder to verify (30B's claims are more numerous and
more colloquial/ad-adjacent — see the "30B extra claims" analysis: a real
subset are genuine facts 14B's extraction missed, transcript-confirmed, but
another subset is ad noise).

Method: point Qwen3-14B (as the model) at the **30B eval database**
(`deep_research_eval_qwen3_30b`) and run `verify-source --force` against its
already-extracted claims — i.e. start the 14B server, then run against the
30B config (see the config-cleanup note above for the exact reproduction
recipe). If 14B gets a meaningfully better resolution rate on 30B's *own*
claims than 30B got verifying itself, that points at the verifying model. If
it's similarly low, that points at the claim set.

**Before running this**, the 30B eval database was backed up in full via
`pg_dump` to
`/home/matheau/code/deep_research/evals/backups/qwen3-30b-round4-baseline.dump`
(same convention the `deep-research-eval backup` command uses) —
Round 4's "30B verifies itself" baseline is preserved and restorable
regardless of what this overwrites. Restore with:
`docker cp evals/backups/qwen3-30b-round4-baseline.dump deep-research-postgres:/tmp/restore.dump && docker exec deep-research-postgres pg_restore -U deep_research -d deep_research_eval_qwen3_30b --clean --if-exists /tmp/restore.dump`

Status: complete. All 3 sources ran to completion (130 Years 07:11-08:47,
AI Bubble 08:47-11:40, Inequality 11:40-14:15, 2026-07-13).

### Results

Same 3 sources, `importance_score >= 0.8`, counted with `count(DISTINCT
c.id)` (not the naive `count(*)`, which double-counts claims with multiple
`claim_evidence` rows — see the Eval Registry section's `report.py` note
below). Claim counts differ slightly between the baseline and cross-verify
snapshots (136 vs 130 total) because `resolve_and_promote` re-ran during
the `--force` pass and merged/deduped a handful of claims; this doesn't
affect the resolution-rate comparison.

| Verifier -> claims from | Claims | Resolved (supported+contradicted+mixed) | Resolution rate | Contradicted |
| --- | ---: | ---: | ---: | ---: |
| 30B verifying its own claims (baseline, Round 4) | 136 | 27 | 19.9% | 6 (4.4%) |
| **14B verifying 30B's claims (cross-verify)** | 130 | 47 | **36.2%** | 13 (10.0%) |

Per-source breakdown (cross-verify vs baseline, supported/contradicted/mixed/unverified):

| Source | Baseline (30B on 30B) | Cross-verify (14B on 30B) |
| --- | --- | --- |
| 130 Years | 4/4/0/29 (37 claims) | 4/6/1/24 (35 claims) |
| Does Inequality Only Ever Get Worse? | 2/0/0/35 (37 claims) | 8/3/0/25 (36 claims) |
| The AI Bubble... We Need to Talk | 15/2/0/45 (62 claims) | 18/4/3/34 (59 claims) |

### Conclusion

This answers the question the experiment was designed to answer: **the
resolution-rate gap is about the verifying model, not the claim set.**
Pointing 14B at the *exact same claims* 30B extracted and already tried
(and mostly failed) to verify against itself nearly doubles the resolution
rate (19.9% -> 36.2%), and does so on both sides of the ledger —
contradicted count nearly doubles as a share of claims (4.4% -> 10.0%) right
alongside the rise in supported. A model that was simply "more lenient"
would inflate supported without touching contradicted; seeing both rise
together is evidence of genuinely better search-query formulation and/or
evidence-judgment, not a shifted decision threshold.

This doesn't yet distinguish the two remaining hypotheses for *why* 14B
verifies better (dense vs MoE active-parameter reasoning depth, vs. the
`_parse_json_object` malformed-JSON-defaults-to-unrelated confound noted
above) — that would need per-call JSON-parse-failure instrumentation, not
done here. But it does rule out "30B's claims are just unusually hard to
verify" as the explanation.

## Eval Registry Tooling (2026-07-13)

The manual process (hand-copy a config, hand-write a server-lifecycle
bash script, hand-write an asyncpg comparison query) is now real tooling:
`deep-research-eval` (new CLI, `cli/eval.py` +
`deep_research/evals/{registry,server,report}.py`). Both eval databases and
all 3 test sources from this whole session are already registered.

```bash
# from /home/matheau/code/deep_research
.venv/bin/python -m cli.eval list-models
.venv/bin/python -m cli.eval list-sources
.venv/bin/python -m cli.eval report --source 130years   # cross-model comparison table, live-queried
.venv/bin/python -m cli.eval start-server qwen3-14b      # or qwen3-30b
.venv/bin/python -m cli.eval stop-server qwen3-14b
.venv/bin/python -m cli.eval backup qwen3-30b            # pg_dump -> evals/backups/
```

To register a *new* model for the next comparison:

```bash
.venv/bin/python -m cli.eval register-model <slug> \
  --model-path /path/to/model.gguf --display-name "Name" --port 18080 \
  --gpu-layers 99 --tensor-split 1,1 --parallel 2
```

This provisions the Postgres DB, snapshot dir, and
`evals/configs/<slug>.yaml` — then use that config with the existing
`deep-research-kb` commands (`ingest-youtube`, `chunk-source`,
`extract-source`, `verify-source`) exactly as this whole eval has been run
all along. `report`'s numbers are more accurate than every stats query
hand-run through Bash during Rounds 1-4: those all used plain `count(*)`,
which double-counts any claim with more than one `claim_evidence` row
against the same source (confirmed: several claims per source have this);
`report` uses `count(DISTINCT claim id)`.

**Important**: `evals/` (configs/logs/backups) is gitignored at the repo
root — but `deep_research/evals/` (the actual Python package) is not; the
`.gitignore` pattern is anchored (`/evals/`) specifically so it doesn't
also swallow the nested package, which it did on the first attempt.

## Eval Next Steps

(Build/product next steps live in [ROADMAP.md](ROADMAP.md).)

- Add a fourth source using a fresh eval-only canonical URL to get a second
  clean (non-contaminated) 14B-vs-30B pair, to corroborate the verification
  finding above — now much cheaper via `deep-research-eval
  register-model`/`add-source`.
- Decide whether the AI Bubble leftover source should be excluded from
  future model-comparison conclusions given its contamination, while still
  keeping it in the KB for topic/report purposes.
- Instrument `verification.py` to log raw LLM responses (or at least a
  parse-success flag) to separate "model judged no relationship" from
  "model's response didn't parse" — currently indistinguishable, see
  Cross-Verify Experiment above. (Also listed in ROADMAP.md, since the
  decision-log work is the natural place to add it.)

## Reference: server launch + per-source flow

Use llama.cpp on a free port such as `18080`, because `8080` was occupied by Open WebUI internals.

Example 14B server:

```bash
/home/matheau/llama/llama.cpp/build/bin/llama-server \
  -m /home/matheau/.cache/llama.cpp/Qwen_Qwen3-14B-GGUF_Qwen3-14B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18080 -ngl 99 -fa on -c 32768 -b 4096 -ub 512 \
  -dev CUDA0,CUDA1 -sm layer -ts 1,1 --parallel 2
```

Example 30B server:

```bash
/home/matheau/llama/llama.cpp/build/bin/llama-server \
  -m /home/matheau/llama/models/gguf-large/Qwen3-30B-A3B-Q4_K_M.gguf \
  --host 127.0.0.1 --port 18080 -ngl 99 -fa on -c 32768 -b 4096 -ub 512 \
  -dev CUDA0,CUDA1 -sm layer -ts 1,1 --parallel 2
```

(The examples above are kept for reference — `deep-research-eval
register-model` now generates the equivalent config + tracks these same
launch params per model, and `start-server`/`stop-server` wrap the actual
process lifecycle.)

Typical per-source CLI flow from `/home/matheau/code/deep_research`:

```bash
.venv/bin/python -m cli.kb --config evals/configs/qwen3-14b.yaml ingest-youtube <URL>
.venv/bin/python -m cli.kb --config evals/configs/qwen3-14b.yaml chunk-source <SOURCE_ID>
.venv/bin/python -m cli.kb --config evals/configs/qwen3-14b.yaml extract-source <SOURCE_ID> --force
```

Repeat the same flow with `evals/configs/qwen3-30b.yaml`.

Before deciding to use 30B, require a quality win, not just higher count or faster extraction.
