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
3. Add a real source-adapter interface, beginning with academic and repository
   adapters, then official-documentation/domain targeting. Do not call these
   merely query rewrites.
4. Add an actual local-KB adapter only after defining freshness/provenance
   rules. Keep it disabled for model-collection benchmarks to avoid bias.
5. Upgrade source fitness with semantic directness assessment plus deterministic
   URL/duplicate/extractability checks.
6. Run a plan-only comparison across models first. Score valid facets, useful
   search queries, capability selection, and no raw-question queries.
7. After approving a plan, run source-only collection again. Compare raw
   bundles before running extraction or synthesis.
8. Freeze one approved evidence bundle, then evaluate extractor, synthesizer,
   and fact-checker models separately.

## Relevant Documents

- `PLAN_RESEARCH_SOURCE_ROUTING.md` — longer architecture and benchmark plan.
- `RESEARCH_WORK_HANDOFF.md` — this operational handoff.
