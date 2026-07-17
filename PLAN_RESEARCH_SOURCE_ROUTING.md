# Generalized Deep-Research Source Routing Plan

## Purpose

Improve source collection for any research question without encoding rules for
a particular domain. The current evidence-facet workflow identifies what a
question needs, but it sends every facet through general web search. The next
iteration will choose appropriate source adapters, preserve approved evidence,
and make collection quality observable before synthesis.

This work begins **after the current six source-collection runs finish**. Those
runs are the baseline; do not overwrite them before their plans, raw sources,
coverage, and elapsed time have been exported for comparison.

## Observations From the Baseline

- A local model can produce plausible queries while still retrieving generic,
  duplicated, or irrelevant sources.
- A source can be authoritative but fail to cover the facet it was retrieved
  for; authority and fitness must be measured separately.
- Different models may be better at candidate discovery, factual retrieval,
  source assessment, or synthesis. Collection needs its own benchmark.
- Broad web search is useful as a fallback, but it should not be the only
  retrieval path for scholarly, official-documentation, repository, news, or
  user-provided evidence.

## Target Architecture

```text
Question
  -> research plan (ambiguities, facets, success criteria)
  -> optional user review
  -> source router chooses adapters per facet
  -> bounded collection and source-fitness assessment
  -> coverage matrix and targeted gap closure
  -> approved frozen evidence bundle
  -> extraction -> synthesis -> fact check
```

### Core Records

All records are generic and must not contain domain-specific fields such as
`model_card`, `price`, or `paper` as required concepts.

- `ResearchPlan`: question, ambiguities, facets, user constraints, breadth and
  depth budgets, and completion criteria.
- `ResearchFacet`: evidence question, purpose, preferred source capabilities,
  coverage state, and gap-closure history.
- `SourceAdapter`: capability declaration, query method, fetch method, source
  provenance, rate limits, and availability status.
- `SourceAssessment`: authority, directness to facet, independence, freshness
  when applicable, duplication relationship, extraction quality, and notes.
- `EvidenceBundle`: plan snapshot, accepted/rejected sources and reasons,
  facet coverage, source text snapshots, and a stable bundle ID.

## Implementation Phases

### Phase 1 — Preserve and Review the Baseline

1. Wait for all six current collection-only jobs to reach a terminal state.
2. Export each job's plan, facet coverage, attempts, source metadata, source
   excerpts, elapsed time, model, context, and reasoning setting.
3. Review the bundles manually and label sources as useful, weak, duplicate,
   irrelevant, or broken.
4. Record the labels as the first comparison rubric. Do not infer model quality
   from a final report in this phase.

### Phase 2 — Source Adapter Interface

1. Define a `SourceAdapter` protocol with `capabilities`, `search`, `fetch`,
   `health`, and optional `cost/rate_limit` metadata.
2. Put the existing layered web search behind a `web` adapter so behavior is
   preserved during migration.
3. Add adapters incrementally, each optional and independently health-checked:

   - Academic: arXiv, OpenAlex, Semantic Scholar, PubMed where configured.
   - Official documentation: domain-restricted web search plus direct fetch.
   - Code/repository: GitHub search and repository/documentation fetch.
   - News/current events: reputable-news search adapter with publication dates.
   - Local knowledge base: the project's existing KB and saved source library.

4. An unavailable adapter must yield a diagnostic and fall back safely; it must
   never silently turn a specialized facet into an arbitrary web result.

### Phase 3 — Router and Budgeting

1. Extend `ResearchFacet` with source capabilities, not specific sites. Example
   capabilities: `primary`, `scholarly`, `official_documentation`, `repository`,
   `news`, `firsthand`, `local_knowledge`.
2. Create a deterministic router that maps capabilities to available adapters.
   The LLM proposes capabilities; application code validates them against an
   allow-list and applies fallback rules.
3. Add user-visible breadth and depth settings:

   - Breadth: maximum facets and independent candidate branches.
   - Depth: maximum gap-closure rounds per facet.
   - Source budget: maximum accepted and fetched sources.
   - Time/search budget: bounded requests and elapsed time.

4. Persist every routing decision: requested capability, selected adapter,
   fallback reason, query, result count, and accepted source count.

### Phase 4 — Source Fitness and Coverage Gate

1. Assess each fetched source against the facet that requested it:

   - `authority`: provenance and publication quality.
   - `directness`: whether it answers the facet rather than merely mentioning it.
   - `independence`: whether it is a duplicate, repack, mirror, or derivative.
   - `freshness`: only where the plan marks the facet time-sensitive.
   - `extractability`: whether usable text and source-quoted claims are present.

2. Use deterministic checks for URL normalization, duplicate families, page
   quality, dates, and quote validation. Use the LLM only for semantic
   directness, with strict JSON validation and a recorded fallback outcome.
3. Mark facets `covered`, `partial`, or `uncovered`; retain rejected sources
   and reasons for audit rather than discarding them invisibly.
4. Run targeted gap closure only for uncovered/partial facets. Stop when the
   budget is exhausted and state the remaining evidence gaps clearly.
5. Require an approved coverage threshold before synthesis. The threshold is
   plan-driven, not a universal requirement for papers or official sources.

### Phase 5 — Plan Review and Reusable Evidence Library

1. Add an optional UI/API pause after plan creation that shows ambiguities,
   facets, preferred evidence capabilities, and budget. The user can edit or
   approve it before collection begins.
2. Store accepted source snapshots, assessments, and extracted claims in a
   reusable local evidence library linked to the existing KB.
3. On later research, the local-library adapter may propose existing evidence
   but must report its age, original facet, and whether live corroboration is
   still required.
4. Keep collection bundles immutable so model comparisons can replay the exact
   same evidence.

### Phase 6 — Model Roles and Comparison Harness

Separate the workflow into independently measurable roles:

| Role | Input | Output | Primary measures |
|---|---|---|---|
| Planner | Question | Research plan | valid facets, useful ambiguity detection |
| Collector | Plan + adapters | Raw sources | coverage, diversity, source fitness |
| Extractor | Frozen sources | Quoted claim ledger | valid/relevant claims, quote accuracy |
| Synthesizer | Frozen ledger | Draft | completeness, decision usefulness, citation adherence |
| Fact checker | Draft + ledger | Final | unsupported claims removed, supported claims retained |

This permits choosing different local models for different roles, as with the
existing YouTube claim-extraction and verification pipeline.

## Before/After Comparison Test

Run only after Phase 4 is complete.

1. Use the same representative question set across several domains:
   technical/product comparison, current-event question, academic question,
   software/documentation question, and local-KB question.
2. Run the current baseline workflow and routed workflow under the same source,
   breadth/depth, model, context, and reasoning settings where possible.
3. Evaluate raw collection before judging any final prose:

   - facet coverage and remaining gaps;
   - accepted-source authority, directness, independence, and diversity;
   - duplicate/irrelevant/broken-source rate;
   - source text and quoted-claim yield;
   - request count, elapsed time, and model tokens where available.

4. Freeze one evidence bundle per question and run the extractor/synthesizer
   comparison separately, so source collection does not confound model quality.
5. Keep both the baseline and improved bundles/results as labeled, immutable
   artifacts. Do not delete the new benchmark results until a user explicitly
   retires them.

## Acceptance Criteria

- The same code path can research multiple domains without domain-specific
  required fields.
- Every accepted source is linked to a facet and has an assessment record.
- The UI/API exposes plan, routing, coverage, rejected-source reasons, and
  remaining gaps before synthesis.
- Adapter outages are visible and have deterministic fallbacks.
- A source-only benchmark can compare models without generating final reports.
- A frozen evidence bundle can be replayed for extraction, synthesis, and
  fact-checking evaluations.
- The before/after test shows a measurable reduction in duplicate/irrelevant
  sources and an improvement in covered facets at an acceptable resource cost.
