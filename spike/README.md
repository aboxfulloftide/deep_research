# Extraction + Resolution Spike (throwaway)

This is step 0 from `PLAN_KB_ARCHITECTURE.md`. It is a disposable harness, not
production code — its job is to produce ~100 real extracted claims from two
overlapping sources (one article, one YouTube transcript) so they can be read
by hand and answer:

1. Is first-pass claim extraction with the intended fast local model
   (`qwen3:14b` via Ollama) good enough to build on?
2. How bad is entity/claim/event duplication across sources?

Run:

```
../.venv/bin/python run.py
```

Output lands in `output/`:

- `spike.db` — `spike_chunks`, `spike_extraction_runs`, `spike_extracted_observations`,
  `spike_resolution_candidates`
- `observations.jsonl` — every extracted observation, one per line, for hand-reading
- summary counts printed to stdout

Sources used:

- Article: https://www.oliverwyman.com/our-expertise/insights/2026/jan/impact-ai-bubble-burst-on-global-financial-markets.html
- YouTube: https://www.youtube.com/watch?v=2J2Fb1bBufA

Both cover the same topic (AI investment bubble / financial market risk) so the
resolution question is actually testable.

Do not extend this into production code. When the spike is done, keep the
learnings and this directory's read-out; the schema and pipeline get built
properly against PostgreSQL per the plan's build order, not on top of this
harness.
