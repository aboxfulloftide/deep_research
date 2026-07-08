# Spike Findings (Step 0)

Ran `run.py` against:

- Article: Oliver Wyman, "How An AI Bubble Burst Could Shake Global Financial Markets"
  (9,188 chars, 8 chunks)
- YouTube transcript: a video on the same AI-bubble topic (19,634 chars, 17 chunks)

Model: `Qwen3-14B-Q4_K_M` served locally via `llama.cpp` (`llama-server`, port 8080),
`/no_think` in the system prompt to suppress reasoning tokens. Result: 158 extracted
claims (53 article, 105 transcript) — enough to hand-read against the ~100-claim exit
target.

## Q1: Is first-pass extraction quality good enough to build on?

**Yes, with caveats.** Read all 158 rows in `output/observations.jsonl`.

- **Atomicity**: claims are consistently one-fact-per-row. No blobs of multiple facts
  jammed into one `claim_text`.
- **Faithfulness / hallucination**: 0 fabricated claims found on manual read. Every
  claim traces to real chunk content.
- **Evidence linking (`supporting_quote`)**: 152/158 (96%) matched the chunk text
  verbatim (exact or case-insensitive substring). The 6 misses were all genuine
  excerpts with minor drift — a dropped leading "In short,", an inserted/dropped
  article word, or an ellipsis where the model condensed a longer quote — never
  invented text. **Conclusion: exact character-offset evidence linking needs
  fuzzy/normalized matching, not strict substring search**, which the schema already
  anticipated (`excerpt_hash`, normalized-text comparison). Strict substring lookup
  alone would wrongly drop ~4% of otherwise-good evidence links.
- **Confidence/importance scores**: narrow range (confidence 0.75–0.98, importance
  0.50–0.95) and only weakly discriminative. Vague/rhetorical statements ("The stakes
  for the economy are high", "The more leverage, the more fuel, the hotter the fire")
  got extracted as claims and still scored 0.75–0.8 confidence — not clearly separated
  from well-sourced numeric claims at 0.9+. **Conclusion: treat self-reported
  confidence/importance as weak, coarse metadata (roughly 3 usable buckets: low/mid/high),
  not a fine-grained ranking signal.** It's good enough to gate verification budget
  (decision: skip verifying anything under ~0.8) but not to rank claims precisely.
- One extraction pass also incidentally exposed a bug in the existing
  `deep_research/tools/scrape.py` product-detection heuristic: it misfires on
  prose paragraphs containing "$" amounts (mistook the Oliver Wyman article for an
  e-commerce product listing). The spike bypassed it by calling `_extract_text`
  directly; the scraper itself should get a stricter product-detection gate before
  it's reused for general article ingestion.

## Q2: How bad is the duplication / resolution problem?

**Entity resolution is easy for exact names, but naive fuzzy matching is unusably
noisy. Claim resolution needs semantic (embedding) matching — lexical similarity
found zero cross-source duplicates.**

Breakdown of the 385 `entity_duplicate` candidates the spike's naive resolver flagged:

| method | count | quality |
|---|---|---|
| exact normalized match | 76 | reliable — e.g. `Nvidia` == `Nvidia` across both sources, correctly unified |
| substring match | 309 | mostly garbage — e.g. `AI` flagged as a "duplicate" of `Mark Twain`, `Britain`, `railway lines`, because those strings happen to contain the substring "ai" |
| trigram (`SequenceMatcher` ratio ≥ 0.82) | 0 | never fired — ratio is unstable on short entity names |

Real-world duplicates that *did* resolve correctly via exact match: `Nvidia` appears
in both the article and the transcript and collapses to one identity, as intended.

Claim-level duplication: **0 claim-duplicate candidates** found via `difflib`
similarity (threshold 0.6) across sources, even between claims covering the same
underlying story. This isn't because the two sources never overlap in substance — it's
because the model paraphrases each claim independently, so lexical similarity between
differently-phrased statements of the same fact is low. Lexical/trigram matching alone
will not catch same-fact claims from different sources; it only catches near-verbatim
repeats within one source.

### Feeds directly into the step-1 resolution strategy decision

- **Entities**: exact normalized-name match as the automatic-merge tier; require a
  minimum name length (e.g. ≥ 4 chars) and/or word-boundary-aware matching before any
  fuzzy method runs, to kill the substring false-positive class seen here. Route
  anything fuzzy into `resolution_candidates` for review — never auto-merge on a fuzzy
  match.
- **Claims**: lexical/trigram matching is not sufficient by itself. Claim resolution
  needs an embedding-similarity pass (or entity+event co-occurrence as a coarse
  pre-filter) to catch same-fact claims phrased differently — this pulls "Role C:
  embeddings" out of "can be deferred" for at least the resolution step, even if v1
  reporting doesn't use vector retrieval otherwise.

## Minimum quality gates — verdict

- Atomic, dedup-able claims: **pass**
- Evidence links point to the correct chunk: **pass**, with the caveat that offset
  matching must tolerate minor normalization drift
- Hallucinated specifics rare: **pass** (0 observed in 158 claims)
- Duplicate clusters measurable with a proposed v1 strategy: **pass** — measured above,
  strategy is exact-match-first for entities + embeddings for claims
- Confidence/importance judged useful or explicitly weak: **judged weak** — usable only
  as a coarse verification-budget gate, not a ranking signal

All gates clear enough to proceed to schema implementation (build order step 1:
lock the resolution strategy) with the two concrete adjustments above baked in rather
than left as open questions.

## Follow-up: does embedding similarity actually fix claim resolution?

The claim-resolution finding above was a negative result (lexical matching found
zero cross-source duplicates). Before locking "use embeddings" into the resolution
strategy, ran a direct validation: embedded all 158 claim texts with
`nomic-embed-text:v1.5` (`clustering:` prefix, already running locally via Ollama) and
ranked all 5,565 cross-source claim pairs by cosine similarity
(`spike/validate_embeddings.py`).

**Result: embeddings catch real duplicates lexical matching missed, but precision
degrades fast past the top few pairs — this must feed a review queue, not an
auto-merge.**

- Top pair (0.884): article's *"Harvard economist Jason Furman estimates that
  AI-driven infrastructure investment accounted for 92% of US GDP growth in the first
  half of 2025"* vs. transcript's *"If the AI buildout is stripped from the equation,
  the US economy only grew at about 1/10 of 1% in the first half of 2025"* — the same
  underlying fact, complementary framing, zero lexical overlap. A genuine catch.
- 2nd pair (0.875): the same article claim vs. transcript's *"investment in
  information processing equipment and software accounted for about 92% of all the
  growth in the US economy"* — same fact, same 92% figure, different attribution
  wording. Also genuine.
- 3rd pair (0.864): the **same article Furman claim** paired with *"Railway investment
  reached approximately 7% of the entire British economy at its peak"* — topically
  related (both are "X% of the economy" statistics about historical/economic bubbles)
  but not the same fact at all. A false positive, one rank below two true positives.
- Similarity distribution across all 5,565 cross-source pairs: p50 = 0.686, p90 =
  0.753, max = 0.884. Only 4 pairs clear 0.85, and only 2 of those 4 are genuine
  duplicates (50% precision at that cutoff). Zero pairs clear 0.90 — including the two
  genuine duplicates, so a stricter threshold would have thrown away the true
  positives along with the noise.

Conclusion: embedding cosine similarity is the right signal to generate claim-merge
*candidates* (lexical matching provably cannot do this job), but no fixed threshold
separates true duplicates from topically-similar-but-distinct claims cleanly enough to
auto-merge on. This locks in as: embedding similarity feeds `resolution_candidates`
for review, same as the fuzzy entity-match tier — never a direct promotion path to
`claim_links.duplicate_of`.
