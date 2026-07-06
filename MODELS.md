# Local Model Plan

Local model/runtime guidance for the knowledge-base system described in
[PLAN_KB_ARCHITECTURE.md](PLAN_KB_ARCHITECTURE.md). Split out of the main plan
because model and runtime choices change faster than the data model and pipeline
design.

This section assumes the likely local runtime choice becomes:

- `llama.cpp`
- `2 x 16 GB` NVIDIA GPUs
- `128 GB` system RAM already available

The goal is not to choose one perfect model for everything.
The goal is to assign different model classes to different jobs in the pipeline.

## Overall strategy

Use at least two practical model tiers:

1. a fast day-to-day model
2. a slower higher-quality model for extraction, verification, and harder report synthesis

Avoid designing the system around one giant "do everything" model.
This project will benefit more from role-specific model selection than from forcing every task through one model.

## Runtime direction

Primary runtime recommendation:

- `llama.cpp` for local text generation and local serving

Why:

- explicit multi-GPU support
- GGUF quantized model ecosystem
- OpenAI-compatible `llama-server`
- direct control over split behavior, context, and model choice

Current relevant notes from upstream docs:

- multi-GPU is supported
- `layer` split is the default and most compatible mode
- `tensor` split exists but is marked experimental
- performance guidance is strongest for multiple NVIDIA GPUs with CUDA

## Recommended model roles

### Role A: fast ingestion / classification / lightweight extraction

Use for:

- source metadata cleanup
- broad topic tagging
- identifying candidate entities/events
- deciding whether a source is worth deeper processing
- UI responsiveness

Target class:

- roughly `7B` to `14B`

Good fit:

- `Qwen3-14B`
- comparable `12B` to `14B` instruction models that work well in GGUF form

Why this tier:

- fast enough for repeated source passes
- good enough for first-pass extraction
- lower cost for chunk-by-chunk processing

### Role B: main extraction / verification / report synthesis

Use for:

- claim extraction
- evidence-aware rewriting
- contradiction analysis
- better timeline synthesis
- higher-value topic reports

Target class:

- strongest practical `20B` to `30B`-class quantized model you can run comfortably across `2 x 16 GB`

Good fit:

- `Qwen3-30B-A3B`
- `Gemma 3 27B` if a stable GGUF/runtime path fits your testing

Why this tier:

- better reasoning and synthesis quality than the small fast tier
- realistic for a `32 GB total VRAM` dual-GPU plan using quantized GGUF models

### Role C: optional embeddings / retrieval helper

Use for:

- later retrieval improvements
- clustering or lightweight semantic matching

This can be deferred.
Do not let embeddings block v1.

## Initial model family recommendation

If starting today, the simplest plan is:

1. fast default model:
- `Qwen3-14B`

2. heavy model:
- `Qwen3-30B-A3B`

Why Qwen first:

- current Qwen3 family is strong for general instruction following and reasoning
- there are practical local-app and quantization pathways around it
- it gives a cleaner two-tier family than mixing too many architectures early

Secondary candidate:

- `Gemma 3 27B`

Gemma is still worth evaluating, but I would not make it the default first choice for this project until you test it locally in your exact stack.

## Suggested task mapping

Start with this routing:

- source intake and broad tagging:
  - fast model
- first-pass extraction from chunks:
  - fast model
- second-pass extraction on important sources:
  - heavy model
- verification / contradiction resolution:
  - heavy model
- final topic summary / timeline report:
  - heavy model
- UI chat answer from already-curated data:
  - fast model by default, heavy model on demand

## Thinking vs non-thinking models

Default recommendation:

- prefer non-thinking / instruction-tuned modes for pipeline work

Why:

- easier to control output format
- lower latency
- fewer long chain-of-thought style digressions
- better fit for extraction and structured JSON-ish workflows

If a model family offers both reasoning and non-reasoning variants:

- use non-reasoning for most extraction
- reserve reasoning-heavy variants for difficult verification or synthesis tasks

## Context strategy

Do not plan around huge monolithic prompts.
Even with more VRAM, the architecture should stay chunk-first.

Recommended approach:

- chunk sources aggressively
- extract locally per chunk
- merge claims/evidence in code and database
- only use larger context windows for targeted synthesis, not raw whole-source ingestion

That matters because your project is a knowledge system, not a single-prompt summarizer.

## Multi-GPU strategy in llama.cpp

Start simple:

- use `llama-server`
- start with default `layer` split
- only experiment with `tensor` split after the system is stable

Reason:

- `layer` split is the current default and most compatible path
- `tensor` split may eventually be better for some workloads, but upstream still marks it experimental

## What to avoid early

- building the whole system around one very large model
- depending on experimental split modes on day one
- mixing many unrelated model families before the pipeline is stable
- assuming long context alone will fix extraction quality

## Recommended v1 model stack

If the project were started immediately, the practical v1 stack would be:

- primary runtime:
  - `llama.cpp`
- fast general model:
  - `Qwen3-14B`
- heavy synthesis/extraction model:
  - `Qwen3-30B-A3B`

Optional later evaluations:

- `Gemma 3 27B`
- other strong `12B` to `14B` instruction models if speed or formatting is better

## Why this is a good fit for the project

This plan matches the project's actual workload:

- lots of source-level repeated passes
- extraction and re-analysis matter more than chatbot personality
- claim verification matters more than benchmark bragging rights
- the system benefits from a fast model plus a stronger review/synthesis model

## Sources used for this model plan

- llama.cpp README and server support:
  - https://github.com/ggml-org/llama.cpp
- llama.cpp multi-GPU guide:
  - https://github.com/ggml-org/llama.cpp/blob/master/docs/multi-gpu.md
- Qwen3 14B model card:
  - https://huggingface.co/Qwen/Qwen3-14B
- Qwen3 30B-A3B model card:
  - https://huggingface.co/Qwen/Qwen3-30B-A3B
- Gemma 3 27B model card:
  - https://huggingface.co/google/gemma-3-27b-it
