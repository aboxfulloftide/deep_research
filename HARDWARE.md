# Hardware Planning

Hardware guidance for the knowledge-base system described in
[PLAN_KB_ARCHITECTURE.md](PLAN_KB_ARCHITECTURE.md). Split out of the main plan
because it rots at a different rate than the data model and pipeline design.

## Resource Planning

### Current tool

For the tool as it exists today:

- 16 GB VRAM is generally enough
- larger context alone is not a huge win because the app truncates and compacts inputs

### Planned system

For the planned knowledge-base version:

- 32 GB VRAM becomes much easier to justify
- not because the app needs giant raw context
- because the system will benefit from stronger local models and more concurrent extraction/verification work

### Minimum workable local box

- `32 GB` system RAM
- `16 GB` VRAM
- `1-2 TB` NVMe storage

### Better target

- `64 GB` system RAM
- `32 GB` VRAM
- `2 TB+` NVMe storage

### Why

- PostgreSQL + parsers + chunking + workers need RAM
- local extraction and verification benefit from larger models
- transcripts/documents create a lot of intermediate text and indexes
- SSD speed matters for snapshots, chunks, caches, and retrieval

### Current machine guidance

If budget is limited, current priority order recommendation:

1. `64 GB` system RAM
2. fast `2 TB+` NVMe
3. `32 GB` total VRAM
4. more CPU cores after that

Interpretation:

- if the goal stays close to the current app, 16 GB VRAM is fine
- if the goal becomes a real local KB with ingestion, verification, and reporting, 32 GB VRAM is a reasonable target

## Revised Hardware Recommendation

This section revisits hardware after the schema and workflow definition became more concrete.

### What changed versus the earlier recommendation

The earlier recommendation was based on the current lightweight scraper/summarizer app.
The newer recommendation reflects the intended system:

- PostgreSQL-backed knowledge base
- raw source snapshots on disk
- chunking and re-analysis
- YouTube transcript ingestion
- document ingestion
- claim extraction and verification
- source trust/ranking
- web UI as the primary interface
- optional background jobs for selected topics

That shifts the bottlenecks away from just "single prompt context size" and toward:

- local model quality for extraction and verification
- system RAM for database, parsers, and workers
- SSD capacity and I/O for snapshots and chunk stores
- GPU memory for stronger local models and parallel job headroom

### Practical conclusion

For this planned system, `32 GB` of total VRAM is no longer just a nice-to-have.
It is the point where the machine starts to make sense as a serious fully local knowledge-base and research box.

`16 GB` VRAM is still viable for a lean v1, but it will force more compromises:

- smaller local models
- less headroom for verification passes
- less comfortable concurrency
- tighter batching and chunking limits

### Recommended machine tiers

#### Tier 1: Minimum viable v1

Use this only if budget is tight and you want to get started without overbuilding.

- CPU: 8 to 12 strong cores
- RAM: `64 GB`
- GPU: `16 GB` VRAM
- Storage:
  - `2 TB` NVMe primary
  - optional second SSD later for raw source snapshots

Expected behavior:

- works for v1
- good for web ingestion, transcript ingestion, document parsing, and smaller-model extraction
- acceptable for manual workflows
- not ideal for heavier verification, re-extraction campaigns, or larger local models

#### Tier 2: Recommended balanced build

This is the best target for the project as currently defined.

- CPU: 12 to 16 strong cores
- RAM: `96 GB` preferred, `64 GB` acceptable
- GPU: `32 GB` total VRAM
- Storage:
  - `2 TB` fast NVMe for OS, PostgreSQL, app, active artifacts
  - `2 TB+` second SSD/NVMe for raw source snapshots, transcripts, and archived artifacts

Expected behavior:

- good local extraction quality
- much better room for verification and re-analysis jobs
- enough headroom for selected-topic monitoring
- far better long-term fit for medium-to-large source volume

#### Tier 3: Comfortable long-term local box

Use this if you want to push quality and reduce future rebuild pressure.

- CPU: 16+ strong cores
- RAM: `128 GB`
- GPU: `32 GB` to `48 GB+` VRAM
- Storage:
  - `2 TB` fast NVMe primary
  - `4 TB+` secondary SSD/NVMe for source archives and growth

Expected behavior:

- comfortable for broader ingestion, more aggressive verification, and batch re-extraction
- better fit for "best practical local capability"
- much less likely to feel constrained after the first implementation phase

### GPU guidance

#### Preferred vendor direction

For this project, NVIDIA remains the safest choice if the priority is fully local inference with the least tooling friction.

Reasons:

- strongest local inference ecosystem
- best compatibility with current LLM tooling
- best chance of fewer surprises across model runtimes beyond Ollama

#### Why not optimize purely for VRAM dollars

Lower-cost alternatives can be attractive on paper, but this project is not just a toy inference box.
It will likely use:

- Ollama or similar local serving
- multiple ingestion/extraction utilities
- evolving model/runtime choices over time

Because of that, ecosystem maturity matters almost as much as raw VRAM size.

#### Current relevant GPU classes

As of `July 5, 2026`, current official spec pages show:

- NVIDIA `RTX 5080`: `16 GB` GDDR7
- NVIDIA `RTX 5090`: `32 GB` GDDR7
- NVIDIA `RTX PRO 6000 Blackwell`: `96 GB` GDDR7 ECC
- AMD Radeon PRO `W7900`: `48 GB` GDDR6
- Intel Arc Pro `B70`: `32 GB`

Interpretation:

- `RTX 5080` class is still a 16 GB compromise tier
- `RTX 5090` class is the first straightforward single-card 32 GB option in the consumer NVIDIA stack
- workstation cards like `W7900` and `RTX PRO 6000` are capacity-first options, but usually for a very different budget
- Intel `B70` is interesting as a 32 GB workstation card, but I would not make it the default recommendation for this project unless budget pressure is dominant and you are willing to accept a less proven local AI software path

### Single 32 GB card vs two 16 GB cards

For this project, a single `32 GB` card is better than two `16 GB` cards if the budget allows it.

Why:

- simpler setup
- fewer multi-GPU edge cases
- cleaner support across runtimes
- better behavior for single-model workloads

Two `16 GB` cards can still help, especially because Ollama improved multi-GPU scheduling in `September 2025`.
But it is still a more complex path than a single larger card.

Important practical point:

- two `16 GB` cards do not behave exactly like a single `32 GB` card in every runtime or workflow
- some serving stacks handle sharding better than others
- concurrency and scheduling behavior are still easier when one card can hold the working model comfortably

### RAM recommendation

For this project, I would now treat `64 GB` system RAM as the floor, not the target.

My recommendation:

- minimum: `64 GB`
- preferred: `96 GB`
- comfortable: `128 GB`

Why RAM matters here:

- PostgreSQL caching
- document parsing
- chunk generation
- background job workers
- browser/UI plus local services on the same machine
- future embeddings or reranking if added later

### Storage recommendation

Because raw sources and snapshots will be stored as files, storage planning matters more than in the current app.

Recommended:

- primary NVMe: OS + PostgreSQL + active app data
- secondary SSD/NVMe: raw source snapshots, transcripts, parsed artifacts, exported outputs

Suggested sizes:

- minimum total: `2 TB`
- recommended total: `4 TB`

For medium-to-large usage, `1 TB` is too small for comfort.

### CPU recommendation

This project is not GPU-only.
Parsing, chunking, DB work, HTML cleanup, transcript processing, and job orchestration all consume CPU.

Recommended:

- minimum: modern 8-core CPU
- better target: modern 12 to 16-core CPU

Do not overspend on CPU before RAM and storage are in good shape, but also do not underbuild CPU if you expect background jobs and frequent refreshes.

### Operating system direction

For a fully local research box, Linux is the safest default.

Why:

- best support for PostgreSQL and background services
- fewer issues with local AI tooling stacks
- better control over GPU/runtime setup

Windows can work, but Linux is still the cleaner default if you are building this primarily as a local AI/data workstation.

### Recommended purchase logic

If you are deciding whether the next dollar should go to GPU, RAM, or storage:

1. Get to `64 GB` RAM if you are below that now
2. Ensure at least `2 TB` of fast SSD storage
3. Then move from `16 GB` VRAM to `32 GB` VRAM if the project is staying on the current roadmap

### Bottom-line recommendation

If you want the shortest answer:

- for the current lightweight app, keep the existing `16 GB` card
- for the planned KB-driven local research system, aim for:
  - `32 GB` VRAM
  - `64-96 GB` RAM
  - `2-4 TB` fast SSD storage

If the budget only allows one major upgrade, the best balanced sequence is:

1. `64 GB` or `96 GB` system RAM
2. enough NVMe storage for snapshots and artifacts
3. then `32 GB` total VRAM
