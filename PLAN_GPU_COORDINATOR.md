# Host GPU Time Coordinator Plan

## Purpose

Provide one local, host-level coordinator for **cooperating GPU consumers**.
It gives a tool a way to inspect GPU availability, reserve the GPU resources it
actually needs, wait fairly in a shared queue, move an important request to the
next position, and release its reservation when finished.

The coordinator is intentionally independent of Deep Research's PostgreSQL
database, model registry, and process lifecycle. Deep Research is one client;
future clients may include Ollama workflows, ComfyUI, benchmark runners, video
tools, or another local application.

## Non-Goals

- Start, stop, configure, or otherwise manage llama.cpp, Ollama, ComfyUI, or
  any other workload. The requesting tool owns that work.
- Preempt arbitrary GPU processes. A tool can only cooperate by acquiring and
  releasing a coordinator lease.
- Guarantee that unmanaged external work is absent. It reports unknown GPU
  work and fails closed for exclusive requests.
- Replace model-specific concurrency. For example, llama.cpp's `--parallel 3`
  remains llama.cpp's own request scheduling; an exclusive coordinator lease
  is for a tool that needs the GPUs for a model load, swap, batch job, or other
  GPU-intensive phase.

## User-Facing Contract

Install a host command named `local-gpu` (the final name can change without
changing the protocol). Any tool may call it:

```bash
# Human or tool inspection
local-gpu status
local-gpu queue

# Request a declared resource envelope. `--wait` blocks until granted and
# renews the lease. A request can fit on one GPU while another tool uses the
# remaining capacity of the other GPU.
local-gpu acquire --owner deep-research --purpose 'Qwen3-14B research' \
  --vram-mib 12000 --gpus 1 --wait
local-gpu acquire --owner comfyui --purpose 'image batch' \
  --vram-mib 14000 --gpus 1 --wait

# A user-approved request can move from fifth place to the next runnable spot.
local-gpu move-next <request-id>

# Release normally; abandoned leases expire by heartbeat timeout.
local-gpu release <lease-id>
```

`status` must clearly distinguish:

- **available** — no active lease and no blocking unmanaged GPU work;
- **leased** — owner, purpose, GPU indices, reserved VRAM, start time,
  heartbeat age, and queue length;
- **external-busy** — process detected but not managed by the coordinator;
- **unavailable** — detection failed, so exclusive work is not started.

`move-next` is deliberately not a blanket priority escalation. It moves one
identified waiting request immediately after the active lease (or to the head
when idle), preserving the relative order of every other request. The action
is recorded with who requested it and when, so the UI/CLI can explain why the
order changed.

## Shared State and Protocol

Use a host-owned SQLite database under the user's runtime/state directory,
for example `~/.local/share/local-gpu-coordinator/coordinator.db`. SQLite is
appropriate because this is single-host coordination and supports atomic queue
operations without requiring every client to share Deep Research's PostgreSQL
credentials.

The coordinator exposes the CLI above and a local-only Unix-socket HTTP/JSON
API with the same operations. The CLI is sufficient for shell tools; the API
allows ComfyUI extensions or future UIs to integrate without shell parsing.

Core records:

| Record | Required fields | Meaning |
| --- | --- | --- |
| `requests` | id, owner, purpose, resource request, priority, state, requested_at, moved_next_at | FIFO queue entry; states include waiting, granted, cancelled, expired, completed. |
| `leases` | id, request_id, owner, GPU allocation, reserved VRAM, acquired_at, heartbeat_at, expires_at | A granted reservation; several leases may coexist only when their declared allocations fit safely. |
| `events` | timestamp, request_id, action, actor, detail | Auditable acquire, release, expiry, cancellation, and move-next actions. |
| `runtimes` | owner, name, GPU group, optional endpoint/unit/PID, heartbeat | Informational registration of resident services such as llama.cpp, Ollama, or ComfyUI. It does not grant exclusive time. |

All queue mutations use one SQLite transaction (`BEGIN IMMEDIATE`). Leases use
a short TTL (for example 60 seconds) and a client heartbeat every 15 seconds.
Expired leases are reclaimed atomically before granting the next request.

Each request declares a resource envelope rather than merely "the GPU":

- minimum and preferred GPU counts or an explicit GPU set;
- per-GPU VRAM reservation, including model weights, KV cache, and a safety
  margin;
- workload class (`latency-sensitive`, `interactive`, or `batch`);
- whether it can share a GPU with another managed workload;
- optional launch variants, such as one-GPU and two-GPU forms of the same
  model.

The coordinator measures total VRAM and current process allocations per GPU.
It subtracts active managed reservations and an unallocatable safety reserve;
observed memory alone is never treated as guaranteed free capacity.

A range alone is insufficient for correct placement: one GPU versus two GPUs
can have different VRAM layouts, context limits, and throughput. Therefore a
request may provide several **viable variants**, each with a concrete resource
shape and optional launch hint. The coordinator chooses the best fitting
variant; the requesting tool owns what that hint means operationally.

## Detection Model

The coordinator separates **physical observation** from **permission**:

1. Query GPU processes through NVML when available, with `nvidia-smi` as a
   fallback. Record PID, process name, memory, utilization, and GPU index.
2. Match processes to registered runtimes only as status enrichment; do not
   infer ownership solely from a process name.
3. Let a registered runtime expose an optional health/idle probe. For llama.cpp
   that is `/slots`; for a future tool it may be an HTTP endpoint, Unix socket,
   or no probe at all.
4. An unknown compute process yields `external-busy` for the affected GPU. The
   coordinator may still grant a request that explicitly fits on other known,
   safe GPUs; it never kills the unknown process.
5. A resident but idle service (for example an unloaded ComfyUI worker or an
   idle llama.cpp server) is visible in status but does not hold the exclusive
   lease.

This fixes the present ambiguity where checking only llama.cpp port 8080 misses
an alternate server on port 18080, and where GPU memory residency is confused
with active GPU work.

## Capacity Packing and Automatic Backfill

The scheduler is resource-aware. It evaluates the queue in FIFO order, but may
grant a later **backfillable** request when the earlier request cannot fit and
the later one safely fits the currently unused capacity. This is the automatic
version of "jumping in line": a small Ollama embedding job or ComfyUI request
can use a free GPU while a larger dual-GPU request waits for both GPUs.

Backfill rules:

- Never interrupt or shrink an active lease.
- Never allocate beyond declared reservations plus the per-GPU safety margin.
- Do not bypass an earlier request that already fits.
- Record every bypass event and show it in `queue` output.
- Apply starvation protection: after a configurable number of bypasses or
  maximum wait time, reserve compatible capacity for the oldest blocked job.
- `move-next` remains a user action; resource backfill is automatic only when
  it uses otherwise unusable capacity.

The result is not a simplistic global FIFO queue: it is fair FIFO with safe
capacity packing.

## Deep Research Resource-Aware Launching

Deep Research currently launches registered llama.cpp profiles across both
GPUs by default. Its coordinator adapter must add resource variants for each
profile, for example:

| Profile variant | Coordinator request | llama.cpp launch intent |
| --- | --- | --- |
| Qwen3-14B / single GPU | one GPU, model+KV safety envelope | all layers on the granted GPU; no two-GPU tensor split |
| Qwen3-14B / dual GPU | two GPUs, split envelope | current `-ts 1,1 -dev CUDA0,CUDA1` launch |
| Qwen3-30B / dual GPU | two GPUs, larger split envelope | split across both GPUs |

When a Deep Research job reaches the head of the queue, it submits a capability
contract rather than trying to inspect machine capacity itself. For example,
Qwen3-14B can declare **minimum: one GPU** and **preferred: two GPUs**, with
the concrete one- and two-GPU variants in the table above.

The coordinator selects the preferred two-GPU grant when there is no active or
waiting request that needs the remaining capacity. This lets an otherwise idle
machine use all available resources. If the dual-GPU variant is blocked, or if
choosing one GPU leaves the other GPU available for useful queued work, the
coordinator returns the single-GPU grant and the Deep Research adapter launches
that registered single-GPU variant. The adapter, not the coordinator, still
owns the actual llama.cpp stop/start command.

When new work enters the queue during an already-running dual-GPU job, the
coordinator does not shrink or interrupt that job by default. It applies the
smarter allocation decision at the next safe job boundary. This preserves
current work while making subsequent jobs maximize aggregate queue throughput.

This decision must be made at a safe lifecycle boundary: before loading a
model, between jobs, or after an idle server has been deliberately swapped.
The coordinator must never force a running dual-GPU llama.cpp process to
change topology mid-generation. A user-visible status should state which
variant was selected and why.

## Fairness and Priority

- Default scheduling is FIFO by `requested_at`.
- A tool can cancel its own waiting request.
- `move-next` requires an explicit user-facing action or a caller permission
  flag. It cannot move a request ahead of an active lease.
- Optionally cap consecutive move-next actions per owner to avoid starvation.
- The queue exposes position, active owner, and an estimated wait based on
  completed lease durations where enough history exists.

## Deep Research Integration

Deep Research becomes a normal client:

1. Before a model experiment, model swap, source extraction, verification
   batch, or direct llama.cpp chat generation, request the appropriate
   resource envelope and honour the granted launch variant.
2. Register managed runtimes for the primary llama.cpp server and temporary
   evaluation servers, including port and systemd unit for observability.
3. Keep all existing llama.cpp start/stop logic in Deep Research. The
   coordinator only says when this project may use the GPU.
4. Replace the PostgreSQL-only speculative GPU gate with coordinator status
   plus lease acquisition. Keep the PostgreSQL advisory lock for protecting
   Deep Research's own database jobs.
5. Replace shell `pgrep` handoffs with durable job completion plus lease
   release/acquisition.

Ollama and ComfyUI integrations follow the same contract: they acquire before
a GPU-intensive action and release afterward. Their own services, models,
queues, and APIs remain their responsibility.

## Failure Handling

- Client crash: heartbeat expires; lease becomes expired and queue advances.
- Coordinator restart: persisted state is recovered; stale leases are expired
  based on TTL before any new grant.
- GPU query failure: report `unavailable` and refuse new exclusive leases.
- A tool fails after acquiring: release in a `finally` block; TTL is fallback.
- Service remains resident after its job: show it as an idle runtime, not as a
  phantom active job. Lifecycle cleanup remains with its owning tool.

## Delivery Phases

1. **Coordinator core:** SQLite state, CLI, resource-envelope acquire/release/
   wait, TTL, `status`, `queue`, `move-next`, and tests with a fake
   clock/probe.
2. **Host service and API:** user systemd service, Unix socket API, GPU process
   probe, runtime registration, and human-readable status.
3. **Deep Research adoption:** register one- and two-GPU profile variants;
   wrap experiments, model switching, and queued GPU phases; expose
   coordinator status, selected variant, and move-next in the web UI.
4. **Other-tool adapters:** small optional wrappers/examples for Ollama and
   ComfyUI. They remain opt-in and do not become dependencies of Deep Research.
5. **Operational hardening:** audit view, lease-duration metrics, starvation
   safeguards, and documented recovery commands.

## Acceptance Criteria

- Two independent local tools requesting GPU resources run in one visible
  queue and never receive allocations exceeding any GPU's safe capacity.
- A small one-GPU request behind a blocked two-GPU request can be granted when
  it uses otherwise unused capacity; the bypass and its reason are visible.
- Deep Research can choose a registered single-GPU model variant when that
  enables another compatible request to use the second GPU, without changing
  a running model mid-generation.
- When the queue is otherwise empty, a request declaring minimum one GPU and
  preferred two GPUs receives the two-GPU variant when it safely fits.
- A user can move a waiting request from fifth to next; the action is visible
  in queue history and does not interrupt active work.
- A crashed client no longer blocks the queue after its lease TTL.
- `local-gpu status` identifies managed idle runtimes, active leases, and
  unknown external GPU work separately.
- Deep Research can show why a job is waiting and which external owner holds
  the resource, without relying on a project-local PostgreSQL job record.
