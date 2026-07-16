# Host GPU Time Coordinator Plan

## Purpose

Provide one local, host-level coordinator for **cooperating GPU consumers**.
It gives a tool a way to inspect GPU availability, reserve exclusive GPU time,
wait fairly in a shared queue, move an important request to the next position,
and release its reservation when finished.

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

# Request exclusive time. `--wait` blocks until granted and renews the lease.
local-gpu acquire --owner deep-research --purpose 'Qwen3-32B experiment' --wait
local-gpu acquire --owner comfyui --purpose 'image batch' --wait

# A user-approved request can move from fifth place to the next runnable spot.
local-gpu move-next <request-id>

# Release normally; abandoned leases expire by heartbeat timeout.
local-gpu release <lease-id>
```

`status` must clearly distinguish:

- **available** — no active lease and no blocking unmanaged GPU work;
- **leased** — owner, purpose, start time, heartbeat age, and queue length;
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
| `requests` | id, owner, purpose, priority, state, requested_at, moved_next_at | FIFO queue entry; states include waiting, granted, cancelled, expired, completed. |
| `leases` | id, request_id, owner, acquired_at, heartbeat_at, expires_at | One active exclusive reservation per configured GPU group. |
| `events` | timestamp, request_id, action, actor, detail | Auditable acquire, release, expiry, cancellation, and move-next actions. |
| `runtimes` | owner, name, GPU group, optional endpoint/unit/PID, heartbeat | Informational registration of resident services such as llama.cpp, Ollama, or ComfyUI. It does not grant exclusive time. |

All queue mutations use one SQLite transaction (`BEGIN IMMEDIATE`). Leases use
a short TTL (for example 60 seconds) and a client heartbeat every 15 seconds.
Expired leases are reclaimed atomically before granting the next request.

## Detection Model

The coordinator separates **physical observation** from **permission**:

1. Query GPU processes through NVML when available, with `nvidia-smi` as a
   fallback. Record PID, process name, memory, utilization, and GPU index.
2. Match processes to registered runtimes only as status enrichment; do not
   infer ownership solely from a process name.
3. Let a registered runtime expose an optional health/idle probe. For llama.cpp
   that is `/slots`; for a future tool it may be an HTTP endpoint, Unix socket,
   or no probe at all.
4. An unknown compute process yields `external-busy` for an exclusive request.
   The coordinator does not kill it.
5. A resident but idle service (for example an unloaded ComfyUI worker or an
   idle llama.cpp server) is visible in status but does not hold the exclusive
   lease.

This fixes the present ambiguity where checking only llama.cpp port 8080 misses
an alternate server on port 18080, and where GPU memory residency is confused
with active GPU work.

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
   batch, or direct llama.cpp chat generation, acquire the appropriate lease.
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

1. **Coordinator core:** SQLite state, CLI, FIFO acquire/release/wait, TTL,
   `status`, `queue`, `move-next`, and tests with a fake clock/probe.
2. **Host service and API:** user systemd service, Unix socket API, GPU process
   probe, runtime registration, and human-readable status.
3. **Deep Research adoption:** wrap experiments, model switching, and queued
   GPU phases; expose coordinator status and move-next in the web UI.
4. **Other-tool adapters:** small optional wrappers/examples for Ollama and
   ComfyUI. They remain opt-in and do not become dependencies of Deep Research.
5. **Operational hardening:** audit view, lease-duration metrics, starvation
   safeguards, and documented recovery commands.

## Acceptance Criteria

- Two independent local tools requesting exclusive time run in one visible
  queue and never receive a lease concurrently for the same GPU group.
- A user can move a waiting request from fifth to next; the action is visible
  in queue history and does not interrupt active work.
- A crashed client no longer blocks the queue after its lease TTL.
- `local-gpu status` identifies managed idle runtimes, active leases, and
  unknown external GPU work separately.
- Deep Research can show why a job is waiting and which external owner holds
  the resource, without relying on a project-local PostgreSQL job record.
