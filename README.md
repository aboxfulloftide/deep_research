# Deep Research

Deep Research is a local, LLM-assisted research workspace. Create a topic,
add a few URLs, YouTube videos, or files, and the knowledge base automatically
ingests, extracts, checks, connects, and summarizes the material. Automated
actions are visible in decision history and preserve human control over
editorial choices such as topic framing and contradiction adjudication.

## Quick start

Requirements: Python 3.10+, Docker, a local OpenAI-compatible LLM endpoint
for extraction/verification, and Ollama (or another compatible embedding
endpoint).

```bash
cp config.example.yaml config.yaml
docker compose up -d
python -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/uvicorn web.app:app --reload
```

Open the web app at `http://localhost:8000`. Configure local model endpoints
in `config.yaml`; `config.example.yaml` documents every supported KB setting.

## Start automatically at boot

On this host, the app is installed as the `deep-research.service` systemd user
service. It starts the registered Qwen3-14B primary llama.cpp server on port
8080 and Docker's PostgreSQL/SearXNG dependencies before running the web app
and durable knowledge-base worker. Check them with:

```bash
systemctl --user status deep-research.service
systemctl --user status deep-research-llama.service
journalctl --user -u deep-research.service -f
```

The checked-in units are in `systemd/`. The llama unit is boot-only on purpose:
the app's experiment workflow can stop, swap, and restore the primary model
without systemd racing to relaunch it. If the project is moved, update their
`WorkingDirectory` and `ExecStart` paths before linking them.

## Normal workflow

1. Create a topic.
2. Add several sources directly from the topic page.
3. Read the automatically refreshed overview, clearly separated into
   supported, unverified, and competing claims.
4. Use the decision history, evidence links, and review controls whenever you
   want to correct an automated result.

No manual chunk/extract/verify sequence is needed for ordinary use.

## Interactive research modes

The Research page defaults to **Standard** mode: a fast web-first answer with
full text from the strongest available sources. Choose **Extra · 4 levels**
when a question needs deeper investigation. It asks the local model to plan
two complementary starting searches alongside the original question, follows
the evidence through two bounded follow-up levels, then closes one remaining
evidence gap with an extra source. Pages are saved to the research session,
analyzed separately, combined into a draft, and checked against the original
source excerpts before the cited answer is returned. Progress is shown while
each level runs; if a local model is slow at planning follow-up queries, the
app continues with evidence-focused fallback queries instead of stalling.

Research-answer links open in a new browser tab.

The Research page also has a **Chat · llama.cpp** mode for direct streamed
conversation with the currently loaded model. Selecting a registered alternate
profile changes the primary model only when normal processing work is drained
and the GPU is idle; the selected model then remains loaded for later chat.

### llama.cpp model experiments

Interactive research uses the managed llama.cpp server only. The Research page
can queue a model experiment against the current model or a registered
alternate profile, with an optional context-window override and reasoning
enabled or disabled. Experiments are deliberately low priority: they wait for
all ingestion and verification jobs to drain and for the GPU to be idle.
Alternate profiles run temporarily on their evaluation port and are stopped
afterward. If a larger alternate profile cannot fit alongside the primary
model, the worker waits until the queue is idle, temporarily swaps models for
the experiment, and restores the primary server before normal work resumes.

## CLI

The same durable pipeline is available through `deep-research-kb`:

```bash
deep-research-kb ingest-url https://example.com/article
deep-research-kb ingest-youtube 'https://www.youtube.com/watch?v=...'
deep-research-kb track-playlist 'https://www.youtube.com/playlist?list=...'
deep-research-kb extract-pending
deep-research-kb verify-unverified --trigger cron
```

The production host schedules that verification command at 11:00 PM with
`systemd/deep-research-nightly-verification.timer`. The timer is persistent,
so a run missed while the machine is asleep starts after the next wake instead
of being silently skipped like an ordinary cron entry.

After a successful cron-triggered verification sweep, the worker also queues
counter-view checks for up to 50 supported claims that have never been checked.
These are low-priority, idempotent jobs and drain while the local model is idle;
set `DEEP_RESEARCH_KB_NIGHTLY_COUNTER_EVIDENCE_LIMIT` to change the batch size.

For registered local model profiles, `deep-research-kb nightly-role-split`
runs the configured extraction model, then the verifier model, and leaves the
verifier loaded for daytime use.

## Development

```bash
.venv/bin/pytest -q
cd frontend && npm install && npm run build
```

`ROADMAP.md` is the current product and implementation record. `HANDOFF.md`
is the preserved model-evaluation record.
