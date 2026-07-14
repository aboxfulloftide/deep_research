"""Topic report generation (build order step 7).

Reports are outputs, not truth storage (Core Design Rule 5) — this synthesizes
a markdown report from stored claims/evidence via the LLM, citing sources,
never inventing facts beyond what it's given. Decision 27: a new `reports`
row is written every generation, but only the latest is ever surfaced — there
is no report-history browsing feature in v1.

Sizing is map-reduce, not truncation. The first version of this module built
one big prompt and dropped whatever didn't fit a guessed character budget —
that silently loses content, and the "budget" was a static guess anyway, not
the server's real limit. Now: detect the server's actual context window
(llama.cpp's /slots endpoint), and if the topic's content doesn't fit in one
pass, batch it, summarize each batch (preserving every date/number/citation),
and recursively re-summarize the summaries until they fit — nothing is ever
left out, a large topic just costs more LLM calls. If map-reduce had to run
at all, the result carries a human-readable suggestion about the server's
context configuration — surfaced as information, never acted on
automatically; restarting the user's inference server is not this module's
call to make.
"""

from dataclasses import dataclass

import httpx

from deep_research.config import Config, LLMConfig
from deep_research.kb.db import KBDatabase
from deep_research.kb.extraction import detect_context_size, detect_model
from deep_research.kb.timeline import get_topic_timeline
from deep_research.llm import LLMClient

REPORT_SYSTEM_PROMPT = """/no_think
You are a research report writer for a knowledge base. You will be given a topic name/description, a chronological timeline of dated events with supporting claims, and a list of additional undated claims. Write a clear, well-organized markdown report summarizing what is known about this topic.

Rules:
- Base the report ONLY on the claims and evidence provided. Do not invent facts, dates, or numbers not present in the input.
- Cite the source title in parentheses after each claim you use, e.g. "(Source: Fortune)".
- Organize with a short introduction, a "Timeline" section (chronological), and an "Other notable claims" section for undated claims.
- If claims are marked [CONTRADICTED] or [MIXED], note the disagreement explicitly rather than silently picking one side.
- Keep it concise — this is a summary, not an essay. Use markdown headers and bullet points.
"""

BATCH_SUMMARY_SYSTEM_PROMPT = """/no_think
You are condensing one batch of claims from a knowledge base topic into a compact intermediate summary. This summary will later be combined with other batches (or other summaries) to write a final report — your job is to preserve everything a later writer would need, not to write the final report yourself.

Rules:
- Preserve every specific date, number, and source citation exactly as given — condense the prose around them, never the facts themselves.
- Preserve [CONTRADICTED]/[MIXED] status flags exactly as given, attached to the claim they describe.
- Do not add commentary, opinions, or facts not present in the input.
- Output compact markdown bullet points. No headers, no introduction — just the condensed facts.
"""

CHARS_PER_TOKEN_ESTIMATE = 4  # rough heuristic, used only to turn a token budget into a char budget
# Two different reserves, not one: a batch/reduce call is explicitly instructed
# to produce compact bullet points (short output), but the final synthesis
# call writes a full multi-section report (much longer). Using one reserve
# for both meant the final call's own response could get cut off mid-sentence
# even after map-reduce correctly shrank the *input* — the input budget was
# never the only constraint; generation room is too.
BATCH_RESPONSE_TOKEN_RESERVE = 700
FINAL_RESPONSE_TOKEN_RESERVE = 2200
MAX_REDUCE_ROUNDS = 5  # safety cap against a pathological non-converging reduce loop
MAX_BISECTION_DEPTH = 6  # safety cap on _summarize_batch's split-in-half retries below


@dataclass
class ReportResult:
    report_id: str
    content_markdown: str
    used_map_reduce: bool = False
    batch_count: int = 0
    context_tokens_detected: int | None = None
    suggestion: str | None = None


def _format_claim_line(claim: dict, source_title: str | None) -> str:
    cite = f" (Source: {source_title})" if source_title else ""
    # An assertion with no independent check must never read like settled
    # fact in the synthesis prompt. The final report is asked to preserve
    # these markers, separating open assertions from corroborated material.
    flag = f" [{claim['status'].upper()}]" if claim["status"] in ("unverified", "contradicted", "mixed") else ""
    return f"- {claim['canonical_text']}{flag}{cite}"


async def _claim_source_title(kb_db: KBDatabase, claim: dict) -> str | None:
    if claim.get("preferred_source_id"):
        source = await kb_db.get_source(claim["preferred_source_id"])
        if source:
            return source.get("title") or source.get("canonical_uri")
    evidence = await kb_db.list_claim_evidence(claim["id"])
    if evidence:
        return evidence[0].get("source_title")
    return None


async def _detect_context_tokens(base_url: str, fallback_tokens: int) -> int | None:
    """Returns the detected context size, or None if detection failed (caller
    uses fallback_tokens in that case — kept separate from the returned value
    so callers can still report "detection failed" accurately)."""
    return await detect_context_size(base_url)


def _budget_chars(context_tokens: int, reserve_tokens: int) -> int:
    usable_tokens = max(context_tokens - reserve_tokens, 500)
    return usable_tokens * CHARS_PER_TOKEN_ESTIMATE


def _batch_blocks(blocks: list[str], budget_chars: int) -> list[list[str]]:
    """Greedily groups text blocks into batches (each a list of the original
    blocks, not pre-joined — _summarize_batch needs the list form to bisect a
    batch that turns out to be oversized) that fit within budget_chars. A
    single oversized block becomes its own over-budget batch rather than
    being silently dropped."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for block in blocks:
        block_len = len(block) + 2
        if current and current_len + block_len > budget_chars:
            batches.append(current)
            current, current_len = [], 0
        current.append(block)
        current_len += block_len
    if current:
        batches.append(current)
    return batches


def _is_context_exceeded_error(exc: Exception) -> bool:
    """llama.cpp returns a 400 with `type: exceed_context_size_error` when a
    request genuinely doesn't fit — distinct from other 400s (bad params,
    etc.) that bisecting and retrying wouldn't fix and would just waste calls
    on."""
    if not isinstance(exc, httpx.HTTPStatusError) or exc.response.status_code != 400:
        return False
    try:
        return exc.response.json().get("error", {}).get("type") == "exceed_context_size_error"
    except Exception:
        return False


async def _summarize_batch(llm: LLMClient, topic_name: str, blocks: list[str], depth: int = 0) -> str:
    """Our batch sizing is a char-per-token *estimate* (CHARS_PER_TOKEN_ESTIMATE)
    — it can still be wrong, and _batch_blocks explicitly allows a single
    pathologically large block to become its own over-budget batch rather
    than dropping it. If the server rejects a batch as genuinely too large for
    its context, split the blocks in half and summarize each half separately
    instead of failing the whole report — nothing gets dropped, it just costs
    an extra LLM call. Only bisects on the specific exceed-context-size error;
    any other failure (including a single unsplittable block) propagates."""
    batch_text = "\n\n".join(blocks)
    try:
        resp = await llm.chat([
            {"role": "system", "content": BATCH_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Topic: {topic_name}\n\n{batch_text}"},
        ])
        return resp["choices"][0]["message"]["content"] or ""
    except httpx.HTTPStatusError as e:
        if not _is_context_exceeded_error(e) or len(blocks) <= 1 or depth >= MAX_BISECTION_DEPTH:
            raise
        mid = len(blocks) // 2
        left = await _summarize_batch(llm, topic_name, blocks[:mid], depth + 1)
        right = await _summarize_batch(llm, topic_name, blocks[mid:], depth + 1)
        return f"{left}\n\n{right}"


async def _reduce_to_single_input(
    llm: LLMClient, topic_name: str, blocks: list[str], batch_budget_chars: int, final_budget_chars: int,
) -> tuple[str, bool, int]:
    """Map-reduce over `blocks`: if everything already fits in the *final*
    synthesis call's budget, this is a no-op (fast path, no extra LLM calls).
    Otherwise batches and summarizes (map) using batch_budget_chars — batch
    summaries are short by instruction, so they can pack more raw input per
    call — then recursively re-batches and re-summarizes the summaries
    (reduce) until they fit final_budget_chars specifically, which reserves
    much more headroom than the batch budget since the final call writes a
    full multi-section report, not a compact bullet list. Nothing is ever
    dropped — a topic too large for one context window just costs more LLM
    calls. Returns (combined_text, used_map_reduce, batch_count)."""
    total_len = sum(len(b) for b in blocks) + len(blocks)
    if total_len <= final_budget_chars:
        return "\n\n".join(blocks), False, 0

    batches = _batch_blocks(blocks, batch_budget_chars)
    total_batches = len(batches)
    summaries = [await _summarize_batch(llm, topic_name, b) for b in batches]

    rounds = 0
    while sum(len(s) for s in summaries) + len(summaries) > final_budget_chars and len(summaries) > 1:
        rounds += 1
        if rounds > MAX_REDUCE_ROUNDS:
            break  # pathological non-converging case — proceed with what we have
        batches = _batch_blocks(summaries, batch_budget_chars)
        summaries = [await _summarize_batch(llm, topic_name, b) for b in batches]

    return "\n\n".join(summaries), True, total_batches


async def generate_topic_report(kb_db: KBDatabase, config: Config, topic_id: str) -> ReportResult:
    topic = await kb_db.get_topic(topic_id)
    if topic is None:
        raise ValueError(f"No such topic: {topic_id}")

    timeline = await get_topic_timeline(kb_db, topic_id)
    all_claims = await kb_db.list_topic_claims(topic_id, link_status="attached")
    timelined_claim_ids = {c["id"] for entry in timeline for c in entry.claims}
    undated_claims = sorted(
        (c for c in all_claims if c["id"] not in timelined_claim_ids),
        key=lambda c: c["importance_score"] or 0, reverse=True,
    )

    timeline_blocks: list[str] = []
    for entry in timeline:
        date_label = entry.event.get("start_at") or "?"
        lines = [f"### {entry.event['title']} ({date_label})"]
        for claim in entry.claims:
            title = await _claim_source_title(kb_db, claim)
            lines.append(_format_claim_line(claim, title))
        timeline_blocks.append("\n".join(lines))

    undated_blocks = []
    for claim in undated_claims:
        title = await _claim_source_title(kb_db, claim)
        undated_blocks.append(_format_claim_line(claim, title))

    all_blocks = (
        ["## Timeline"] + (timeline_blocks or ["(no dated events)"])
        + ["## Other claims (undated)"] + (undated_blocks or ["(none)"])
    )

    base_url = config.kb.extraction_llm_base_url
    model = config.kb.extraction_llm_model or await detect_model(base_url)
    detected = await _detect_context_tokens(base_url, config.kb.report_context_fallback_tokens)
    context_tokens = detected if detected is not None else config.kb.report_context_fallback_tokens
    batch_budget_chars = _budget_chars(context_tokens, BATCH_RESPONSE_TOKEN_RESERVE)
    final_budget_chars = _budget_chars(context_tokens, FINAL_RESPONSE_TOKEN_RESERVE)

    llm = LLMClient(Config(llm=LLMConfig(base_url=base_url, model=model, api_key="not-needed")))
    try:
        combined_content, used_map_reduce, batch_count = await _reduce_to_single_input(
            llm, topic["name"], all_blocks, batch_budget_chars, final_budget_chars,
        )

        input_text = (
            f"Topic: {topic['name']}\nDescription: {topic.get('description') or '(none)'}\n\n{combined_content}"
        )
        resp = await llm.chat([
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ])
        content = resp["choices"][0]["message"]["content"] or ""
    finally:
        await llm.close()

    suggestion = None
    if used_map_reduce:
        ctx_label = f"{detected}-token" if detected is not None else f"undetected (assumed {context_tokens}-token)"
        suggestion = (
            f"This report needed {batch_count} extra batch(es) to fit the server's current "
            f"{ctx_label} context per slot. For faster single-pass reports on large topics, "
            "consider restarting llama-server with a larger --ctx-size or a lower --parallel "
            "(more context per slot, fewer concurrent slots)."
        )

    report = await kb_db.add_report(
        topic_id=topic_id, content_markdown=content, report_type="timeline",
        title=f"{topic['name']} — timeline report",
        generated_from_scope={
            "claim_count": len(all_claims),
            "timeline_event_count": len(timeline),
            "undated_claim_count": len(undated_claims),
            "used_map_reduce": used_map_reduce,
            "batch_count": batch_count,
            "context_tokens_detected": detected,
        },
    )
    return ReportResult(
        report_id=report["id"], content_markdown=content, used_map_reduce=used_map_reduce,
        batch_count=batch_count, context_tokens_detected=detected, suggestion=suggestion,
    )
