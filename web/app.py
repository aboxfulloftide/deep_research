import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from deep_research.agent import ResearchAgent, _analyze_products, _compact_product_list, _URL_RE
from deep_research.config import load_config
from deep_research.db import Database
from deep_research.llm import LLMClient
from deep_research.model_backends import apply_backend, list_models as backend_list_models
from deep_research.kb.extraction import detect_model
from deep_research.model_switching import ModelSwitchUnavailable, switch_primary_profile
from deep_research.tools.scrape import scrape_page
from deep_research.tools.search import check_providers_now, web_search
from deep_research.tools.search_usage import get_usage_summary
from web import kb_routes

# Global state
config = None
db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, db
    config = load_config()
    db = Database(config.db_path)
    await db.init()
    await kb_routes.init_kb(config)
    try:
        yield
    finally:
        await kb_routes.close_kb()


app = FastAPI(title="Deep Research", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(kb_routes.router)


# --- Request/Response models ---

class QueryRequest(BaseModel):
    query: str
    model: str | None = None
    session_id: str | None = None
    prioritize_kb: bool = False
    research_mode: Literal["standard", "extra"] = "standard"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class LlamaChatRequest(BaseModel):
    message: str
    messages: list[ChatMessage] = []
    profile_slug: str = "current"
    session_id: str | None = None


class ResearchPlanRequest(BaseModel):
    query: str


class SessionResponse(BaseModel):
    id: str
    title: str | None
    created_at: str
    updated_at: str


# --- API Routes ---

@app.get("/api/models")
async def list_models():
    """List models on the project's llama.cpp server.

    The interactive research app deliberately uses one managed local runtime.
    Ollama remains available for embeddings, but is not a chat backend here.
    """
    cfg = apply_backend(config.model_copy(deep=True), "llama_cpp")
    try:
        models = await backend_list_models("llama_cpp", cfg.llm.base_url)
        default = cfg.llm.model if cfg.llm.model in models else (models[0] if models else "")
        return {"models": models, "default": default, "backend": "llama_cpp"}
    except Exception as e:
        return {"models": [], "default": "", "backend": "llama_cpp", "error": str(e)}


@app.get("/api/search-usage")
async def search_usage():
    """Per-provider search call counts/status (duckduckgo via SearXNG scrape,
    brave/tavily via their real APIs) -- how many calls, ok/empty/error
    breakdown, and the most recent call's outcome per provider."""
    return await get_usage_summary(config)


@app.post("/api/search-usage/check")
async def search_usage_check():
    """Fires one live probe query at each provider right now and returns
    whether it's actually responding -- the historical log above can be
    stale if nothing's called a given provider in a while."""
    return await check_providers_now(config)


@app.post("/api/research-plan")
async def preview_research_plan(request: ResearchPlanRequest):
    """Create a research plan without calling a search or scrape provider."""
    question = request.query.strip()
    if not question:
        raise HTTPException(400, "A research question is required")
    from deep_research.tools.extra_research import plan_research

    cfg = config.model_copy(deep=True)
    try:
        cfg.llm.model = await detect_model(cfg.llm.llama_cpp_base_url)
    except Exception:
        pass
    llm = LLMClient(cfg)
    try:
        plan = await plan_research(llm, question)
    finally:
        await llm.close()
    return {
        "question": plan.question,
        "ambiguities": plan.ambiguities,
        "facets": [
            {
                "id": facet.id, "evidence_question": facet.question,
                "search_query": facet.search_query, "purpose": facet.purpose,
                "capabilities": facet.capabilities,
            }
            for facet in plan.facets
        ],
        "searches_performed": 0,
    }


@app.get("/api/sessions")
async def list_sessions(limit: int = 50):
    """List past research sessions."""
    sessions = await db.list_sessions(limit=limit)
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details and messages."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    messages = await db.get_session_messages(session_id)
    return {"session": session, "messages": messages}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    # Delete messages and session
    import aiosqlite
    async with aiosqlite.connect(db.db_path) as conn:
        await conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await conn.execute("DELETE FROM scraped_pages WHERE session_id = ?", (session_id,))
        await conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await conn.commit()
    return {"status": "deleted"}


@app.post("/api/research")
async def research(req: QueryRequest):
    """Run a research query and stream results via SSE."""
    cfg = apply_backend(config.model_copy(deep=True), "llama_cpp")
    cfg.llm.model = req.model or cfg.llm.model

    return EventSourceResponse(
        _stream_research(cfg, req.query, req.session_id, req.prioritize_kb, req.research_mode),
        media_type="text/event-stream",
    )


@app.post("/api/llama-chat")
async def llama_chat(req: LlamaChatRequest):
    """Direct streamed chat with the loaded llama.cpp model.

    A selected registered profile replaces the primary model only after the
    worker queue and GPU are idle. ``current`` always chats with whichever
    model is already loaded.
    """
    message = req.message.strip()
    if not message:
        raise HTTPException(400, "A chat message is required")
    if len(req.messages) > 100:
        raise HTTPException(400, "Chat history is limited to 100 messages")

    async def stream():
        try:
            if req.session_id:
                session = await db.get_session(req.session_id)
                if not session:
                    raise ValueError("Session not found")
                chat_session_id = req.session_id
            else:
                chat_session_id = await db.create_session()
            await db.add_message(chat_session_id, "user", content=message)
            session = await db.get_session(chat_session_id)
            if not session.get("title"):
                await db.update_session_title(chat_session_id, message[:80] + ("..." if len(message) > 80 else ""))
            yield {"event": "session", "data": json.dumps({"session_id": chat_session_id})}
            if req.profile_slug != "current":
                if kb_routes.kb_db is None:
                    raise RuntimeError("The knowledge-base worker is unavailable, so safe model switching is unavailable")
                yield {"event": "status", "data": json.dumps({"detail": "Checking whether it is safe to switch models..."})}
                loaded_model = await switch_primary_profile(kb_routes.kb_db, config, req.profile_slug)
            else:
                loaded_model = await detect_model(config.llm.llama_cpp_base_url)

            yield {"event": "model", "data": json.dumps({"model": loaded_model})}
            cfg = apply_backend(config.model_copy(deep=True), "llama_cpp")
            cfg.llm.model = loaded_model
            llm = LLMClient(cfg)
            try:
                history = [message.model_dump() for message in req.messages]
                history.append({"role": "user", "content": message})
                answer_parts = []
                async for token in llm.chat_stream(history):
                    answer_parts.append(token)
                    yield {"event": "token", "data": json.dumps({"content": token})}
            finally:
                await llm.close()
            await db.add_message(chat_session_id, "assistant", content="".join(answer_parts))
            yield {"event": "done", "data": json.dumps({})}
        except (ModelSwitchUnavailable, ValueError) as exc:
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}
        except Exception as exc:
            yield {"event": "error", "data": json.dumps({"error": str(exc)})}

    return EventSourceResponse(stream(), media_type="text/event-stream")


async def _stream_research(
    cfg, query: str, session_id: str | None, prioritize_kb: bool = False, research_mode: str = "standard",
):
    """Generator that yields SSE events during research."""
    llm = LLMClient(cfg)

    try:
        # Create/resume session
        if session_id:
            existing = await db.get_session(session_id)
            if not existing:
                yield {"event": "error", "data": json.dumps({"error": "Session not found"})}
                return
        else:
            session_id = await db.create_session()

        yield {"event": "session", "data": json.dumps({"session_id": session_id})}

        await db.add_message(session_id, "user", content=query)

        # Auto-title
        session = await db.get_session(session_id)
        if not session.get("title"):
            title = query[:80] + ("..." if len(query) > 80 else "")
            await db.update_session_title(session_id, title)

        # Pre-gather URLs
        urls = _URL_RE.findall(query)
        gathered_parts = []
        all_products = []

        for url in urls:
            yield {"event": "status", "data": json.dumps({"step": "scraping", "detail": url[:80]})}
            try:
                page = await scrape_page(url, cfg)
                await db.save_scraped_page(
                    session_id, page.url, page.title,
                    page.text_content, page.structured_data,
                )
                gathered_parts.append(f"=== Scraped: {page.title} ({url}) ===\n{page.text_content}")
                if page.structured_data and page.structured_data.get("products"):
                    all_products.extend(page.structured_data["products"])
            except Exception as e:
                gathered_parts.append(f"=== Error scraping {url}: {e} ===")

        gathered_data = "\n\n".join(gathered_parts)

        if all_products:
            analysis = _analyze_products(all_products, query)
            compact = _compact_product_list(all_products)
            gathered_data = f"{analysis}\n\n{compact}"

        if research_mode == "extra":
            answer = ""
            async for event in _extra_research_answer(llm, query, cfg, session_id):
                if event.get("event") == "answer":
                    answer = event["data"]
                else:
                    yield event
        else:
            # Check tool support
            if llm.supports_tools is None:
                try:
                    from deep_research.prompts import TOOL_DEFINITIONS
                    await llm.chat(
                        [{"role": "user", "content": "hi"}],
                        tools=[TOOL_DEFINITIONS[0]],
                    )
                except Exception:
                    pass

            if gathered_data or not llm.supports_tools:
                # Text mode — direct answer
                yield {"event": "status", "data": json.dumps({"step": "generating", "detail": "Composing answer..."})}
                answer = await _text_mode_answer(llm, query, gathered_data, cfg, prioritize_kb)
            else:
                # Tool loop
                answer = ""
                async for event in _tool_loop(llm, query, session_id, cfg, prioritize_kb):
                    if event.get("event") == "answer":
                        answer = event["data"]
                    else:
                        yield event

        await db.add_message(session_id, "assistant", content=answer)
        yield {"event": "answer", "data": json.dumps({"answer": answer, "session_id": session_id})}
        yield {"event": "done", "data": json.dumps({"session_id": session_id})}

    except Exception as e:
        yield {"event": "error", "data": json.dumps({"error": str(e)})}
    finally:
        await llm.close()


async def _extra_research_answer(llm: LLMClient, query: str, cfg, session_id: str | None = None):
    """Run a bounded, source-preserving, four-level Extra Research workflow."""
    from deep_research.tools.extra_research import (
        analysis_context,
        analyze_sources_separately,
        build_claim_ledger,
        claim_ledger_context,
        has_authoritative_source,
        collect_research_bundle,
        source_context,
    )

    yield {
        "event": "status",
        "data": json.dumps({"step": "thinking", "detail": "Planning question-specific evidence facets with the local model..."}),
    }
    yield {"event": "status", "data": json.dumps({"step": "researching", "detail": "Collecting evidence for each facet, then closing uncovered facets..."})}
    research_bundle = await collect_research_bundle(llm, query, cfg)
    sources = research_bundle.sources
    if session_id:
        for source in sources:
            await db.save_scraped_page(
                session_id, source.url, source.title, source.full_content or source.content,
                {"extra_research": {"level": source.level, "query": source.query, "research_facets": research_bundle.coverage}},
            )

    if not sources or not has_authoritative_source(sources):
        yield {"event": "answer", "data": "I could not retrieve usable sources for this extra research run."}
        return

    yield {
        "event": "status",
        "data": json.dumps({"step": "thinking", "detail": f"Analyzing {len(sources)} sources separately before combining them..."}),
    }
    analyses = await analyze_sources_separately(llm, query, sources)
    yield {
        "event": "status",
        "data": json.dumps({"step": "thinking", "detail": "Building a source-quoted claim ledger and excluding unsupported facts..."}),
    }
    claims = await build_claim_ledger(llm, query, sources)
    if not claims:
        yield {"event": "answer", "data": "I found sources but could not extract source-quoted evidence reliably enough to produce a research answer."}
        return
    yield {
        "event": "status",
        "data": json.dumps({"step": "generating", "detail": "Combining source analyses into a draft answer..."}),
    }
    briefs = analysis_context(analyses)
    ledger = claim_ledger_context(claims)
    messages = [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are a careful deep-research analyst. Synthesize source-by-source analyses into a "
                "decision memo using ONLY the supplied claim ledger. Do not add facts from your general knowledge or "
                "from the source analyses. Separate verified specifications, reproducible estimates, and unknowns. "
                "Every factual or numerical sentence must include an exact Markdown source link from the ledger. "
                "Never use [citation: N], a bare citation number, or a source not in the ledger. If evidence is "
                "insufficient, say so. Never call a finding official unless its ledger tier is primary or paper; "
                "otherwise label it as technical-reference evidence. Prefer a ranked shortlist and concise tradeoffs "
                "over an encyclopedia."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Research question: {query}\n\nClaim ledger (authoritative):\n{ledger}\n\n"
                f"Source analyses (context only; not evidence):\n{briefs}\n\nWrite the decision memo now."
            ),
        },
    ]
    response = await llm.chat(messages)
    draft = response["choices"][0]["message"].get("content", "No answer produced.")
    yield {
        "event": "status",
        "data": json.dumps({"step": "thinking", "detail": "Checking the draft against the original question and source evidence..."}),
    }
    evidence = source_context(sources, per_source_chars=900)
    fact_check = [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are a strict final fact checker. Correct the draft only where the supplied evidence "
                "does not support it, it overstates certainty, has an uncited number, or uses a citation not present in "
                "the ledger. Remove a claim rather than guessing. Keep only facts that map to a ledger row, preserve exact "
                "Markdown source links, and return the corrected final answer only. Never output [citation: N] or "
                "label secondary/technical-reference evidence as official."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question: {query}\n\nDraft answer:\n{draft}\n\n"
                f"Authoritative claim ledger:\n{ledger}\n\nSource evidence for quote checks:\n{evidence}"
            ),
        },
    ]
    response = await llm.chat(fact_check)
    answer = response["choices"][0]["message"].get("content", "").strip() or draft
    source_links = "\n".join(f"- [{source.title}]({source.url})" for source in sources)
    answer = f"{answer.rstrip()}\n\n### Sources consulted\n{source_links}"
    yield {"event": "answer", "data": answer}


async def _text_mode_answer(llm: LLMClient, query: str, gathered_data: str, cfg, prioritize_kb: bool = False) -> str:
    """Generate answer in text mode. This is the common fallback for models
    without tool-calling support, so prioritize_kb has to affect it too, not
    just the tool loop, to actually change behavior for most local models."""
    from deep_research.prompts import SYSTEM_PROMPT_NO_TOOLS
    from deep_research.tools.kb_search import kb_search

    if not gathered_data and prioritize_kb and kb_routes.kb_db:
        kb_result = await kb_search(query[:200], kb_routes.kb_db, cfg)
        if kb_result and not kb_result.startswith("No results found"):
            gathered_data = kb_result

    if not gathered_data:
        try:
            results = await web_search(query[:200], cfg)
            if results:
                search_results = "\n".join(
                    f"- {r.title}: {r.url}\n  {r.snippet}" for r in results
                )
                gathered_data = search_results

                # Models without reliable tool calling use this path. Search
                # snippets are useful for finding sources but too abbreviated
                # to answer questions about contested wording, qualifications,
                # or chronology. Read the strongest HTML sources as well and
                # give the model their actual text, while retaining the full
                # result list for citations and fallback context.
                candidates = [
                    result for result in results
                    if not result.url.lower().split("?", 1)[0].endswith(".pdf")
                ][:4]

                async def read_result(result):
                    try:
                        page = await scrape_page(result.url, cfg)
                        return result, page
                    except Exception:
                        return result, None

                fetched = await asyncio.gather(*(read_result(result) for result in candidates))
                pages = []
                for result, page in fetched:
                    if page and page.text_content:
                        pages.append(
                            f"=== Source: {page.title or result.title} ({result.url}) ===\n"
                            # The lead and opening sections normally contain
                            # the claim, quote, and the source's conclusion.
                            # Keeping each source bounded leaves the local
                            # model enough room to compare both sources and
                            # answer, rather than spending its whole context
                            # window on one long article.
                            f"{page.text_content[:4000]}"
                        )
                    if len(pages) == 2:
                        break
                if pages:
                    gathered_data = f"{search_results}\n\n" + "\n\n".join(pages)
        except Exception:
            gathered_data = "(Search failed)"

    import re
    clean_query = _URL_RE.sub("", query).strip()
    for prefix in ("scrape", "scan", "check", "read", "look at", "go to"):
        if clean_query.lower().startswith(prefix):
            clean_query = clean_query[len(prefix):].strip()
    clean_query = clean_query.lstrip("and").strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_NO_TOOLS},
        {
            "role": "user",
            "content": (
                f"DATA:\n\n{gathered_data}\n\n"
                f"---\n\n"
                f"QUESTION: {clean_query}\n\n"
                f"Rules:\n"
                f"1. Answer ONLY this question using the DATA above.\n"
                f"2. Copy product URLs exactly from the data (do NOT invent URLs).\n"
                f"3. Start your response with the answer immediately.\n"
                f"4. Do NOT include any thinking, reasoning, or analysis process.\n"
                f"5. Keep your response under 200 words."
                f"6. For disputed quotes or claims, distinguish the exact words used, "
                f"their surrounding qualification, and the scope of the claim; do not "
                f"collapse them into a misleading yes/no answer.\n"
                f"7. When a source gives a Claim, Rating, or Context, report that finding "
                f"accurately. Do not say a speaker praised a specific subgroup unless the "
                f"source says the speaker was referring to that subgroup.\n"
                f"8. Cite the most relevant web sources as Markdown links using the URLs "
                f"provided in DATA."
            ),
        },
    ]

    resp = await llm.chat(messages)
    return resp["choices"][0]["message"].get("content", "No answer produced.")


async def _tool_loop(llm: LLMClient, query: str, session_id: str, cfg, prioritize_kb: bool = False):
    """Run the agent tool loop, yielding SSE events."""
    from deep_research.prompts import (
        KB_SEARCH_TOOL_DEFINITION,
        SYSTEM_PROMPT_KB_FIRST,
        SYSTEM_PROMPT_WEB_FIRST,
        TOOL_DEFINITIONS,
    )
    from deep_research.tools.kb_search import kb_search
    from deep_research.tools.scrape import scrape_and_extract, scrape_page
    from deep_research.tools.search import web_search

    # kb_search is only offered as a tool when a KBDatabase is actually up —
    # see kb_routes.init_kb's best-effort connection.
    tools = TOOL_DEFINITIONS + ([KB_SEARCH_TOOL_DEFINITION] if kb_routes.kb_db else [])
    system_prompt = SYSTEM_PROMPT_KB_FIRST if (prioritize_kb and kb_routes.kb_db) else SYSTEM_PROMPT_WEB_FIRST

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    for step in range(cfg.agent.max_steps):
        yield {"event": "status", "data": json.dumps({"step": "thinking", "detail": f"Step {step + 1}"})}

        resp = await llm.chat(messages, tools=tools)
        msg = resp["choices"][0]["message"]

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            yield {"event": "answer", "data": msg.get("content", "")}
            return

        assistant_msg = {"role": "assistant", "content": msg.get("content")}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        for tc in tool_calls:
            tool_name = tc["function"]["name"]
            try:
                tool_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                tool_args = {}

            tool_call_id = tc.get("id", tool_name)
            yield {"event": "tool", "data": json.dumps({"tool": tool_name, "args": tool_args})}

            # Execute tool
            result = ""
            try:
                if tool_name == "web_search":
                    results = await web_search(tool_args["query"], cfg)
                    if results:
                        result = "\n".join(f"**{r.title}**\n{r.url}\n{r.snippet}\n" for r in results)
                    else:
                        result = "No results found."
                elif tool_name == "kb_search":
                    if kb_routes.kb_db:
                        result = await kb_search(tool_args["query"], kb_routes.kb_db, cfg)
                    else:
                        result = "Local knowledge base is not available."
                elif tool_name == "scrape_webpage":
                    page = await scrape_page(tool_args["url"], cfg)
                    if page.structured_data and page.structured_data.get("products"):
                        await db.save_scraped_page(
                            session_id, page.url, page.title,
                            page.text_content, page.structured_data,
                        )
                        result = f"Title: {page.title}\n\n{page.text_content}"
                    else:
                        result = f"Title: {page.title}\n\n{page.text_content}"
                elif tool_name == "finish":
                    yield {"event": "answer", "data": tool_args.get("answer", "")}
                    return
            except Exception as e:
                result = f"Error: {e}"

            messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": result})

    # Max steps
    yield {"event": "answer", "data": "Max steps reached. Could not complete research."}


class SPAStaticFiles(StaticFiles):
    """StaticFiles serves index.html for `/` but raises a 404 on any other
    client-side route (e.g. /topics, /history) since no such file exists on
    disk -- breaking direct navigation and page refresh in the Vue Router SPA.
    Falls back to index.html on that 404 so the client-side router can take
    over, while still serving real assets (JS/CSS/images) normally."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                return await super().get_response("index.html", scope)
            raise


# Mount static frontend (built Vue app)
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", SPAStaticFiles(directory=str(frontend_dist), html=True), name="frontend")
