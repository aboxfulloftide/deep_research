import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from deep_research.agent import ResearchAgent, _analyze_products, _compact_product_list, _URL_RE
from deep_research.config import load_config
from deep_research.db import Database
from deep_research.llm import LLMClient
from deep_research.tools.scrape import scrape_page
from deep_research.tools.search import web_search
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


class SessionResponse(BaseModel):
    id: str
    title: str | None
    created_at: str
    updated_at: str


# --- API Routes ---

@app.get("/api/models")
async def list_models():
    """List available models from the Ollama server."""
    ollama_url = config.llm.base_url.replace("/v1", "")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"models": models, "default": config.llm.model}
    except Exception as e:
        return {"models": [], "default": config.llm.model, "error": str(e)}


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
    model = req.model or config.llm.model

    # Create LLM client with selected model
    cfg = config.model_copy()
    cfg.llm.model = model

    return EventSourceResponse(
        _stream_research(cfg, req.query, req.session_id, req.prioritize_kb),
        media_type="text/event-stream",
    )


async def _stream_research(cfg, query: str, session_id: str | None, prioritize_kb: bool = False):
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


async def _text_mode_answer(llm: LLMClient, query: str, gathered_data: str, cfg, prioritize_kb: bool = False) -> str:
    """Generate answer in text mode. This is the common fallback for models
    without tool-calling support, so prioritize_kb has to affect it too, not
    just the tool loop, to actually change behavior for most local models."""
    from deep_research.prompts import SYSTEM_PROMPT_NO_TOOLS
    from deep_research.tools.kb_search import kb_search

    if not gathered_data and prioritize_kb and kb_routes.kb_db:
        kb_result = await kb_search(query[:200], kb_routes.kb_db)
        if kb_result and not kb_result.startswith("No results found"):
            gathered_data = kb_result

    if not gathered_data:
        try:
            results = await web_search(query[:200], cfg)
            if results:
                gathered_data = "\n".join(
                    f"- {r.title}: {r.url}\n  {r.snippet}" for r in results
                )
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
                        result = await kb_search(tool_args["query"], kb_routes.kb_db)
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


# Mount static frontend (built Vue app)
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
