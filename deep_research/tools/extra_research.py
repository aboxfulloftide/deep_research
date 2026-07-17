"""Bounded multi-level web research for the interactive Extra mode.

This intentionally has hard caps. "Deep" should mean progressively better
evidence, not an unbounded number of browser requests or a prompt too large
for a practical local model to synthesize.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from deep_research.config import Config
from deep_research.llm import LLMClient
from deep_research.models import SearchResult
from deep_research.tools.scrape import scrape_page
from deep_research.tools.search import web_search

SOURCES_PER_QUERY = 1
MIN_EVIDENCE_QUALITY = 3
SOURCE_EXCERPT_CHARS = 3_000
FOLLOW_UP_QUERY_LIMIT = 2
INITIAL_QUERY_LIMIT = 2
GAP_CLOSING_QUERY_LIMIT = 1
SOURCE_ANALYSIS_CHARS = 2_500
OFFICIAL_HF_ORGANIZATIONS = {
    "qwen", "mistralai", "meta-llama", "google", "microsoft", "deepseek-ai",
    "nvidia", "ibm-granite", "tiiuae", "allenai", "cohereforai", "openai",
}
BAD_SCRAPE_MARKERS = ("found 2 products:", "found 1 product:", "did you describe any potential participant risks")


@dataclass
class ResearchSource:
    title: str
    url: str
    content: str
    level: int
    query: str
    full_content: str = ""
    source_kind: str = "secondary"
    quality_score: int = 0


@dataclass(frozen=True)
class EvidenceClaim:
    """A source-grounded fact that is allowed into the final synthesis."""

    statement: str
    quote: str
    source_title: str
    source_url: str
    source_kind: str
    confidence: float


@dataclass(frozen=True)
class ResearchFacet:
    """One question-specific evidence need, independent of any domain."""

    id: str
    question: str
    purpose: str
    capabilities: list[str] = field(default_factory=lambda: ["web"])
    search_query: str = ""


@dataclass(frozen=True)
class ResearchPlan:
    question: str
    ambiguities: list[str]
    facets: list[ResearchFacet]


@dataclass
class ResearchBundle:
    """An inspectable pre-synthesis research artifact."""

    plan: ResearchPlan
    sources: list[ResearchSource]
    collection_attempts: list[dict]
    coverage: dict
    assessments: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchBudget:
    """Bounded, user-configurable collection budget for any question."""

    max_facets: int = 4
    max_adapters_per_facet: int = 2
    max_sources: int = 10
    max_gap_rounds: int = 1


SOURCE_CAPABILITIES = {
    "web", "primary", "scholarly", "official_documentation", "repository", "news", "local_knowledge",
}


def _adapter_query(capability: str, question: str) -> str:
    """Route a generic evidence capability through the available web adapter.

    The adapter boundary is intentional: domain-specific APIs can replace
    these query constraints later without changing plans or coverage logic.
    """
    suffixes = {
        "primary": "primary source official",
        "scholarly": "site:arxiv.org OR site:openreview.net research paper",
        "official_documentation": "official documentation specifications",
        "repository": "site:github.com repository documentation",
        "news": "reputable news publication date",
        "local_knowledge": "",  # No local-KB adapter is registered yet; web remains transparent fallback.
        "web": "",
    }
    return f"{question} {suffixes.get(capability, '')}".strip()


def _is_html_result(result: SearchResult) -> bool:
    return not result.url.lower().split("?", 1)[0].endswith(".pdf")


def _title_key(title: str) -> str:
    """Identify syndicated copies that have different URLs but the same story."""
    article_title = re.split(r"\s+[|–—-]\s+", title, maxsplit=1)[0]
    return "title:" + re.sub(r"\W+", " ", article_title.lower()).strip()


def _model_family_key(title: str, url: str) -> str | None:
    """Collapse FP8/AWQ/GGUF repacks into the base model's evidence slot."""
    if (urlparse(url).hostname or "").lower().removeprefix("www.") != "huggingface.co":
        return None
    name = title.split("·", 1)[0].strip().lower().rsplit("/", 1)[-1]
    name = re.sub(r"[-_ ](?:fp\d+|awq|gptq|gguf|exl2|iq\d+|q\d+(?:_[a-z]+)*)$", "", name)
    return f"model-family:{re.sub(r'\s+', ' ', name)}" if name else None


def _canonical_source_key(url: str) -> str | None:
    """Recognize the arXiv abstract and HTML rendering as the same paper."""
    parsed = urlparse(url)
    if parsed.hostname and parsed.hostname.lower() == "arxiv.org":
        match = re.match(r"/(?:abs|html)/(\d{4}\.\d{4,5})", parsed.path)
        if match:
            return f"arxiv:{match.group(1)}"
    return None


def _usable_scrape(content: str) -> bool:
    normalised = _normalise(content)
    return len(normalised) >= 240 and not any(marker in normalised for marker in BAD_SCRAPE_MARKERS)


def classify_source(url: str) -> tuple[str, int]:
    """Give primary technical material priority without silently excluding corroboration."""
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if host in {"arxiv.org", "openreview.net"}:
        return "paper", 5
    if host == "huggingface.co":
        path_parts = [part for part in urlparse(url).path.split("/") if part]
        first_path = path_parts[0] if path_parts else ""
        if any(part in {"discussions", "community"} for part in path_parts):
            return "community", 1
        if first_path.lower() in OFFICIAL_HF_ORGANIZATIONS:
            return "primary", 5
        if first_path not in {"blog", "docs", "spaces", "collections"}:
            return "technical_reference", 3
        return "technical_reference", 3
    if host in {"qwenlm.github.io", "mistral.ai", "docs.mistral.ai", "llama.com", "ai.meta.com"}:
        return "primary", 5
    if host in {"github.com", "paperswithcode.com", "swebench.com", "livecodebench.github.io"}:
        return "technical_reference", 4
    if host.startswith(("docs.", "developer.")) or any(marker in host for marker in ("benchmark", "lmarena", "lmsys")):
        return "technical_reference", 4
    if host in {"reddit.com", "news.ycombinator.com", "medium.com"}:
        return "community", 1
    return "secondary", 2


async def collect_sources(
    queries: list[str], config: Config, level: int, seen_urls: set[str], *, sources_per_query: int | None = None,
) -> list[ResearchSource]:
    """Search each query and read new HTML sources from it.

    The first level combines the original wording with planned subquestions.
    Follow-up levels use focused evidence branches; the final level uses one
    gap-closing query. This keeps the whole run bounded while diversifying
    evidence beyond the user's exact wording.
    """
    selections: list[tuple[str, SearchResult]] = []
    pending_urls: set[str] = set()
    pending_titles: set[str] = set()
    per_query_limit = sources_per_query if sources_per_query is not None else (2 if len(queries) == 1 else SOURCES_PER_QUERY)
    for query in queries:
        try:
            results = await web_search(query, config)
        except Exception:
            continue
        added = 0
        ranked_results = sorted(results, key=lambda result: classify_source(result.url)[1], reverse=True)
        for result in ranked_results:
            _, quality_score = classify_source(result.url)
            if quality_score < MIN_EVIDENCE_QUALITY:
                continue
            title_key = _title_key(result.title)
            family_key = _model_family_key(result.title, result.url)
            canonical_key = _canonical_source_key(result.url)
            if (
                result.url in seen_urls
                or title_key in seen_urls
                or (family_key is not None and family_key in seen_urls)
                or (canonical_key is not None and canonical_key in seen_urls)
                or result.url in pending_urls
                or title_key in pending_titles
                or (canonical_key is not None and canonical_key in pending_titles)
                or not _is_html_result(result)
            ):
                continue
            selections.append((query, result))
            pending_urls.add(result.url)
            pending_titles.add(title_key)
            if family_key is not None:
                pending_titles.add(family_key)
            if canonical_key is not None:
                pending_titles.add(canonical_key)
            added += 1
            if added >= per_query_limit:
                break

    async def read(query: str, result: SearchResult) -> ResearchSource:
        try:
            page = await scrape_page(result.url, config)
            content = page.text_content or result.snippet
            title = page.title or result.title
        except Exception:
            content = result.snippet
            title = result.title
        source_kind, quality_score = classify_source(result.url)
        return ResearchSource(
            title=title,
            url=result.url,
            content=content[:SOURCE_EXCERPT_CHARS],
            level=level,
            query=query,
            full_content=content,
            source_kind=source_kind,
            quality_score=quality_score,
        )

    sources = [source for source in await asyncio.gather(*(read(query, result) for query, result in selections)) if _usable_scrape(source.content)]
    for source in sources:
        seen_urls.add(source.url)
        seen_urls.add(_title_key(source.title))
        family_key = _model_family_key(source.title, source.url)
        if family_key is not None:
            seen_urls.add(family_key)
        canonical_key = _canonical_source_key(source.url)
        if canonical_key is not None:
            seen_urls.add(canonical_key)
    return sources


def source_context(sources: list[ResearchSource], *, per_source_chars: int = SOURCE_EXCERPT_CHARS) -> str:
    return "\n\n".join(
        f"=== Level {source.level} source: {source.title} ({source.url}) ===\n"
        f"Found while searching: {source.query}\n"
        f"{source.content[:per_source_chars]}"
        for source in sources
    )


def _parse_queries(content: str, original_query: str, limit: int = FOLLOW_UP_QUERY_LIMIT) -> list[str]:
    queries: list[str] = []
    seen = {original_query.strip().lower()}
    for line in content.splitlines():
        candidate = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if len(candidate) < 8 or len(candidate) > 220:
            continue
        normalized = candidate.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        queries.append(candidate)
        if len(queries) == limit:
            break
    return queries


def _fallback_research_plan(question: str) -> ResearchPlan:
    """Safe generic plan when the local planner cannot return structured JSON."""
    keywords = " ".join(
        word for word in re.findall(r"[a-zA-Z0-9]+", question.lower())
        if word not in {"research", "deeply", "best", "that", "with", "from", "this", "should", "would", "could", "usable", "main", "also", "used", "less", "than", "under"}
    )[:180]
    return ResearchPlan(question, [], [
        ResearchFacet("core", question, "Direct evidence that answers the central question.", ["web", "primary"], f"{keywords} official specifications"),
        ResearchFacet("constraints", "Definitions, limits, and assumptions relevant to the question.", "Definitions, limits, and assumptions.", ["official_documentation", "web"], f"{keywords} requirements limitations"),
        ResearchFacet("corroboration", "Independent corroboration relevant to the question.", "Independent or primary corroboration.", ["scholarly", "repository"], f"{keywords} independent benchmark evidence"),
    ])


async def plan_research(llm: LLMClient, question: str) -> ResearchPlan:
    """Turn any research question into evidence needs before searching.

    The plan is deliberately domain-neutral: it describes what must be shown,
    rather than assuming that all questions need model cards, papers, prices,
    or a particular source type.
    """
    messages = [
        {"role": "system", "content": (
            "/no_think\nPlan research for any question. Return ONLY JSON: "
            '{"ambiguities":["..."],"facets":[{"id":"short_slug","question":"evidence need","search_query":"short search-engine query, not a restatement of the user question","purpose":"why this evidence matters","capabilities":["web","primary"]}]}. '
            "Return 2-4 complementary facets. Facets must cover the central answer, constraints/definitions where relevant, "
            "and corroboration or tradeoffs where relevant. Do not assume a domain or answer the question."
        )},
        {"role": "user", "content": question},
    ]
    try:
        response = await asyncio.wait_for(llm.chat(messages), timeout=25)
        payload = json.loads(response["choices"][0]["message"].get("content", "").strip())
        rows = payload.get("facets") if isinstance(payload, dict) else None
        facets = []
        if isinstance(rows, list):
            for index, row in enumerate(rows[:4], 1):
                if not isinstance(row, dict):
                    continue
                facet_question = str(row.get("question") or "").strip()
                search_query = str(row.get("search_query") or "").strip()
                purpose = str(row.get("purpose") or "").strip()
                facet_id = re.sub(r"[^a-z0-9_-]+", "-", str(row.get("id") or f"facet-{index}").lower()).strip("-")
                raw_capabilities = row.get("capabilities", ["web"])
                capabilities = [str(capability) for capability in raw_capabilities if str(capability) in SOURCE_CAPABILITIES] if isinstance(raw_capabilities, list) else []
                if len(facet_question) >= 12 and len(purpose) >= 8 and facet_id and _normalise(search_query) != _normalise(question) and 12 <= len(search_query) <= 220:
                    facets.append(ResearchFacet(facet_id, facet_question[:260], purpose[:260], capabilities or ["web"], search_query))
        if len(facets) >= 2:
            ambiguities = payload.get("ambiguities", [])
            return ResearchPlan(question, [str(item)[:240] for item in ambiguities if str(item).strip()][:4], facets)
    except Exception:
        pass
    return _fallback_research_plan(question)


def _source_assessment(source: ResearchSource, facet: ResearchFacet) -> dict:
    """Deterministic source-fitness signal; semantic scoring can refine it later."""
    terms = {term for term in re.findall(r"[a-z0-9]{4,}", facet.question.lower()) if term not in {"with", "that", "this", "from", "under", "what", "which"}}
    haystack = f"{source.title} {source.content}".lower()
    overlap = sum(term in haystack for term in terms) / max(1, len(terms))
    directness = round(min(1.0, 0.25 + overlap * 1.5), 2)
    return {
        "url": source.url, "facet_id": facet.id, "authority": source.quality_score,
        "directness": directness, "independent": True,
        "extractable": len(_normalise(source.content)) >= 240,
        "accepted": source.quality_score >= MIN_EVIDENCE_QUALITY and directness >= 0.45,
    }


def _coverage_for(plan: ResearchPlan, sources: list[ResearchSource], assessments: list[dict] | None = None) -> dict:
    assessments = assessments or []
    facets = []
    for facet in plan.facets:
        matching = [source for source in sources if source.query == facet.question]
        facet_assessments = [assessment for assessment in assessments if assessment["facet_id"] == facet.id]
        facets.append({
            "id": facet.id, "purpose": facet.purpose, "question": facet.question, "search_query": facet.search_query,
            "source_count": len(matching), "best_quality": max((source.quality_score for source in matching), default=0),
            "best_directness": max((assessment["directness"] for assessment in facet_assessments), default=0),
            "source_urls": [source.url for source in matching],
        })
    covered = [facet["id"] for facet in facets if facet["source_count"] and facet["best_quality"] >= MIN_EVIDENCE_QUALITY and facet["best_directness"] >= 0.45]
    return {
        "facets": facets, "covered_facet_ids": covered,
        "missing_facet_ids": [facet.id for facet in plan.facets if facet.id not in covered],
        "has_high_authority_source": any(source.quality_score >= 5 for source in sources),
    }


async def collect_research_bundle(llm: LLMClient, question: str, config: Config, budget: ResearchBudget | None = None) -> ResearchBundle:
    """Collect diverse evidence by research facet, then close uncovered facets.

    This is the common source-collection stage used before any analysis. Its
    output is intentionally preserved so users and evaluators can reject a
    weak bundle rather than mistaking a polished synthesis for research.
    """
    budget = budget or ResearchBudget()
    plan = await plan_research(llm, question)
    plan = ResearchPlan(plan.question, plan.ambiguities, plan.facets[:budget.max_facets])
    seen_urls: set[str] = set()
    sources: list[ResearchSource] = []
    attempts: list[dict] = []
    assessments: list[dict] = []

    async def collect_for(facet: ResearchFacet, level: int, capability: str, query: str) -> None:
        if len(sources) >= budget.max_sources:
            return
        found = await collect_sources([query], config, level, seen_urls, sources_per_query=1)
        found = found[:max(0, budget.max_sources - len(sources))]
        sources.extend(found)
        # Attribute retrieved evidence to its facet even if a recovery query
        # supplied the wording, so coverage remains inspectable.
        for source in found:
            source.query = facet.question
            assessments.append(_source_assessment(source, facet))
        attempts.append({"level": level, "facet_id": facet.id, "adapter": capability, "queries": [query], "source_count": len(found), "source_urls": [source.url for source in found]})

    for facet in plan.facets:
        for capability in facet.capabilities[:budget.max_adapters_per_facet]:
            await collect_for(facet, 1, capability, _adapter_query(capability, facet.search_query))

    coverage = _coverage_for(plan, sources, assessments)
    for _ in range(budget.max_gap_rounds):
        missing = [facet for facet in plan.facets if facet.id in coverage["missing_facet_ids"]]
        if not missing:
            break
        for facet in missing:
            recovery = "primary" if "primary" not in facet.capabilities else "web"
            await collect_for(facet, 2, recovery, _adapter_query(recovery, f"{facet.search_query} independent corroboration"))
        coverage = _coverage_for(plan, sources, assessments)

    return ResearchBundle(plan, sources, attempts, coverage, assessments)


async def derive_starting_queries(llm: LLMClient, original_query: str) -> list[str]:
    """Plan two complementary searches before querying the web.

    The original wording remains a search query. These additions turn a broad
    question into smaller factual branches instead of betting the first round
    on whichever exact words the user happened to type.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "/no_think\nYou plan a concise web-research search set. Return exactly two search queries, one per line. "
                "First identify ambiguous constraints that must not be assumed (for example total RAM versus VRAM). "
                "Then break the question into complementary factual branches: official specifications, primary data, "
                "independently reproducible benchmarks, or decision criteria. Keep each query independently searchable. "
                "Do not answer the question."
            ),
        },
        {"role": "user", "content": f"Research question: {original_query}"},
    ]
    try:
        response = await asyncio.wait_for(llm.chat(messages), timeout=20)
        queries = _parse_queries(
            response["choices"][0]["message"].get("content", ""), original_query, INITIAL_QUERY_LIMIT,
        )
        if len(queries) == INITIAL_QUERY_LIMIT:
            return queries
    except Exception:
        pass
    # Never silently steer arbitrary research toward one model vendor. This
    # fallback retains the user's actual constraints while asking for sources
    # that can support a later evidence gate.
    return [
        f"{original_query} official model card specifications context hardware requirements",
        f"{original_query} benchmark owner methodology results",
    ]


async def derive_follow_up_queries(
    llm: LLMClient, original_query: str, evidence: list[ResearchSource], level: int,
) -> list[str]:
    """Use the evidence itself to choose the next, narrower research branch."""
    evidence_brief = source_context(evidence[-3:], per_source_chars=900)
    messages = [
        {
            "role": "system",
            "content": (
                "/no_think\nYou plan web research. Return exactly two concise search queries, one per line. "
                "Use the evidence to find an official model card, technical paper, benchmark owner, or independent "
                "corroboration for a specific unresolved tradeoff. Avoid generic reviews and search-result summaries. "
                "Do not answer the original question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Original question: {original_query}\n"
                f"This is research level {level}; choose the next level's queries from this evidence:\n\n"
                f"{evidence_brief}"
            ),
        },
    ]
    try:
        # Query planning is helpful but must never hold up the whole research
        # run. Smaller local models can spend a long time in reasoning mode
        # for this tiny task, so fall back promptly to evidence-derived
        # searches instead.
        response = await asyncio.wait_for(llm.chat(messages), timeout=20)
        content = response["choices"][0]["message"].get("content", "")
        queries = _parse_queries(content, original_query, FOLLOW_UP_QUERY_LIMIT)
    except Exception:
        queries = []

    # A reliable fallback keeps Extra mode useful with smaller models that
    # cannot follow the query-planning format.
    if len(queries) < FOLLOW_UP_QUERY_LIMIT:
        anchor = next((source.title for source in reversed(evidence) if source.title), original_query)
        anchor = anchor[:120]
        fallbacks = [
            f"{anchor} official documentation technical details",
            f"{anchor} independent comparison limitations benchmarks",
        ]
        known = {query.lower() for query in queries} | {original_query.lower()}
        queries.extend(query for query in fallbacks if query.lower() not in known)
    return queries[:FOLLOW_UP_QUERY_LIMIT]


async def derive_gap_closing_query(
    llm: LLMClient, original_query: str, evidence: list[ResearchSource],
) -> list[str]:
    """Choose one final source that can corroborate the biggest remaining gap."""
    evidence_brief = source_context(evidence[-4:], per_source_chars=700)
    messages = [
        {
            "role": "system",
            "content": (
                "/no_think\nYou are closing a web-research evidence gap. Return exactly one concise search query. "
                "Prefer a primary source, authoritative dataset, or independent corroboration for the most important "
                "unresolved factual claim. Do not answer the question."
            ),
        },
        {
            "role": "user",
            "content": f"Original question: {original_query}\n\nEvidence so far:\n{evidence_brief}",
        },
    ]
    try:
        response = await asyncio.wait_for(llm.chat(messages), timeout=20)
        content = response["choices"][0]["message"].get("content", "")
        queries = _parse_queries(content, original_query, GAP_CLOSING_QUERY_LIMIT)
    except Exception:
        queries = []

    if not queries:
        anchor = next((source.title for source in reversed(evidence) if source.title), original_query)[:120]
        queries = [f"{anchor} primary source verification"]
    return queries[:GAP_CLOSING_QUERY_LIMIT]


async def analyze_sources_separately(
    llm: LLMClient, original_query: str, sources: list[ResearchSource],
) -> list[str]:
    """Create compact, source-attributed briefs before cross-source synthesis."""
    semaphore = asyncio.Semaphore(2)

    async def analyze(source: ResearchSource) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "/no_think\nAnalyze one research source for a later synthesis. Extract only facts relevant to the "
                    "original question, distinguish claims from evidence, note limitations or uncertainty, and do not "
                    "invent information. Keep it under 180 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Original question: {original_query}\n"
                    f"Source: {source.title} ({source.url})\n\n"
                    f"Source text:\n{source.content[:SOURCE_ANALYSIS_CHARS]}"
                ),
            },
        ]
        try:
            async with semaphore:
                response = await llm.chat(messages)
            content = response["choices"][0]["message"].get("content", "").strip()
        except Exception:
            content = ""
        if not content:
            content = source.content[:900]
        return f"=== {source.title} ({source.url}) ===\n{content}"

    return await asyncio.gather(*(analyze(source) for source in sources))


def analysis_context(analyses: list[str]) -> str:
    return "\n\n".join(analyses)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _parse_ledger(content: str, source: ResearchSource) -> list[EvidenceClaim]:
    """Accept only claims whose quoted evidence actually occurs in that source."""
    try:
        rows = json.loads(content.strip())
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(rows, list):
        return []
    source_text = _normalise(source.content)
    claims: list[EvidenceClaim] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        statement = str(row.get("statement") or "").strip()
        quote = str(row.get("quote") or "").strip()
        if len(statement) < 12 or len(quote) < 12 or _normalise(quote) not in source_text:
            continue
        try:
            confidence = float(row.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        claims.append(EvidenceClaim(
            statement=statement, quote=quote, source_title=source.title, source_url=source.url,
            source_kind=source.source_kind, confidence=max(0.0, min(confidence, 1.0)),
        ))
    return claims


async def build_claim_ledger(
    llm: LLMClient, original_query: str, sources: list[ResearchSource],
) -> list[EvidenceClaim]:
    """Extract auditable claims before synthesis; unsupported prose never enters the answer context."""
    semaphore = asyncio.Semaphore(2)

    async def extract(source: ResearchSource) -> list[EvidenceClaim]:
        if source.quality_score < MIN_EVIDENCE_QUALITY:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "/no_think\nExtract evidence for a research claim ledger. Return ONLY a JSON array. Each item must be "
                    '{"statement":"atomic fact relevant to the question","quote":"verbatim quote from the supplied source, max 35 words","confidence":0.0}. '
                    "Do not infer, estimate, combine facts, or include a claim unless the quote directly supports it. "
                    "For numerical claims, preserve the exact units and qualifiers."
                ),
            },
            {"role": "user", "content": f"Question: {original_query}\n\nSource text:\n{source.content[:SOURCE_ANALYSIS_CHARS]}"},
        ]
        try:
            async with semaphore:
                response = await llm.chat(messages)
            return _parse_ledger(response["choices"][0]["message"].get("content", ""), source)
        except Exception:
            return []

    per_source = await asyncio.gather(*(extract(source) for source in sources))
    return [claim for claims in per_source for claim in claims]


def claim_ledger_context(claims: list[EvidenceClaim]) -> str:
    return "\n".join(
        f"- {claim.statement}\n  Evidence: {claim.quote}\n  Source: [{claim.source_title}]({claim.source_url}) "
        f"(evidence tier: {claim.source_kind}; confidence {claim.confidence:.2f})"
        for claim in claims
    )


def has_authoritative_source(sources: list[ResearchSource]) -> bool:
    """A decision memo needs at least one model card or paper, not only commentary."""
    return any(source.source_kind in {"primary", "paper"} for source in sources)
