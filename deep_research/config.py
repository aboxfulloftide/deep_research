import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    # base_url/model/api_key are the *currently active* triple that LLMClient
    # actually reads. backend + the preset fields below exist so the CLI/web
    # layer can switch the agent between Ollama and llama.cpp (see
    # deep_research/model_backends.py) without touching LLMClient at all --
    # switching just means resolving a preset's base_url/api_key into these
    # three fields before constructing an LLMClient.
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3"
    api_key: str = "ollama"
    backend: str = "llama_cpp"  # Interactive chat is served by the managed llama.cpp runtime.
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"
    llama_cpp_base_url: str = "http://localhost:8080/v1"
    llama_cpp_api_key: str = "not-needed"


class SearXNGConfig(BaseModel):
    url: str = "http://localhost:8888"
    # Minimum gap enforced between consecutive SearXNG calls (see
    # search.py's _throttle_searxng), even across concurrent verification
    # tasks -- duckduckgo/mojeek have no documented rate limit but
    # visibly start CAPTCHA'ing/429ing under rapid-fire bursts. This isn't a
    # workaround for that (see the CAPTCHA-solving conversation) -- it's
    # just not hitting them faster than they're willing to serve.
    min_interval_seconds: float = 1.5


class BraveConfig(BaseModel):
    # SearXNG's built-in "brave" engine scrapes search.brave.com unauthenticated
    # (no api key support) and gets rate-limited by Brave's anti-bot heuristics
    # under sustained query volume. When that happens, web_search() falls back
    # to Brave's real Search API using this key, instead of paying for the
    # metered official API on every query up front.
    # Primary key: the free-tier subscription -- observed limits ~2000
    # queries/month and 1 request/second.
    api_key: str = ""
    # Paid backup key (separate Brave subscription: up to 50 req/s, budgeted
    # ~3000 searches/month). Only spent once the primary key actually fails
    # (monthly quota exhausted, persistent 429, subscription error) -- never
    # while the free key is still answering, and never on a merely-empty
    # result. See search.py's _brave_search_layered.
    fallback_api_key: str = ""


class TavilyConfig(BaseModel):
    # General-purpose fallback (not tied to one SearXNG engine like brave is)
    # -- used when SearXNG's combined results are thin, regardless of which
    # engine(s) failed (google cse, startpage, or a fully dead SearXNG). Both
    # this is only invoked when the primary SearXNG + Brave + Serper layer is
    # still thin, not on every query.
    api_key: str = ""


class SerperConfig(BaseModel):
    # Primary provider alongside SearXNG/Bing and Brave. The current account
    # allowance is large enough for routine use; Tavily remains the fallback
    # when the combined primary results are still thin.
    api_key: str = ""


class WikipediaConfig(BaseModel):
    # Wikimedia's REST search API is free/no-key, but its edge WAF 403s
    # requests whose User-Agent doesn't follow their identification policy
    # (https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy --
    # needs real contact info, not a generic client string). Kept out of
    # source (an email is PII) the same way API keys are -- set via env var.
    contact: str = ""


class ScrapingConfig(BaseModel):
    timeout: int = 15
    max_content_length: int = 8000


class AgentConfig(BaseModel):
    max_steps: int = 20


class DBConfig(BaseModel):
    path: str = "~/.local/share/deep_research/research.db"


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class KBConfig(BaseModel):
    """Knowledge base storage — deliberately separate from chat session storage
    (which stays on SQLite; see deep_research/db.py). PostgreSQL per build order
    step 5, migrated once the schema had been exercised end-to-end on SQLite."""
    postgres_dsn: str = "postgresql://deep_research:deep_research@localhost:5432/deep_research_kb"
    snapshot_dir: str = "~/.local/share/deep_research/kb_snapshots"
    # Extraction/resolution use their own model config, independent of the
    # interactive research agent's `llm` section (MODELS.md: per-role model
    # assignment, not one model for everything). Defaults match what the step-0
    # spike validated: local llama.cpp for extraction, Ollama for embeddings.
    extraction_llm_base_url: str = "http://localhost:8080/v1"
    extraction_llm_model: str = ""  # empty = auto-detect from the llama.cpp server
    # Optional verifier role. Leaving these unset preserves the current
    # single-model deployment while allowing 30B extraction / 14B checking.
    verification_llm_base_url: str | None = None
    verification_llm_model: str = ""
    embedding_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text:v1.5"
    claim_duplicate_threshold: float = 0.85
    # Verification budget (build order step 6, "Verification Policy and Budget"
    # in PLAN_KB_ARCHITECTURE.md) — bounds cost so verifying a claim can't fan
    # out into unbounded searches/scrapes/extraction passes.
    verification_max_web_searches: int = 2
    # Shared cap for one whole unattended sweep. Per-claim limits alone can
    # turn a large backlog into hundreds of external calls overnight.
    verification_run_max_web_searches: int = 50
    verification_max_sources_examined: int = 3
    verification_importance_threshold: float = 0.8
    # A web-fallback source only needs extraction run on the handful of
    # chunks actually relevant to the claim being checked, not the whole
    # page -- extracting a whole page is the same expensive full-source
    # extraction pass used for deliberate ingestion, and on a long page can
    # dump hundreds to (observed) 1000+ tangential claims into the KB just
    # to check one fact. Chunks are ranked by embedding similarity to the
    # claim first; only the top N get extracted.
    verification_max_chunks_per_page: int = 3
    # How many claims to verify concurrently in a batch (nightly sweep, or
    # verifying every unverified claim from one source). Should match
    # llama-server's --parallel so a batch actually uses both slots instead
    # of leaving one idle -- see HARDWARE.md's "Verification bottleneck
    # measurement" section for the concurrency test this was validated with.
    verification_concurrency: int = 2
    # Counter-view checks can each make several local-model calls. Drain a
    # bounded set of supported, never-checked claims after the nightly sweep
    # rather than letting balance checks monopolize the shared worker.
    nightly_counter_evidence_limit: int = 50
    # Report generation auto-detects the server's actual context window via
    # llama.cpp's /slots endpoint and batches (map-reduce) whatever doesn't
    # fit in one pass, rather than dropping content. This fallback is only
    # used if that detection fails (a non-llama.cpp backend, or /slots
    # disabled) — a deliberately conservative guess, not a tuning knob.
    report_context_fallback_tokens: int = 4096
    playlist_max_videos_per_run: int = 3


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    searxng: SearXNGConfig = SearXNGConfig()
    brave: BraveConfig = BraveConfig()
    tavily: TavilyConfig = TavilyConfig()
    serper: SerperConfig = SerperConfig()
    wikipedia: WikipediaConfig = WikipediaConfig()
    scraping: ScrapingConfig = ScrapingConfig()
    agent: AgentConfig = AgentConfig()
    db: DBConfig = DBConfig()
    web: WebConfig = WebConfig()
    kb: KBConfig = KBConfig()

    @property
    def db_path(self) -> Path:
        return Path(self.db.path).expanduser()

    @property
    def kb_snapshot_dir(self) -> Path:
        return Path(self.kb.snapshot_dir).expanduser()


def _apply_env_overrides(config: Config) -> Config:
    """Override config values with DEEP_RESEARCH_* environment variables."""
    env_map = {
        "DEEP_RESEARCH_LLM_BASE_URL": ("llm", "base_url"),
        "DEEP_RESEARCH_LLM_MODEL": ("llm", "model"),
        "DEEP_RESEARCH_LLM_API_KEY": ("llm", "api_key"),
        "DEEP_RESEARCH_LLM_BACKEND": ("llm", "backend"),
        "DEEP_RESEARCH_LLM_OLLAMA_BASE_URL": ("llm", "ollama_base_url"),
        "DEEP_RESEARCH_LLM_OLLAMA_API_KEY": ("llm", "ollama_api_key"),
        "DEEP_RESEARCH_LLM_LLAMA_CPP_BASE_URL": ("llm", "llama_cpp_base_url"),
        "DEEP_RESEARCH_LLM_LLAMA_CPP_API_KEY": ("llm", "llama_cpp_api_key"),
        "DEEP_RESEARCH_SEARXNG_URL": ("searxng", "url"),
        "DEEP_RESEARCH_SEARXNG_MIN_INTERVAL_SECONDS": ("searxng", "min_interval_seconds"),
        "DEEP_RESEARCH_BRAVE_API_KEY": ("brave", "api_key"),
        "DEEP_RESEARCH_BRAVE_FALLBACK_API_KEY": ("brave", "fallback_api_key"),
        "DEEP_RESEARCH_TAVILY_API_KEY": ("tavily", "api_key"),
        "DEEP_RESEARCH_SERPER_API_KEY": ("serper", "api_key"),
        "DEEP_RESEARCH_WIKIPEDIA_CONTACT": ("wikipedia", "contact"),
        "DEEP_RESEARCH_SCRAPING_TIMEOUT": ("scraping", "timeout"),
        "DEEP_RESEARCH_SCRAPING_MAX_CONTENT_LENGTH": ("scraping", "max_content_length"),
        "DEEP_RESEARCH_AGENT_MAX_STEPS": ("agent", "max_steps"),
        "DEEP_RESEARCH_DB_PATH": ("db", "path"),
        "DEEP_RESEARCH_WEB_HOST": ("web", "host"),
        "DEEP_RESEARCH_WEB_PORT": ("web", "port"),
        "DEEP_RESEARCH_KB_POSTGRES_DSN": ("kb", "postgres_dsn"),
        "DEEP_RESEARCH_KB_SNAPSHOT_DIR": ("kb", "snapshot_dir"),
        "DEEP_RESEARCH_KB_EXTRACTION_LLM_BASE_URL": ("kb", "extraction_llm_base_url"),
        "DEEP_RESEARCH_KB_EXTRACTION_LLM_MODEL": ("kb", "extraction_llm_model"),
        "DEEP_RESEARCH_KB_VERIFICATION_LLM_BASE_URL": ("kb", "verification_llm_base_url"),
        "DEEP_RESEARCH_KB_VERIFICATION_LLM_MODEL": ("kb", "verification_llm_model"),
        "DEEP_RESEARCH_KB_EMBEDDING_BASE_URL": ("kb", "embedding_base_url"),
        "DEEP_RESEARCH_KB_EMBEDDING_MODEL": ("kb", "embedding_model"),
        "DEEP_RESEARCH_KB_CLAIM_DUPLICATE_THRESHOLD": ("kb", "claim_duplicate_threshold"),
        "DEEP_RESEARCH_KB_VERIFICATION_MAX_WEB_SEARCHES": ("kb", "verification_max_web_searches"),
        "DEEP_RESEARCH_KB_VERIFICATION_MAX_SOURCES_EXAMINED": ("kb", "verification_max_sources_examined"),
        "DEEP_RESEARCH_KB_VERIFICATION_IMPORTANCE_THRESHOLD": ("kb", "verification_importance_threshold"),
        "DEEP_RESEARCH_KB_VERIFICATION_RUN_MAX_WEB_SEARCHES": ("kb", "verification_run_max_web_searches"),
        "DEEP_RESEARCH_KB_VERIFICATION_CONCURRENCY": ("kb", "verification_concurrency"),
        "DEEP_RESEARCH_KB_NIGHTLY_COUNTER_EVIDENCE_LIMIT": ("kb", "nightly_counter_evidence_limit"),
        "DEEP_RESEARCH_KB_VERIFICATION_MAX_CHUNKS_PER_PAGE": ("kb", "verification_max_chunks_per_page"),
        "DEEP_RESEARCH_KB_REPORT_CONTEXT_FALLBACK_TOKENS": ("kb", "report_context_fallback_tokens"),
        "DEEP_RESEARCH_KB_PLAYLIST_MAX_VIDEOS_PER_RUN": ("kb", "playlist_max_videos_per_run"),
    }
    data = config.model_dump()
    for env_var, (section, key) in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            # Coerce to the field's declared scalar type
            field = config.__class__.model_fields[section].annotation
            sub_field = field.model_fields[key]
            if sub_field.annotation is int:
                val = int(val)
            elif sub_field.annotation is float:
                val = float(val)
            data[section][key] = val
    return Config(**data)


def load_config(config_path: str | None = None) -> Config:
    """Load config from YAML file with env var overrides."""
    search_paths = []
    if config_path:
        search_paths.append(Path(config_path))
    # Check CWD, project root (relative to this file), and user config dir
    project_root = Path(__file__).parent.parent
    search_paths.extend([
        Path("config.yaml"),
        project_root / "config.yaml",
        Path.home() / ".config" / "deep_research" / "config.yaml",
    ])

    data = {}
    for path in search_paths:
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            break

    config = Config(**data)
    return _apply_env_overrides(config)
