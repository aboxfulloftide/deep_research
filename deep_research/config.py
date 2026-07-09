import os
from pathlib import Path

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    model: str = "llama3"
    api_key: str = "ollama"


class SearXNGConfig(BaseModel):
    url: str = "http://localhost:8888"


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
    embedding_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text:v1.5"
    claim_duplicate_threshold: float = 0.85
    # Verification budget (build order step 6, "Verification Policy and Budget"
    # in PLAN_KB_ARCHITECTURE.md) — bounds cost so verifying a claim can't fan
    # out into unbounded searches/scrapes/extraction passes.
    verification_max_web_searches: int = 2
    verification_max_sources_examined: int = 3
    verification_importance_threshold: float = 0.8
    # Report generation auto-detects the server's actual context window via
    # llama.cpp's /slots endpoint and batches (map-reduce) whatever doesn't
    # fit in one pass, rather than dropping content. This fallback is only
    # used if that detection fails (a non-llama.cpp backend, or /slots
    # disabled) — a deliberately conservative guess, not a tuning knob.
    report_context_fallback_tokens: int = 4096


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    searxng: SearXNGConfig = SearXNGConfig()
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
        "DEEP_RESEARCH_SEARXNG_URL": ("searxng", "url"),
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
        "DEEP_RESEARCH_KB_EMBEDDING_BASE_URL": ("kb", "embedding_base_url"),
        "DEEP_RESEARCH_KB_EMBEDDING_MODEL": ("kb", "embedding_model"),
        "DEEP_RESEARCH_KB_CLAIM_DUPLICATE_THRESHOLD": ("kb", "claim_duplicate_threshold"),
        "DEEP_RESEARCH_KB_VERIFICATION_MAX_WEB_SEARCHES": ("kb", "verification_max_web_searches"),
        "DEEP_RESEARCH_KB_VERIFICATION_MAX_SOURCES_EXAMINED": ("kb", "verification_max_sources_examined"),
        "DEEP_RESEARCH_KB_VERIFICATION_IMPORTANCE_THRESHOLD": ("kb", "verification_importance_threshold"),
        "DEEP_RESEARCH_KB_REPORT_CONTEXT_FALLBACK_TOKENS": ("kb", "report_context_fallback_tokens"),
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
