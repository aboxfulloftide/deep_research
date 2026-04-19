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


class Config(BaseModel):
    llm: LLMConfig = LLMConfig()
    searxng: SearXNGConfig = SearXNGConfig()
    scraping: ScrapingConfig = ScrapingConfig()
    agent: AgentConfig = AgentConfig()
    db: DBConfig = DBConfig()
    web: WebConfig = WebConfig()

    @property
    def db_path(self) -> Path:
        return Path(self.db.path).expanduser()


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
    }
    data = config.model_dump()
    for env_var, (section, key) in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            # Coerce to int for integer fields
            field = config.__class__.model_fields[section].annotation
            sub_field = field.model_fields[key]
            if sub_field.annotation is int:
                val = int(val)
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
