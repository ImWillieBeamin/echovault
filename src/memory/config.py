import os
import stat
import sys
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class EmbeddingConfig:
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    base_url: Optional[str] = "http://localhost:11434"
    api_key: Optional[str] = None


@dataclass
class EnrichmentConfig:
    provider: str = "none"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class ContextConfig:
    semantic: str = "auto"
    topup_recent: bool = True


@dataclass
class MemoryConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    enrichment: EnrichmentConfig = field(default_factory=EnrichmentConfig)
    context: ContextConfig = field(default_factory=ContextConfig)


def get_memory_home() -> str:
    return os.environ.get("MEMORY_HOME", os.path.join(os.path.expanduser("~"), ".memory"))


# Environment variable names for API keys (more secure than config file)
ENV_OPENAI_KEY = "ECHOVAULT_OPENAI_KEY"
ENV_OPENROUTER_KEY = "ECHOVAULT_OPENROUTER_KEY"


def check_config_permissions(path: str) -> None:
    """Check if config file has secure permissions and warn if not.

    On Unix-like systems, warns if config file is readable by group or others.
    Does nothing on Windows or if file doesn't exist.

    Args:
        path: Path to the config file
    """
    if os.name == "nt":
        # Windows doesn't use Unix permissions
        return

    try:
        file_stat = os.stat(path)
    except (FileNotFoundError, OSError):
        return

    mode = file_stat.st_mode

    # Check if group or others can read (0o044 = r--r--)
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        print(
            f"Warning: Config file '{path}' is readable by group or others. "
            "Consider running: chmod 600 " + path,
            file=sys.stderr,
        )


def load_config(path: str) -> MemoryConfig:
    """Load configuration from YAML file with environment variable overrides.

    Environment variables take precedence over config file values for API keys:
    - ECHOVAULT_OPENAI_KEY: Overrides embedding.api_key
    - ECHOVAULT_OPENROUTER_KEY: Overrides enrichment.api_key

    Args:
        path: Path to the config YAML file

    Returns:
        MemoryConfig with values from file and environment
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        data = {}

    config = MemoryConfig()
    if "embedding" in data:
        e = data["embedding"]
        config.embedding = EmbeddingConfig(
            provider=e.get("provider", "ollama"),
            model=e.get("model", "nomic-embed-text"),
            base_url=e.get("base_url", "http://localhost:11434"),
            api_key=e.get("api_key"),
        )
    if "enrichment" in data:
        en = data["enrichment"]
        config.enrichment = EnrichmentConfig(
            provider=en.get("provider", "none"),
            model=en.get("model"),
            base_url=en.get("base_url"),
            api_key=en.get("api_key"),
        )
    if "context" in data:
        cx = data["context"]
        config.context = ContextConfig(
            semantic=cx.get("semantic", "auto"),
            topup_recent=cx.get("topup_recent", True),
        )

    # Apply environment variable overrides for API keys
    # Non-empty env var takes precedence over config file
    env_openai = os.environ.get(ENV_OPENAI_KEY, "").strip()
    if env_openai:
        config.embedding.api_key = env_openai

    env_openrouter = os.environ.get(ENV_OPENROUTER_KEY, "").strip()
    if env_openrouter:
        config.enrichment.api_key = env_openrouter

    return config
