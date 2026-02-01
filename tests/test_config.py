import os
import tempfile
from pathlib import Path

import pytest
import yaml

from memory.config import (
    EmbeddingConfig,
    EnrichmentConfig,
    MemoryConfig,
    get_memory_home,
    load_config,
)


def test_default_config_has_correct_defaults():
    """Test that default config has expected default values."""
    config = MemoryConfig()

    assert config.embedding.provider == "ollama"
    assert config.embedding.model == "nomic-embed-text"
    assert config.embedding.base_url == "http://localhost:11434"
    assert config.embedding.api_key is None

    assert config.enrichment.provider == "none"
    assert config.enrichment.model is None
    assert config.enrichment.base_url is None
    assert config.enrichment.api_key is None


def test_load_config_with_all_fields():
    """Test loading config from YAML with all fields populated."""
    config_data = {
        "embedding": {
            "provider": "openai",
            "model": "text-embedding-3-small",
            "base_url": "https://api.openai.com/v1",
            "api_key": "openai-key",
        },
        "enrichment": {
            "provider": "openrouter",
            "model": "anthropic/claude-3.5-sonnet",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "openrouter-key",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        config = load_config(config_path)

        assert config.embedding.provider == "openai"
        assert config.embedding.model == "text-embedding-3-small"
        assert config.embedding.base_url == "https://api.openai.com/v1"
        assert config.embedding.api_key == "openai-key"

        assert config.enrichment.provider == "openrouter"
        assert config.enrichment.model == "anthropic/claude-3.5-sonnet"
        assert config.enrichment.base_url == "https://openrouter.ai/api/v1"
        assert config.enrichment.api_key == "openrouter-key"
    finally:
        os.unlink(config_path)


def test_load_config_missing_file_returns_defaults():
    """Test that loading a non-existent file returns default config."""
    config = load_config("/nonexistent/path/to/config.yaml")

    assert config.embedding.provider == "ollama"
    assert config.embedding.model == "nomic-embed-text"
    assert config.enrichment.provider == "none"


def test_load_config_partial_fields():
    """Test loading config with only some fields specified."""
    config_data = {
        "embedding": {
            "provider": "ollama",
            "model": "nomic-embed-text",
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    try:
        config = load_config(config_path)

        assert config.embedding.provider == "ollama"
        assert config.embedding.model == "nomic-embed-text"
        assert config.embedding.base_url == "http://localhost:11434"
        assert config.embedding.api_key is None

        # Enrichment should have defaults
        assert config.enrichment.provider == "none"
        assert config.enrichment.model is None
    finally:
        os.unlink(config_path)


def test_load_config_empty_file():
    """Test loading an empty YAML file returns defaults."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")
        config_path = f.name

    try:
        config = load_config(config_path)

        assert config.embedding.provider == "ollama"
        assert config.embedding.model == "nomic-embed-text"
        assert config.enrichment.provider == "none"
    finally:
        os.unlink(config_path)


def test_get_memory_home_defaults_to_home_directory():
    """Test that get_memory_home defaults to ~/.memory."""
    # Temporarily remove MEMORY_HOME if it exists
    old_value = os.environ.get("MEMORY_HOME")
    if "MEMORY_HOME" in os.environ:
        del os.environ["MEMORY_HOME"]

    try:
        memory_home = get_memory_home()
        expected = os.path.join(os.path.expanduser("~"), ".memory")
        assert memory_home == expected
    finally:
        if old_value is not None:
            os.environ["MEMORY_HOME"] = old_value


def test_get_memory_home_respects_env_var():
    """Test that get_memory_home respects MEMORY_HOME env var."""
    custom_path = "/custom/memory/path"
    old_value = os.environ.get("MEMORY_HOME")

    try:
        os.environ["MEMORY_HOME"] = custom_path
        memory_home = get_memory_home()
        assert memory_home == custom_path
    finally:
        if old_value is not None:
            os.environ["MEMORY_HOME"] = old_value
        else:
            del os.environ["MEMORY_HOME"]
