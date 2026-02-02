import random
from typing import Optional
from unittest.mock import patch

import pytest

from memory.embeddings.base import EmbeddingProvider
from memory.enrichment.base import EnrichmentProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic fake embedding provider for tests.

    Returns reproducible vectors based on text hash so that
    identical inputs produce identical embeddings.
    """

    def __init__(self, dim: int = 768):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        rng = random.Random(text)
        vec = [rng.gauss(0, 1) for _ in range(self.dim)]
        # L2 normalize
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]


@pytest.fixture
def tmp_vault(tmp_path):
    """Provides a temporary vault directory for tests."""
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    return tmp_path


@pytest.fixture
def env_home(tmp_vault, monkeypatch):
    """Overrides MEMORY_HOME and patches embedding provider for tests."""
    monkeypatch.setenv("MEMORY_HOME", str(tmp_vault))

    fake = FakeEmbeddingProvider(dim=768)

    with patch.object(
        __import__("memory.core", fromlist=["MemoryService"]).MemoryService,
        "_create_embedding_provider",
        return_value=fake,
    ):
        yield tmp_vault


class FakeEnrichmentProvider(EnrichmentProvider):
    """Fake enrichment provider for tests.

    Returns predetermined results for extract_tags and extract_memory.
    """

    def __init__(self, tags: list[str] | None = None, memory: dict | None = None):
        self._tags = tags or ["test-tag"]
        self._memory = memory

    def extract_tags(self, text: str, max_tags: int = 8) -> list[str]:
        return self._tags[:max_tags]

    def extract_memory(self, response: str, max_chars: int = 4000) -> Optional[dict]:
        return self._memory
