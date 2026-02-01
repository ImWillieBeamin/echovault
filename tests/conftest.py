import random
from unittest.mock import patch

import pytest

from memory.embeddings.base import EmbeddingProvider


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
