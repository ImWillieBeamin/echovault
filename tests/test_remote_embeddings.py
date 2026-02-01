import pytest
from memory.embeddings.ollama import OllamaEmbedding
from memory.embeddings.openai_embed import OpenAIEmbedding
from memory.embeddings.openrouter import OpenRouterEmbedding
from memory.embeddings.base import EmbeddingProvider


def test_ollama_is_embedding_provider():
    assert issubclass(OllamaEmbedding, EmbeddingProvider)


def test_openai_is_embedding_provider():
    assert issubclass(OpenAIEmbedding, EmbeddingProvider)


def test_openrouter_is_embedding_provider():
    assert issubclass(OpenRouterEmbedding, EmbeddingProvider)


def test_ollama_default_config():
    p = OllamaEmbedding()
    assert p.model == "nomic-embed-text"
    assert p.base_url == "http://localhost:11434"


def test_openai_default_config():
    p = OpenAIEmbedding()
    assert p.model == "text-embedding-3-small"


def test_openrouter_default_config():
    p = OpenRouterEmbedding()
    assert p.model == "openai/text-embedding-3-small"
