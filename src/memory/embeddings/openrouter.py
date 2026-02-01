import httpx
from memory.embeddings.base import EmbeddingProvider


class OpenRouterEmbedding(EmbeddingProvider):
    def __init__(self, model: str = "openai/text-embedding-3-small",
                 api_key: str | None = None):
        self.model = model
        self.api_key = api_key or ""

    def embed(self, text: str) -> list[float]:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model, "input": text},
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]
