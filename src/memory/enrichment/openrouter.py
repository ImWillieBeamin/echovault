import json
from typing import Optional

import httpx

from memory.enrichment.base import (
    EXTRACT_MEMORY_PROMPT,
    EXTRACT_MEMORY_SYSTEM,
    EnrichmentProvider,
    dedupe_tags,
    parse_memory_response,
)


class OpenRouterEnrichment(EnrichmentProvider):
    def __init__(self, model: str = "openai/gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or ""

    def extract_tags(self, text: str, max_tags: int = 8) -> list[str]:
        if not self.api_key:
            raise RuntimeError("OpenRouter API key not configured")

        prompt = (
            "Extract concise, lowercase tags for this memory. "
            "Return ONLY a JSON array of 5-10 short tags, no prose.\n\n"
            f"Memory:\n{text}\n"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a tagging assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        try:
            tags = json.loads(content)
            if isinstance(tags, list):
                return dedupe_tags([str(t) for t in tags], max_tags=max_tags)
        except json.JSONDecodeError:
            pass

        # Fallback: split by commas/newlines
        rough = [t.strip() for t in content.replace("\n", ",").split(",") if t.strip()]
        return dedupe_tags(rough, max_tags=max_tags)

    def extract_memory(self, response: str, max_chars: int = 4000) -> Optional[dict]:
        if not self.api_key:
            return None

        truncated = response[:max_chars]
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": EXTRACT_MEMORY_SYSTEM},
                {"role": "user", "content": EXTRACT_MEMORY_PROMPT.format(response=truncated)},
            ],
            "temperature": 0.2,
        }
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json=payload,
            timeout=30.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        return parse_memory_response(content)
