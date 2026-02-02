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


class OllamaEnrichment(EnrichmentProvider):
    def __init__(self, model: str = "llama3.1:8b",
                 base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def extract_tags(self, text: str, max_tags: int = 8) -> list[str]:
        prompt = (
            "Extract concise, lowercase tags for this memory. "
            "Return ONLY a JSON array of 5-10 short tags, no prose.\n\n"
            f"Memory:\n{text}\n"
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        resp = httpx.post(
            f"{self.base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json().get("response", "").strip()

        try:
            tags = json.loads(content)
            if isinstance(tags, list):
                return dedupe_tags([str(t) for t in tags], max_tags=max_tags)
        except json.JSONDecodeError:
            pass

        rough = [t.strip() for t in content.replace("\n", ",").split(",") if t.strip()]
        return dedupe_tags(rough, max_tags=max_tags)

    def extract_memory(self, response: str, max_chars: int = 4000) -> Optional[dict]:
        truncated = response[:max_chars]
        payload = {
            "model": self.model,
            "system": EXTRACT_MEMORY_SYSTEM,
            "prompt": EXTRACT_MEMORY_PROMPT.format(response=truncated),
            "stream": False,
        }
        resp = httpx.post(
            f"{self.base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json().get("response", "").strip()
        return parse_memory_response(content)
