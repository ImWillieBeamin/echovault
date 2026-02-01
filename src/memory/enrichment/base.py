from __future__ import annotations

import re
from abc import ABC, abstractmethod


def normalize_tag(tag: str) -> str:
    cleaned = tag.strip().lstrip("#")
    cleaned = re.sub(r"\s+", "-", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "", cleaned)
    return cleaned.lower()


def dedupe_tags(tags: list[str], max_tags: int = 8) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for tag in tags:
        norm = normalize_tag(tag)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        result.append(norm)
        if len(result) >= max_tags:
            break
    return result


class EnrichmentProvider(ABC):
    @abstractmethod
    def extract_tags(self, text: str, max_tags: int = 8) -> list[str]:
        ...
