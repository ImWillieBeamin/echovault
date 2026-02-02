from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Optional


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


EXTRACT_MEMORY_SYSTEM = "You are a memory extraction assistant for a coding agent."

EXTRACT_MEMORY_PROMPT = """\
Given the agent's response below, decide if it contains anything worth remembering for future sessions.

Worth saving: decisions, bug fixes, discoveries, architecture discussions, configuration changes, patterns learned, tradeoffs considered, and non-code discussions (e.g. documentation drafts, planning conversations).

NOT worth saving: greetings, simple file reads, acknowledgments, clarifying questions, trivial responses.

If NOT worth saving, respond with: {{"save": false}}
If worth saving, respond with:
{{"save": true, "title": "Short descriptive title (under 80 chars)", "what": "What happened or was decided", "why": "Reasoning behind it", "impact": "What changed as a result", "category": "decision|bug|pattern|context|learning|miscellaneous", "tags": ["tag1", "tag2", "tag3"], "details": "Full context with all important details"}}

Respond ONLY with valid JSON, no prose.

Agent response:
{response}"""


VALID_CATEGORIES = {"decision", "bug", "pattern", "context", "learning", "miscellaneous"}


def parse_memory_response(content: str) -> Optional[dict]:
    """Parse LLM JSON response into a memory dict or None."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict) or not data.get("save"):
        return None

    title = data.get("title", "").strip()
    what = data.get("what", "").strip()
    if not title or not what:
        return None

    category = data.get("category", "").strip().lower()
    if category not in VALID_CATEGORIES:
        category = "context"

    tags = data.get("tags", [])
    if isinstance(tags, list):
        tags = dedupe_tags([str(t) for t in tags], max_tags=8)
    else:
        tags = []

    return {
        "title": title[:80],
        "what": what,
        "why": data.get("why", "").strip() or None,
        "impact": data.get("impact", "").strip() or None,
        "category": category,
        "tags": tags,
        "details": data.get("details", "").strip() or None,
    }


class EnrichmentProvider(ABC):
    @abstractmethod
    def extract_tags(self, text: str, max_tags: int = 8) -> list[str]:
        ...

    @abstractmethod
    def extract_memory(self, response: str, max_chars: int = 4000) -> Optional[dict]:
        """Extract a structured memory from an agent response.

        Returns None if the response isn't worth saving.
        Returns dict with keys: title, what, why, impact, category, tags, details.
        """
        ...
