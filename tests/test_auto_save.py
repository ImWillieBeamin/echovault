"""Tests for auto-save feature (Stop hook memory extraction)."""

from typing import Optional
from unittest.mock import patch

from click.testing import CliRunner

from memory.cli import main
from memory.core import MemoryService
from memory.enrichment.base import EnrichmentProvider, parse_memory_response


class FakeEnrichmentProvider(EnrichmentProvider):
    """Fake enrichment provider for tests."""

    def __init__(self, tags=None, memory=None):
        self._tags = tags or ["test-tag"]
        self._memory = memory

    def extract_tags(self, text: str, max_tags: int = 8) -> list[str]:
        return self._tags[:max_tags]

    def extract_memory(self, response: str, max_chars: int = 4000) -> Optional[dict]:
        return self._memory


# --- parse_memory_response tests ---


def test_parse_memory_response_returns_none_for_save_false():
    assert parse_memory_response('{"save": false}') is None


def test_parse_memory_response_returns_none_for_invalid_json():
    assert parse_memory_response("not json at all") is None


def test_parse_memory_response_returns_none_for_missing_title():
    raw = '{"save": true, "what": "something"}'
    assert parse_memory_response(raw) is None


def test_parse_memory_response_returns_none_for_missing_what():
    raw = '{"save": true, "title": "A title"}'
    assert parse_memory_response(raw) is None


def test_parse_memory_response_extracts_valid_memory():
    raw = (
        '{"save": true, "title": "Fixed search bug", "what": "Search was slow",'
        ' "why": "Double embedding call", "impact": "5s to 2s",'
        ' "category": "bug", "tags": ["search", "performance"],'
        ' "details": "Full details here"}'
    )
    result = parse_memory_response(raw)
    assert result is not None
    assert result["title"] == "Fixed search bug"
    assert result["what"] == "Search was slow"
    assert result["why"] == "Double embedding call"
    assert result["impact"] == "5s to 2s"
    assert result["category"] == "bug"
    assert result["tags"] == ["search", "performance"]
    assert result["details"] == "Full details here"


def test_parse_memory_response_defaults_invalid_category():
    raw = '{"save": true, "title": "Title", "what": "What", "category": "invalid"}'
    result = parse_memory_response(raw)
    assert result is not None
    assert result["category"] == "context"


def test_parse_memory_response_truncates_long_title():
    raw = '{"save": true, "title": "' + "A" * 100 + '", "what": "What"}'
    result = parse_memory_response(raw)
    assert result is not None
    assert len(result["title"]) == 80


# --- MemoryService.auto_save tests ---


def test_auto_save_skips_when_no_enrichment(env_home):
    svc = MemoryService(memory_home=str(env_home))
    with patch.object(svc, "_create_enrichment_provider", return_value=None):
        svc._enrichment_provider = None
        result = svc.auto_save("Some agent response")
    svc.close()
    assert result is None


def test_auto_save_skips_when_not_worth_saving(env_home):
    fake = FakeEnrichmentProvider(memory=None)
    svc = MemoryService(memory_home=str(env_home))
    svc._enrichment_provider = fake
    result = svc.auto_save("Sure, here you go.")
    svc.close()
    assert result is None


def test_auto_save_saves_when_worth_saving(env_home):
    extracted = {
        "title": "Fixed the search bug",
        "what": "Search was calling embed twice",
        "why": "Redundant call in hybrid_search",
        "impact": "Reduced latency from 5s to 2s",
        "category": "bug",
        "tags": ["search", "performance"],
        "details": "Full details about the fix",
    }
    fake = FakeEnrichmentProvider(memory=extracted)
    svc = MemoryService(memory_home=str(env_home))
    svc._enrichment_provider = fake

    result = svc.auto_save("I found the bug...", project="test-project", source="claude-code")
    svc.close()

    assert result is not None
    assert "id" in result
    assert "file_path" in result


def test_auto_save_handles_enrichment_exception(env_home):
    fake = FakeEnrichmentProvider()
    svc = MemoryService(memory_home=str(env_home))
    svc._enrichment_provider = fake

    # Make extract_memory raise
    def boom(response, max_chars=4000):
        raise RuntimeError("API down")

    fake.extract_memory = boom

    result = svc.auto_save("Some response")
    svc.close()
    assert result is None


def test_auto_save_sets_source(env_home):
    extracted = {
        "title": "A decision",
        "what": "Chose approach A",
        "category": "decision",
        "tags": ["arch"],
    }
    fake = FakeEnrichmentProvider(memory=extracted)
    svc = MemoryService(memory_home=str(env_home))
    svc._enrichment_provider = fake

    result = svc.auto_save("We decided...", project="myproj", source="claude-code")

    # Verify the memory was stored with correct source
    results = svc.search("decision", project="myproj", use_vectors=False)
    svc.close()

    assert result is not None
    assert len(results) >= 1
    assert results[0]["source"] == "claude-code"


# --- CLI auto-save command tests ---


def test_cli_auto_save_empty_stdin(env_home):
    runner = CliRunner()
    result = runner.invoke(main, ["auto-save", "--source", "test"], input="")
    assert result.exit_code == 0
    assert result.output == ""


def test_cli_auto_save_with_response(env_home):
    extracted = {
        "title": "Found a pattern",
        "what": "Retry logic needed",
        "category": "pattern",
        "tags": ["resilience"],
    }
    fake_enrichment = FakeEnrichmentProvider(memory=extracted)

    with patch.object(
        MemoryService,
        "_create_enrichment_provider",
        return_value=fake_enrichment,
    ):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["auto-save", "--project", "--source", "claude-code"],
            input="I discovered that retry logic is needed for API calls.",
        )

    assert result.exit_code == 0
    assert "Auto-saved:" in result.output
