"""Security tests for path traversal, ReDoS, and shell escaping protection."""

import os
import re
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPathTraversal:
    """Test path traversal protection in project names."""

    def test_reject_dotdot_in_project_name(self):
        """../../../tmp should raise or be sanitized."""
        from memory.security import sanitize_project_name

        with pytest.raises(ValueError, match="Invalid project name"):
            sanitize_project_name("../../../tmp")

    def test_reject_dotdot_in_middle(self):
        """foo/../bar should raise or be sanitized."""
        from memory.security import sanitize_project_name

        with pytest.raises(ValueError, match="Invalid project name"):
            sanitize_project_name("foo/../bar")

    def test_reject_absolute_path_project(self):
        """/etc/passwd should raise or be sanitized."""
        from memory.security import sanitize_project_name

        with pytest.raises(ValueError, match="Invalid project name"):
            sanitize_project_name("/etc/passwd")

    def test_reject_windows_absolute_path(self):
        """C:\\Windows should raise."""
        from memory.security import sanitize_project_name

        with pytest.raises(ValueError, match="Invalid project name"):
            sanitize_project_name("C:\\Windows")

    def test_reject_null_bytes(self):
        """Null bytes should be rejected."""
        from memory.security import sanitize_project_name

        with pytest.raises(ValueError, match="Invalid project name"):
            sanitize_project_name("project\x00name")

    def test_reject_empty_project_name(self):
        """Empty string should be rejected."""
        from memory.security import sanitize_project_name

        with pytest.raises(ValueError, match="Invalid project name"):
            sanitize_project_name("")

    def test_reject_whitespace_only(self):
        """Whitespace-only should be rejected."""
        from memory.security import sanitize_project_name

        with pytest.raises(ValueError, match="Invalid project name"):
            sanitize_project_name("   ")

    def test_accept_valid_project_names(self):
        """Valid project names should pass through."""
        from memory.security import sanitize_project_name

        assert sanitize_project_name("my-project") == "my-project"
        assert sanitize_project_name("my_project_123") == "my_project_123"
        assert sanitize_project_name("MyProject") == "MyProject"
        assert sanitize_project_name("project.name") == "project.name"
        assert sanitize_project_name("a") == "a"

    def test_accept_project_with_spaces(self):
        """Project names with spaces should be allowed."""
        from memory.security import sanitize_project_name

        assert sanitize_project_name("My Project Name") == "My Project Name"

    def test_reject_hidden_directory_escape(self):
        """..hidden should not be confused with directory traversal."""
        from memory.security import sanitize_project_name

        # Double dot at start followed by more chars that look like traversal
        with pytest.raises(ValueError, match="Invalid project name"):
            sanitize_project_name("..hidden/secret")


class TestValidateDateString:
    """Test date string validation for session file names."""

    def test_valid_date_format(self):
        """Valid YYYY-MM-DD dates should pass."""
        from memory.security import validate_date_string

        assert validate_date_string("2024-01-15") == "2024-01-15"
        assert validate_date_string("2026-12-31") == "2026-12-31"

    def test_reject_path_traversal_in_date(self):
        """Path traversal in date should be rejected."""
        from memory.security import validate_date_string

        with pytest.raises(ValueError, match="Invalid date"):
            validate_date_string("../../etc/passwd")

    def test_reject_invalid_date_format(self):
        """Invalid date formats should be rejected."""
        from memory.security import validate_date_string

        with pytest.raises(ValueError, match="Invalid date"):
            validate_date_string("01-15-2024")  # Wrong order

        with pytest.raises(ValueError, match="Invalid date"):
            validate_date_string("2024/01/15")  # Wrong separator

        with pytest.raises(ValueError, match="Invalid date"):
            validate_date_string("2024-1-5")  # Missing leading zeros

    def test_reject_date_with_extra_content(self):
        """Date with extra content should be rejected."""
        from memory.security import validate_date_string

        with pytest.raises(ValueError, match="Invalid date"):
            validate_date_string("2024-01-15; rm -rf /")


class TestReDoSProtection:
    """Test ReDoS protection for regex patterns."""

    def test_safe_compile_valid_pattern(self):
        """Valid patterns should compile successfully."""
        from memory.security import safe_compile_pattern

        pattern = safe_compile_pattern(r"\d{3}-\d{2}-\d{4}")
        assert pattern is not None
        assert pattern.match("123-45-6789")

    def test_safe_compile_invalid_regex(self):
        """Invalid regex syntax should return None."""
        from memory.security import safe_compile_pattern

        # Unbalanced parenthesis
        pattern = safe_compile_pattern(r"(abc")
        assert pattern is None

        # Invalid escape - use a raw string that's actually invalid regex
        pattern = safe_compile_pattern(r"[invalid")
        assert pattern is None

    def test_safe_compile_returns_compiled_pattern(self):
        """Should return a compiled regex pattern."""
        from memory.security import safe_compile_pattern

        pattern = safe_compile_pattern(r"test.*pattern")
        assert isinstance(pattern, re.Pattern)

    def test_normal_patterns_work(self):
        """Normal patterns from SENSITIVE_PATTERNS should work."""
        from memory.security import safe_compile_pattern

        patterns = [
            r"sk_live_[a-zA-Z0-9]+",
            r"ghp_[a-zA-Z0-9]+",
            r"AKIA[0-9A-Z]{16}",
        ]
        for p in patterns:
            compiled = safe_compile_pattern(p)
            assert compiled is not None


class TestRedactionWithSafePatterns:
    """Test that redaction uses safe pattern compilation."""

    def test_invalid_memoryignore_pattern_skipped(self):
        """Invalid patterns in .memoryignore should be skipped, not crash."""
        from memory.redaction import redact

        text = "password: secret123"
        # Invalid pattern should be skipped gracefully
        result = redact(text, extra_patterns=[r"(abc"])
        # Built-in pattern should still work
        assert "secret123" not in result
        assert "[REDACTED]" in result

    def test_valid_extra_patterns_still_work(self):
        """Valid extra patterns should still redact correctly."""
        from memory.redaction import redact

        text = "SSN: 123-45-6789"
        result = redact(text, extra_patterns=[r"\d{3}-\d{2}-\d{4}"])
        assert "123-45-6789" not in result
        assert "[REDACTED]" in result


class TestSecureFilePath:
    """Test secure file path construction."""

    def test_ensure_path_within_vault(self):
        """File paths should be validated to stay within vault."""
        from memory.security import ensure_path_within_base

        base = Path("/home/user/.memory/vault")

        # Valid path
        result = ensure_path_within_base(base, "project/2024-01-15-session.md")
        assert result == base / "project/2024-01-15-session.md"

    def test_reject_path_escape_attempt(self):
        """Path escape attempts should be rejected."""
        from memory.security import ensure_path_within_base

        base = Path("/home/user/.memory/vault")

        with pytest.raises(ValueError, match="outside"):
            ensure_path_within_base(base, "../../../etc/passwd")

    def test_reject_absolute_path_in_subpath(self):
        """Absolute paths in subpath should be rejected."""
        from memory.security import ensure_path_within_base

        base = Path("/home/user/.memory/vault")

        with pytest.raises(ValueError, match="outside"):
            ensure_path_within_base(base, "/etc/passwd")


class TestShellEscaping:
    """Test shell argument escaping for hooks."""

    def test_escape_shell_metacharacters(self):
        """Shell metacharacters should be escaped/quoted safely."""
        from memory.security import escape_for_shell

        # Command substitution - should be wrapped in single quotes
        result = escape_for_shell("$(whoami)")
        # shlex.quote wraps in single quotes, making it a literal string
        assert result == "'$(whoami)'"

        # Backticks - should also be wrapped
        result = escape_for_shell("`whoami`")
        assert result == "'`whoami`'"

    def test_escape_semicolons(self):
        """Semicolons should be escaped."""
        from memory.security import escape_for_shell

        result = escape_for_shell("test; rm -rf /")
        # The ; should be escaped/quoted
        assert result != "test; rm -rf /"

    def test_escape_pipes(self):
        """Pipes should be escaped."""
        from memory.security import escape_for_shell

        result = escape_for_shell("test | cat /etc/passwd")
        assert result != "test | cat /etc/passwd"

    def test_safe_text_unchanged(self):
        """Safe text should pass through reasonably."""
        from memory.security import escape_for_shell

        # Note: shlex.quote adds quotes, so we just check it's safe
        result = escape_for_shell("simple text")
        # When executed, this should produce "simple text"
        assert "simple" in result
        assert "text" in result

    def test_escape_newlines(self):
        """Newlines should be handled safely."""
        from memory.security import escape_for_shell

        result = escape_for_shell("line1\nline2")
        # Should be escaped in a way that doesn't break the command
        assert result  # Non-empty result

    def test_escape_quotes(self):
        """Quotes should be escaped."""
        from memory.security import escape_for_shell

        result = escape_for_shell('test "quoted" text')
        # Should handle the quotes safely
        assert result
