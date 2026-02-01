"""Tests for the three-layer secret redaction pipeline."""

import pytest
from memory.redaction import redact, load_memoryignore
import tempfile
import os


class TestRedactExplicitTags:
    """Test Layer 1: Explicit <redacted> tags."""

    def test_single_redacted_tag(self):
        text = "My password is <redacted>secret123</redacted> here"
        result = redact(text)
        assert result == "My password is [REDACTED] here"

    def test_multiple_redacted_tags(self):
        text = "Key: <redacted>key1</redacted> and <redacted>key2</redacted>"
        result = redact(text)
        assert result == "Key: [REDACTED] and [REDACTED]"

    def test_nested_redacted_tags(self):
        # Non-greedy regex matches innermost pair first, then outer pair
        text = "Data: <redacted>outer <redacted>inner</redacted> text</redacted>"
        result = redact(text)
        assert "inner" not in result
        assert "outer" not in result
        assert "[REDACTED]" in result

    def test_multiline_redacted_tags(self):
        text = """Start
<redacted>
line1
line2
line3
</redacted>
End"""
        result = redact(text)
        assert "[REDACTED]" in result
        assert "line1" not in result
        assert "Start" in result
        assert "End" in result


class TestRedactPatterns:
    """Test Layer 2: Automatic pattern detection."""

    def test_stripe_live_key(self):
        text = "Stripe key: sk_live_1234567890abcdefg"
        result = redact(text)
        assert "sk_live_" not in result
        assert "[REDACTED]" in result

    def test_stripe_test_key(self):
        text = "Test key: sk_test_abcdefghijklmnop"
        result = redact(text)
        assert "sk_test_" not in result
        assert "[REDACTED]" in result

    def test_github_token(self):
        text = "GitHub: ghp_1234567890abcdefghijklmnopqrstuvwxyz"
        result = redact(text)
        assert "ghp_" not in result
        assert "[REDACTED]" in result

    def test_aws_access_key(self):
        text = "AWS: AKIAIOSFODNN7EXAMPLE"
        result = redact(text)
        assert "AKIA" not in result
        assert "[REDACTED]" in result

    def test_slack_token(self):
        text = "Slack: xoxb-123-456-abc-def"
        result = redact(text)
        assert "xoxb-" not in result
        assert "[REDACTED]" in result

    def test_private_key(self):
        text = "Key: -----BEGIN PRIVATE KEY-----"
        result = redact(text)
        assert "BEGIN PRIVATE KEY" not in result
        assert "[REDACTED]" in result

    def test_rsa_private_key(self):
        text = "Key: -----BEGIN RSA PRIVATE KEY-----"
        result = redact(text)
        assert "BEGIN RSA PRIVATE KEY" not in result
        assert "[REDACTED]" in result

    def test_jwt_token(self):
        text = "JWT: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
        result = redact(text)
        assert "eyJ" not in result
        assert "[REDACTED]" in result

    def test_password_field_colon(self):
        text = "password: mysecretpass123"
        result = redact(text)
        assert "mysecretpass123" not in result
        assert "[REDACTED]" in result

    def test_password_field_equals(self):
        text = 'password = "admin123"'
        result = redact(text)
        assert "admin123" not in result
        assert "[REDACTED]" in result

    def test_secret_field(self):
        text = "secret: my-secret-value"
        result = redact(text)
        assert "my-secret-value" not in result
        assert "[REDACTED]" in result

    def test_api_key_field(self):
        text = "api_key: 1234567890abcdef"
        result = redact(text)
        assert "1234567890abcdef" not in result
        assert "[REDACTED]" in result

    def test_api_key_hyphen(self):
        text = "api-key = test-key-value"
        result = redact(text)
        assert "test-key-value" not in result
        assert "[REDACTED]" in result


class TestPreserveNormalText:
    """Test that normal text is preserved unchanged."""

    def test_plain_text(self):
        text = "This is just normal text without secrets"
        result = redact(text)
        assert result == text

    def test_code_snippet(self):
        text = "def hello(): return 'world'"
        result = redact(text)
        assert result == text

    def test_urls(self):
        text = "Visit https://example.com for more info"
        result = redact(text)
        assert result == text

    def test_email(self):
        text = "Contact us at support@example.com"
        result = redact(text)
        assert result == text


class TestExtraPatterns:
    """Test custom patterns from .memoryignore."""

    def test_extra_patterns_redacted(self):
        text = "SSN: 123-45-6789 and Phone: 555-1234"
        extra = [r"\d{3}-\d{2}-\d{4}", r"555-\d{4}"]
        result = redact(text, extra_patterns=extra)
        assert "123-45-6789" not in result
        assert "555-1234" not in result
        assert "[REDACTED]" in result

    def test_extra_patterns_none(self):
        text = "password: secret123"
        result = redact(text, extra_patterns=None)
        assert "secret123" not in result


class TestLoadMemoryignore:
    """Test .memoryignore file loading."""

    def test_load_valid_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.memoryignore') as f:
            f.write("# This is a comment\n")
            f.write(r"\d{3}-\d{2}-\d{4}" + "\n")
            f.write("\n")  # Empty line
            f.write(r"custom_pattern_\d+" + "\n")
            f.name_to_delete = f.name

        try:
            patterns = load_memoryignore(f.name_to_delete)
            assert len(patterns) == 2
            assert r"\d{3}-\d{2}-\d{4}" in patterns
            assert r"custom_pattern_\d+" in patterns
        finally:
            os.unlink(f.name_to_delete)

    def test_load_missing_file(self):
        patterns = load_memoryignore("/nonexistent/path/.memoryignore")
        assert patterns == []

    def test_load_empty_file(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.memoryignore') as f:
            f.write("")
            f.name_to_delete = f.name

        try:
            patterns = load_memoryignore(f.name_to_delete)
            assert patterns == []
        finally:
            os.unlink(f.name_to_delete)

    def test_load_only_comments(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.memoryignore') as f:
            f.write("# Comment 1\n")
            f.write("# Comment 2\n")
            f.name_to_delete = f.name

        try:
            patterns = load_memoryignore(f.name_to_delete)
            assert patterns == []
        finally:
            os.unlink(f.name_to_delete)


class TestLayeredRedaction:
    """Test that all layers work together."""

    def test_all_layers_combined(self):
        text = """
        Explicit: <redacted>secret1</redacted>
        Stripe: sk_live_abcdefghijk
        GitHub: ghp_xyz123456
        Custom: SSN-123-45-6789
        Normal: Just regular text
        """
        extra = [r"SSN-\d{3}-\d{2}-\d{4}"]
        result = redact(text, extra_patterns=extra)

        assert "secret1" not in result
        assert "sk_live_" not in result
        assert "ghp_" not in result
        assert "SSN-123-45-6789" not in result
        assert "Normal: Just regular text" in result
        assert result.count("[REDACTED]") == 4
