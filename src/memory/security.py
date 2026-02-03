"""Security utilities for path validation, ReDoS protection, and shell escaping.

This module provides defense-in-depth security functions:

1. Path traversal protection - Validates project names and file paths
2. ReDoS protection - Safe regex compilation with error handling
3. Shell escaping - Safe argument escaping for hook commands
"""

import re
import shlex
from pathlib import Path
from typing import Optional


# Valid date format: YYYY-MM-DD
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Characters that are never allowed in project names
FORBIDDEN_CHARS = frozenset("\x00\n\r")


def sanitize_project_name(name: str) -> str:
    """Validate and sanitize a project name to prevent path traversal.

    Args:
        name: The project name to validate

    Returns:
        The validated project name (unchanged if valid)

    Raises:
        ValueError: If the project name is invalid or contains path traversal

    Examples:
        >>> sanitize_project_name("my-project")
        'my-project'
        >>> sanitize_project_name("../../../etc")
        Traceback (most recent call last):
        ValueError: Invalid project name: contains path traversal
    """
    if not name or not name.strip():
        raise ValueError("Invalid project name: empty or whitespace-only")

    # Check for forbidden characters
    if any(c in name for c in FORBIDDEN_CHARS):
        raise ValueError("Invalid project name: contains forbidden characters")

    # Check for path traversal patterns
    # Normalize path separators for cross-platform checking
    normalized = name.replace("\\", "/")

    # Check for .. sequences (path traversal)
    if ".." in normalized:
        raise ValueError("Invalid project name: contains path traversal")

    # Check for absolute paths
    if normalized.startswith("/"):
        raise ValueError("Invalid project name: absolute paths not allowed")

    # Check for Windows absolute paths (C:, D:, etc.)
    if len(name) >= 2 and name[1] == ":" and name[0].isalpha():
        raise ValueError("Invalid project name: absolute paths not allowed")

    return name


def validate_date_string(date_str: str) -> str:
    """Validate a date string for use in session file names.

    Args:
        date_str: Date string to validate (expected format: YYYY-MM-DD)

    Returns:
        The validated date string

    Raises:
        ValueError: If the date string is invalid or contains path traversal

    Examples:
        >>> validate_date_string("2024-01-15")
        '2024-01-15'
        >>> validate_date_string("../../etc/passwd")
        Traceback (most recent call last):
        ValueError: Invalid date format
    """
    if not DATE_PATTERN.match(date_str):
        raise ValueError(f"Invalid date format: expected YYYY-MM-DD, got '{date_str}'")

    return date_str


def safe_compile_pattern(pattern: str) -> Optional[re.Pattern]:
    """Safely compile a regex pattern with error handling.

    This protects against:
    1. Invalid regex syntax that would crash the application
    2. Provides a safe fallback for malformed patterns

    Note: ReDoS protection is limited in Python's re module. For patterns
    from untrusted sources, consider using the 'regex' library with
    timeout support or limit input length.

    Args:
        pattern: The regex pattern to compile

    Returns:
        Compiled regex pattern, or None if invalid

    Examples:
        >>> p = safe_compile_pattern(r"\\d{3}-\\d{2}-\\d{4}")
        >>> p.match("123-45-6789") is not None
        True
        >>> safe_compile_pattern(r"(abc") is None
        True
    """
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


def ensure_path_within_base(base: Path, subpath: str) -> Path:
    """Ensure a constructed path stays within a base directory.

    Args:
        base: The base directory that paths must stay within
        subpath: The relative path to append

    Returns:
        The resolved path within base

    Raises:
        ValueError: If the resulting path would escape the base directory

    Examples:
        >>> base = Path("/home/user/.memory/vault")
        >>> ensure_path_within_base(base, "project/file.md")
        PosixPath('/home/user/.memory/vault/project/file.md')
    """
    # Resolve the base to absolute path
    base_resolved = base.resolve()

    # Construct and resolve the full path
    full_path = (base / subpath).resolve()

    # Check if the resolved path is within base
    try:
        full_path.relative_to(base_resolved)
    except ValueError:
        raise ValueError(f"Path '{subpath}' resolves outside base directory")

    return full_path


def escape_for_shell(text: str) -> str:
    """Escape text for safe use in shell commands.

    Uses shlex.quote() to properly escape text for POSIX shells.
    This prevents command injection via shell metacharacters.

    Args:
        text: The text to escape

    Returns:
        Shell-escaped text safe for use in command strings

    Examples:
        >>> escape_for_shell("simple text")
        "'simple text'"
        >>> escape_for_shell("$(whoami)")
        "'$(whoami)'"
    """
    return shlex.quote(text)
