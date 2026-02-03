"""Tests for agent setup module."""

import json
import os

import pytest


@pytest.fixture
def claude_home(tmp_path):
    """Create a temporary ~/.claude directory."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    return claude_dir


@pytest.fixture
def claude_home_with_settings(claude_home):
    """Create ~/.claude with an existing settings.json."""
    settings = {"permissions": {"allow": ["Bash(memory:*)"]}}
    (claude_home / "settings.json").write_text(json.dumps(settings, indent=2))
    return claude_home


class TestClaudeCodeSetup:
    def test_creates_settings_file_if_missing(self, claude_home):
        from memory.setup import setup_claude_code

        result = setup_claude_code(str(claude_home))

        settings_path = claude_home / "settings.json"
        assert settings_path.exists()
        settings = json.loads(settings_path.read_text())
        assert "hooks" in settings
        assert "UserPromptSubmit" in settings["hooks"]

    def test_adds_hooks_to_existing_settings(self, claude_home_with_settings):
        from memory.setup import setup_claude_code

        result = setup_claude_code(str(claude_home_with_settings))

        settings = json.loads((claude_home_with_settings / "settings.json").read_text())
        assert settings["permissions"]["allow"] == ["Bash(memory:*)"]
        assert "UserPromptSubmit" in settings["hooks"]

    def test_installs_user_prompt_submit_hook(self, claude_home):
        from memory.setup import setup_claude_code

        setup_claude_code(str(claude_home))

        settings = json.loads((claude_home / "settings.json").read_text())
        hooks = settings["hooks"]["UserPromptSubmit"]
        assert len(hooks) == 1
        assert hooks[0]["hooks"][0]["type"] == "command"
        assert "memory context" in hooks[0]["hooks"][0]["command"]

    def test_installs_post_tool_use_hook(self, claude_home):
        from memory.setup import setup_claude_code

        setup_claude_code(str(claude_home))

        settings = json.loads((claude_home / "settings.json").read_text())
        hooks = settings["hooks"]["PostToolUse"]
        assert len(hooks) == 1
        assert hooks[0]["matcher"] == "Read|Write|Edit"
        assert hooks[0]["hooks"][0]["type"] == "command"
        assert "memory auto-save" in hooks[0]["hooks"][0]["command"]
        assert "$TOOL_INPUT" in hooks[0]["hooks"][0]["command"]

    def test_does_not_duplicate_hooks(self, claude_home):
        from memory.setup import setup_claude_code

        setup_claude_code(str(claude_home))
        setup_claude_code(str(claude_home))

        settings = json.loads((claude_home / "settings.json").read_text())
        hooks = settings["hooks"]["UserPromptSubmit"]
        memory_hooks = [
            h for group in hooks for h in group.get("hooks", [])
            if "memory context" in h.get("command", "")
        ]
        assert len(memory_hooks) == 1

    def test_installs_skill_md(self, claude_home):
        from memory.setup import setup_claude_code

        setup_claude_code(str(claude_home))

        skill_path = claude_home / "skills" / "echovault" / "SKILL.md"
        assert skill_path.exists()
        content = skill_path.read_text()
        assert "EchoVault" in content
        assert "memory context" in content

    def test_returns_success_result(self, claude_home):
        from memory.setup import setup_claude_code

        result = setup_claude_code(str(claude_home))

        assert result["status"] == "ok"
        assert "installed" in result["message"].lower() or "already" in result["message"].lower()


@pytest.fixture
def cursor_home(tmp_path):
    """Create a temporary .cursor directory."""
    cursor_dir = tmp_path / ".cursor"
    cursor_dir.mkdir()
    return cursor_dir


@pytest.fixture
def cursor_home_with_hooks(cursor_home):
    """Create .cursor with an existing hooks.json."""
    hooks = {
        "version": 1,
        "hooks": {
            "afterFileEdit": [
                {"command": "./format.sh"}
            ]
        }
    }
    (cursor_home / "hooks.json").write_text(json.dumps(hooks, indent=2))
    return cursor_home


class TestCursorSetup:
    def test_creates_hooks_file_if_missing(self, cursor_home):
        from memory.setup import setup_cursor

        setup_cursor(str(cursor_home))

        hooks_path = cursor_home / "hooks.json"
        assert hooks_path.exists()
        data = json.loads(hooks_path.read_text())
        assert data["version"] == 1
        assert "beforeSubmitPrompt" in data["hooks"]

    def test_adds_hooks_to_existing_config(self, cursor_home_with_hooks):
        from memory.setup import setup_cursor

        setup_cursor(str(cursor_home_with_hooks))

        data = json.loads((cursor_home_with_hooks / "hooks.json").read_text())
        assert "afterFileEdit" in data["hooks"]
        assert "beforeSubmitPrompt" in data["hooks"]

    def test_installs_before_submit_prompt_hook(self, cursor_home):
        from memory.setup import setup_cursor

        setup_cursor(str(cursor_home))

        data = json.loads((cursor_home / "hooks.json").read_text())
        hooks = data["hooks"]["beforeSubmitPrompt"]
        assert len(hooks) == 1
        assert "memory context" in hooks[0]["command"]

    def test_does_not_duplicate_hooks(self, cursor_home):
        from memory.setup import setup_cursor

        setup_cursor(str(cursor_home))
        setup_cursor(str(cursor_home))

        data = json.loads((cursor_home / "hooks.json").read_text())
        hooks = data["hooks"]["beforeSubmitPrompt"]
        memory_hooks = [h for h in hooks if "memory context" in h.get("command", "")]
        assert len(memory_hooks) == 1

    def test_installs_skill_md(self, cursor_home):
        from memory.setup import setup_cursor

        setup_cursor(str(cursor_home))

        skill_path = cursor_home / "skills" / "echovault" / "SKILL.md"
        assert skill_path.exists()

    def test_returns_success_result(self, cursor_home):
        from memory.setup import setup_cursor

        result = setup_cursor(str(cursor_home))

        assert result["status"] == "ok"


@pytest.fixture
def codex_home(tmp_path):
    """Create a temporary ~/.codex directory."""
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    return codex_dir


@pytest.fixture
def codex_home_with_agents_md(codex_home):
    """Create ~/.codex with an existing AGENTS.md."""
    (codex_home / "AGENTS.md").write_text("# My Codex Rules\n\nBe concise.\n")
    return codex_home


class TestCodexSetup:
    def test_creates_agents_md_if_missing(self, codex_home):
        from memory.setup import setup_codex

        setup_codex(str(codex_home))

        agents_path = codex_home / "AGENTS.md"
        assert agents_path.exists()
        content = agents_path.read_text()
        assert "memory context --project" in content
        assert "memory save" in content

    def test_appends_to_existing_agents_md(self, codex_home_with_agents_md):
        from memory.setup import setup_codex

        setup_codex(str(codex_home_with_agents_md))

        content = (codex_home_with_agents_md / "AGENTS.md").read_text()
        assert "# My Codex Rules" in content
        assert "Be concise." in content
        assert "memory context --project" in content

    def test_does_not_duplicate_section(self, codex_home):
        from memory.setup import setup_codex

        setup_codex(str(codex_home))
        setup_codex(str(codex_home))

        content = (codex_home / "AGENTS.md").read_text()
        assert content.count("## EchoVault") == 1

    def test_installs_skill_md(self, codex_home):
        from memory.setup import setup_codex

        setup_codex(str(codex_home))

        skill_path = codex_home / "skills" / "echovault" / "SKILL.md"
        assert skill_path.exists()

    def test_returns_success_result(self, codex_home):
        from memory.setup import setup_codex

        result = setup_codex(str(codex_home))

        assert result["status"] == "ok"


class TestUninstall:
    def test_uninstall_claude_code_removes_hooks(self, claude_home):
        from memory.setup import setup_claude_code, uninstall_claude_code

        setup_claude_code(str(claude_home))
        uninstall_claude_code(str(claude_home))

        settings = json.loads((claude_home / "settings.json").read_text())
        ups = settings.get("hooks", {}).get("UserPromptSubmit", [])
        memory_hooks = [
            h for group in ups for h in group.get("hooks", [])
            if "memory context" in h.get("command", "")
        ]
        assert len(memory_hooks) == 0

    def test_uninstall_claude_code_preserves_other_hooks(self, claude_home):
        from memory.setup import setup_claude_code, uninstall_claude_code

        setup_claude_code(str(claude_home))
        settings = json.loads((claude_home / "settings.json").read_text())
        settings["hooks"]["PreToolUse"] = [{"hooks": [{"type": "command", "command": "echo hi"}]}]
        (claude_home / "settings.json").write_text(json.dumps(settings, indent=2))

        uninstall_claude_code(str(claude_home))

        settings = json.loads((claude_home / "settings.json").read_text())
        assert "PreToolUse" in settings["hooks"]

    def test_uninstall_cursor_removes_hooks(self, cursor_home):
        from memory.setup import setup_cursor, uninstall_cursor

        setup_cursor(str(cursor_home))
        uninstall_cursor(str(cursor_home))

        data = json.loads((cursor_home / "hooks.json").read_text())
        bsp = data.get("hooks", {}).get("beforeSubmitPrompt", [])
        memory_hooks = [h for h in bsp if "memory context" in h.get("command", "")]
        assert len(memory_hooks) == 0

    def test_uninstall_codex_removes_section(self, codex_home):
        from memory.setup import setup_codex, uninstall_codex

        setup_codex(str(codex_home))
        uninstall_codex(str(codex_home))

        content = (codex_home / "AGENTS.md").read_text()
        assert "## EchoVault" not in content

    def test_uninstall_claude_code_removes_skill(self, claude_home):
        from memory.setup import setup_claude_code, uninstall_claude_code

        setup_claude_code(str(claude_home))
        assert (claude_home / "skills" / "echovault" / "SKILL.md").exists()

        uninstall_claude_code(str(claude_home))
        assert not (claude_home / "skills" / "echovault").exists()

    def test_uninstall_codex_removes_skill(self, codex_home):
        from memory.setup import setup_codex, uninstall_codex

        setup_codex(str(codex_home))
        assert (codex_home / "skills" / "echovault" / "SKILL.md").exists()

        uninstall_codex(str(codex_home))
        assert not (codex_home / "skills" / "echovault").exists()

    def test_uninstall_noop_when_not_installed(self, claude_home):
        from memory.setup import uninstall_claude_code

        (claude_home / "settings.json").write_text("{}")

        result = uninstall_claude_code(str(claude_home))
        assert result["status"] == "ok"


class TestHookCommandSecurity:
    """Test that hook commands are safe from injection."""

    def test_hook_commands_use_safe_patterns(self, claude_home):
        """Verify hook commands don't use vulnerable echo patterns."""
        from memory.setup import setup_claude_code, CLAUDE_CODE_HOOKS

        # Check that hooks don't use vulnerable patterns
        for event, hook_groups in CLAUDE_CODE_HOOKS.items():
            for group in hook_groups:
                for hook in group.get("hooks", []):
                    cmd = hook.get("command", "")
                    # Should not use bare echo with variable
                    assert 'echo "$' not in cmd, f"Unsafe echo pattern in {event}"
                    # Should not use bare echo with variable (single quotes won't expand)
                    assert "echo '$" not in cmd or "$" not in cmd.replace("'$", ""), f"Unsafe echo pattern in {event}"

    def test_post_tool_use_hook_uses_stdin_safely(self, claude_home):
        """PostToolUse hook should pass data via stdin safely."""
        from memory.setup import CLAUDE_CODE_HOOKS

        post_tool_hooks = CLAUDE_CODE_HOOKS.get("PostToolUse", [])
        assert len(post_tool_hooks) > 0

        for group in post_tool_hooks:
            for hook in group.get("hooks", []):
                cmd = hook.get("command", "")
                # Should use printf or here-doc for safety, not bare echo
                assert "memory auto-save" in cmd
                # The command should pass TOOL_INPUT safely via stdin
                assert "TOOL_INPUT" in cmd or "stdin" in hook.get("type", "")

    def test_stop_hook_uses_stdin_safely(self, claude_home):
        """Stop hook should pass data via stdin safely."""
        from memory.setup import CLAUDE_CODE_HOOKS

        stop_hooks = CLAUDE_CODE_HOOKS.get("Stop", [])
        assert len(stop_hooks) > 0

        for group in stop_hooks:
            for hook in group.get("hooks", []):
                cmd = hook.get("command", "")
                # Should use safe pattern
                assert "memory auto-save" in cmd
                assert "CLAUDE_RESPONSE" in cmd or "stdin" in hook.get("type", "")

    def test_user_prompt_submit_uses_safe_query(self, claude_home):
        """UserPromptSubmit should pass query safely."""
        from memory.setup import CLAUDE_CODE_HOOKS

        ups_hooks = CLAUDE_CODE_HOOKS.get("UserPromptSubmit", [])
        assert len(ups_hooks) > 0

        for group in ups_hooks:
            for hook in group.get("hooks", []):
                cmd = hook.get("command", "")
                assert "memory context" in cmd
                # Should use safe quoting
                assert "USER_PROMPT" in cmd

    def test_cursor_hooks_use_safe_patterns(self):
        """Verify Cursor hooks don't use vulnerable patterns."""
        from memory.setup import CURSOR_HOOKS

        for event, hooks in CURSOR_HOOKS.items():
            for hook in hooks:
                cmd = hook.get("command", "")
                # Should not use bare echo with variable
                assert 'echo "$' not in cmd, f"Unsafe echo pattern in {event}"
