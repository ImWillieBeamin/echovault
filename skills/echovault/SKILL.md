---
name: echovault
description: Local-first memory for coding agents. Save decisions, bugs, and context as Markdown, search with FTS5 + semantic search, and retrieve prior context at session start. Use when you need to persist or recall project knowledge across sessions.
---

# EchoVault

Use the `memory` CLI to persist decisions, bugs, context, and learnings across sessions.

## Quick start

```bash
memory context --project
memory search "<query>"
```

When search results show "Details: available", run:

```bash
memory details <memory-id>
```

## Save memories

Save whenever you make a decision, fix a bug, discover a pattern, set up infra, or learn something non-obvious:

```bash
memory save \
  --title "Short descriptive title" \
  --what "What happened or was decided" \
  --why "Reasoning behind it" \
  --impact "What changed as a result" \
  --tags "tag1,tag2,tag3" \
  --category "decision" \
  --related-files "src/auth.ts,src/middleware.ts" \
  --source "claude-code" \
  --details "Full context with all important details. Be thorough.
             Include alternatives considered, tradeoffs, config values,
             and anything someone would need to understand this fully later."
```

Use `--source` to identify the agent: `claude-code`, `codex`, or `cursor`.

## Claude Code hooks (optional)

Automatically inject relevant memories into new prompts by adding a hook:

```json
{
  "hooks": {
    "UserPromptSubmit": [{
      "command": "memory context --project --query \"$USER_PROMPT\"",
      "timeout": 5000
    }]
  }
}
```

## Useful commands

```bash
memory config
memory sessions
```

## Rules

- Always capture thorough details.
- Never include API keys, secrets, or credentials.
- Wrap sensitive values in `<redacted>` tags if referencing them.
- Search before deciding.
- Save after doing.
