<p align="center">
  <img src="assets/echovault-icon.svg" width="120" height="120" alt="EchoVault" />
</p>

<h1 align="center">EchoVault</h1>

<p align="center">
  Local memory for coding agents. Your agent remembers decisions, bugs, and context across sessions — no cloud, no API keys, no cost.
</p>

<p align="center">
  <a href="#install">Install</a> · <a href="#features">Features</a> · <a href="#how-it-works">How it works</a> · <a href="#commands">Commands</a>
</p>

---
EchoVault gives your agent persistent memory. Every decision, bug fix, and lesson learned is saved locally and automatically surfaced in future sessions. Your agent gets better the more you use it.

### Why I built this tool

I built this to solve my own problem. So coding agents forget everything between sessions. They re-discover the same patterns, repeat the same mistakes, and forget the decisions you made yesterday. Now I do have used other tools like Supermemory and Claude Mem, No doubt they are great tools, but they do have some pros and cons which I found it unsatisfactory based on my use case. 

Supermemory recently announced there MCP and I was tempted to use it, but I found it was saving everything in the cloud and that was a deal breaker for me as I do not want to store my codebase decisions in the cloud as I work with multiple companies as a consultant.

Initially claude-mem was the first tool I used for memory persistence but later I found out my claude sessions was taaking up a lot of memory and It was becoming a bottleneck for me to run multiple sessions at the same time.


TLDR; I built this tool to solve my own problem for local memory persistence for coding agents while keeping it simple and easy to use.

## Features

**Works with most agents** — Claude Code, Cursor, Codex. One command sets up hooks and skills for your agent.

**Local-first** — Everything stays on your machine. Memories are stored as Markdown in `~/.memory/vault/`, readable in Obsidian or any editor. No data leaves your machine unless you opt into cloud embeddings.

**Zero idle cost** — No background processes, no daemon, no RAM overhead. Memory operations only run when explicitly called.

**Automatic retrieval** — Hooks inject relevant memories into every prompt. Your agent sees prior context before it starts working.

**Automatic saving** — The skill instructs agents to save decisions, bugs, patterns, and learnings as they work. No manual intervention needed.

**Hybrid search** — FTS5 keyword search works out of the box. Add Ollama, OpenAI, or OpenRouter for semantic vector search.

**Secret redaction** — 3-layer redaction strips API keys, passwords, and credentials before anything hits disk. Supports explicit `<redacted>` tags, pattern detection, and custom `.memoryignore` rules.

**Cross-agent** — Memories saved by Claude Code are searchable in Cursor and vice versa. One vault, many agents.

**Obsidian-compatible** — Session files are valid Markdown with YAML frontmatter. Point Obsidian at `~/.memory/vault/` and browse your agent's memory visually.

## Install

```bash
pip install git+https://github.com/mraza007/echovault.git
memory init
memory setup claude-code   # or: cursor, codex
```

That's it. `memory setup` installs both the agent skill and hooks automatically.

By default hooks are installed globally (`~/.claude`). To install for a specific project, `cd` into that project first:

```bash
cd ~/my-project
memory setup claude-code --project   # writes to ./my-project/.claude/
```

### Configure embeddings (optional)

Embeddings enable semantic search. Without them, you still get fast keyword search via FTS5.

Generate a starter config:

```bash
memory config init
```

This creates `~/.memory/config.yaml` with sensible defaults:

```yaml
embedding:
  provider: ollama              # ollama | openai | openrouter
  model: nomic-embed-text

enrichment:
  provider: none                # none | ollama | openai | openrouter

context:
  semantic: auto                # auto | always | never
  topup_recent: true
```

**What each section does:**

- **`embedding`** — How memories get turned into vectors for semantic search. `ollama` runs locally, `openai` and `openrouter` call cloud APIs. `nomic-embed-text` is a good local model for Ollama.
- **`enrichment`** — Optional LLM step that enhances memories before storing (better summaries, auto-tags). Set to `none` to skip.
- **`context`** — Controls how memories are retrieved at session start. `auto` uses vector search when embeddings are available, falls back to keywords. `topup_recent` also includes recent memories so the agent has fresh context.

For cloud providers, add `api_key` under the provider section. API keys are redacted in `memory config` output.

## Usage

Once set up, your agent uses memory automatically:

- **Session start** — retrieves relevant memories before doing any work
- **Session end** — saves decisions, bugs, and learnings before finishing

You can also use the CLI directly:

```bash
memory save --title "Switched to JWT auth" \
  --what "Replaced session cookies with JWT" \
  --why "Needed stateless auth for API" \
  --impact "All endpoints now require Bearer token" \
  --tags "auth,jwt" --category "decision"

memory search "authentication"
memory details <id>
memory context --project
```

## How it works

```
~/.memory/
├── vault/                    # Obsidian-compatible Markdown
│   └── my-project/
│       └── 2026-02-01-session.md
├── index.db                  # SQLite: FTS5 + sqlite-vec
└── config.yaml               # Embedding provider config
```

- **Markdown vault** — one file per session per project, with YAML frontmatter
- **SQLite index** — FTS5 for keywords, sqlite-vec for semantic vectors
- **Compact pointers** — search returns ~50-token summaries; full details fetched on demand
- **3-layer redaction** — explicit tags, pattern matching, and `.memoryignore` rules

## Supported agents

| Agent | Setup command | What gets installed |
|-------|-------------|-------------------|
| Claude Code | `memory setup claude-code` | `UserPromptSubmit` hook + skill |
| Cursor | `memory setup cursor` | `beforeSubmitPrompt` hook + skill |
| Codex | `memory setup codex` | `AGENTS.md` instructions + skill |

## Commands

| Command | Description |
|---------|-------------|
| `memory init` | Create `~/.memory` |
| `memory setup <agent>` | Install hooks + skill for an agent |
| `memory uninstall <agent>` | Remove hooks + skill for an agent |
| `memory save ...` | Save a memory |
| `memory search "query"` | Hybrid FTS + semantic search |
| `memory details <id>` | Full details for a memory |
| `memory delete <id>` | Delete a memory by ID or prefix |
| `memory context --project` | List memories for current project |
| `memory sessions` | List session files |
| `memory config` | Show effective config |
| `memory config init` | Generate a starter config.yaml |
| `memory reindex` | Rebuild vectors after changing provider |

## Uninstall

```bash
memory uninstall claude-code   # or: cursor, codex — removes hooks + skill
pip uninstall echovault
```

To also remove all stored memories: `rm -rf ~/.memory/`

## Privacy

Everything stays local by default. If you configure OpenAI or OpenRouter for embeddings, those API calls go to their servers. Use Ollama for fully local operation.


### NOTE: I have mostly used this with Claude Code

## License

MIT — see [LICENSE](LICENSE).
