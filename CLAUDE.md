# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanobot is a lightweight, open-source AI agent framework written in Python with a React/TypeScript WebUI. It centers around a small agent loop that receives messages from chat channels, invokes an LLM provider, executes tools, and manages session context and memory.

## Development Commands

```bash
# Python: run single test / lint
pytest tests/test_openai_api.py::test_function -v
ruff check nanobot/

# WebUI: dev server (proxies API/WS to gateway :8765), build, test
# Build outputs to ../nanobot/web/dist (bundled into the Python wheel)
cd webui && bun run dev      # or NANOBOT_API_URL=... bun run dev
cd webui && bun run build
cd webui && bun run test

# Gateway
nanobot gateway
```

## High-Level Architecture

### Core Data Flow

Messages flow through an async `MessageBus` (`nanobot/bus/queue.py`) that decouples chat channels from the agent core:

1. **Channels** (`nanobot/channels/`) receive messages from external platforms and publish `InboundMessage` events to the bus.
2. **`AgentLoop`** (`nanobot/agent/loop.py`) consumes inbound messages, builds context, and coordinates the turn.
3. **`AgentRunner`** (`nanobot/agent/runner.py`) handles the actual LLM conversation loop: send messages to the provider, receive tool calls, execute tools, and stream responses.
4. Responses are published as `OutboundMessage` events back to the appropriate channel.

### Key Subsystems

- **Agent Loop** (`nanobot/agent/loop.py`, `runner.py`): The core processing engine. `AgentLoop` manages session keys, hooks, context building, and an optional external memory bridge. `AgentRunner` executes the multi-turn LLM conversation with tool execution.
- **LLM Providers** (`nanobot/providers/`): Provider implementations (Anthropic, OpenAI-compatible, Azure, GitHub Copilot, etc.) built on a common base (`base.py`). `factory.py` and `registry.py` handle instantiation and model discovery.
- **Channels** (`nanobot/channels/`): Platform integrations (Telegram, Discord, Slack, Feishu, Matrix, WhatsApp, QQ, WeChat, WebSocket, etc.). `manager.py` discovers and coordinates them. Channels are auto-discovered via `pkgutil` scan + entry-point plugins. `websocket.py` also hosts the embedded WebUI HTTP surface for bootstrap, sessions, settings, signed media, and WebUI transcript routes.
- **Tools** (`nanobot/agent/tools/`): Agent capabilities exposed to the LLM: filesystem (read/write/edit/list), shell execution, web search/fetch, MCP servers, cron, notebook editing, subagent spawning, and `MyTool` for self-modification.
- **Session Management** (`nanobot/session/manager.py`): Per-session conversation persistence in workspace-scoped `sessions/*.jsonl`, including load/repair/save, listing, deletion, history slicing, and fsync-backed shutdown flushes.
- **Local Memory** (`nanobot/agent/memory.py`): Workspace memory files (`SOUL.md`, `USER.md`, `memory/MEMORY.md`, `memory/history.jsonl`), token-based consolidation, Dream two-phase long-term memory updates, and Git-backed memory versioning.
- **External Memory Service** (`nanobot/memory_service/`): Optional local SQLite + FTS5 memory service and `ExternalMemoryBridge`. When explicitly enabled or injected, `AgentLoop` performs pre-turn recall into `# Relevant Memory` and post-turn writeback after local session save.
- **Config** (`nanobot/config/schema.py`, `loader.py`): Pydantic-based configuration loaded from `~/.nanobot/config.json`. Supports camelCase aliases for JSON compatibility.
- **Bridge** (`bridge/`): TypeScript services (e.g. WhatsApp bridge) bundled into the wheel via `pyproject.toml` `force-include`.
- **WebUI** (`webui/`): Vite-based React SPA that talks to the gateway over a WebSocket multiplex protocol. The dev server proxies `/api`, `/webui`, `/auth`, and WebSocket traffic to the gateway.

### Entry Points

- **CLI**: `nanobot/cli/commands.py`
- **Python SDK**: `nanobot/nanobot.py`

## Working Rules

### Instruction Priority

When instructions overlap, use this order:

1. The current user request
2. The actual filesystem and repository docs
3. This `CLAUDE.md`
4. Reusable workflows and skills

Do not invent missing files, commands, or system behavior. If something is missing, read the closest real source of truth instead.

### Task Sizing and Escalation

Before changing code, classify the task:

- **Read-only / exploratory**: inspect, explain, locate files, compare approaches
- **Small change**: narrow fix or focused implementation with limited blast radius
- **Medium change**: multi-file change with shared types, UI flows, or nontrivial behavior shifts
- **High-risk change**: public interfaces, schemas, persistence, auth, payment, CI, dependencies, wide refactors, or anything hard to reverse

Rules:

- Small changes should stay narrow and avoid extra ceremony.
- Medium changes should define scope, constraints, done criteria, and verification before execution.
- High-risk changes should be planned first, then executed, then reviewed with at least one explicit verification pass.

### Do Not Introduce

Unless the user explicitly asks for it, do not introduce:

- new dependencies or package managers
- database migrations or schema churn
- CI/CD workflow changes
- new background services or global state
- secret-handling changes or credential files
- broad refactors unrelated to the requested outcome

### Verification Rules

Do not claim work is complete without evidence.

- Run the smallest relevant verification for the changed surface.
- If verification is not possible, say exactly what could not be verified and why.
- Do not invent command results, test outcomes, screenshots, or external state.
- For UI fixes, prefer focused tests first, then browser verification when the surface is user-visible.

### Blast Radius Check

When touching shared surfaces, search the impact before editing and include that impact in verification.

Shared surfaces include:

- public functions and shared types
- config schema and config serialization
- auth, persistence, provider contracts, and channel protocols
- WebUI hooks/components reused across multiple screens
- root config files, release/build wiring, and deployment scripts

## Project-Specific Notes

- Architecture constraints: [`.agent/design.md`](.agent/design.md)
- Security boundaries: [`.agent/security.md`](.agent/security.md)
- Common gotchas: [`.agent/gotchas.md`](.agent/gotchas.md)

## Branching Strategy

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for the full two-branch model (`main` vs `nightly`) and PR guidelines.

## Code Style

- Python 3.11+, asyncio throughout.
- Line length: 100.
- Linting: `ruff` with rules E, F, I, N, W (E501 ignored).
- pytest with `asyncio_mode = "auto"`.

## Common File Locations

- Config schema: `nanobot/config/schema.py`
- Provider base / new provider template: `nanobot/providers/base.py`
- Channel base / new channel template: `nanobot/channels/base.py`
- Tool registry: `nanobot/agent/tools/registry.py`
- WebUI dev proxy config: `webui/vite.config.ts`
- Tests mirror the `nanobot/` package structure.
