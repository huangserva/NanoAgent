"""Built-in slash command handlers."""

from __future__ import annotations

import asyncio
import os
import sys
import time
from contextlib import suppress
from dataclasses import dataclass

from nanobot import __version__
from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.memory_service.models import TypedMemoryRecord
from nanobot.utils.helpers import build_status_content, truncate_text
from nanobot.utils.restart import set_restart_notice_to_env


@dataclass(frozen=True)
class BuiltinCommandSpec:
    command: str
    title: str
    description: str
    icon: str
    arg_hint: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "command": self.command,
            "title": self.title,
            "description": self.description,
            "icon": self.icon,
            "arg_hint": self.arg_hint,
        }


BUILTIN_COMMAND_SPECS: tuple[BuiltinCommandSpec, ...] = (
    BuiltinCommandSpec(
        "/new",
        "New chat",
        "Stop the current task and start a fresh conversation.",
        "square-pen",
    ),
    BuiltinCommandSpec(
        "/stop",
        "Stop current task",
        "Cancel the active agent turn for this chat.",
        "square",
    ),
    BuiltinCommandSpec(
        "/restart",
        "Restart nanobot",
        "Restart the bot process in place.",
        "rotate-cw",
    ),
    BuiltinCommandSpec(
        "/status",
        "Show status",
        "Display runtime, provider, and channel status.",
        "activity",
    ),
    BuiltinCommandSpec(
        "/model",
        "Switch model preset",
        "Show or switch the active model preset.",
        "brain",
        "[preset]",
    ),
    BuiltinCommandSpec(
        "/history",
        "Show conversation history",
        "Print the last N persisted conversation messages.",
        "history",
        "[n]",
    ),
    BuiltinCommandSpec(
        "/memory",
        "Show memory",
        "List active structured memories.",
        "brain",
        "[list|forget <target>]",
    ),
    BuiltinCommandSpec(
        "/forget",
        "Forget memory",
        "Delete matching structured memories.",
        "trash-2",
        "[type] <target>",
    ),
    BuiltinCommandSpec(
        "/goal",
        "Start long-running goal",
        "Tell the agent to treat the request as a long-running goal.",
        "activity",
        "<goal>",
    ),
    BuiltinCommandSpec(
        "/dream",
        "Run Dream",
        "Manually trigger memory consolidation.",
        "sparkles",
    ),
    BuiltinCommandSpec(
        "/dream-log",
        "Show Dream log",
        "Show what the last Dream consolidation changed.",
        "book-open",
    ),
    BuiltinCommandSpec(
        "/dream-restore",
        "Restore memory",
        "Revert memory to a previous Dream snapshot.",
        "undo-2",
    ),
    BuiltinCommandSpec(
        "/help",
        "Show help",
        "List available slash commands.",
        "circle-help",
    ),
    BuiltinCommandSpec(
        "/pairing",
        "Manage pairing",
        "List, approve, deny or revoke pairing requests.",
        "shield",
        "[list|approve <code>|deny <code>|revoke <user_id>]",
    ),
)


def builtin_command_palette() -> list[dict[str, str]]:
    """Return structured command metadata for UI command palettes."""
    return [spec.as_dict() for spec in BUILTIN_COMMAND_SPECS]


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    total = await loop._cancel_active_tasks(msg.session_key)
    content = f"Stopped {total} task(s)." if total else "No active task to stop."
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content=content,
        metadata=dict(msg.metadata or {})
    )


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process in-place via os.execv."""
    msg = ctx.msg
    set_restart_notice_to_env(
        channel=msg.channel,
        chat_id=msg.chat_id,
        metadata=dict(msg.metadata or {}),
    )

    async def _do_restart():
        await asyncio.sleep(1)
        os.execv(sys.executable, [sys.executable, "-m", "nanobot"] + sys.argv[1:])

    asyncio.create_task(_do_restart())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        metadata=dict(msg.metadata or {})
    )


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    ctx_est = 0
    with suppress(Exception):
        ctx_est, _ = loop.consolidator.estimate_session_prompt_tokens(session)
    if ctx_est <= 0:
        ctx_est = loop._last_usage.get("prompt_tokens", 0)

    # Fetch web search provider usage (best-effort, never blocks the response)
    search_usage_text: str | None = None
    # Never let usage fetch break /status
    with suppress(Exception):
        from nanobot.utils.searchusage import fetch_search_usage
        web_cfg = getattr(loop, "web_config", None)
        search_cfg = getattr(web_cfg, "search", None) if web_cfg else None
        if search_cfg is not None:
            provider = getattr(search_cfg, "provider", "duckduckgo")
            api_key = getattr(search_cfg, "api_key", "") or None
            usage = await fetch_search_usage(provider=provider, api_key=api_key)
            search_usage_text = usage.format()
    active_tasks = loop._active_tasks.get(ctx.key, [])
    task_count = sum(1 for t in active_tasks if not t.done())
    with suppress(Exception):
        task_count += loop.subagents.get_running_count_by_session(ctx.key)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=loop.model,
            start_time=loop._start_time, last_usage=loop._last_usage,
            context_window_tokens=loop.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
            search_usage_text=search_usage_text,
            active_task_count=task_count,
            max_completion_tokens=getattr(
                getattr(loop.provider, "generation", None), "max_tokens", 8192
            ),
        ),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Stop active task and start a fresh session."""
    loop = ctx.loop
    await loop._cancel_active_tasks(ctx.key)
    session = ctx.session or loop.sessions.get_or_create(ctx.key)
    snapshot = session.messages[session.last_consolidated:]
    session.clear()
    loop.sessions.save(session)
    loop.sessions.invalidate(session.key)
    if snapshot:
        loop._schedule_background(loop.consolidator.archive(snapshot))
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="New session started.",
        metadata=dict(ctx.msg.metadata or {})
    )


def _format_preset_names(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "(none configured)"


def _model_preset_names(loop) -> list[str]:
    names = set(loop.model_presets)
    names.add("default")
    return ["default", *sorted(name for name in names if name != "default")]


def _active_model_preset_name(loop) -> str:
    return loop.model_preset or "default"


def _command_error_message(exc: Exception) -> str:
    return str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)


def _model_command_status(loop) -> str:
    names = _model_preset_names(loop)
    active = _active_model_preset_name(loop)
    return "\n".join([
        "## Model",
        f"- Current model: `{loop.model}`",
        f"- Current preset: `{active}`",
        f"- Available presets: {_format_preset_names(names)}",
    ])


async def cmd_model(ctx: CommandContext) -> OutboundMessage:
    """Show or switch model presets."""
    loop = ctx.loop
    args = ctx.args.strip()
    metadata = {**dict(ctx.msg.metadata or {}), "render_as": "text"}

    if not args:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=_model_command_status(loop),
            metadata=metadata,
        )

    parts = args.split()
    if len(parts) != 1:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="Usage: `/model [preset]`",
            metadata=metadata,
        )

    name = parts[0]
    try:
        loop.set_model_preset(name)
    except (KeyError, ValueError) as exc:
        names = _model_preset_names(loop)
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=(
                f"Could not switch model preset: {_command_error_message(exc)}\n\n"
                f"Available presets: {_format_preset_names(names)}"
            ),
            metadata=metadata,
        )

    max_tokens = getattr(getattr(loop.provider, "generation", None), "max_tokens", None)
    lines = [
        f"Switched model preset to `{loop.model_preset}`.",
        f"- Model: `{loop.model}`",
        f"- Context window: {loop.context_window_tokens}",
    ]
    if max_tokens is not None:
        lines.append(f"- Max output tokens: {max_tokens}")
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="\n".join(lines),
        metadata=metadata,
    )


async def cmd_dream(ctx: CommandContext) -> OutboundMessage:
    """Manually trigger a Dream consolidation run."""
    import time

    loop = ctx.loop
    msg = ctx.msg

    async def _run_dream():
        t0 = time.monotonic()
        try:
            did_work = await loop.dream.run()
            elapsed = time.monotonic() - t0
            if did_work:
                content = f"Dream completed in {elapsed:.1f}s."
            else:
                content = "Dream: nothing to process."
        except Exception as e:
            elapsed = time.monotonic() - t0
            content = f"Dream failed after {elapsed:.1f}s: {e}"
        await loop.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    asyncio.create_task(_run_dream())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="Dreaming...",
    )


def _extract_changed_files(diff: str) -> list[str]:
    """Extract changed file paths from a unified diff."""
    files: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def _format_changed_files(diff: str) -> str:
    files = _extract_changed_files(diff)
    if not files:
        return "No tracked memory files changed."
    return ", ".join(f"`{path}`" for path in files)


def _format_dream_log_content(commit, diff: str, *, requested_sha: str | None = None) -> str:
    files_line = _format_changed_files(diff)
    lines = [
        "## Dream Update",
        "",
        "Here is the selected Dream memory change." if requested_sha else "Here is the latest Dream memory change.",
        "",
        f"- Commit: `{commit.sha}`",
        f"- Time: {commit.timestamp}",
        f"- Changed files: {files_line}",
    ]
    if diff:
        lines.extend([
            "",
            f"Use `/dream-restore {commit.sha}` to undo this change.",
            "",
            "```diff",
            diff.rstrip(),
            "```",
        ])
    else:
        lines.extend([
            "",
            "Dream recorded this version, but there is no file diff to display.",
        ])
    return "\n".join(lines)


def _format_dream_restore_list(commits: list) -> str:
    lines = [
        "## Dream Restore",
        "",
        "Choose a Dream memory version to restore. Latest first:",
        "",
    ]
    for c in commits:
        lines.append(f"- `{c.sha}` {c.timestamp} - {c.message.splitlines()[0]}")
    lines.extend([
        "",
        "Preview a version with `/dream-log <sha>` before restoring it.",
        "Restore a version with `/dream-restore <sha>`.",
    ])
    return "\n".join(lines)


async def cmd_dream_log(ctx: CommandContext) -> OutboundMessage:
    """Show what the last Dream changed.

    Default: diff of the latest commit (HEAD~1 vs HEAD).
    With /dream-log <sha>: diff of that specific commit.
    """
    store = ctx.loop.consolidator.store
    git = store.git

    if not git.is_initialized():
        if store.get_last_dream_cursor() == 0:
            msg = "Dream has not run yet. Run `/dream`, or wait for the next scheduled Dream cycle."
        else:
            msg = "Dream history is not available because memory versioning is not initialized."
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=msg, metadata={"render_as": "text"},
        )

    args = ctx.args.strip()

    if args:
        # Show diff of a specific commit
        sha = args.split()[0]
        result = git.show_commit_diff(sha)
        if not result:
            content = (
                f"Couldn't find Dream change `{sha}`.\n\n"
                "Use `/dream-restore` to list recent versions, "
                "or `/dream-log` to inspect the latest one."
            )
        else:
            commit, diff = result
            content = _format_dream_log_content(commit, diff, requested_sha=sha)
    else:
        # Default: show the latest commit's diff
        commits = git.log(max_entries=1)
        result = git.show_commit_diff(commits[0].sha) if commits else None
        if result:
            commit, diff = result
            content = _format_dream_log_content(commit, diff)
        else:
            content = "Dream memory has no saved versions yet."

    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


async def cmd_dream_restore(ctx: CommandContext) -> OutboundMessage:
    """Restore memory files from a previous dream commit.

    Usage:
        /dream-restore          — list recent commits
        /dream-restore <sha>    — revert a specific commit
    """
    store = ctx.loop.consolidator.store
    git = store.git
    if not git.is_initialized():
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="Dream history is not available because memory versioning is not initialized.",
        )

    args = ctx.args.strip()
    if not args:
        # Show recent commits for the user to pick
        commits = git.log(max_entries=10)
        if not commits:
            content = "Dream memory has no saved versions to restore yet."
        else:
            content = _format_dream_restore_list(commits)
    else:
        sha = args.split()[0]
        result = git.show_commit_diff(sha)
        changed_files = _format_changed_files(result[1]) if result else "the tracked memory files"
        new_sha = git.revert(sha)
        if new_sha:
            content = (
                f"Restored Dream memory to the state before `{sha}`.\n\n"
                f"- New safety commit: `{new_sha}`\n"
                f"- Restored files: {changed_files}\n\n"
                f"Use `/dream-log {new_sha}` to inspect the restore diff."
            )
        else:
            content = (
                f"Couldn't restore Dream change `{sha}`.\n\n"
                "It may not exist, or it may be the first saved version with no earlier state to restore."
            )
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


_HISTORY_DEFAULT_COUNT = 10
_HISTORY_MAX_COUNT = 50
_HISTORY_MAX_CONTENT_CHARS = 200


def _format_history_message(msg: dict) -> str | None:
    """Format a single history message for display. Returns None to skip."""
    role = msg.get("role")
    if role not in ("user", "assistant"):
        return None
    content = msg.get("content") or ""
    if isinstance(content, list):
        parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        content = " ".join(parts)
    content = str(content).strip()
    if not content:
        return None
    if len(content) > _HISTORY_MAX_CONTENT_CHARS:
        content = content[:_HISTORY_MAX_CONTENT_CHARS] + "…"
    label = "👤 You" if role == "user" else "🤖 Bot"
    return f"{label}: {content}"


async def cmd_history(ctx: CommandContext) -> OutboundMessage:
    """Show the last N messages of the current session (default 10, max 50).

    Usage: /history [count]
    """
    count = _HISTORY_DEFAULT_COUNT
    if ctx.args.strip():
        try:
            count = max(1, min(int(ctx.args.strip()), _HISTORY_MAX_COUNT))
        except ValueError:
            return OutboundMessage(
                channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
                content="Usage: /history [count] — e.g. /history 5 (default: 10, max: 50)",
                metadata=dict(ctx.msg.metadata or {}),
            )

    session = ctx.session or ctx.loop.sessions.get_or_create(ctx.key)
    history = session.get_history(max_messages=0)
    visible = [_format_history_message(m) for m in history]
    visible = [m for m in visible if m is not None]
    recent = visible[-count:]

    if not recent:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="No conversation history yet.",
            metadata=dict(ctx.msg.metadata or {}),
        )

    header = f"Last {len(recent)} message(s):\n"
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=header + "\n".join(recent),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


_MEMORY_LIST_LIMIT = 50
_MEMORY_TEXT_LIMIT = 180
_MEMORY_DISABLED = "External memory is not enabled for this agent session."
_MEMORY_TYPE_HINT = "preference|profile_fact|task_state|project_fact"
_MEMORY_FORGET_USAGE = (
    f"Usage: `/forget latest [{_MEMORY_TYPE_HINT}]`, `/forget #N`, "
    f"or `/forget [{_MEMORY_TYPE_HINT}] <target>`"
)


def _memory_bridge(ctx: CommandContext):
    return getattr(ctx.loop, "external_memory", None)


@dataclass(frozen=True)
class _ForgetCommand:
    mode: str
    memory_type: str | None = None
    target: str = ""
    index: int | None = None


def _memory_type_from_token(value: str) -> str | None:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in {"preference", "preferences"}:
        return "preference"
    if normalized in {"profile_fact", "profile", "fact", "profile_facts"}:
        return "profile_fact"
    if normalized in {"task_state", "task", "tasks", "task_states"}:
        return "task_state"
    if normalized in {"project_fact", "project", "projects", "project_facts"}:
        return "project_fact"
    return None


def _parse_memory_list_args(args: str) -> tuple[str | None, str | None]:
    args = args.strip()
    if not args:
        return None, None

    first, _sep, rest = args.partition(" ")
    if first.lower() in {"list", "show"}:
        if not rest.strip():
            return None, None
        memory_type = _memory_type_from_token(rest)
        if memory_type is not None:
            return memory_type, None
        return None, f"Usage: `/memory [list|show] [{_MEMORY_TYPE_HINT}]`"

    memory_type = _memory_type_from_token(args)
    if memory_type is not None:
        return memory_type, None
    return None, f"Usage: `/memory [list|show] [{_MEMORY_TYPE_HINT}]`"


def _parse_memory_type_and_target(args: str) -> tuple[str | None, str]:
    args = args.strip()
    if not args:
        return None, ""
    first, _sep, rest = args.partition(" ")
    memory_type = _memory_type_from_token(first)
    if memory_type is not None:
        return memory_type, rest.strip()
    return None, args


def _parse_memory_index(value: str) -> int | None:
    value = value.strip()
    if not value.startswith("#"):
        return None
    number = value[1:]
    if not number.isdecimal():
        return 0
    return int(number)


def _parse_forget_command(args: str) -> _ForgetCommand:
    args = args.strip()
    if not args:
        return _ForgetCommand("usage")

    first, _sep, rest = args.partition(" ")
    memory_type = _memory_type_from_token(first)
    if memory_type is not None:
        return _parse_forget_tail(rest.strip(), memory_type=memory_type)

    if first.lower() == "latest":
        if not rest.strip():
            return _ForgetCommand("latest")
        memory_type = _memory_type_from_token(rest.strip())
        if memory_type is None:
            return _ForgetCommand("usage")
        return _ForgetCommand("latest", memory_type=memory_type)

    index = _parse_memory_index(first)
    if index is not None:
        if not rest.strip():
            return _ForgetCommand("index", index=index)
        memory_type = _memory_type_from_token(rest.strip())
        if memory_type is None:
            return _ForgetCommand("usage")
        return _ForgetCommand("index", memory_type=memory_type, index=index)

    memory_type, target = _parse_memory_type_and_target(args)
    if not target:
        return _ForgetCommand("usage")
    return _ForgetCommand("target", memory_type=memory_type, target=target)


def _parse_forget_tail(args: str, *, memory_type: str) -> _ForgetCommand:
    if not args:
        return _ForgetCommand("usage")
    if args.lower() == "latest":
        return _ForgetCommand("latest", memory_type=memory_type)
    index = _parse_memory_index(args)
    if index is not None:
        return _ForgetCommand("index", memory_type=memory_type, index=index)
    return _ForgetCommand("target", memory_type=memory_type, target=args)


def _short_memory_id(value: str | None) -> str:
    return value[:8] if value else "unknown"


def _format_memory_line(memory: TypedMemoryRecord, index: int | None = None) -> str:
    prefix = f"#{index} " if index is not None else "- "
    text = truncate_text(memory.text, _MEMORY_TEXT_LIMIT)
    evidence = _short_memory_id(memory.evidence_event_id)
    return (
        f"{prefix}[{memory.memory_type}] {text}\n"
        f"   updated={memory.updated_at} evidence={evidence}"
    )


def _format_memory_list(memories: list[TypedMemoryRecord], *, memory_type: str | None = None) -> str:
    lines = ["External memory: enabled"]
    if not memories:
        typed = f" {memory_type}" if memory_type else ""
        lines.append(f"No active{typed} structured memories.")
        return "\n".join(lines)
    if memory_type:
        lines.append(f"Active structured memories (type={memory_type}, updated desc):")
    else:
        lines.append("Active structured memories (updated desc):")
    lines.extend(_format_memory_line(memory, index) for index, memory in enumerate(memories, 1))
    return "\n".join(lines)


def _memory_list_command(memory_type: str | None) -> str:
    return f"`/memory list {memory_type}`" if memory_type else "`/memory list`"


def _no_active_memory_message(memory_type: str | None) -> str:
    typed = f" {memory_type}" if memory_type else ""
    return f"External memory: enabled\nNo active{typed} structured memories to forget."


def _invalid_memory_index_message(index: int | None, memory_type: str | None, total: int) -> str:
    label = f"#{index}" if index is not None and index > 0 else "#N"
    typed = f" {memory_type}" if memory_type else ""
    return (
        "External memory: enabled\n"
        f"No active{typed} structured memory numbered {label}. "
        f"Current list has {total} item(s). Use {_memory_list_command(memory_type)} "
        "to see current numbers."
    )


def _format_forget_result(
    forgotten: list[TypedMemoryRecord],
    *,
    memory_type: str | None = None,
    target: str | None = None,
) -> str:
    if forgotten:
        lines = [
            "External memory: enabled",
            f"Forgot {len(forgotten)} structured memory item(s):",
        ]
        lines.extend(_format_memory_line(memory) for memory in forgotten)
        return "\n".join(lines)

    typed = f" {memory_type}" if memory_type else ""
    if target:
        return (
            "External memory: enabled\n"
            f"No active{typed} structured memories matched `{target}`."
        )
    return _no_active_memory_message(memory_type)


async def cmd_memory(ctx: CommandContext) -> OutboundMessage:
    """List active structured memories or forget matching entries."""
    bridge = _memory_bridge(ctx)
    metadata = {**dict(ctx.msg.metadata or {}), "render_as": "text"}
    if bridge is None:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=_MEMORY_DISABLED,
            metadata=metadata,
        )

    args = ctx.args.strip()
    lower = args.lower()
    if lower == "forget":
        return await _cmd_forget_target(ctx, "", metadata=metadata)
    if lower.startswith("forget "):
        return await _cmd_forget_target(ctx, args[len("forget "):].strip(), metadata=metadata)

    memory_type, error = _parse_memory_list_args(args)
    if error is None:
        try:
            memories = bridge.list_structured_memories(
                sender_id=ctx.msg.sender_id,
                memory_type=memory_type,
                limit=_MEMORY_LIST_LIMIT,
            )
        except Exception:
            content = "External memory is enabled, but structured memory is currently unavailable."
        else:
            content = _format_memory_list(memories, memory_type=memory_type)
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=content,
            metadata=metadata,
        )

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=(
            f"{error}\n"
            f"Forget with: `/memory forget latest [{_MEMORY_TYPE_HINT}]`, "
            f"`/memory forget #N`, or `/memory forget [{_MEMORY_TYPE_HINT}] <target>`"
        ),
        metadata=metadata,
    )


async def cmd_forget(ctx: CommandContext) -> OutboundMessage:
    """Delete matching active structured memories."""
    return await _cmd_forget_target(
        ctx,
        ctx.args.strip(),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def _cmd_forget_target(
    ctx: CommandContext,
    args: str,
    *,
    metadata: dict,
) -> OutboundMessage:
    bridge = _memory_bridge(ctx)
    if bridge is None:
        content = _MEMORY_DISABLED
    else:
        parsed = _parse_forget_command(args)
        if parsed.mode == "usage":
            content = _MEMORY_FORGET_USAGE
        else:
            try:
                if parsed.mode == "target":
                    forgotten = bridge.forget_structured_memories(
                        sender_id=ctx.msg.sender_id,
                        target=parsed.target,
                        memory_type=parsed.memory_type,
                        limit=_MEMORY_LIST_LIMIT,
                    )
                    content = _format_forget_result(
                        forgotten,
                        memory_type=parsed.memory_type,
                        target=parsed.target,
                    )
                else:
                    memories = bridge.list_structured_memories(
                        sender_id=ctx.msg.sender_id,
                        memory_type=parsed.memory_type,
                        limit=_MEMORY_LIST_LIMIT,
                    )
                    selected: TypedMemoryRecord | None = None
                    if parsed.mode == "latest":
                        selected = memories[0] if memories else None
                    elif parsed.mode == "index":
                        if parsed.index is not None and 1 <= parsed.index <= len(memories):
                            selected = memories[parsed.index - 1]
                    if selected is None:
                        if parsed.mode == "index":
                            content = _invalid_memory_index_message(
                                parsed.index,
                                parsed.memory_type,
                                len(memories),
                            )
                        else:
                            content = _no_active_memory_message(parsed.memory_type)
                    else:
                        forgotten = [bridge.delete_structured_memory(selected)]
                        content = _format_forget_result(
                            forgotten,
                            memory_type=parsed.memory_type,
                        )
            except Exception:
                content = "External memory is enabled, but structured memory is currently unavailable."
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata=metadata,
    )


_GOAL_PROMPT_TEMPLATE = """The user declared a sustained objective for this thread.

Inspect or clarify if needed, then call `long_task` with the refined objective (and optional short ui_summary). Work proceeds as normal assistant turns using your usual tools. When the objective is fully done and verified, call `complete_goal` with a brief recap. If the user later cancels or changes direction, still call `complete_goal` with an honest recap (then `long_task` again only after there is no active goal). Do not use `long_task` / `complete_goal` for trivial one-shot answers.

Goal:
{goal}
"""


async def cmd_goal(ctx: CommandContext) -> OutboundMessage | None:
    """Rewrite /goal into a normal agent turn that nudges long_task use."""
    goal = ctx.args.strip()
    if not goal:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content="Usage: /goal <long-running task description>",
            metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
        )
    if ctx.session is None:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=(
                "A task is already running for this chat. "
                "Use `/stop` first, then send `/goal <long-running task description>` again."
            ),
            metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
        )

    ctx.msg.metadata = {
        **dict(ctx.msg.metadata or {}),
        "original_command": "/goal",
        "original_content": ctx.raw,
        "goal_started_at": time.time(),
    }
    ctx.msg.content = _GOAL_PROMPT_TEMPLATE.format(goal=goal)
    return None


async def cmd_pairing(ctx: CommandContext) -> OutboundMessage:
    """List, approve, deny or revoke pairing requests."""
    from nanobot.pairing import PAIRING_COMMAND_META_KEY, handle_pairing_command

    reply = handle_pairing_command(ctx.msg.channel, ctx.args)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=reply,
        metadata={PAIRING_COMMAND_META_KEY: True},
    )


async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_help_text(),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


def build_help_text() -> str:
    """Build canonical help text shared across channels."""
    lines = ["🐈 nanobot commands:"]
    for spec in BUILTIN_COMMAND_SPECS:
        command = spec.command
        if spec.arg_hint:
            command = f"{command} {spec.arg_hint}"
        lines.append(f"{command} — {spec.description}")
    return "\n".join(lines)


def register_builtin_commands(router: CommandRouter) -> None:
    """Register the default set of slash commands."""
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/status", cmd_status)
    router.exact("/model", cmd_model)
    router.prefix("/model ", cmd_model)
    router.exact("/history", cmd_history)
    router.prefix("/history ", cmd_history)
    router.exact("/memory", cmd_memory)
    router.prefix("/memory ", cmd_memory)
    router.exact("/forget", cmd_forget)
    router.prefix("/forget ", cmd_forget)
    router.exact("/goal", cmd_goal)
    router.prefix("/goal ", cmd_goal)
    router.exact("/dream", cmd_dream)
    router.exact("/dream-log", cmd_dream_log)
    router.prefix("/dream-log ", cmd_dream_log)
    router.exact("/dream-restore", cmd_dream_restore)
    router.prefix("/dream-restore ", cmd_dream_restore)
    router.exact("/help", cmd_help)
    router.exact("/pairing", cmd_pairing)
    router.prefix("/pairing ", cmd_pairing)
