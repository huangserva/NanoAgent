"""External memory tools for model-driven recall, store, and forget."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.agent.tools.context import ContextAware, RequestContext, ToolContext
from nanobot.memory_service.bridge import TypedMemoryCandidate

_MEMORY_TYPES = ["preference", "profile_fact", "task_state", "project_fact"]


class _MemoryToolBase(Tool, ContextAware):
    def __init__(self, external_memory: Any) -> None:
        self._external_memory = external_memory
        self._request_ctx = RequestContext(channel="", chat_id="")

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        external_memory = getattr(ctx, "external_memory", None)
        return (
            external_memory is not None
            and getattr(external_memory, "injection_mode", "both") != "auto_inject"
        )

    @classmethod
    def create(cls, ctx: ToolContext) -> "_MemoryToolBase":
        return cls(ctx.external_memory)

    def set_context(self, ctx: RequestContext) -> None:
        self._request_ctx = ctx


class MemoryRecallTool(_MemoryToolBase):
    @property
    def name(self) -> str:
        return "memory_recall"

    @property
    def description(self) -> str:
        return (
            "Recall durable external memory when cross-session context, past user "
            "preferences, profile facts, task state, or project facts are needed. "
            "This is separate from the current conversation history."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Natural language query for external memory.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": _MEMORY_TYPES,
                    "description": "Optional structured memory type filter.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5,
                    "description": "Maximum memories/events to return.",
                },
            },
            "required": ["query"],
        }

    @property
    def read_only(self) -> bool:
        return True

    @property
    def concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        query: str,
        memory_type: str | None = None,
        limit: int = 5,
    ) -> str:
        packet = self._external_memory.recall(
            query,
            sender_id=self._request_ctx.sender_id,
            memory_type=memory_type,
            limit=limit,
        )
        if packet == "No matching memories.":
            return packet
        return f"# Memory Recall\n\n{packet}"


class MemoryStoreTool(_MemoryToolBase):
    @property
    def name(self) -> str:
        return "memory_store"

    @property
    def description(self) -> str:
        return (
            "Store only durable external memory that should persist across sessions: "
            "user preferences, profile facts, task state, or project facts. Do not "
            "store trivial, temporary, or already obvious facts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 500,
                    "description": "Memory body to persist.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": _MEMORY_TYPES,
                    "description": "Structured memory type.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.8,
                    "description": "Confidence that the memory is durable and correct.",
                },
            },
            "required": ["text", "memory_type"],
        }

    async def execute(
        self,
        text: str,
        memory_type: str,
        confidence: float = 0.8,
    ) -> str:
        confidence = max(0.0, min(float(confidence), 1.0))
        candidate = TypedMemoryCandidate(
            memory_type=memory_type,
            text=text.strip(),
            confidence=confidence,
        )
        result = self._external_memory.service.upsert_typed_memory(
            user_id=self._external_memory.subject_key(self._request_ctx.sender_id),
            memory_type=memory_type,
            text=candidate.text,
            confidence=confidence,
            dedupe_key=self._external_memory._typed_dedupe_key(candidate),
        )
        short_id = result.memory.id[:8]
        if not result.created:
            return f"Memory already exists (id={short_id})"
        return f"Stored {memory_type} memory (id={short_id}, confidence={confidence:.2f})"


class MemoryForgetTool(_MemoryToolBase):
    @property
    def name(self) -> str:
        return "memory_forget"

    @property
    def description(self) -> str:
        return (
            "Delete or retire external memory when it is wrong, stale, superseded, "
            "or the user asks to forget it. Match by a phrase describing the stored "
            "memory, optionally narrowed by type."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Substring or phrase describing the memory to retire.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": _MEMORY_TYPES,
                    "description": "Optional structured memory type filter.",
                },
            },
            "required": ["target"],
        }

    async def execute(
        self,
        target: str,
        memory_type: str | None = None,
    ) -> str:
        retired = self._external_memory.forget_structured_memories(
            sender_id=self._request_ctx.sender_id,
            target=target,
            memory_type=memory_type,
        )
        if not retired:
            return "No matching memory found."
        lines = ["Retired memories:"]
        for memory in retired:
            lines.append(f"- id={memory.id[:8]} type={memory.memory_type}: {memory.text}")
        return "\n".join(lines)


class SceneReadTool(_MemoryToolBase):
    @property
    def name(self) -> str:
        return "scene_read"

    @property
    def description(self) -> str:
        return (
            "Read a durable cross-session scene block: an LLM-readable topic dossier "
            "or situational playbook for recurring workflows, project background, "
            "or user-specific scenarios. Scene blocks are longer Markdown context, "
            "distinct from typed memory short fact rows."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "pattern": r"^[a-z0-9][a-z0-9-]{0,63}$",
                    "description": "Scene slug, e.g. project-review-playbook.",
                },
            },
            "required": ["slug"],
        }

    @property
    def read_only(self) -> bool:
        return True

    @property
    def concurrency_safe(self) -> bool:
        return True

    async def execute(self, slug: str) -> str:
        result = self._external_memory.read_scene(
            slug,
            sender_id=self._request_ctx.sender_id,
        )
        if result is None:
            return f"Scene '{slug}' does not exist."
        record, body = result
        return f"# Scene: {record.slug} (updated {record.updated_at})\n\n{body}"


class SceneWriteTool(_MemoryToolBase):
    @property
    def name(self) -> str:
        return "scene_write"

    @property
    def description(self) -> str:
        return (
            "Create or update one durable cross-session scene block. Use this for "
            "focused Markdown context about one recurring scenario, workflow, or "
            "topic dossier. Do not use it as a kitchen sink or frequently rewrite "
            "trivial scenes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "slug": {
                    "type": "string",
                    "pattern": r"^[a-z0-9][a-z0-9-]{0,63}$",
                },
                "body": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 65536,
                },
                "title": {
                    "type": "string",
                    "maxLength": 200,
                },
                "tags": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {
                        "type": "string",
                        "maxLength": 32,
                    },
                },
                "summary": {
                    "type": "string",
                    "maxLength": 400,
                },
            },
            "required": ["slug", "body"],
        }

    async def execute(
        self,
        slug: str,
        body: str,
        title: str | None = None,
        tags: list[str] | None = None,
        summary: str | None = None,
    ) -> str:
        record = self._external_memory.write_scene(
            slug,
            body,
            sender_id=self._request_ctx.sender_id,
            title=title,
            tags=tags,
            summary=summary,
        )
        return f"Scene '{record.slug}' saved ({record.char_count} chars)."


class SceneListTool(_MemoryToolBase):
    @property
    def name(self) -> str:
        return "scene_list"

    @property
    def description(self) -> str:
        return (
            "List or search durable scene blocks for the current user. Supports "
            "tag filtering and free-text search across scene title, summary, and body."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional free-text search query.",
                },
                "tag": {
                    "type": "string",
                    "maxLength": 32,
                    "description": "Optional scene tag filter.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 20,
                },
            },
        }

    @property
    def read_only(self) -> bool:
        return True

    @property
    def concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        query: str | None = None,
        tag: str | None = None,
        limit: int = 20,
    ) -> str:
        effective_limit = max(1, min(int(limit), 50))
        clean_query = query.strip() if isinstance(query, str) else ""
        clean_tag = tag.strip() if isinstance(tag, str) else ""
        if clean_query:
            records = self._external_memory.search_scenes(
                clean_query,
                sender_id=self._request_ctx.sender_id,
                limit=effective_limit,
            )
            if clean_tag:
                records = [record for record in records if clean_tag in record.tags]
        else:
            records = self._external_memory.list_scenes(
                sender_id=self._request_ctx.sender_id,
                tag=clean_tag or None,
                limit=effective_limit,
            )
        if not records:
            return "No scenes found."
        lines = ["# Scenes"]
        for record in records:
            title = record.title or "(untitled)"
            tags = ", ".join(record.tags) if record.tags else "-"
            lines.append(
                f"- `{record.slug}` | {title} | tags: {tags} | "
                f"updated: {record.updated_at} | {record.char_count} chars"
            )
        return "\n".join(lines)
