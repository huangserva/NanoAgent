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
