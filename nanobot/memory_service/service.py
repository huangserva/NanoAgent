"""Business layer for the external memory service."""

from __future__ import annotations

from typing import Any

from nanobot.config.paths import get_memory_service_db_path
from nanobot.memory_service.models import (
    EventWriteResult,
    JobRecord,
    JobWriteResult,
    SearchRequest,
    SearchResult,
    TurnEndRequest,
    TypedMemoryRecord,
    TypedMemoryWriteResult,
)
from nanobot.memory_service.store import MemoryStore


class MemoryService:
    """Small service wrapper around the SQLite memory store."""

    typed_memory_types = frozenset({
        "preference",
        "profile_fact",
        "task_state",
        "project_fact",
    })

    def __init__(self, store: MemoryStore | None = None):
        self.store = store or MemoryStore(get_memory_service_db_path())

    def turn_end(self, request: TurnEndRequest) -> EventWriteResult:
        _require_text(request.user_id, "user_id")
        _require_text(request.source_type, "source_type")
        _require_text(request.event_type, "event_type")
        _require_text(request.content, "content")
        return self.store.insert_event(request)

    def search(self, request: SearchRequest) -> list[SearchResult]:
        _require_text(request.user_id, "user_id")
        query = _require_text(request.query, "query")
        return self.store.search_events(user_id=request.user_id, query=query, limit=request.limit)

    def upsert_typed_memory(
        self,
        *,
        user_id: str,
        memory_type: str,
        text: str,
        confidence: float,
        evidence_event_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        scope: dict[str, Any] | None = None,
        dedupe_key: str | None = None,
    ) -> TypedMemoryWriteResult:
        _require_text(user_id, "user_id")
        memory_type = _require_text(memory_type, "memory_type")
        if memory_type not in self.typed_memory_types:
            raise ValueError(f"unsupported memory_type: {memory_type}")
        text = _require_text(text, "text")
        confidence = max(0.0, min(float(confidence), 1.0))
        return self.store.upsert_typed_memory(
            user_id=user_id,
            memory_type=memory_type,
            text=text,
            confidence=confidence,
            evidence_event_id=evidence_event_id,
            provenance=provenance,
            scope=scope,
            dedupe_key=dedupe_key,
        )

    def search_typed_memories(self, request: SearchRequest) -> list[TypedMemoryRecord]:
        _require_text(request.user_id, "user_id")
        query = _require_text(request.query, "query")
        return self.store.search_typed_memories(
            user_id=request.user_id,
            query=query,
            limit=request.limit,
        )

    def list_active_typed_memories(
        self,
        *,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 100,
    ) -> list[TypedMemoryRecord]:
        _require_text(user_id, "user_id")
        if memory_type is not None:
            memory_type = _require_text(memory_type, "memory_type")
            if memory_type not in self.typed_memory_types:
                raise ValueError(f"unsupported memory_type: {memory_type}")
        return self.store.list_active_typed_memories(
            user_id=user_id,
            memory_type=memory_type,
            limit=limit,
        )

    def retire_typed_memory(
        self,
        memory_id: str,
        *,
        status: str = "inactive",
        reason: str | None = None,
        evidence_event_id: str | None = None,
        superseded_by_id: str | None = None,
    ) -> TypedMemoryRecord | None:
        _require_text(memory_id, "memory_id")
        if status not in {"inactive", "deleted"}:
            raise ValueError(f"unsupported typed memory status: {status}")
        return self.store.retire_typed_memory(
            memory_id,
            status=status,
            reason=reason,
            evidence_event_id=evidence_event_id,
            superseded_by_id=superseded_by_id,
        )

    def create_job(
        self,
        *,
        job_type: str,
        input_data: dict[str, Any],
        dedupe_key: str | None = None,
        status: str = "pending",
    ) -> JobWriteResult:
        _require_text(job_type, "job_type")
        return self.store.create_job(
            job_type=job_type,
            input_data=input_data,
            dedupe_key=dedupe_key,
            status=status,
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        _require_text(job_id, "job_id")
        return self.store.get_job(job_id)


def _require_text(value: str | None, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()
