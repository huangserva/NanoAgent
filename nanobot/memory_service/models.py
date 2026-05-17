"""Lightweight models for the external memory service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class TurnEndRequest:
    user_id: str
    source_type: str
    event_type: str
    content: str
    session_id: str | None = None
    source_id: str | None = None
    metadata: JsonDict | None = None
    provenance: JsonDict | None = None
    dedupe_key: str | None = None


@dataclass(frozen=True)
class EventRecord:
    id: str
    user_id: str
    session_id: str | None
    source_type: str
    source_id: str | None
    event_type: str
    content: str
    metadata: JsonDict | None
    provenance: JsonDict
    dedupe_key: str | None
    created_at: str
    deleted_at: str | None = None


@dataclass(frozen=True)
class EventWriteResult:
    event: EventRecord
    created: bool


@dataclass(frozen=True)
class SearchRequest:
    user_id: str
    query: str
    limit: int = 10


@dataclass(frozen=True)
class SearchResult:
    id: str
    content: str
    event_type: str
    source_type: str
    created_at: str


@dataclass(frozen=True)
class TypedMemoryRecord:
    id: str
    user_id: str
    memory_type: str
    text: str
    confidence: float
    evidence_event_id: str | None
    provenance: JsonDict | None
    scope: JsonDict | None
    dedupe_key: str | None
    created_at: str
    updated_at: str
    status: str = "active"
    retired_at: str | None = None
    retired_reason: str | None = None
    retired_evidence_event_id: str | None = None
    superseded_by_id: str | None = None
    deleted_at: str | None = None


@dataclass(frozen=True)
class TypedMemoryWriteResult:
    memory: TypedMemoryRecord
    created: bool


@dataclass(frozen=True)
class JobRecord:
    id: str
    job_type: str
    status: str
    dedupe_key: str | None
    input: JsonDict
    output: JsonDict | None
    error: str | None
    retry_count: int
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class JobWriteResult:
    job: JobRecord
    created: bool
