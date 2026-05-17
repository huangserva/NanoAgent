from __future__ import annotations

import sqlite3

from nanobot.memory_service.models import TurnEndRequest
from nanobot.memory_service.store import MemoryStore


def test_store_initializes_schema_and_writes_event(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.initialize()

    with sqlite3.connect(tmp_path / "memory.db") as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            ).fetchall()
        }
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(typed_memories)").fetchall()
        }
    assert "events" in tables
    assert "events_fts" in tables
    assert "typed_memories" in tables
    assert "typed_memories_fts" in tables
    assert "jobs" in tables
    assert {"status", "retired_at", "retired_reason", "superseded_by_id"} <= columns

    result = store.insert_event(
        TurnEndRequest(
            user_id="huang",
            session_id="s_123",
            source_type="nanobot",
            source_id="nanobot:cli",
            event_type="turn_end",
            content="用户讨论了 nanobot memory service",
            metadata={"channel": "cli"},
            provenance={"agent": "nanobot"},
        )
    )

    assert result.created is True
    event = store.get_event(result.event.id)
    assert event is not None
    assert event.content == "用户讨论了 nanobot memory service"
    assert event.metadata == {"channel": "cli"}
    assert event.provenance == {"agent": "nanobot"}


def test_store_upserts_and_searches_typed_memory(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.db")

    created = store.upsert_typed_memory(
        user_id="huang",
        memory_type="preference",
        text="Preference: SQLite memory service for local agent memory",
        confidence=0.82,
        evidence_event_id="event-1",
        provenance={"agent": "nanobot"},
        scope={"workspace": "test"},
        dedupe_key="pref-sqlite",
    )
    duplicate = store.upsert_typed_memory(
        user_id="huang",
        memory_type="preference",
        text="Preference: ignored duplicate",
        confidence=0.5,
        dedupe_key="pref-sqlite",
    )
    results = store.search_typed_memories(
        user_id="huang",
        query="local memory",
        limit=5,
    )

    assert created.created is True
    assert duplicate.created is False
    assert duplicate.memory.id == created.memory.id
    assert len(results) == 1
    assert results[0].memory_type == "preference"
    assert results[0].evidence_event_id == "event-1"
    assert results[0].status == "active"


def test_store_retired_typed_memory_is_not_searched(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    created = store.upsert_typed_memory(
        user_id="huang",
        memory_type="preference",
        text="Preference: SQLite memory service",
        confidence=0.82,
        evidence_event_id="event-1",
        dedupe_key="pref-sqlite",
    )

    retired = store.retire_typed_memory(
        created.memory.id,
        status="inactive",
        reason="overwrite",
        evidence_event_id="event-2",
        superseded_by_id="replacement",
    )
    results = store.search_typed_memories(
        user_id="huang",
        query="SQLite memory",
        limit=5,
    )

    assert retired is not None
    assert retired.status == "inactive"
    assert retired.retired_reason == "overwrite"
    assert retired.superseded_by_id == "replacement"
    assert results == []


def test_store_create_and_get_job(tmp_path) -> None:
    store = MemoryStore(tmp_path / "memory.db")

    created = store.create_job(
        job_type="ingest",
        input_data={"source": "reserved"},
        dedupe_key="job-key",
    )
    duplicate = store.create_job(
        job_type="ingest",
        input_data={"source": "ignored"},
        dedupe_key="job-key",
    )

    assert created.created is True
    assert duplicate.created is False
    assert duplicate.job.id == created.job.id
    job = store.get_job(created.job.id)
    assert job is not None
    assert job.status == "pending"
    assert job.input == {"source": "reserved"}
