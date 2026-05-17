"""SQLite store for the external memory service."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanobot.memory_service.models import (
    EventRecord,
    EventWriteResult,
    JobRecord,
    JobWriteResult,
    SearchResult,
    TurnEndRequest,
    TypedMemoryRecord,
    TypedMemoryWriteResult,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _json_loads_object(value: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        payload = json.loads(value)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


_FTS_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_FTS_TRIGRAM_TOKENIZER_RE = re.compile(
    r"tokenize\s*=\s*['\"]?trigram\b",
    re.IGNORECASE,
)


def _is_trigram_fts(sql: str) -> bool:
    return bool(_FTS_TRIGRAM_TOKENIZER_RE.search(sql or ""))


def _fts_query_from_user_query(query: str) -> str:
    tokens = [token for token in _FTS_TOKEN_RE.findall(query) if token.strip()]
    return " ".join(f'"{token}"' for token in tokens)


def _fts_or_query_from_user_query(query: str) -> str:
    tokens = [token for token in _FTS_TOKEN_RE.findall(query) if token.strip()]
    return " OR ".join(f'"{token}"' for token in tokens)


def _like_pattern_from_user_query(query: str) -> str:
    escaped = query.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class MemoryStore:
    """Synchronous SQLite store for M1 memory events and jobs."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).expanduser()

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;

                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    source_type TEXT NOT NULL,
                    source_id TEXT,
                    event_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT,
                    provenance_json TEXT NOT NULL,
                    dedupe_key TEXT,
                    created_at TEXT NOT NULL,
                    deleted_at TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_events_user_dedupe
                    ON events(user_id, dedupe_key)
                    WHERE dedupe_key IS NOT NULL AND dedupe_key != '';

                CREATE INDEX IF NOT EXISTS idx_events_user_created
                    ON events(user_id, created_at);

                CREATE TABLE IF NOT EXISTS typed_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    text TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_event_id TEXT,
                    provenance_json TEXT,
                    scope_json TEXT,
                    dedupe_key TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    retired_at TEXT,
                    retired_reason TEXT,
                    retired_evidence_event_id TEXT,
                    superseded_by_id TEXT,
                    deleted_at TEXT
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_typed_memories_user_type_dedupe
                    ON typed_memories(user_id, memory_type, dedupe_key)
                    WHERE dedupe_key IS NOT NULL AND dedupe_key != '';

                CREATE INDEX IF NOT EXISTS idx_typed_memories_user_type_updated
                    ON typed_memories(user_id, memory_type, updated_at);

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dedupe_key TEXT,
                    input_json TEXT NOT NULL,
                    output_json TEXT,
                    error TEXT,
                    retry_count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_type_dedupe
                    ON jobs(job_type, dedupe_key)
                    WHERE dedupe_key IS NOT NULL AND dedupe_key != '';
                """
            )
            self._ensure_typed_memory_lifecycle_columns(conn)
            self._ensure_fts_trigram_tables(conn)

    def _ensure_typed_memory_lifecycle_columns(self, conn: sqlite3.Connection) -> None:
        columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(typed_memories)").fetchall()
        }
        additions = {
            "status": "ALTER TABLE typed_memories ADD COLUMN status TEXT NOT NULL DEFAULT 'active'",
            "retired_at": "ALTER TABLE typed_memories ADD COLUMN retired_at TEXT",
            "retired_reason": "ALTER TABLE typed_memories ADD COLUMN retired_reason TEXT",
            "retired_evidence_event_id": "ALTER TABLE typed_memories ADD COLUMN retired_evidence_event_id TEXT",
            "superseded_by_id": "ALTER TABLE typed_memories ADD COLUMN superseded_by_id TEXT",
        }
        for name, statement in additions.items():
            if name not in columns:
                conn.execute(statement)

    def _ensure_fts_trigram_tables(self, conn: sqlite3.Connection) -> None:
        self._ensure_fts_trigram_table(
            conn,
            table_name="events_fts",
            create_sql=(
                "CREATE VIRTUAL TABLE events_fts "
                "USING fts5(event_id UNINDEXED, content, tokenize='trigram')"
            ),
            repopulate_sql=(
                "INSERT INTO events_fts(event_id, content) "
                "SELECT id, content FROM events WHERE deleted_at IS NULL"
            ),
        )
        self._ensure_fts_trigram_table(
            conn,
            table_name="typed_memories_fts",
            create_sql=(
                "CREATE VIRTUAL TABLE typed_memories_fts "
                "USING fts5(memory_id UNINDEXED, text, tokenize='trigram')"
            ),
            repopulate_sql=(
                "INSERT INTO typed_memories_fts(memory_id, text) "
                "SELECT id, text FROM typed_memories WHERE deleted_at IS NULL"
            ),
        )

    def _ensure_fts_trigram_table(
        self,
        conn: sqlite3.Connection,
        *,
        table_name: str,
        create_sql: str,
        repopulate_sql: str,
    ) -> None:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table_name,),
        ).fetchone()
        sql = str(row["sql"]) if row is not None else ""
        if row is not None and _is_trigram_fts(sql):
            return
        if row is not None:
            conn.execute(f"DROP TABLE {table_name}")
        conn.execute(create_sql)
        conn.execute(repopulate_sql)

    def insert_event(self, request: TurnEndRequest) -> EventWriteResult:
        self.initialize()
        if request.dedupe_key:
            existing = self.get_event_by_dedupe_key(request.user_id, request.dedupe_key)
            if existing is not None:
                return EventWriteResult(event=existing, created=False)

        event_id = uuid.uuid4().hex
        created_at = _now_iso()
        metadata_json = _json_dumps(request.metadata) if request.metadata is not None else None
        provenance_json = _json_dumps(request.provenance or {})

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO events (
                        id, user_id, session_id, source_type, source_id, event_type,
                        content, metadata_json, provenance_json, dedupe_key, created_at, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        event_id,
                        request.user_id,
                        request.session_id,
                        request.source_type,
                        request.source_id,
                        request.event_type,
                        request.content,
                        metadata_json,
                        provenance_json,
                        request.dedupe_key,
                        created_at,
                    ),
                )
                conn.execute(
                    "INSERT INTO events_fts(event_id, content) VALUES (?, ?)",
                    (event_id, request.content),
                )
            except sqlite3.IntegrityError:
                if request.dedupe_key:
                    existing = self.get_event_by_dedupe_key(request.user_id, request.dedupe_key)
                    if existing is not None:
                        return EventWriteResult(event=existing, created=False)
                raise

        event = EventRecord(
            id=event_id,
            user_id=request.user_id,
            session_id=request.session_id,
            source_type=request.source_type,
            source_id=request.source_id,
            event_type=request.event_type,
            content=request.content,
            metadata=request.metadata,
            provenance=request.provenance or {},
            dedupe_key=request.dedupe_key,
            created_at=created_at,
        )
        return EventWriteResult(event=event, created=True)

    def get_event(self, event_id: str) -> EventRecord | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return _event_from_row(row) if row is not None else None

    def get_event_by_dedupe_key(self, user_id: str, dedupe_key: str) -> EventRecord | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM events
                WHERE user_id = ? AND dedupe_key = ? AND deleted_at IS NULL
                """,
                (user_id, dedupe_key),
            ).fetchone()
        return _event_from_row(row) if row is not None else None

    def search_events(self, *, user_id: str, query: str, limit: int = 10) -> list[SearchResult]:
        self.initialize()
        limit = max(1, min(int(limit), 100))
        fts_query = _fts_query_from_user_query(query)
        if not fts_query:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT e.id, e.content, e.event_type, e.source_type, e.created_at
                FROM events_fts
                JOIN events e ON e.id = events_fts.event_id
                WHERE events_fts.content MATCH ?
                    AND e.user_id = ?
                    AND e.deleted_at IS NULL
                ORDER BY e.created_at DESC
                LIMIT ?
                """,
                (fts_query, user_id, limit),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    """
                    SELECT e.id, e.content, e.event_type, e.source_type, e.created_at
                    FROM events e
                    WHERE e.user_id = ?
                        AND e.content LIKE ? ESCAPE '\\'
                        AND e.deleted_at IS NULL
                    ORDER BY e.created_at DESC
                    LIMIT ?
                    """,
                    (user_id, _like_pattern_from_user_query(query), limit),
                ).fetchall()
        return [
            SearchResult(
                id=str(row["id"]),
                content=str(row["content"]),
                event_type=str(row["event_type"]),
                source_type=str(row["source_type"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        ]

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
        self.initialize()
        if dedupe_key:
            existing = self.get_typed_memory_by_dedupe_key(user_id, memory_type, dedupe_key)
            if existing is not None:
                return TypedMemoryWriteResult(memory=existing, created=False)
            retired = self.get_typed_memory_by_dedupe_key(
                user_id,
                memory_type,
                dedupe_key,
                active_only=False,
            )
            if retired is not None:
                return self._revive_typed_memory(
                    retired.id,
                    text=text,
                    confidence=confidence,
                    evidence_event_id=evidence_event_id,
                    provenance=provenance,
                    scope=scope,
                )

        memory_id = uuid.uuid4().hex
        now = _now_iso()
        provenance_json = _json_dumps(provenance) if provenance is not None else None
        scope_json = _json_dumps(scope) if scope is not None else None

        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO typed_memories (
                        id, user_id, memory_type, text, confidence, evidence_event_id,
                        provenance_json, scope_json, dedupe_key, created_at, updated_at,
                        status, retired_at, retired_reason, retired_evidence_event_id,
                        superseded_by_id, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', NULL, NULL, NULL, NULL, NULL)
                    """,
                    (
                        memory_id,
                        user_id,
                        memory_type,
                        text,
                        confidence,
                        evidence_event_id,
                        provenance_json,
                        scope_json,
                        dedupe_key,
                        now,
                        now,
                    ),
                )
                conn.execute(
                    "INSERT INTO typed_memories_fts(memory_id, text) VALUES (?, ?)",
                    (memory_id, text),
                )
            except sqlite3.IntegrityError:
                if dedupe_key:
                    existing = self.get_typed_memory_by_dedupe_key(user_id, memory_type, dedupe_key)
                    if existing is not None:
                        return TypedMemoryWriteResult(memory=existing, created=False)
                raise

        return TypedMemoryWriteResult(
            memory=TypedMemoryRecord(
                id=memory_id,
                user_id=user_id,
                memory_type=memory_type,
                text=text,
                confidence=float(confidence),
                evidence_event_id=evidence_event_id,
                provenance=provenance,
                scope=scope,
                dedupe_key=dedupe_key,
                created_at=now,
                updated_at=now,
                status="active",
            ),
            created=True,
        )

    def get_typed_memory_by_dedupe_key(
        self,
        user_id: str,
        memory_type: str,
        dedupe_key: str,
        *,
        active_only: bool = True,
    ) -> TypedMemoryRecord | None:
        self.initialize()
        filters = [
            "user_id = ?",
            "memory_type = ?",
            "dedupe_key = ?",
        ]
        if active_only:
            filters.extend(["status = 'active'", "deleted_at IS NULL"])
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT * FROM typed_memories
                WHERE {' AND '.join(filters)}
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (user_id, memory_type, dedupe_key),
            ).fetchone()
        return _typed_memory_from_row(row) if row is not None else None

    def get_typed_memory(self, memory_id: str) -> TypedMemoryRecord | None:
        self.initialize()
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id is required")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM typed_memories WHERE id = ?",
                (memory_id.strip(),),
            ).fetchone()
        return _typed_memory_from_row(row) if row is not None else None

    def _revive_typed_memory(
        self,
        memory_id: str,
        *,
        text: str,
        confidence: float,
        evidence_event_id: str | None = None,
        provenance: dict[str, Any] | None = None,
        scope: dict[str, Any] | None = None,
    ) -> TypedMemoryWriteResult:
        now = _now_iso()
        provenance_json = _json_dumps(provenance) if provenance is not None else None
        scope_json = _json_dumps(scope) if scope is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE typed_memories
                SET text = ?,
                    confidence = ?,
                    evidence_event_id = ?,
                    provenance_json = ?,
                    scope_json = ?,
                    status = 'active',
                    retired_at = NULL,
                    retired_reason = NULL,
                    retired_evidence_event_id = NULL,
                    superseded_by_id = NULL,
                    deleted_at = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    text,
                    confidence,
                    evidence_event_id,
                    provenance_json,
                    scope_json,
                    now,
                    memory_id,
                ),
            )
            conn.execute("DELETE FROM typed_memories_fts WHERE memory_id = ?", (memory_id,))
            conn.execute(
                "INSERT INTO typed_memories_fts(memory_id, text) VALUES (?, ?)",
                (memory_id, text),
            )
            row = conn.execute(
                "SELECT * FROM typed_memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return TypedMemoryWriteResult(memory=_typed_memory_from_row(row), created=True)

    def search_typed_memories(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> list[TypedMemoryRecord]:
        self.initialize()
        limit = max(1, min(int(limit), 50))
        fts_query = _fts_or_query_from_user_query(query)
        if not fts_query:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT tm.*
                FROM typed_memories_fts
                JOIN typed_memories tm ON tm.id = typed_memories_fts.memory_id
                WHERE typed_memories_fts.text MATCH ?
                    AND tm.user_id = ?
                    AND tm.status = 'active'
                    AND tm.deleted_at IS NULL
                ORDER BY tm.confidence DESC, tm.updated_at DESC
                LIMIT ?
                """,
                (fts_query, user_id, limit),
            ).fetchall()
            if not rows:
                rows = conn.execute(
                    """
                    SELECT tm.*
                    FROM typed_memories tm
                    WHERE tm.user_id = ?
                        AND tm.text LIKE ? ESCAPE '\\'
                        AND tm.status = 'active'
                        AND tm.deleted_at IS NULL
                    ORDER BY tm.confidence DESC, tm.updated_at DESC
                    LIMIT ?
                    """,
                    (user_id, _like_pattern_from_user_query(query), limit),
                ).fetchall()
        return [_typed_memory_from_row(row) for row in rows]

    def list_active_typed_memories(
        self,
        *,
        user_id: str,
        memory_type: str | None = None,
        limit: int = 100,
    ) -> list[TypedMemoryRecord]:
        self.initialize()
        limit = max(1, min(int(limit), 500))
        clauses = ["user_id = ?", "status = 'active'", "deleted_at IS NULL"]
        params: list[Any] = [user_id]
        if memory_type:
            clauses.append("memory_type = ?")
            params.append(memory_type)
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM typed_memories
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [_typed_memory_from_row(row) for row in rows]

    def retire_typed_memory(
        self,
        memory_id: str,
        *,
        status: str = "inactive",
        reason: str | None = None,
        evidence_event_id: str | None = None,
        superseded_by_id: str | None = None,
    ) -> TypedMemoryRecord | None:
        self.initialize()
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE typed_memories
                SET status = ?,
                    retired_at = ?,
                    retired_reason = ?,
                    retired_evidence_event_id = ?,
                    superseded_by_id = ?,
                    deleted_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    now,
                    reason,
                    evidence_event_id,
                    superseded_by_id,
                    now if status == "deleted" else None,
                    now,
                    memory_id,
                ),
            )
            row = conn.execute(
                "SELECT * FROM typed_memories WHERE id = ?",
                (memory_id,),
            ).fetchone()
        return _typed_memory_from_row(row) if row is not None else None

    def create_job(
        self,
        *,
        job_type: str,
        input_data: dict[str, Any],
        dedupe_key: str | None = None,
        status: str = "pending",
    ) -> JobWriteResult:
        self.initialize()
        if dedupe_key:
            existing = self.get_job_by_dedupe_key(job_type, dedupe_key)
            if existing is not None:
                return JobWriteResult(job=existing, created=False)

        job_id = uuid.uuid4().hex
        now = _now_iso()
        with self._connect() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO jobs (
                        id, job_type, status, dedupe_key, input_json, output_json,
                        error, retry_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, NULL, NULL, 0, ?, ?)
                    """,
                    (job_id, job_type, status, dedupe_key, _json_dumps(input_data), now, now),
                )
            except sqlite3.IntegrityError:
                if dedupe_key:
                    existing = self.get_job_by_dedupe_key(job_type, dedupe_key)
                    if existing is not None:
                        return JobWriteResult(job=existing, created=False)
                raise

        return JobWriteResult(
            job=JobRecord(
                id=job_id,
                job_type=job_type,
                status=status,
                dedupe_key=dedupe_key,
                input=input_data,
                output=None,
                error=None,
                retry_count=0,
                created_at=now,
                updated_at=now,
            ),
            created=True,
        )

    def get_job(self, job_id: str) -> JobRecord | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _job_from_row(row) if row is not None else None

    def get_job_by_dedupe_key(self, job_type: str, dedupe_key: str) -> JobRecord | None:
        self.initialize()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE job_type = ? AND dedupe_key = ?",
                (job_type, dedupe_key),
            ).fetchone()
        return _job_from_row(row) if row is not None else None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


def _event_from_row(row: sqlite3.Row) -> EventRecord:
    return EventRecord(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        session_id=row["session_id"],
        source_type=str(row["source_type"]),
        source_id=row["source_id"],
        event_type=str(row["event_type"]),
        content=str(row["content"]),
        metadata=_json_loads_object(row["metadata_json"]),
        provenance=_json_loads_object(row["provenance_json"]) or {},
        dedupe_key=row["dedupe_key"],
        created_at=str(row["created_at"]),
        deleted_at=row["deleted_at"],
    )


def _typed_memory_from_row(row: sqlite3.Row) -> TypedMemoryRecord:
    return TypedMemoryRecord(
        id=str(row["id"]),
        user_id=str(row["user_id"]),
        memory_type=str(row["memory_type"]),
        text=str(row["text"]),
        confidence=float(row["confidence"]),
        evidence_event_id=row["evidence_event_id"],
        provenance=_json_loads_object(row["provenance_json"]),
        scope=_json_loads_object(row["scope_json"]),
        dedupe_key=row["dedupe_key"],
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        status=str(row["status"]),
        retired_at=row["retired_at"],
        retired_reason=row["retired_reason"],
        retired_evidence_event_id=row["retired_evidence_event_id"],
        superseded_by_id=row["superseded_by_id"],
        deleted_at=row["deleted_at"],
    )


def _job_from_row(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=str(row["id"]),
        job_type=str(row["job_type"]),
        status=str(row["status"]),
        dedupe_key=row["dedupe_key"],
        input=_json_loads_object(row["input_json"]) or {},
        output=_json_loads_object(row["output_json"]),
        error=row["error"],
        retry_count=int(row["retry_count"] or 0),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )
