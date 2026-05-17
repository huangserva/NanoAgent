"""Business layer for the external memory service."""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_memory_service_db_path
from nanobot.memory_service.models import (
    EventWriteResult,
    JobRecord,
    JobWriteResult,
    SceneRecord,
    SearchRequest,
    SearchResult,
    TurnEndRequest,
    TypedMemoryRecord,
    TypedMemoryWriteResult,
)
from nanobot.memory_service.store import MemoryStore

_SCENE_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_SCENE_BODY_CHAR_LIMIT = 65_536


class MemoryService:
    """Small service wrapper around the SQLite memory store."""

    typed_memory_types = frozenset({
        "preference",
        "profile_fact",
        "task_state",
        "project_fact",
    })

    def __init__(
        self,
        store: MemoryStore | None = None,
        *,
        workspace_path: str | Path | None = None,
    ):
        self.store = store or MemoryStore(get_memory_service_db_path())
        self.workspace_path = Path(workspace_path).expanduser() if workspace_path else None
        if self.workspace_path is not None and self.store.workspace_path is None:
            self.store.workspace_path = self.workspace_path

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

    def upsert_scene(
        self,
        *,
        user_id: str,
        slug: str,
        body: str,
        title: str | None = None,
        tags: list[str] | None = None,
        summary: str | None = None,
    ) -> SceneRecord:
        _require_text(user_id, "user_id")
        slug = _validate_scene_slug(slug)
        if not isinstance(body, str) or not body:
            raise ValueError("body is required")
        if len(body) > _SCENE_BODY_CHAR_LIMIT:
            raise ValueError("body must be 65536 characters or fewer")
        scene_dir = self._scene_dir_for(user_id)
        scene_dir.mkdir(parents=True, exist_ok=True)
        path = scene_dir / f"{slug}.md"
        tmp_path = scene_dir / f"{slug}.{uuid.uuid4().hex}.tmp"

        clean_title = _clean_optional_text(title, "title", max_length=200)
        if clean_title is None:
            clean_title = _extract_scene_title(body)
        clean_summary = _clean_optional_text(summary, "summary", max_length=400)
        clean_tags = _clean_scene_tags(tags)
        db_written = False
        self._sync_store_workspace()
        try:
            tmp_path.write_text(body, encoding="utf-8")
            record = self.store.upsert_scene(
                slug=slug,
                user_id=user_id,
                title=clean_title,
                tags=clean_tags,
                summary=clean_summary,
                body=body,
                char_count=len(body),
            )
            db_written = True
            os.replace(tmp_path, path)
            return record
        except Exception as exc:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            if isinstance(exc, OSError):
                # The DB row is written before the final atomic rename so FTS stays transactional.
                # If the disk commit fails, remove the scoped row instead of leaving an index
                # entry whose body was never committed to the durable scene file.
                if db_written:
                    self.store.delete_scene(slug, user_id)
                raise OSError(f"failed to write scene file: {path}") from exc
            raise

    def read_scene(self, *, user_id: str, slug: str) -> tuple[SceneRecord, str] | None:
        _require_text(user_id, "user_id")
        slug = _validate_scene_slug(slug)
        self._sync_store_workspace()
        record = self.store.get_scene(slug, user_id)
        if record is None:
            return None
        path = self._scene_dir_for(user_id) / f"{slug}.md"
        try:
            body = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.warning("Scene row exists but file is missing: {}", path)
            body = ""
        except OSError:
            logger.exception("Failed to read scene file: {}", path)
            body = ""
        return record, body

    def list_scenes(
        self,
        *,
        user_id: str,
        tag: str | None = None,
        limit: int = 50,
    ) -> list[SceneRecord]:
        _require_text(user_id, "user_id")
        clean_tag = _clean_optional_text(tag, "tag", max_length=32)
        self._sync_store_workspace()
        return self.store.list_scenes(user_id=user_id, tag=clean_tag, limit=limit)

    def search_scenes(
        self,
        *,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> list[SceneRecord]:
        _require_text(user_id, "user_id")
        query = _require_text(query, "query")
        self._sync_store_workspace()
        return self.store.search_scenes(user_id=user_id, query=query, limit=limit)

    def delete_scene(self, *, user_id: str, slug: str) -> SceneRecord | None:
        _require_text(user_id, "user_id")
        slug = _validate_scene_slug(slug)
        self._sync_store_workspace()
        return self.store.delete_scene(slug, user_id)

    def _scene_dir(self) -> Path:
        if self.workspace_path is None:
            raise ValueError("workspace_path is required for scene storage")
        return self.workspace_path / "memory" / "scenes"

    def _scene_dir_for(self, user_id: str) -> Path:
        return self._scene_dir() / hashlib.sha256(user_id.encode()).hexdigest()[:12]

    def _sync_store_workspace(self) -> None:
        if self.workspace_path is not None:
            self.store.workspace_path = self.workspace_path


def _require_text(value: str | None, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _validate_scene_slug(value: str | None) -> str:
    slug = _require_text(value, "slug")
    if (
        not _SCENE_SLUG_RE.fullmatch(slug)
        or ".." in slug
        or "/" in slug
        or "\\" in slug
        or slug.startswith("-")
    ):
        raise ValueError("slug must match ^[a-z0-9][a-z0-9-]{0,63}$")
    return slug


def _clean_optional_text(value: str | None, field: str, *, max_length: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if not text:
        return None
    if len(text) > max_length:
        raise ValueError(f"{field} must be {max_length} characters or fewer")
    return text


def _clean_scene_tags(tags: list[str] | None) -> list[str]:
    if tags is None:
        return []
    if not isinstance(tags, list):
        raise ValueError("tags must be a list")
    if len(tags) > 10:
        raise ValueError("tags must contain 10 items or fewer")
    clean: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError("tags must contain strings")
        normalized = tag.strip()
        if not normalized:
            continue
        if len(normalized) > 32:
            raise ValueError("tags must be 32 characters or fewer")
        if normalized not in seen:
            seen.add(normalized)
            clean.append(normalized)
    return clean


def _extract_scene_title(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            return title[:200] if title else None
    return None
