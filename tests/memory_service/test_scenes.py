from __future__ import annotations

import hashlib
import os
import sqlite3

import pytest

from nanobot.memory_service import service as service_module
from nanobot.memory_service.service import MemoryService
from nanobot.memory_service.store import MemoryStore


@pytest.fixture
def service(tmp_path) -> MemoryService:
    return MemoryService(MemoryStore(tmp_path / "memory.db"), workspace_path=tmp_path)


def _scene_path(tmp_path, user_id: str, slug: str):
    user_dir = hashlib.sha256(user_id.encode()).hexdigest()[:12]
    return tmp_path / "memory" / "scenes" / user_dir / f"{slug}.md"


def test_upsert_scene_writes_file_and_indexes(service: MemoryService, tmp_path) -> None:
    record = service.upsert_scene(
        user_id="huang",
        slug="review-playbook",
        body="# Review Playbook\n\n用户喜欢先看风险。",
        tags=["review"],
        summary="Code review context",
    )

    assert record.slug == "review-playbook"
    assert record.title == "Review Playbook"
    assert record.tags == ["review"]
    assert _scene_path(tmp_path, "huang", "review-playbook").read_text(
        encoding="utf-8"
    ) == "# Review Playbook\n\n用户喜欢先看风险。"
    results = service.search_scenes(user_id="huang", query="风险", limit=10)
    assert [item.slug for item in results] == ["review-playbook"]


def test_upsert_scene_updates_in_place_preserves_created_at(service: MemoryService) -> None:
    first = service.upsert_scene(
        user_id="huang",
        slug="project-notes",
        body="# Project Notes\n\nFirst body",
    )
    second = service.upsert_scene(
        user_id="huang",
        slug="project-notes",
        body="# Project Notes\n\nSecond body",
        tags=["project"],
    )

    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    assert second.char_count == len("# Project Notes\n\nSecond body")
    read = service.read_scene(user_id="huang", slug="project-notes")
    assert read is not None
    assert read[1].endswith("Second body")


def test_get_scene_returns_none_for_missing(service: MemoryService) -> None:
    assert service.read_scene(user_id="huang", slug="missing-scene") is None


def test_get_scene_scoped_by_user(service: MemoryService) -> None:
    service.upsert_scene(user_id="huang", slug="private-scene", body="# Private\n\nBody")

    assert service.read_scene(user_id="other", slug="private-scene") is None
    service.upsert_scene(user_id="other", slug="private-scene", body="# Other\n\nBody")
    other = service.read_scene(user_id="other", slug="private-scene")
    assert other is not None
    assert other[1].endswith("Other\n\nBody")


def test_scenes_isolated_between_users(service: MemoryService, tmp_path) -> None:
    service.upsert_scene(user_id="user-a", slug="shared", body="# Shared\n\nA body")
    service.upsert_scene(user_id="user-b", slug="shared", body="# Shared\n\nB body")

    read_a = service.read_scene(user_id="user-a", slug="shared")
    read_b = service.read_scene(user_id="user-b", slug="shared")

    assert read_a is not None
    assert read_b is not None
    assert read_a[1] == "# Shared\n\nA body"
    assert read_b[1] == "# Shared\n\nB body"
    path_a = _scene_path(tmp_path, "user-a", "shared")
    path_b = _scene_path(tmp_path, "user-b", "shared")
    assert path_a != path_b
    assert path_a.exists()
    assert path_b.exists()
    assert [record.user_id for record in service.list_scenes(user_id="user-a")] == ["user-a"]


def test_list_scenes_orders_by_updated_desc(service: MemoryService) -> None:
    service.upsert_scene(user_id="huang", slug="older", body="# Older\n\nBody")
    service.upsert_scene(user_id="huang", slug="newer", body="# Newer\n\nBody")

    records = service.list_scenes(user_id="huang", limit=10)

    assert [record.slug for record in records] == ["newer", "older"]


def test_list_scenes_filters_by_tag(service: MemoryService) -> None:
    service.upsert_scene(
        user_id="huang",
        slug="with-tag",
        body="# Tagged\n\nBody",
        tags=["workflow", "review"],
    )
    service.upsert_scene(user_id="huang", slug="without-tag", body="# Untagged\n\nBody")

    records = service.list_scenes(user_id="huang", tag="workflow", limit=10)

    assert [record.slug for record in records] == ["with-tag"]


def test_search_scenes_fts_and_cjk_fallback(service: MemoryService) -> None:
    service.upsert_scene(
        user_id="huang",
        slug="agent-memory",
        body="# Agent Memory\n\nScene body mentions durable scene blocks.",
    )
    service.upsert_scene(
        user_id="huang",
        slug="chinese-scene",
        body="# 中文场景\n\n用户偏好本地优先方案。",
    )

    english = service.search_scenes(user_id="huang", query="durable scene", limit=10)
    chinese = service.search_scenes(user_id="huang", query="用户", limit=10)

    assert [record.slug for record in english] == ["agent-memory"]
    assert [record.slug for record in chinese] == ["chinese-scene"]


def test_delete_scene_soft_deletes_from_listing(service: MemoryService) -> None:
    service.upsert_scene(user_id="huang", slug="old-scene", body="# Old\n\nBody")

    deleted = service.delete_scene(user_id="huang", slug="old-scene")

    assert deleted is not None
    assert deleted.deleted_at is not None
    assert service.list_scenes(user_id="huang") == []
    with sqlite3.connect(service.store.db_path) as conn:
        row = conn.execute(
            "SELECT deleted_at FROM scenes WHERE slug = ?",
            ("old-scene",),
        ).fetchone()
    assert row is not None
    assert row[0] is not None


@pytest.mark.parametrize(
    "slug",
    [
        "Uppercase",
        "has.dot",
        "has/slash",
        "has\\back",
        "double..dot",
        "-foo",
        "",
        "用户",
        "a" * 65,
    ],
)
def test_invalid_slug_rejected(service: MemoryService, slug: str) -> None:
    with pytest.raises(ValueError):
        service.upsert_scene(user_id="huang", slug=slug, body="body")


def test_body_size_limit_enforced(service: MemoryService) -> None:
    with pytest.raises(ValueError, match="65536"):
        service.upsert_scene(user_id="huang", slug="too-large", body="x" * 65_537)


def test_legacy_scenes_table_migrates_to_composite_pk(tmp_path) -> None:
    db_path = tmp_path / "memory.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE scenes (
                slug TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                tags_json TEXT,
                summary TEXT,
                char_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO scenes (
                slug, user_id, title, tags_json, summary, char_count,
                created_at, updated_at, deleted_at
            ) VALUES ('legacy', 'huang', 'Legacy', '[]', 'summary', 4,
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', NULL)
            """
        )

    MemoryStore(db_path).initialize()

    with sqlite3.connect(db_path) as conn:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'scenes'"
        ).fetchone()[0]
        row = conn.execute(
            "SELECT user_id, slug, title FROM scenes WHERE user_id = ? AND slug = ?",
            ("huang", "legacy"),
        ).fetchone()
    assert "PRIMARY KEY (user_id, slug)" in sql
    assert row == ("huang", "legacy", "Legacy")


def test_fts_trigram_migration_repopulates_scene_body_from_disk(tmp_path) -> None:
    db_path = tmp_path / "memory.db"
    scene_dir = tmp_path / "memory" / "scenes"
    scene_dir.mkdir(parents=True)
    (scene_dir / "legacy.md").write_text("# Legacy\n\nsecretneedle body", encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE scenes (
                user_id TEXT NOT NULL,
                slug TEXT NOT NULL,
                title TEXT,
                tags_json TEXT,
                summary TEXT,
                char_count INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted_at TEXT,
                PRIMARY KEY (user_id, slug)
            )
            """
        )
        conn.execute(
            """
            INSERT INTO scenes (
                user_id, slug, title, tags_json, summary, char_count,
                created_at, updated_at, deleted_at
            ) VALUES ('huang', 'legacy', 'Legacy', '[]', '', 27,
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', NULL)
            """
        )
        conn.execute("CREATE VIRTUAL TABLE scenes_fts USING fts5(slug UNINDEXED, title, summary, body)")
        conn.execute(
            "INSERT INTO scenes_fts(slug, title, summary, body) VALUES ('legacy', 'Legacy', '', '')"
        )

    store = MemoryStore(db_path)
    store.initialize()

    results = store.search_scenes(user_id="huang", query="secretneedle", limit=10)
    assert [record.slug for record in results] == ["legacy"]


def test_concurrent_tmp_filenames_do_not_collide(
    service: MemoryService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeUUID:
        hex = "abc123def456"

    seen: dict[str, str] = {}
    real_replace = os.replace

    def capture_replace(src, dst) -> None:
        seen["tmp_name"] = os.fspath(src).split(os.sep)[-1]
        real_replace(src, dst)

    monkeypatch.setattr(service_module.uuid, "uuid4", lambda: FakeUUID())
    monkeypatch.setattr(service_module.os, "replace", capture_replace)

    service.upsert_scene(user_id="huang", slug="tmp-check", body="# Tmp\n\nBody")

    assert seen["tmp_name"] == "tmp-check.abc123def456.tmp"
