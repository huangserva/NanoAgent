from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import pytest_asyncio

from nanobot.api.server import create_app
from nanobot.memory_service.service import MemoryService
from nanobot.memory_service.store import MemoryStore

try:
    from aiohttp.test_utils import TestClient, TestServer

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

pytest_plugins = ("pytest_asyncio",)


@pytest_asyncio.fixture
async def aiohttp_client():
    clients: list[TestClient] = []

    async def _make_client(app):
        client = TestClient(TestServer(app))
        await client.start_server()
        clients.append(client)
        return client

    try:
        yield _make_client
    finally:
        for client in clients:
            await client.close()


def _service(tmp_path) -> MemoryService:
    return MemoryService(MemoryStore(tmp_path / "memory.db"))


def _app(tmp_path):
    agent = MagicMock()
    return create_app(agent, memory_service=_service(tmp_path))


def _turn_payload(**overrides):
    payload = {
        "user_id": "huang",
        "session_id": "s_123",
        "source_type": "nanobot",
        "source_id": "nanobot:cli",
        "event_type": "turn_end",
        "content": "用户刚刚讨论了 nanobot memory service",
        "metadata": {"channel": "cli"},
        "provenance": {"agent": "nanobot"},
    }
    payload.update(overrides)
    return payload


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_turn_end_writes_event(aiohttp_client, tmp_path) -> None:
    client = await aiohttp_client(_app(tmp_path))

    resp = await client.post("/v1/memory/turn/end", json=_turn_payload())

    assert resp.status == 200
    body = await resp.json()
    assert body["event_id"]
    assert body["created"] is True
    assert body["created_at"]


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_turn_end_dedupe_key_returns_existing_event(aiohttp_client, tmp_path) -> None:
    client = await aiohttp_client(_app(tmp_path))
    payload = _turn_payload(dedupe_key="turn-1")

    first = await client.post("/v1/memory/turn/end", json=payload)
    second = await client.post("/v1/memory/turn/end", json=payload | {"content": "ignored"})

    assert first.status == 200
    assert second.status == 200
    first_body = await first.json()
    second_body = await second.json()
    assert first_body["created"] is True
    assert second_body["created"] is False
    assert second_body["event_id"] == first_body["event_id"]


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_search_returns_matching_events(aiohttp_client, tmp_path) -> None:
    client = await aiohttp_client(_app(tmp_path))
    await client.post("/v1/memory/turn/end", json=_turn_payload())

    resp = await client.post(
        "/v1/memory/search",
        json={"user_id": "huang", "query": "memory service", "limit": 10},
    )

    assert resp.status == 200
    body = await resp.json()
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result["content"] == "用户刚刚讨论了 nanobot memory service"
    assert result["event_type"] == "turn_end"
    assert result["source_type"] == "nanobot"
    assert result["created_at"]


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected_min_results"),
    [
        ("memory-service", 1),
        ("foo/bar", 0),
        ("OR", 0),
        ("memory:", 1),
    ],
)
async def test_search_handles_fts_syntax_characters(
    aiohttp_client, tmp_path, query, expected_min_results
) -> None:
    client = await aiohttp_client(_app(tmp_path))
    await client.post("/v1/memory/turn/end", json=_turn_payload())

    resp = await client.post(
        "/v1/memory/search",
        json={"user_id": "huang", "query": query, "limit": 10},
    )

    assert resp.status == 200
    body = await resp.json()
    assert isinstance(body["results"], list)
    assert len(body["results"]) >= expected_min_results


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_search_documents_chinese_tokenizer_boundary(aiohttp_client, tmp_path) -> None:
    client = await aiohttp_client(_app(tmp_path))
    await client.post("/v1/memory/turn/end", json=_turn_payload())

    english = await client.post(
        "/v1/memory/search",
        json={"user_id": "huang", "query": "nanobot", "limit": 10},
    )
    chinese = await client.post(
        "/v1/memory/search",
        json={"user_id": "huang", "query": "用户", "limit": 10},
    )

    assert english.status == 200
    assert len((await english.json())["results"]) == 1
    assert chinese.status == 200
    assert (await chinese.json())["results"] == []


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_search_empty_query_returns_400(aiohttp_client, tmp_path) -> None:
    client = await aiohttp_client(_app(tmp_path))

    resp = await client.post(
        "/v1/memory/search",
        json={"user_id": "huang", "query": "   "},
    )

    assert resp.status == 400


@pytest.mark.skipif(not HAS_AIOHTTP, reason="aiohttp not installed")
@pytest.mark.asyncio
async def test_get_job_existing_and_missing(aiohttp_client, tmp_path) -> None:
    service = _service(tmp_path)
    job = service.create_job(job_type="ingest", input_data={"source": "reserved"}).job
    client = await aiohttp_client(create_app(MagicMock(), memory_service=service))

    existing = await client.get(f"/v1/memory/jobs/{job.id}")
    missing = await client.get("/v1/memory/jobs/missing")

    assert existing.status == 200
    body = await existing.json()
    assert body["id"] == job.id
    assert body["status"] == "pending"
    assert body["input"] == {"source": "reserved"}
    assert missing.status == 404
