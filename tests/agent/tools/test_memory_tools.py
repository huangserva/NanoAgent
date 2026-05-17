from __future__ import annotations

import pytest

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.memory import MemoryForgetTool, MemoryRecallTool, MemoryStoreTool
from nanobot.memory_service.bridge import ExternalMemoryBridge
from nanobot.memory_service.models import SearchRequest
from nanobot.memory_service.service import MemoryService
from nanobot.memory_service.store import MemoryStore


@pytest.fixture
def service(tmp_path) -> MemoryService:
    return MemoryService(MemoryStore(tmp_path / "memory.db"))


@pytest.fixture
def bridge(service: MemoryService, tmp_path) -> ExternalMemoryBridge:
    return ExternalMemoryBridge(service, workspace=tmp_path, injection_mode="tools_only")


def _set_sender(tool, sender_id: str = "huang") -> None:
    tool.set_context(
        RequestContext(
            channel="cli",
            chat_id="memory",
            session_key="cli:memory",
            sender_id=sender_id,
        )
    )


def _store(
    bridge: ExternalMemoryBridge,
    service: MemoryService,
    *,
    text: str,
    memory_type: str = "preference",
    sender_id: str = "huang",
):
    return service.upsert_typed_memory(
        user_id=bridge.subject_key(sender_id),
        memory_type=memory_type,
        text=text,
        confidence=0.91,
        dedupe_key=f"test:{memory_type}:{text.casefold()}",
    )


@pytest.mark.asyncio
async def test_memory_recall_returns_markdown_packet(
    bridge: ExternalMemoryBridge,
    service: MemoryService,
) -> None:
    _store(bridge, service, text="Preference: use SQLite for local memory")
    tool = MemoryRecallTool(bridge)
    _set_sender(tool)

    result = await tool.execute(query="SQLite local memory", limit=5)

    assert result.startswith("# Memory Recall")
    assert "## Structured Memory" in result
    assert "type=preference" in result
    assert "SQLite for local memory" in result


@pytest.mark.asyncio
async def test_memory_recall_returns_no_match_message(bridge: ExternalMemoryBridge) -> None:
    tool = MemoryRecallTool(bridge)
    _set_sender(tool)

    result = await tool.execute(query="nothing here")

    assert result == "No matching memories."


@pytest.mark.asyncio
async def test_memory_recall_filters_by_memory_type(
    bridge: ExternalMemoryBridge,
    service: MemoryService,
) -> None:
    _store(bridge, service, text="Preference: nanobot memory uses SQLite", memory_type="preference")
    _store(bridge, service, text="Project fact: nanobot memory uses SQLite", memory_type="project_fact")
    tool = MemoryRecallTool(bridge)
    _set_sender(tool)

    result = await tool.execute(
        query="nanobot memory SQLite",
        memory_type="project_fact",
        limit=5,
    )

    assert "type=project_fact" in result
    assert "type=preference" not in result


@pytest.mark.asyncio
async def test_memory_store_creates_typed_memory(
    bridge: ExternalMemoryBridge,
    service: MemoryService,
) -> None:
    tool = MemoryStoreTool(bridge)
    _set_sender(tool)

    result = await tool.execute(
        text="Preference: concise answers first",
        memory_type="preference",
        confidence=0.87,
    )

    assert result.startswith("Stored preference memory")
    memories = service.list_active_typed_memories(
        user_id=bridge.subject_key("huang"),
        memory_type="preference",
    )
    assert len(memories) == 1
    assert memories[0].text == "Preference: concise answers first"
    assert memories[0].confidence == pytest.approx(0.87)


@pytest.mark.asyncio
async def test_memory_store_dedupe_returns_existing_id(
    bridge: ExternalMemoryBridge,
    service: MemoryService,
) -> None:
    tool = MemoryStoreTool(bridge)
    _set_sender(tool)

    first = await tool.execute(text="Project fact: no new dependencies", memory_type="project_fact")
    second = await tool.execute(text="Project fact: no new dependencies", memory_type="project_fact")

    assert first.startswith("Stored project_fact memory")
    assert second.startswith("Memory already exists")
    memories = service.list_active_typed_memories(
        user_id=bridge.subject_key("huang"),
        memory_type="project_fact",
    )
    assert len(memories) == 1


@pytest.mark.asyncio
async def test_memory_forget_retires_matching_memory(
    bridge: ExternalMemoryBridge,
    service: MemoryService,
) -> None:
    _store(bridge, service, text="Task state: finish memory tools")
    tool = MemoryForgetTool(bridge)
    _set_sender(tool)

    result = await tool.execute(target="finish memory tools")

    assert "Retired memories:" in result
    assert "finish memory tools" in result
    assert service.list_active_typed_memories(user_id=bridge.subject_key("huang")) == []


@pytest.mark.asyncio
async def test_memory_forget_no_match(bridge: ExternalMemoryBridge) -> None:
    tool = MemoryForgetTool(bridge)
    _set_sender(tool)

    result = await tool.execute(target="missing memory")

    assert result == "No matching memory found."


def test_memory_store_scopes_by_sender(
    bridge: ExternalMemoryBridge,
    service: MemoryService,
) -> None:
    _store(bridge, service, text="Preference: scoped memory", sender_id="other")

    hits = service.search_typed_memories(
        SearchRequest(
            user_id=bridge.subject_key("huang"),
            query="scoped memory",
            limit=5,
        )
    )

    assert hits == []
