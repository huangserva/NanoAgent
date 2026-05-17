from __future__ import annotations

import hashlib

import pytest

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.memory import SceneListTool, SceneReadTool, SceneWriteTool
from nanobot.memory_service.bridge import ExternalMemoryBridge
from nanobot.memory_service.service import MemoryService
from nanobot.memory_service.store import MemoryStore


@pytest.fixture
def service(tmp_path) -> MemoryService:
    return MemoryService(MemoryStore(tmp_path / "memory.db"), workspace_path=tmp_path)


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


def _scene_path(tmp_path, sender_id: str, slug: str):
    user_id = bridge_subject_key(tmp_path, sender_id)
    user_dir = hashlib.sha256(user_id.encode()).hexdigest()[:12]
    return tmp_path / "memory" / "scenes" / user_dir / f"{slug}.md"


def bridge_subject_key(tmp_path, sender_id: str) -> str:
    workspace_digest = hashlib.sha256(str(tmp_path.expanduser().resolve()).encode("utf-8")).hexdigest()[:12]
    return f"{sender_id}@workspace:{workspace_digest}"


@pytest.mark.asyncio
async def test_scene_write_creates_file_and_returns_confirmation(
    bridge: ExternalMemoryBridge,
    tmp_path,
) -> None:
    tool = SceneWriteTool(bridge)
    _set_sender(tool)

    result = await tool.execute(
        slug="agent-playbook",
        body="# Agent Playbook\n\nUse concise answers.",
        tags=["agent"],
    )

    assert result == "Scene 'agent-playbook' saved (38 chars)."
    assert _scene_path(tmp_path, "huang", "agent-playbook").exists()


@pytest.mark.asyncio
async def test_scene_write_updates_existing_scene(bridge: ExternalMemoryBridge) -> None:
    tool = SceneWriteTool(bridge)
    _set_sender(tool)

    await tool.execute(slug="agent-playbook", body="# Agent Playbook\n\nOld")
    result = await tool.execute(slug="agent-playbook", body="# Agent Playbook\n\nNew")
    read = bridge.read_scene("agent-playbook", sender_id="huang")

    assert result == "Scene 'agent-playbook' saved (21 chars)."
    assert read is not None
    assert read[1].endswith("New")


@pytest.mark.asyncio
async def test_scene_read_returns_body_with_header(bridge: ExternalMemoryBridge) -> None:
    bridge.write_scene("agent-playbook", "# Agent Playbook\n\nBody", sender_id="huang")
    tool = SceneReadTool(bridge)
    _set_sender(tool)

    result = await tool.execute(slug="agent-playbook")

    assert result.startswith("# Scene: agent-playbook (updated ")
    assert "# Agent Playbook\n\nBody" in result


@pytest.mark.asyncio
async def test_scene_read_returns_not_found_message(bridge: ExternalMemoryBridge) -> None:
    tool = SceneReadTool(bridge)
    _set_sender(tool)

    result = await tool.execute(slug="missing")

    assert result == "Scene 'missing' does not exist."


@pytest.mark.asyncio
async def test_scene_list_empty_returns_message(bridge: ExternalMemoryBridge) -> None:
    tool = SceneListTool(bridge)
    _set_sender(tool)

    result = await tool.execute()

    assert result == "No scenes found."


@pytest.mark.asyncio
async def test_scene_list_returns_markdown_list(bridge: ExternalMemoryBridge) -> None:
    bridge.write_scene(
        "agent-playbook",
        "# Agent Playbook\n\nBody",
        sender_id="huang",
        tags=["agent"],
    )
    tool = SceneListTool(bridge)
    _set_sender(tool)

    result = await tool.execute()

    assert result.startswith("# Scenes")
    assert "`agent-playbook` | Agent Playbook | tags: agent" in result


@pytest.mark.asyncio
async def test_scene_list_with_query_searches_fts(bridge: ExternalMemoryBridge) -> None:
    bridge.write_scene("agent-playbook", "# Agent Playbook\n\nDurable context", sender_id="huang")
    bridge.write_scene("other", "# Other\n\nIrrelevant", sender_id="huang")
    tool = SceneListTool(bridge)
    _set_sender(tool)

    result = await tool.execute(query="Durable context")

    assert "`agent-playbook`" in result
    assert "`other`" not in result


@pytest.mark.asyncio
async def test_scene_list_with_tag_filters(bridge: ExternalMemoryBridge) -> None:
    bridge.write_scene("agent-playbook", "# Agent Playbook\n\nBody", sender_id="huang", tags=["agent"])
    bridge.write_scene("other", "# Other\n\nBody", sender_id="huang", tags=["misc"])
    tool = SceneListTool(bridge)
    _set_sender(tool)

    result = await tool.execute(tag="agent")

    assert "`agent-playbook`" in result
    assert "`other`" not in result


def test_scene_tools_not_registered_when_external_memory_none() -> None:
    from nanobot.agent.tools.context import ToolContext

    ctx = ToolContext(config=None, workspace="/tmp/workspace", external_memory=None)

    assert SceneReadTool.enabled(ctx) is False
    assert SceneWriteTool.enabled(ctx) is False
    assert SceneListTool.enabled(ctx) is False


def test_scene_tools_not_registered_when_injection_mode_auto_inject(
    bridge: ExternalMemoryBridge,
) -> None:
    from nanobot.agent.tools.context import ToolContext

    bridge.injection_mode = "auto_inject"
    ctx = ToolContext(config=None, workspace="/tmp/workspace", external_memory=bridge)

    assert SceneReadTool.enabled(ctx) is False
    assert SceneWriteTool.enabled(ctx) is False
    assert SceneListTool.enabled(ctx) is False
