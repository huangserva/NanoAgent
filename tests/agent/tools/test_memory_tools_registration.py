from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.context import ToolContext
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.memory import MemoryForgetTool, MemoryRecallTool, MemoryStoreTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.config.schema import ToolsConfig
from nanobot.memory_service.bridge import ExternalMemoryBridge
from nanobot.memory_service.service import MemoryService
from nanobot.memory_service.store import MemoryStore
from nanobot.providers.base import LLMResponse


def _load_names(external_memory) -> list[str]:
    registry = ToolRegistry()
    loader = ToolLoader(
        test_classes=[MemoryForgetTool, MemoryRecallTool, MemoryStoreTool],
    )
    ctx = ToolContext(
        config=ToolsConfig(),
        workspace="/tmp/workspace",
        external_memory=external_memory,
    )
    return loader.load(ctx, registry)


def _bridge(tmp_path, injection_mode: str) -> ExternalMemoryBridge:
    service = MemoryService(MemoryStore(tmp_path / f"{injection_mode}.db"))
    return ExternalMemoryBridge(
        service,
        workspace=tmp_path,
        injection_mode=injection_mode,
    )


def test_tools_not_registered_when_external_memory_is_none() -> None:
    assert _load_names(None) == []


def test_tools_not_registered_when_injection_mode_auto_inject(tmp_path) -> None:
    assert _load_names(_bridge(tmp_path, "auto_inject")) == []


def test_tools_registered_when_injection_mode_tools_only(tmp_path) -> None:
    assert sorted(_load_names(_bridge(tmp_path, "tools_only"))) == [
        "memory_forget",
        "memory_recall",
        "memory_store",
    ]


def test_tools_registered_when_injection_mode_both(tmp_path) -> None:
    assert sorted(_load_names(_bridge(tmp_path, "both"))) == [
        "memory_forget",
        "memory_recall",
        "memory_store",
    ]


def _capturing_provider(response: str = "answer"):
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = SimpleNamespace(
        max_tokens=4096,
        temperature=0.1,
        reasoning_effort=None,
    )
    provider.estimate_prompt_tokens.return_value = (100, "test")
    captured: list[list[dict]] = []

    async def chat_with_retry(*, messages, **_kwargs):
        captured.append(copy.deepcopy(messages))
        return LLMResponse(content=response, tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    return provider, captured


@pytest.mark.asyncio
async def test_auto_inject_skipped_when_injection_mode_tools_only(tmp_path) -> None:
    provider, captured = _capturing_provider()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        max_iterations=1,
        external_memory_bridge=_bridge(tmp_path, "tools_only"),
    )
    loop._retrieve_external_memory = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("auto-injection should be skipped")
    )

    result = await loop.process_direct(
        "What do you remember?",
        session_key="cli:memory",
        channel="cli",
        chat_id="memory",
    )

    assert result is not None
    assert result.content == "answer"
    assert captured
    assert "# Relevant Memory" not in captured[0][0]["content"]
