from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.memory_service.bridge import ExternalMemoryBridge
from nanobot.memory_service.models import SearchRequest
from nanobot.memory_service.service import MemoryService
from nanobot.memory_service.store import MemoryStore as SQLiteMemoryStore
from nanobot.providers.base import LLMResponse


def _service(tmp_path) -> MemoryService:
    return MemoryService(SQLiteMemoryStore(tmp_path / "memory.db"))


def _capturing_provider(responses: list[str]):
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
        content = responses[len(captured) - 1]
        return LLMResponse(content=content, tool_calls=[], usage={})

    provider.chat_with_retry = chat_with_retry
    return provider, captured


def _loop(tmp_path, provider, service: MemoryService | None = None) -> AgentLoop:
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        max_iterations=1,
        memory_service=service,
    )
    loop.tools.get_definitions = MagicMock(return_value=[])

    async def no_consolidation(*_args, **_kwargs):
        return None

    loop.consolidator.maybe_consolidate_by_tokens = no_consolidation  # type: ignore[method-assign]
    return loop


def _structured_section(system_prompt: str) -> str:
    if "## Structured Memory" not in system_prompt:
        return ""
    section = system_prompt.split("## Structured Memory", 1)[1]
    return section.split("## Related Events", 1)[0]


@pytest.mark.asyncio
async def test_agent_loop_writes_retrieves_and_injects_external_memory(tmp_path):
    service = _service(tmp_path)
    provider, captured = _capturing_provider(["answer-1", "answer-2"])
    loop = _loop(tmp_path, provider, service=service)

    first = await loop.process_direct(
        "I prefer SQLite memory service for local agent memory.",
        session_key="cli:memory",
        channel="cli",
        chat_id="memory",
    )

    assert first is not None
    assert first.content == "answer-1"
    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    hits = service.search(SearchRequest(user_id=subject, query="SQLite memory", limit=5))
    assert len(hits) == 1
    assert "SQLite memory service" in hits[0].content

    second = await loop.process_direct(
        "What did I say about SQLite memory?",
        session_key="cli:memory",
        channel="cli",
        chat_id="memory",
    )

    assert second is not None
    assert second.content == "answer-2"
    assert len(captured) == 2
    assert "# Relevant Memory" not in captured[0][0]["content"]
    second_system = captured[1][0]["content"]
    assert "# Relevant Memory" in second_system
    assert "## Structured Memory" in second_system
    assert "type=preference" in second_system
    assert "evidence=" in second_system
    assert "## Related Events" in second_system
    assert "event=" in second_system
    assert "source=nanobot_turn" in second_system
    assert "SQLite memory service" in second_system


@pytest.mark.asyncio
async def test_agent_loop_extracts_profile_fact_memory(tmp_path):
    service = _service(tmp_path)
    provider, captured = _capturing_provider(["noted", "profile answer"])
    loop = _loop(tmp_path, provider, service=service)

    first = await loop.process_direct(
        "I am responsible for nanobot memory infrastructure.",
        session_key="cli:profile",
        channel="cli",
        chat_id="profile",
    )
    assert first is not None

    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    typed = service.search_typed_memories(
        SearchRequest(user_id=subject, query="nanobot memory infrastructure", limit=5)
    )
    assert len(typed) == 1
    assert typed[0].memory_type == "profile_fact"

    second = await loop.process_direct(
        "Who handles nanobot memory infrastructure?",
        session_key="cli:profile",
        channel="cli",
        chat_id="profile",
    )

    assert second is not None
    second_system = captured[1][0]["content"]
    assert "## Structured Memory" in second_system
    assert "type=profile_fact" in second_system
    assert "nanobot memory infrastructure" in second_system


@pytest.mark.asyncio
async def test_agent_loop_does_not_extract_obviously_temporary_memory(tmp_path):
    service = _service(tmp_path)
    provider, captured = _capturing_provider(["answer-1", "answer-2"])
    loop = _loop(tmp_path, provider, service=service)

    first = await loop.process_direct(
        "I prefer SQLite memory service today.",
        session_key="cli:temporary",
        channel="cli",
        chat_id="temporary",
    )
    assert first is not None

    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    typed = service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite memory service", limit=5)
    )
    assert typed == []

    second = await loop.process_direct(
        "What did I say about SQLite memory service?",
        session_key="cli:temporary",
        channel="cli",
        chat_id="temporary",
    )

    assert second is not None
    second_system = captured[1][0]["content"]
    assert "# Relevant Memory" in second_system
    assert "## Structured Memory" not in second_system
    assert "## Related Events" in second_system
    assert "SQLite memory service today" in second_system


@pytest.mark.asyncio
async def test_agent_loop_extracts_task_state_memory(tmp_path):
    service = _service(tmp_path)
    provider, captured = _capturing_provider(["noted", "task answer"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I'm working on external memory control surface.",
        session_key="cli:task-state",
        channel="cli",
        chat_id="task-state",
    )

    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    typed = service.search_typed_memories(
        SearchRequest(user_id=subject, query="external memory control surface", limit=5)
    )
    assert len(typed) == 1
    assert typed[0].memory_type == "task_state"

    await loop.process_direct(
        "What is the status of external memory control surface?",
        session_key="cli:task-state",
        channel="cli",
        chat_id="task-state",
    )
    structured = _structured_section(captured[1][0]["content"])
    assert "type=task_state" in structured
    assert "external memory control surface" in structured


@pytest.mark.asyncio
async def test_agent_loop_task_completion_retires_matching_task_state(tmp_path):
    service = _service(tmp_path)
    provider, captured = _capturing_provider(["noted", "fixed", "answer"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I'm working on SQLite FTS tokenizer fix.",
        session_key="cli:task-complete",
        channel="cli",
        chat_id="task-complete",
    )
    await loop.process_direct(
        "I fixed SQLite FTS tokenizer fix.",
        session_key="cli:task-complete",
        channel="cli",
        chat_id="task-complete",
    )

    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    assert service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite FTS tokenizer", limit=5)
    ) == []

    await loop.process_direct(
        "What is the SQLite FTS tokenizer status?",
        session_key="cli:task-complete",
        channel="cli",
        chat_id="task-complete",
    )
    system_prompt = captured[2][0]["content"]
    assert "type=task_state" not in system_prompt


@pytest.mark.asyncio
async def test_agent_loop_extracts_project_fact_memory(tmp_path):
    service = _service(tmp_path)
    provider, captured = _capturing_provider(["noted", "project answer"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "This project uses SQLite-first external memory.",
        session_key="cli:project-fact",
        channel="cli",
        chat_id="project-fact",
    )

    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    typed = service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite-first external memory", limit=5)
    )
    assert len(typed) == 1
    assert typed[0].memory_type == "project_fact"

    await loop.process_direct(
        "What does this project use for external memory?",
        session_key="cli:project-fact",
        channel="cli",
        chat_id="project-fact",
    )
    structured = _structured_section(captured[1][0]["content"])
    assert "type=project_fact" in structured
    assert "SQLite-first external memory" in structured


def test_project_fact_extraction_ignores_personal_default_statement():
    personal = ExternalMemoryBridge._extract_typed_memories("By default I use vim.")
    assert all(candidate.memory_type != "project_fact" for candidate in personal)

    project = ExternalMemoryBridge._extract_typed_memories(
        "By default we keep external memory disabled."
    )
    assert [(candidate.memory_type, candidate.text) for candidate in project] == [
        ("project_fact", "Project fact: By default we keep external memory disabled")
    ]


@pytest.mark.asyncio
async def test_agent_loop_does_not_extract_one_shot_question_as_task_state(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["answer"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "Can you help me debug SQLite FTS tokenizer?",
        session_key="cli:not-task-state",
        channel="cli",
        chat_id="not-task-state",
    )

    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    assert service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite FTS tokenizer", limit=5)
    ) == []


@pytest.mark.asyncio
async def test_agent_loop_overwrites_preference_memory(tmp_path):
    service = _service(tmp_path)
    provider, captured = _capturing_provider(["noted sqlite", "noted postgres", "answer"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:overwrite-pref",
        channel="cli",
        chat_id="overwrite-pref",
    )
    await loop.process_direct(
        "I now prefer Postgres for local agent memory.",
        session_key="cli:overwrite-pref",
        channel="cli",
        chat_id="overwrite-pref",
    )

    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    typed = service.search_typed_memories(
        SearchRequest(user_id=subject, query="local agent memory", limit=5)
    )
    assert len(typed) == 1
    assert "Postgres" in typed[0].text
    assert "SQLite" not in typed[0].text

    raw_hits = service.search(
        SearchRequest(user_id=subject, query="SQLite local agent memory", limit=5)
    )
    assert raw_hits, "raw event history should stay available as evidence"

    await loop.process_direct(
        "What is my preference for local agent memory?",
        session_key="cli:overwrite-pref",
        channel="cli",
        chat_id="overwrite-pref",
    )
    structured = _structured_section(captured[2][0]["content"])
    assert "Postgres" in structured
    assert "SQLite" not in structured


@pytest.mark.asyncio
async def test_agent_loop_unrelated_preference_survives_overwrite(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider([
        "noted sqlite",
        "noted theme",
        "noted postgres",
    ])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:unrelated-pref",
        channel="cli",
        chat_id="unrelated-pref",
    )
    await loop.process_direct(
        "I prefer dark mode for editor theme.",
        session_key="cli:unrelated-pref",
        channel="cli",
        chat_id="unrelated-pref",
    )
    await loop.process_direct(
        "I now prefer Postgres for local agent memory.",
        session_key="cli:unrelated-pref",
        channel="cli",
        chat_id="unrelated-pref",
    )

    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    local_memory = service.search_typed_memories(
        SearchRequest(user_id=subject, query="local agent memory", limit=5)
    )
    editor_theme = service.search_typed_memories(
        SearchRequest(user_id=subject, query="editor theme", limit=5)
    )

    assert len(local_memory) == 1
    assert "Postgres" in local_memory[0].text
    assert "SQLite" not in local_memory[0].text
    assert len(editor_theme) == 1
    assert "dark mode" in editor_theme[0].text


@pytest.mark.asyncio
async def test_agent_loop_overwrites_profile_fact_memory(tmp_path):
    service = _service(tmp_path)
    provider, captured = _capturing_provider(["noted infra", "noted product", "answer"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I am responsible for infrastructure platform.",
        session_key="cli:overwrite-profile",
        channel="cli",
        chat_id="overwrite-profile",
    )
    await loop.process_direct(
        "I am not responsible for infrastructure platform; I mainly work on product strategy.",
        session_key="cli:overwrite-profile",
        channel="cli",
        chat_id="overwrite-profile",
    )

    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    typed = service.search_typed_memories(
        SearchRequest(user_id=subject, query="product strategy", limit=5)
    )
    assert len(typed) == 1
    assert typed[0].memory_type == "profile_fact"
    assert "product strategy" in typed[0].text

    await loop.process_direct(
        "What product strategy work did I mention?",
        session_key="cli:overwrite-profile",
        channel="cli",
        chat_id="overwrite-profile",
    )
    structured = _structured_section(captured[2][0]["content"])
    assert "product strategy" in structured
    assert "infrastructure platform" not in structured


@pytest.mark.asyncio
async def test_agent_loop_forget_intent_retires_structured_memory_but_keeps_events(tmp_path):
    service = _service(tmp_path)
    provider, captured = _capturing_provider(["noted", "forgot", "answer"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:forget-pref",
        channel="cli",
        chat_id="forget-pref",
    )
    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    assert service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite", limit=5)
    )

    await loop.process_direct(
        "Forget that I prefer SQLite.",
        session_key="cli:forget-pref",
        channel="cli",
        chat_id="forget-pref",
    )
    assert service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite", limit=5)
    ) == []
    assert service.search(
        SearchRequest(user_id=subject, query="SQLite local agent memory", limit=5)
    ), "raw event evidence should not be removed by forget intent"

    await loop.process_direct(
        "What about SQLite local agent memory?",
        session_key="cli:forget-pref",
        channel="cli",
        chat_id="forget-pref",
    )
    system_prompt = captured[2][0]["content"]
    assert "## Structured Memory" not in system_prompt
    assert "## Related Events" in system_prompt


@pytest.mark.asyncio
async def test_agent_loop_can_relearn_after_forget(tmp_path):
    service = _service(tmp_path)
    provider, captured = _capturing_provider(["noted", "forgot", "relearned", "answer"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:relearn-pref",
        channel="cli",
        chat_id="relearn-pref",
    )
    await loop.process_direct(
        "Forget that I prefer SQLite.",
        session_key="cli:relearn-pref",
        channel="cli",
        chat_id="relearn-pref",
    )
    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:relearn-pref",
        channel="cli",
        chat_id="relearn-pref",
    )

    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    typed = service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite local agent memory", limit=5)
    )
    assert len(typed) == 1
    assert typed[0].status == "active"
    assert "SQLite" in typed[0].text

    await loop.process_direct(
        "What is my local agent memory preference?",
        session_key="cli:relearn-pref",
        channel="cli",
        chat_id="relearn-pref",
    )
    structured = _structured_section(captured[3][0]["content"])
    assert "SQLite" in structured


@pytest.mark.asyncio
async def test_memory_command_lists_active_structured_memories(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["noted"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:memory-command-list",
        channel="cli",
        chat_id="memory-command-list",
    )

    result = await loop.process_direct(
        "/memory",
        session_key="cli:memory-command-list",
        channel="cli",
        chat_id="memory-command-list",
    )

    assert result is not None
    assert "External memory: enabled" in result.content
    assert "Active structured memories (updated desc):" in result.content
    assert "#1 [preference]" in result.content
    assert "[preference]" in result.content
    assert "SQLite for local agent memory" in result.content
    assert "updated=" in result.content
    assert "evidence=" in result.content


@pytest.mark.asyncio
async def test_memory_list_can_filter_by_type(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["noted preference", "noted profile"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:memory-command-filter",
        channel="cli",
        chat_id="memory-command-filter",
    )
    await loop.process_direct(
        "I am responsible for product strategy.",
        session_key="cli:memory-command-filter",
        channel="cli",
        chat_id="memory-command-filter",
    )

    result = await loop.process_direct(
        "/memory list preference",
        session_key="cli:memory-command-filter",
        channel="cli",
        chat_id="memory-command-filter",
    )

    assert result is not None
    assert "Active structured memories (type=preference, updated desc):" in result.content
    assert "#1 [preference]" in result.content
    assert "SQLite for local agent memory" in result.content
    assert "profile_fact" not in result.content
    assert "product strategy" not in result.content


@pytest.mark.asyncio
async def test_memory_list_supports_task_state_and_project_fact_filters(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["noted task", "noted project"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I'm working on memory command polish.",
        session_key="cli:memory-command-new-types",
        channel="cli",
        chat_id="memory-command-new-types",
    )
    await loop.process_direct(
        "This project uses SQLite-first external memory.",
        session_key="cli:memory-command-new-types",
        channel="cli",
        chat_id="memory-command-new-types",
    )

    task_result = await loop.process_direct(
        "/memory list task_state",
        session_key="cli:memory-command-new-types",
        channel="cli",
        chat_id="memory-command-new-types",
    )
    project_result = await loop.process_direct(
        "/memory list project_fact",
        session_key="cli:memory-command-new-types",
        channel="cli",
        chat_id="memory-command-new-types",
    )

    assert task_result is not None
    assert project_result is not None
    assert "Active structured memories (type=task_state, updated desc):" in task_result.content
    assert "#1 [task_state]" in task_result.content
    assert "memory command polish" in task_result.content
    assert "project_fact" not in task_result.content
    assert "Active structured memories (type=project_fact, updated desc):" in project_result.content
    assert "#1 [project_fact]" in project_result.content
    assert "SQLite-first external memory" in project_result.content
    assert "task_state" not in project_result.content


@pytest.mark.asyncio
async def test_forget_command_deletes_structured_memory_but_keeps_raw_event(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["noted"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:memory-command-forget",
        channel="cli",
        chat_id="memory-command-forget",
    )
    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    before_events = service.search(
        SearchRequest(user_id=subject, query="SQLite local agent memory", limit=5)
    )
    assert before_events

    result = await loop.process_direct(
        "/forget SQLite",
        session_key="cli:memory-command-forget",
        channel="cli",
        chat_id="memory-command-forget",
    )

    assert result is not None
    assert "Forgot 1 structured memory item(s):" in result.content
    assert "SQLite for local agent memory" in result.content
    assert service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite", limit=5)
    ) == []
    after_events = service.search(
        SearchRequest(user_id=subject, query="SQLite local agent memory", limit=5)
    )
    assert [event.id for event in after_events] == [event.id for event in before_events]


@pytest.mark.asyncio
async def test_forget_command_supports_task_state_type(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["noted task", "noted project"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I'm working on memory command polish.",
        session_key="cli:memory-command-forget-task",
        channel="cli",
        chat_id="memory-command-forget-task",
    )
    await loop.process_direct(
        "This project uses SQLite-first external memory.",
        session_key="cli:memory-command-forget-task",
        channel="cli",
        chat_id="memory-command-forget-task",
    )

    result = await loop.process_direct(
        "/forget task_state memory command polish",
        session_key="cli:memory-command-forget-task",
        channel="cli",
        chat_id="memory-command-forget-task",
    )

    assert result is not None
    assert "Forgot 1 structured memory item(s):" in result.content
    assert "[task_state]" in result.content
    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    assert service.list_active_typed_memories(
        user_id=subject,
        memory_type="task_state",
    ) == []
    project = service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite-first external memory", limit=5)
    )
    assert len(project) == 1
    assert project[0].memory_type == "project_fact"


@pytest.mark.asyncio
async def test_forget_command_supports_project_fact_type(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["noted project"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "This project uses SQLite-first external memory.",
        session_key="cli:memory-command-forget-project",
        channel="cli",
        chat_id="memory-command-forget-project",
    )

    result = await loop.process_direct(
        "/forget project_fact SQLite-first",
        session_key="cli:memory-command-forget-project",
        channel="cli",
        chat_id="memory-command-forget-project",
    )

    assert result is not None
    assert "Forgot 1 structured memory item(s):" in result.content
    assert "[project_fact]" in result.content
    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    assert service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite-first external memory", limit=5)
    ) == []


@pytest.mark.asyncio
async def test_forget_latest_deletes_only_latest_structured_memory(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["noted preference", "noted profile"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:memory-command-latest",
        channel="cli",
        chat_id="memory-command-latest",
    )
    await loop.process_direct(
        "I am responsible for product strategy.",
        session_key="cli:memory-command-latest",
        channel="cli",
        chat_id="memory-command-latest",
    )

    result = await loop.process_direct(
        "/forget latest",
        session_key="cli:memory-command-latest",
        channel="cli",
        chat_id="memory-command-latest",
    )

    assert result is not None
    assert "Forgot 1 structured memory item(s):" in result.content
    assert "product strategy" in result.content
    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    assert service.search_typed_memories(
        SearchRequest(user_id=subject, query="product strategy", limit=5)
    ) == []
    remaining = service.search_typed_memories(
        SearchRequest(user_id=subject, query="SQLite local agent memory", limit=5)
    )
    assert len(remaining) == 1
    assert "SQLite" in remaining[0].text


@pytest.mark.asyncio
async def test_forget_number_deletes_only_matching_list_position(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["noted sqlite", "noted theme"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:memory-command-number",
        channel="cli",
        chat_id="memory-command-number",
    )
    await loop.process_direct(
        "I prefer dark mode for editor theme.",
        session_key="cli:memory-command-number",
        channel="cli",
        chat_id="memory-command-number",
    )

    result = await loop.process_direct(
        "/forget #2",
        session_key="cli:memory-command-number",
        channel="cli",
        chat_id="memory-command-number",
    )

    assert result is not None
    assert "Forgot 1 structured memory item(s):" in result.content
    assert "SQLite for local agent memory" in result.content
    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    assert service.search_typed_memories(
        SearchRequest(user_id=subject, query="local agent memory", limit=5)
    ) == []
    remaining = service.search_typed_memories(
        SearchRequest(user_id=subject, query="editor theme", limit=5)
    )
    assert len(remaining) == 1
    assert "dark mode" in remaining[0].text


@pytest.mark.asyncio
async def test_forget_invalid_number_does_not_delete_memory(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["noted sqlite", "noted theme"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:memory-command-invalid-number",
        channel="cli",
        chat_id="memory-command-invalid-number",
    )
    await loop.process_direct(
        "I prefer dark mode for editor theme.",
        session_key="cli:memory-command-invalid-number",
        channel="cli",
        chat_id="memory-command-invalid-number",
    )

    result = await loop.process_direct(
        "/forget #99",
        session_key="cli:memory-command-invalid-number",
        channel="cli",
        chat_id="memory-command-invalid-number",
    )

    assert result is not None
    assert "No active structured memory numbered #99" in result.content
    assert "Current list has 2 item(s)" in result.content
    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    assert len(service.search_typed_memories(
        SearchRequest(user_id=subject, query="local agent memory", limit=5)
    )) == 1
    assert len(service.search_typed_memories(
        SearchRequest(user_id=subject, query="editor theme", limit=5)
    )) == 1


@pytest.mark.asyncio
async def test_memory_command_forget_target_uses_same_lifecycle_matching(tmp_path):
    service = _service(tmp_path)
    provider, _captured = _capturing_provider(["noted sqlite", "noted theme"])
    loop = _loop(tmp_path, provider, service=service)

    await loop.process_direct(
        "I prefer SQLite for local agent memory.",
        session_key="cli:memory-command-target",
        channel="cli",
        chat_id="memory-command-target",
    )
    await loop.process_direct(
        "I prefer dark mode for editor theme.",
        session_key="cli:memory-command-target",
        channel="cli",
        chat_id="memory-command-target",
    )

    result = await loop.process_direct(
        "/memory forget SQLite",
        session_key="cli:memory-command-target",
        channel="cli",
        chat_id="memory-command-target",
    )

    assert result is not None
    assert "Forgot 1 structured memory item(s):" in result.content
    subject = loop.external_memory.subject_key("user")  # type: ignore[union-attr]
    local_memory = service.search_typed_memories(
        SearchRequest(user_id=subject, query="local agent memory", limit=5)
    )
    editor_theme = service.search_typed_memories(
        SearchRequest(user_id=subject, query="editor theme", limit=5)
    )
    assert local_memory == []
    assert len(editor_theme) == 1
    assert "dark mode" in editor_theme[0].text


@pytest.mark.asyncio
async def test_memory_commands_report_disabled_when_external_memory_is_off(tmp_path):
    provider, captured = _capturing_provider([])
    loop = _loop(tmp_path, provider, service=None)

    memory_result = await loop.process_direct(
        "/memory",
        session_key="cli:memory-command-disabled",
        channel="cli",
        chat_id="memory-command-disabled",
    )
    forget_result = await loop.process_direct(
        "/forget SQLite",
        session_key="cli:memory-command-disabled",
        channel="cli",
        chat_id="memory-command-disabled",
    )

    assert memory_result is not None
    assert forget_result is not None
    assert "External memory is not enabled" in memory_result.content
    assert "External memory is not enabled" in forget_result.content
    assert captured == []


@pytest.mark.asyncio
async def test_agent_loop_without_external_memory_preserves_prompt_shape(tmp_path):
    provider, captured = _capturing_provider(["ok"])
    loop = _loop(tmp_path, provider, service=None)

    result = await loop.process_direct(
        "No external memory should be injected.",
        session_key="cli:disabled",
        channel="cli",
        chat_id="disabled",
    )

    assert result is not None
    assert result.content == "ok"
    assert len(captured) == 1
    assert "# Relevant Memory" not in captured[0][0]["content"]


@pytest.mark.asyncio
async def test_agent_loop_memory_failures_do_not_break_turn(tmp_path):
    class FailingMemoryService:
        def search(self, _request):
            raise RuntimeError("search unavailable")

        def turn_end(self, _request):
            raise RuntimeError("write unavailable")

    provider, captured = _capturing_provider(["still works"])
    loop = _loop(tmp_path, provider, service=FailingMemoryService())  # type: ignore[arg-type]

    result = await loop.process_direct(
        "Memory service is down.",
        session_key="cli:failing",
        channel="cli",
        chat_id="failing",
    )

    assert result is not None
    assert result.content == "still works"
    assert len(captured) == 1
    assert "# Relevant Memory" not in captured[0][0]["content"]
