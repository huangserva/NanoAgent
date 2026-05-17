"""aiohttp routes for the external memory service."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from aiohttp import web

from nanobot.memory_service.models import SearchRequest, TurnEndRequest
from nanobot.memory_service.service import MemoryService

MEMORY_SERVICE_APP_KEY = web.AppKey("memory_service", MemoryService)


def register_memory_routes(app: web.Application, service: MemoryService | None = None) -> None:
    app[MEMORY_SERVICE_APP_KEY] = service or MemoryService()
    app.router.add_post("/v1/memory/turn/end", handle_turn_end)
    app.router.add_post("/v1/memory/search", handle_search)
    app.router.add_post("/v1/memory/typed", handle_upsert_typed)
    app.router.add_get("/v1/memory/typed", handle_list_typed)
    app.router.add_get("/v1/memory/typed/{id}", handle_get_typed)
    app.router.add_post("/v1/memory/typed/{id}/retire", handle_retire_typed)
    app.router.add_get("/v1/memory/jobs/{id}", handle_get_job)


def create_memory_app(service: MemoryService | None = None) -> web.Application:
    app = web.Application()
    register_memory_routes(app, service=service)
    return app


async def handle_turn_end(request: web.Request) -> web.Response:
    try:
        body = await _read_json(request)
        payload = TurnEndRequest(
            user_id=_string_field(body, "user_id"),
            session_id=_optional_string_field(body, "session_id"),
            source_type=_string_field(body, "source_type"),
            source_id=_optional_string_field(body, "source_id"),
            event_type=_string_field(body, "event_type"),
            content=_string_field(body, "content"),
            metadata=_optional_object_field(body, "metadata"),
            provenance=_optional_object_field(body, "provenance"),
            dedupe_key=_optional_string_field(body, "dedupe_key"),
        )
        result = _service(request).turn_end(payload)
    except ValueError as exc:
        return _error_json(400, str(exc))

    return web.json_response(
        {
            "event_id": result.event.id,
            "created_at": result.event.created_at,
            "created": result.created,
        }
    )


async def handle_search(request: web.Request) -> web.Response:
    try:
        body = await _read_json(request)
        payload = SearchRequest(
            user_id=_string_field(body, "user_id"),
            query=_string_field(body, "query"),
            limit=_limit_field(body.get("limit", 10)),
        )
        results = _service(request).search(payload)
    except ValueError as exc:
        return _error_json(400, str(exc))

    return web.json_response({"results": [asdict(result) for result in results]})


async def handle_get_job(request: web.Request) -> web.Response:
    job_id = request.match_info.get("id", "")
    try:
        job = _service(request).get_job(job_id)
    except ValueError as exc:
        return _error_json(400, str(exc))
    if job is None:
        return _error_json(404, "job not found", err_type="not_found")
    return web.json_response(asdict(job))


async def handle_upsert_typed(request: web.Request) -> web.Response:
    try:
        body = await _read_json(request)
        memory_type = _memory_type_field(body, "memory_type")
        result = _service(request).upsert_typed_memory(
            user_id=_string_field(body, "user_id"),
            memory_type=memory_type,
            text=_string_field(body, "text"),
            confidence=_confidence_field(body.get("confidence")),
            evidence_event_id=_optional_string_field(body, "evidence_event_id"),
            provenance=_optional_object_field(body, "provenance"),
            scope=_optional_object_field(body, "scope"),
            dedupe_key=_optional_string_field(body, "dedupe_key"),
        )
    except ValueError as exc:
        return _error_json(400, str(exc))

    return web.json_response({"memory": asdict(result.memory), "created": result.created})


async def handle_list_typed(request: web.Request) -> web.Response:
    try:
        query = request.query
        user_id = _query_string_field(query, "user_id", "userId")
        memory_type = _query_optional_memory_type(query)
        limit = _limit_field(query.get("limit", 50))
        memories = _service(request).list_active_typed_memories(
            user_id=user_id,
            memory_type=memory_type,
            limit=limit,
        )
    except ValueError as exc:
        return _error_json(400, str(exc))

    return web.json_response({"memories": [asdict(memory) for memory in memories]})


async def handle_get_typed(request: web.Request) -> web.Response:
    memory_id = request.match_info.get("id", "")
    try:
        memory = _service(request).store.get_typed_memory(memory_id)
    except ValueError as exc:
        return _error_json(400, str(exc))
    if memory is None or memory.deleted_at is not None:
        return _error_json(404, "typed memory not found", err_type="not_found")
    return web.json_response(asdict(memory))


async def handle_retire_typed(request: web.Request) -> web.Response:
    memory_id = request.match_info.get("id", "")
    try:
        body = await _read_json(request)
        status = _string_field(body, "status")
        if status not in {"inactive", "deleted"}:
            raise ValueError("status must be inactive or deleted")
        memory = _service(request).retire_typed_memory(
            memory_id,
            status=status,
            reason=_optional_string_field(body, "reason"),
            evidence_event_id=_optional_string_field(body, "evidence_event_id"),
            superseded_by_id=_optional_string_field(body, "superseded_by_id"),
        )
    except ValueError as exc:
        return _error_json(400, str(exc))
    if memory is None:
        return _error_json(404, "typed memory not found", err_type="not_found")
    return web.json_response(asdict(memory))


async def _read_json(request: web.Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception as exc:
        raise ValueError("Invalid JSON body") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _service(request: web.Request) -> MemoryService:
    service = request.app[MEMORY_SERVICE_APP_KEY]
    return service


def _string_field(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()


def _optional_string_field(payload: dict[str, Any], field: str) -> str | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    value = value.strip()
    return value or None


def _optional_object_field(payload: dict[str, Any], field: str) -> dict[str, Any] | None:
    value = payload.get(field)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _memory_type_field(payload: dict[str, Any], field: str) -> str:
    memory_type = _string_field(payload, field)
    if memory_type not in MemoryService.typed_memory_types:
        supported = ", ".join(sorted(MemoryService.typed_memory_types))
        raise ValueError(f"memory_type must be one of: {supported}")
    return memory_type


def _confidence_field(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence is required and must be a number") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return confidence


def _query_string_field(query: Any, *fields: str) -> str:
    for field in fields:
        value = query.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"{fields[0]} is required")


def _query_optional_memory_type(query: Any) -> str | None:
    values: list[str] = []
    for field in ("memory_type", "memoryType"):
        values.extend(value for value in query.getall(field, []) if isinstance(value, str))
    values = [value.strip() for value in values if value.strip()]
    if not values:
        return None
    if len(set(values)) > 1:
        raise ValueError("memory_type must not be repeated with different values")
    memory_type = values[0]
    if memory_type not in MemoryService.typed_memory_types:
        supported = ", ".join(sorted(MemoryService.typed_memory_types))
        raise ValueError(f"memory_type must be one of: {supported}")
    return memory_type


def _limit_field(value: Any) -> int:
    try:
        limit = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("limit must be an integer") from exc
    return max(1, min(limit, 100))


def _error_json(status: int, message: str, err_type: str = "invalid_request_error") -> web.Response:
    return web.json_response(
        {"error": {"message": message, "type": err_type, "code": status}},
        status=status,
    )
