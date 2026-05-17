"""Internal bridge from AgentLoop turns to the external memory service."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.memory_service.models import SearchRequest, TurnEndRequest, TypedMemoryRecord
from nanobot.memory_service.service import MemoryService
from nanobot.utils.helpers import truncate_text

if TYPE_CHECKING:
    from nanobot.agent.loop import TurnContext
    from nanobot.bus.events import InboundMessage


@dataclass(frozen=True)
class TypedMemoryCandidate:
    memory_type: str
    text: str
    confidence: float


@dataclass(frozen=True)
class ForgetIntent:
    memory_type: str | None
    target: str
    latest_only: bool


class ExternalMemoryBridge:
    """Best-effort lifecycle adapter for the SQLite external memory service."""

    event_type = "turn_end"
    source_type = "nanobot_turn"
    _QUERY_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
    _STOPWORDS = frozenset({
        "a", "an", "and", "are", "about", "can", "could", "did", "do", "does",
        "for", "i", "in", "is", "it", "me", "my", "of", "on", "or", "please",
        "remember", "say", "should", "that", "the", "to", "was", "were",
        "what", "would", "you",
    })
    _SENTENCE_SPLIT_RE = re.compile(r"[。！？!?;\n]+")
    _PREFERENCE_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
        (re.compile(r"^(?:我(?:现在|目前)?(?:更)?喜欢|我(?:现在|目前)?偏好|我(?:现在|目前)?通常用|我习惯用|我倾向于)\s*(.+)$"), 0.82),
        (re.compile(r"^(?:I now prefer|I prefer|I like|I usually use|I tend to use)\s+(.+)$", re.IGNORECASE), 0.82),
        (re.compile(r"^my preferred\s+.+?\s+is\s+(.+)$", re.IGNORECASE), 0.78),
    )
    _PROFILE_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
        (re.compile(r"^(?:我(?:现在|目前)?主要负责|我(?:现在|目前)?负责|我的工作是|我是)\s*(.+)$"), 0.78),
        (re.compile(r"^(?:I now am responsible for|I am responsible for|I now mainly work on|I mainly work on|I now work on|I work on)\s+(.+)$", re.IGNORECASE), 0.78),
        (re.compile(r"^(?:I am|I'm)\s+((?:a|an|the)\s+.+)$", re.IGNORECASE), 0.74),
    )
    _TASK_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
        (re.compile(r"^((?:我|我们)(?:正在|在|还在)(?:做|排查|推进|调试|修复).+)$"), 0.76),
        (re.compile(r"^((?:我|我们)(?:还需要|需要继续|还没)(?:完成|修完|解决|排查).+)$"), 0.78),
        (re.compile(r"^((?:我|我们).*(?:卡在|被阻塞在|阻塞在).+)$"), 0.80),
        (re.compile(r"^((?:I am|I'm|we are|we're)\s+(?:working on|debugging|investigating|blocked on).+)$", re.IGNORECASE), 0.78),
        (re.compile(r"^((?:I|we)\s+(?:still need to|need to finish|need to complete|haven't finished|have not finished).+)$", re.IGNORECASE), 0.78),
        (re.compile(r"^((?:we are|we're)\s+(?:moving forward with|pushing forward on).+)$", re.IGNORECASE), 0.74),
    )
    _PROJECT_FACT_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
        (re.compile(r"^((?:这个项目|本项目)(?:现在)?(?:是|使用|采用|默认|需要|要求|保持).+)$"), 0.80),
        (re.compile(r"^((?:这轮|当前阶段|本阶段)(?:不要|不加|不引入|先).+)$"), 0.78),
        (re.compile(r"^((?:this project|the project)\s+(?:uses|is|keeps|defaults to|requires).+)$", re.IGNORECASE), 0.80),
        (re.compile(r"^((?:external memory|nanobot external memory)\s+(?:defaults to|is).+)$", re.IGNORECASE), 0.78),
        (re.compile(r"^((?:by default),?\s+(?:we|this project|the project|external memory|nanobot)\b.+)$", re.IGNORECASE), 0.74),
        (re.compile(r"^((?:we should not|we do not|do not|don't)\s+.+\s+(?:in this phase|for this project|for nanobot))$", re.IGNORECASE), 0.78),
        (re.compile(r"^((?:we keep|we use|we decided to|we default to)\s+.+)$", re.IGNORECASE), 0.76),
    )
    _UNSTABLE_MARKERS = frozenset({
        "today", "tomorrow", "tonight", "right now", "currently", "for now",
        "this time", "temporary", "temporarily", "刚刚", "今天", "明天", "今晚",
        "临时", "暂时", "这次", "马上", "现在",
    })
    _NON_PROFILE_STARTS = (
        "using ", "trying ", "testing ", "debugging ", "asking ", "looking ",
        "running ", "checking ",
    )
    _CORRECTION_MARKERS = (
        "不是", "不再", "改成", "更正", "纠正", "现在偏好", "现在更喜欢",
        "现在主要负责", "现在负责", " now prefer ", " now mainly work on ",
        " now work on ", " not ", " no longer ", "rather than", "instead of",
    )
    _FORGET_MARKERS = (
        "别记", "不要记", "别把", "不要把", "忘掉", "删掉", "删除",
        "forget", "delete", "don't remember", "do not remember", "don't store",
        "do not store",
    )
    _TASK_COMPLETION_PATTERNS: tuple[re.Pattern[str], ...] = (
        re.compile(r"^(?:我|我们)?(?:已经)?(?:修好了?|修复了?|解决了?|完成了?|搞定了?)\s*(.+)$"),
        re.compile(r"^(.+?)(?:已经)?(?:修好了?|修复了?|解决了?|完成了?|搞定了?)$"),
        re.compile(r"^(?:I|we)(?: have|'ve)?\s+(?:fixed|resolved|finished|completed)\s+(.+)$", re.IGNORECASE),
        re.compile(r"^(.+?)\s+(?:is|has been)\s+(?:resolved|fixed|finished|completed)$", re.IGNORECASE),
    )
    _TASK_COMPLETION_LATEST_MARKERS = (
        "it's resolved", "it is resolved", "issue resolved", "problem solved",
        "这个问题解决了", "问题解决了", "已经解决了",
    )
    _TASK_UNSTABLE_MARKERS = frozenset({
        "today", "tomorrow", "tonight", "this time", "temporary", "temporarily",
        "for now", "今天", "明天", "今晚", "临时", "暂时", "这次",
    })
    _PROJECT_UNSTABLE_MARKERS = frozenset({
        "today", "tomorrow", "tonight", "temporary", "temporarily", "今天",
        "明天", "今晚", "临时", "暂时",
    })

    def __init__(
        self,
        service: MemoryService | None = None,
        *,
        workspace: Path | None = None,
        retrieval_limit: int = 3,
        packet_char_limit: int = 2_000,
        result_char_limit: int = 420,
        event_char_limit: int = 4_000,
    ) -> None:
        self._service: MemoryService | None = service
        self.workspace = workspace
        self.retrieval_limit = max(1, retrieval_limit)
        self.packet_char_limit = max(256, packet_char_limit)
        self.result_char_limit = max(120, result_char_limit)
        self.event_char_limit = max(512, event_char_limit)

    @property
    def service(self) -> MemoryService:
        if self._service is None:
            self._service = MemoryService()
        return self._service

    def retrieve_for_turn(self, msg: InboundMessage) -> str | None:
        """Return a bounded prompt packet for the current user message."""
        query = self._retrieval_query(msg.content) if isinstance(msg.content, str) else ""
        if not query:
            return None

        user_id = self.subject_key(msg.sender_id)
        typed_results = []
        event_results = []
        try:
            typed_results = self.service.search_typed_memories(
                SearchRequest(
                    user_id=user_id,
                    query=query,
                    limit=self.retrieval_limit,
                )
            )
        except Exception:
            logger.exception("External typed memory retrieval failed")

        try:
            event_results = self.service.search(
                SearchRequest(
                    user_id=user_id,
                    query=query,
                    limit=self.retrieval_limit,
                )
            )
        except Exception:
            logger.exception("External event memory retrieval failed")

        lines: list[str] = []
        if typed_results:
            lines.append("## Structured Memory")
        for memory in typed_results[: self.retrieval_limit]:
            text = truncate_text(memory.text, self.result_char_limit)
            evidence = memory.evidence_event_id or "unknown"
            lines.append(
                f"- type={memory.memory_type} confidence={memory.confidence:.2f} "
                f"evidence={evidence} updated={memory.updated_at}: {text}"
            )

        if event_results:
            if lines:
                lines.append("")
            lines.append("## Related Events")
        for result in event_results[: self.retrieval_limit]:
            summary = self._format_result_content(result.content)
            summary = truncate_text(summary, self.result_char_limit)
            lines.append(
                f"- [{result.created_at}] event={result.id} "
                f"source={result.source_type} type={result.event_type}: {summary}"
            )

        if not lines:
            return None
        return truncate_text("\n".join(lines), self.packet_char_limit)

    def record_turn(self, ctx: TurnContext) -> None:
        """Write a compact canonical turn event after local session save."""
        if not ctx.final_content:
            return
        try:
            payload = self._turn_payload(ctx)
            event_result = self.service.turn_end(
                TurnEndRequest(
                    user_id=self.subject_key(ctx.msg.sender_id),
                    session_id=ctx.session_key,
                    source_type=self.source_type,
                    source_id=f"{ctx.msg.channel}:{ctx.msg.chat_id}",
                    event_type=self.event_type,
                    content=truncate_text(
                        json.dumps(payload, ensure_ascii=False, sort_keys=True),
                        self.event_char_limit,
                    ),
                    metadata={
                        "turn_id": ctx.turn_id,
                        "channel": ctx.msg.channel,
                        "chat_id": ctx.msg.chat_id,
                        "session_key": ctx.session_key,
                        "stop_reason": ctx.stop_reason,
                        "tools_used": list(ctx.tools_used or []),
                    },
                    provenance=self._provenance(ctx),
                    dedupe_key=self._dedupe_key(ctx),
                )
            )
        except Exception:
            logger.exception("External memory turn write failed")
            return

        try:
            if self._handle_forget_intent(ctx, event_result.event.id):
                return
            self._handle_task_completion(ctx, event_result.event.id)
            self._record_typed_memories(ctx, event_result.event.id)
        except Exception:
            logger.exception("External typed memory extraction failed")

    def subject_key(self, sender_id: str | None) -> str:
        """Scope memory by sender and workspace so runtime DBs do not cross-pollinate."""
        sender = (sender_id or "unknown").strip() or "unknown"
        if self.workspace is None:
            return sender
        digest = hashlib.sha256(str(self.workspace.expanduser().resolve()).encode("utf-8")).hexdigest()[:12]
        return f"{sender}@workspace:{digest}"

    def list_structured_memories(
        self,
        *,
        sender_id: str | None,
        memory_type: str | None = None,
        limit: int = 50,
    ) -> list[TypedMemoryRecord]:
        """Return active structured memories for the current user scope."""
        return self.service.list_active_typed_memories(
            user_id=self.subject_key(sender_id),
            memory_type=memory_type,
            limit=limit,
        )

    def forget_structured_memories(
        self,
        *,
        sender_id: str | None,
        target: str,
        memory_type: str | None = None,
        limit: int = 50,
    ) -> list[TypedMemoryRecord]:
        """Retire active structured memories matching a user-visible command target."""
        target = target.strip()
        if not target:
            return []
        active = self.list_structured_memories(
            sender_id=sender_id,
            memory_type=memory_type,
            limit=limit,
        )
        intent = ForgetIntent(memory_type=memory_type, target=target, latest_only=False)
        matches = self._match_forget_targets(active, intent)
        retired: list[TypedMemoryRecord] = []
        for memory in matches:
            retired.append(self.delete_structured_memory(memory))
        return retired

    def delete_structured_memory(
        self,
        memory: TypedMemoryRecord,
        *,
        reason: str = "command_forget",
    ) -> TypedMemoryRecord:
        """Retire one structured memory selected by a user-visible command."""
        updated = self.service.retire_typed_memory(
            memory.id,
            status="deleted",
            reason=reason,
        )
        return updated or memory

    def _turn_payload(self, ctx: TurnContext) -> dict[str, Any]:
        return {
            "user": ctx.msg.content,
            "assistant": ctx.final_content,
            "tools_used": list(ctx.tools_used or []),
            "stop_reason": ctx.stop_reason,
        }

    def _provenance(self, ctx: TurnContext) -> dict[str, Any]:
        data: dict[str, Any] = {
            "channel": ctx.msg.channel,
            "chat_id": ctx.msg.chat_id,
            "sender_id": ctx.msg.sender_id,
            "session_key": ctx.session_key,
            "turn_id": ctx.turn_id,
        }
        if self.workspace is not None:
            data["workspace"] = str(self.workspace.expanduser().resolve())
        if ctx.msg.metadata.get("message_id"):
            data["message_id"] = ctx.msg.metadata["message_id"]
        return data

    def _dedupe_key(self, ctx: TurnContext) -> str:
        message_id = ctx.msg.metadata.get("message_id")
        if message_id:
            return f"turn:{ctx.msg.channel}:{ctx.msg.chat_id}:{message_id}"

        stable_parts = [
            ctx.session_key,
            ctx.msg.channel,
            ctx.msg.chat_id,
            ctx.msg.sender_id,
            ctx.msg.timestamp.isoformat(),
            ctx.msg.content,
        ]
        digest = hashlib.sha256("\n".join(stable_parts).encode("utf-8")).hexdigest()[:24]
        return f"turn:{digest}"

    def _record_typed_memories(self, ctx: TurnContext, evidence_event_id: str) -> None:
        candidates = self._extract_typed_memories(ctx.msg.content)
        if not candidates:
            return
        user_id = self.subject_key(ctx.msg.sender_id)
        provenance = self._provenance(ctx)
        is_correction = self._is_correction_intent(ctx.msg.content)
        correction_targets = self._correction_targets(ctx.msg.content)
        scope = {
            "channel": ctx.msg.channel,
            "chat_id": ctx.msg.chat_id,
            "session_key": ctx.session_key,
        }
        for candidate in candidates:
            result = self.service.upsert_typed_memory(
                user_id=user_id,
                memory_type=candidate.memory_type,
                text=candidate.text,
                confidence=candidate.confidence,
                evidence_event_id=evidence_event_id,
                provenance=provenance,
                scope=scope,
                dedupe_key=self._typed_dedupe_key(candidate),
            )
            if is_correction:
                self._retire_conflicting_memories(
                    user_id=user_id,
                    memory_type=candidate.memory_type,
                    reason="overwrite",
                    evidence_event_id=evidence_event_id,
                    superseded_by_id=result.memory.id,
                    keep_id=result.memory.id,
                    new_text=candidate.text,
                    correction_targets=correction_targets,
                )

    def _handle_forget_intent(self, ctx: TurnContext, evidence_event_id: str) -> bool:
        intent = self._extract_forget_intent(ctx.msg.content)
        if intent is None:
            return False

        user_id = self.subject_key(ctx.msg.sender_id)
        active = self.service.list_active_typed_memories(
            user_id=user_id,
            memory_type=intent.memory_type,
            limit=50,
        )
        matches = self._match_forget_targets(active, intent)
        for memory in matches:
            self.service.retire_typed_memory(
                memory.id,
                status="deleted",
                reason="forget_intent",
                evidence_event_id=evidence_event_id,
            )
        return True

    def _handle_task_completion(self, ctx: TurnContext, evidence_event_id: str) -> None:
        intent = self._extract_task_completion_intent(ctx.msg.content)
        if intent is None:
            return
        user_id = self.subject_key(ctx.msg.sender_id)
        active = self.service.list_active_typed_memories(
            user_id=user_id,
            memory_type="task_state",
            limit=50,
        )
        matches = self._match_forget_targets(active, intent)
        for memory in matches:
            self.service.retire_typed_memory(
                memory.id,
                status="inactive",
                reason="task_completed",
                evidence_event_id=evidence_event_id,
            )

    def _retire_conflicting_memories(
        self,
        *,
        user_id: str,
        memory_type: str,
        reason: str,
        evidence_event_id: str,
        superseded_by_id: str,
        keep_id: str,
        new_text: str,
        correction_targets: list[str],
    ) -> None:
        active = self.service.list_active_typed_memories(
            user_id=user_id,
            memory_type=memory_type,
            limit=100,
        )
        for memory in active:
            if memory.id == keep_id:
                continue
            if not self._is_conflicting_memory(
                memory.text,
                new_text,
                correction_targets,
            ):
                continue
            self.service.retire_typed_memory(
                memory.id,
                status="inactive",
                reason=reason,
                evidence_event_id=evidence_event_id,
                superseded_by_id=superseded_by_id,
            )

    @classmethod
    def _is_conflicting_memory(
        cls,
        old_text: str,
        new_text: str,
        correction_targets: list[str],
    ) -> bool:
        if any(cls._memory_matches_target(old_text, target) for target in correction_targets):
            return True

        old_tokens = set(cls._meaningful_lifecycle_tokens(cls._normalize_lifecycle_text(old_text)))
        new_tokens = set(cls._meaningful_lifecycle_tokens(cls._normalize_lifecycle_text(new_text)))
        old_tokens -= {"preference", "profile", "project", "task", "state", "fact"}
        new_tokens -= {"preference", "profile", "project", "task", "state", "fact"}
        if len(old_tokens) < 3 or len(new_tokens) < 3:
            return False

        overlap = old_tokens & new_tokens
        smaller = min(len(old_tokens), len(new_tokens))
        return len(overlap) >= 2 and (len(overlap) / smaller) >= 0.5

    @classmethod
    def _correction_targets(cls, content: str) -> list[str]:
        targets: list[str] = []
        patterns = (
            r"(?:not responsible for|no longer responsible for)\s+(.+?)(?:[.;,]|$)",
            r"(?:not|no longer|rather than|instead of)\s+(.+?)(?:[.;,]|$)",
            r"(?:不是|不再是|而不是)\s*(.+?)(?:[。；;，,]|$)",
        )
        for pattern in patterns:
            for match in re.finditer(pattern, content, flags=re.IGNORECASE):
                target = cls._clean_memory_value(match.group(1))
                if target:
                    targets.append(target)
        return targets[:5]

    @classmethod
    def _extract_forget_intent(cls, content: str) -> ForgetIntent | None:
        if not isinstance(content, str) or not content.strip():
            return None
        lower = content.casefold()
        if not any(marker in lower for marker in cls._FORGET_MARKERS):
            return None

        memory_type: str | None = None
        if any(token in lower for token in ("偏好", "喜欢", "preference", "prefer", "like")):
            memory_type = "preference"
        elif any(token in lower for token in ("我是", "身份", "负责", "工程师", "profile", "responsible", "engineer")):
            memory_type = "profile_fact"
        elif any(token in lower for token in ("task_state", "task state", "任务", "待办", "阻塞", "blocked", "working on")):
            memory_type = "task_state"
        elif any(token in lower for token in ("project_fact", "project fact", "项目", "project", "规则", "约束", "default", "默认")):
            memory_type = "project_fact"

        target = cls._forget_target_text(content)
        latest_only = any(token in lower for token in ("刚才", "上次", "latest", "last", "previous"))
        return ForgetIntent(memory_type=memory_type, target=target, latest_only=latest_only)

    @classmethod
    def _extract_task_completion_intent(cls, content: str) -> ForgetIntent | None:
        if not isinstance(content, str) or not content.strip():
            return None
        lower = content.casefold().strip()
        if any(marker in lower for marker in cls._TASK_COMPLETION_LATEST_MARKERS):
            return ForgetIntent(memory_type="task_state", target="", latest_only=True)

        for raw_sentence in cls._SENTENCE_SPLIT_RE.split(content):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            for pattern in cls._TASK_COMPLETION_PATTERNS:
                match = pattern.match(sentence)
                if not match:
                    continue
                target = cls._clean_memory_value(match.group(1))
                if target:
                    return ForgetIntent(
                        memory_type="task_state",
                        target=target,
                        latest_only=False,
                    )
        return None

    @classmethod
    def _forget_target_text(cls, content: str) -> str:
        text = content
        replacements = (
            "别记住", "别记", "不要记住", "不要记", "别把", "不要把", "记下来",
            "忘掉", "删掉", "删除", "我刚才说的", "刚才说的", "我的", "我",
            "喜欢", "偏好", "我是", "负责",
            "forget that", "forget", "delete that", "delete", "don't remember",
            "do not remember", "don't store", "do not store", "my", "that",
            "prefer", "like", "preference", "profile fact", "profile", "task state",
            "task_state", "task", "project fact", "project_fact", "project", "fact",
        )
        for item in replacements:
            text = re.sub(re.escape(item), " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return cls._clean_memory_value(text)

    @classmethod
    def _match_forget_targets(
        cls,
        memories: list[Any],
        intent: ForgetIntent,
    ) -> list[Any]:
        if not memories:
            return []
        if intent.target:
            matches = [
                memory for memory in memories
                if cls._memory_matches_target(memory.text, intent.target)
            ]
            return matches[:3]
        if intent.latest_only and intent.memory_type is not None:
            return memories[:1]
        return []

    @classmethod
    def _memory_matches_target(cls, memory_text: str, target: str) -> bool:
        memory_norm = cls._normalize_lifecycle_text(memory_text)
        target_norm = cls._normalize_lifecycle_text(target)
        if not memory_norm or not target_norm:
            return False
        if target_norm in memory_norm or memory_norm in target_norm:
            return True

        target_tokens = cls._meaningful_lifecycle_tokens(target_norm)
        memory_tokens = set(cls._meaningful_lifecycle_tokens(memory_norm))
        if not target_tokens:
            return False
        overlap = sum(1 for token in target_tokens if token in memory_tokens)
        return overlap >= min(2, len(target_tokens))

    @classmethod
    def _normalize_lifecycle_text(cls, value: str) -> str:
        value = value.casefold()
        value = re.sub(
            r"^(preference|profile fact|task state|project fact):\s*",
            "",
            value,
        )
        value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
        return re.sub(r"\s+", " ", value).strip()

    @classmethod
    def _meaningful_lifecycle_tokens(cls, value: str) -> list[str]:
        return [
            token for token in cls._QUERY_TOKEN_RE.findall(value)
            if len(token) > 1 and token not in cls._STOPWORDS
        ]

    @classmethod
    def _extract_typed_memories(cls, content: str) -> list[TypedMemoryCandidate]:
        if not isinstance(content, str) or not content.strip():
            return []
        if "?" in content or "？" in content:
            return []

        candidates: list[TypedMemoryCandidate] = []
        seen: set[tuple[str, str]] = set()
        for raw_sentence in cls._SENTENCE_SPLIT_RE.split(content):
            sentence = raw_sentence.strip()
            if not sentence:
                continue
            for pattern, confidence in cls._PREFERENCE_PATTERNS:
                match = pattern.match(sentence)
                if match:
                    value = cls._clean_memory_value(match.group(1))
                    if cls._looks_stable(value):
                        text = f"Preference: {value}"
                        key = ("preference", text.casefold())
                        if key not in seen:
                            seen.add(key)
                            candidates.append(TypedMemoryCandidate("preference", text, confidence))
                    break

            for pattern, confidence in cls._PROFILE_PATTERNS:
                match = pattern.match(sentence)
                if match:
                    value = cls._clean_memory_value(match.group(1))
                    if cls._looks_stable(value) and not cls._looks_like_ephemeral_profile(value):
                        text = f"Profile fact: {value}"
                        key = ("profile_fact", text.casefold())
                        if key not in seen:
                            seen.add(key)
                            candidates.append(TypedMemoryCandidate("profile_fact", text, confidence))
                    break

            for pattern, confidence in cls._TASK_PATTERNS:
                match = pattern.match(sentence)
                if match:
                    value = cls._clean_memory_statement(match.group(1))
                    if cls._looks_like_task_state(value):
                        text = f"Task state: {value}"
                        key = ("task_state", text.casefold())
                        if key not in seen:
                            seen.add(key)
                            candidates.append(TypedMemoryCandidate("task_state", text, confidence))
                    break

            for pattern, confidence in cls._PROJECT_FACT_PATTERNS:
                match = pattern.match(sentence)
                if match:
                    value = cls._clean_memory_statement(match.group(1))
                    if cls._looks_like_project_fact(value):
                        text = f"Project fact: {value}"
                        key = ("project_fact", text.casefold())
                        if key not in seen:
                            seen.add(key)
                            candidates.append(TypedMemoryCandidate("project_fact", text, confidence))
                    break
        return candidates

    @classmethod
    def _clean_memory_value(cls, value: str) -> str:
        value = re.sub(r"\s+", " ", value.strip())
        value = re.split(
            r"(?:\s+not\s+|\s+rather than\s+|\s+instead of\s+|，?不是|，?而不是)",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        value = value.strip(" \t\r\n。.!！,，;；:：\"'“”‘’")
        if value.endswith("了") and len(value) > 1:
            value = value[:-1].rstrip()
        return truncate_text(value, 180)

    @classmethod
    def _clean_memory_statement(cls, value: str) -> str:
        value = re.sub(r"\s+", " ", value.strip())
        value = value.strip(" \t\r\n。.!！,，;；:：\"'“”‘’")
        return truncate_text(value, 220)

    @classmethod
    def _looks_stable(cls, value: str) -> bool:
        if len(value) < 2:
            return False
        lower = value.casefold()
        return not any(marker in lower for marker in cls._UNSTABLE_MARKERS)

    @classmethod
    def _looks_like_ephemeral_profile(cls, value: str) -> bool:
        lower = value.casefold()
        if lower.startswith(cls._NON_PROFILE_STARTS):
            return True
        return any(token in value for token in ("来问", "想问", "问一下"))

    @classmethod
    def _looks_like_task_state(cls, value: str) -> bool:
        if len(value) < 6:
            return False
        lower = value.casefold()
        return not any(marker in lower for marker in cls._TASK_UNSTABLE_MARKERS)

    @classmethod
    def _looks_like_project_fact(cls, value: str) -> bool:
        if len(value) < 6:
            return False
        lower = value.casefold()
        return not any(marker in lower for marker in cls._PROJECT_UNSTABLE_MARKERS)

    @classmethod
    def _is_correction_intent(cls, content: str) -> bool:
        padded = f" {content.casefold()} "
        return any(marker in padded for marker in cls._CORRECTION_MARKERS)

    @classmethod
    def _typed_dedupe_key(cls, candidate: TypedMemoryCandidate) -> str:
        normalized = re.sub(r"\s+", " ", candidate.text.casefold()).strip()
        digest = hashlib.sha256(
            f"{candidate.memory_type}:{normalized}".encode("utf-8")
        ).hexdigest()[:24]
        return f"{candidate.memory_type}:{digest}"

    @classmethod
    def _retrieval_query(cls, query: str) -> str:
        tokens = [
            token
            for token in cls._QUERY_TOKEN_RE.findall(query)
            if token and token.lower() not in cls._STOPWORDS
        ]
        return " ".join(tokens[:6]) if tokens else query.strip()

    @staticmethod
    def _format_result_content(content: str) -> str:
        try:
            payload = json.loads(content)
        except Exception:
            return content
        if not isinstance(payload, dict):
            return content

        pieces: list[str] = []
        user = payload.get("user")
        assistant = payload.get("assistant")
        if isinstance(user, str) and user.strip():
            pieces.append(f"User: {user.strip()}")
        if isinstance(assistant, str) and assistant.strip():
            pieces.append(f"Assistant: {assistant.strip()}")
        return " | ".join(pieces) if pieces else content
