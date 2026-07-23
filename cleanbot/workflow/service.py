from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from cleanbot.core.logging import get_logger
from cleanbot.core.schemas import ChatEvent, ChatRequest, SourceRef
from cleanbot.db.database import Database, SessionOwnershipError
from cleanbot.workflow.graph import CleanBotGraph

logger = get_logger(__name__)


class AgentService:
    def __init__(self, database: Database, graph: CleanBotGraph, model: BaseChatModel) -> None:
        self.database = database
        self.graph = graph
        self.model = model

    async def stream(self, request: ChatRequest) -> AsyncIterator[ChatEvent]:
        request_id = str(uuid.uuid4())
        started = time.perf_counter()
        final_text = ""
        sources: list[SourceRef] = []
        intent = "unknown"

        try:
            self.database.ensure_session(request.session_id, request.user_id)
            self.database.add_message(request.session_id, "user", request.message)

            yield self._event("status", request_id, stage="routing", message="正在识别需求并加载会话")
            state = await self.graph.prepare(
                {
                    "session_id": request.session_id,
                    "user_id": request.user_id,
                    "message": request.message,
                    "month": request.month,
                }
            )
            intent = state.get("intent", "unknown")
            sources = state.get("sources", [])
            yield self._event(
                "status",
                request_id,
                stage="prepared",
                message=f"已进入 {intent} 工作流",
            )
            for source in sources:
                yield ChatEvent(
                    event="source",
                    request_id=request_id,
                    data=source.model_dump(mode="json"),
                )

            direct_answer = state.get("direct_answer")
            if direct_answer:
                async for token in self._chunk_text(direct_answer):
                    final_text += token
                    yield self._event("token", request_id, text=token)
            else:
                prompt = state.get("answer_prompt", "")
                if not prompt:
                    raise RuntimeError("Workflow did not produce an answer prompt")
                yield self._event("status", request_id, stage="generating", message="正在生成有依据的回答")
                async for chunk in self.model.astream(prompt):
                    token = self._message_text(chunk)
                    if token:
                        final_text += token
                        yield self._event("token", request_id, text=token)

            if not final_text.strip():
                final_text = "服务没有生成有效内容，请稍后重试。"
                yield self._event("token", request_id, text=final_text)
            self.database.add_message(request.session_id, "assistant", final_text, sources)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.info(
                "request_completed",
                extra={
                    "context": {
                        "request_id": request_id,
                        "session_id": request.session_id,
                        "intent": intent,
                        "source_count": len(sources),
                        "latency_ms": elapsed_ms,
                    }
                },
            )
            yield self._event(
                "done",
                request_id,
                intent=intent,
                source_count=len(sources),
                latency_ms=elapsed_ms,
            )
        except SessionOwnershipError as exc:
            yield self._event(
                "error",
                request_id,
                message="当前会话属于其他用户，请开始新会话后重试。",
                error_type=type(exc).__name__,
            )
        except Exception as exc:
            fallback = "服务暂时无法完成本次请求，请稍后重试。错误已记录，但不会返回伪造结果。"
            if not final_text:
                self.database.add_message(request.session_id, "assistant", fallback)
            logger.exception(
                "request_failed",
                extra={
                    "context": {
                        "request_id": request_id,
                        "session_id": request.session_id,
                        "error_type": type(exc).__name__,
                    }
                },
            )
            yield self._event("error", request_id, message=fallback, error_type=type(exc).__name__)

    @staticmethod
    async def _chunk_text(text: str, size: int = 24) -> AsyncIterator[str]:
        for start in range(0, len(text), size):
            yield text[start : start + size]

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        return str(content) if content is not None else ""

    @staticmethod
    def _event(event: str, request_id: str, **data: Any) -> ChatEvent:
        return ChatEvent(event=event, request_id=request_id, data=data)  # type: ignore[arg-type]
