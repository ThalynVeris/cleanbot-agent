from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from cleanbot.api.container import AppContainer, get_container
from cleanbot.core.logging import configure_logging
from cleanbot.core.schemas import (
    ChatEvent,
    ChatRequest,
    ChatSessionSummary,
    DemoUser,
    DeviceActionDecisionRequest,
    DeviceActionDecisionResponse,
    DeviceActionView,
    IngestResult,
    StoredMessage,
)
from cleanbot.rag.knowledge_base import ALLOWED_SUFFIXES


def _sse(event: ChatEvent) -> str:
    payload = json.dumps(event.data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event.event}\nid: {event.request_id}\ndata: {payload}\n\n"


def create_app(container: AppContainer | None = None) -> FastAPI:
    selected_container = container

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        nonlocal selected_container
        configure_logging()
        selected_container = selected_container or get_container()
        try:
            await asyncio.to_thread(selected_container.initialize)
            application.state.container = selected_container
            yield
        finally:
            close = getattr(
                selected_container,
                "close",
                None,
            )

            if callable(close):
                await asyncio.to_thread(close)

    app = FastAPI(
        title="CleanBot Agent API",
        version="0.2.0",
        description="可评测的扫地机器人 LangGraph + Hybrid RAG 客服服务",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Admin-Token"],
    )

    def services() -> AppContainer:
        return app.state.container

    def verify_admin(token: str | None) -> None:
        expected = services().settings.admin_token
        if not token or not secrets.compare_digest(token, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid admin token")

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, Any]:
        current = services()
        try:
            user_count = len(await asyncio.to_thread(current.database.list_users))
            chunk_count = await asyncio.to_thread(current.knowledge_base.count)
            mcp_healthy = await current.device_mcp.health()

            if not mcp_healthy:
                raise RuntimeError("Device MCP is unhealthy")
            return {
                "status": "ok" if user_count > 0 else "degraded",
                "database": "ok" if user_count > 0 else "empty",
                "vector_store": "ok" if chunk_count > 0 else "empty",
                "device_mcp": "ok",
                "users": user_count,
                "chunks": chunk_count,
                "collection": current.settings.collection_name,
            }
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Health check failed: {type(exc).__name__}") from exc

    @app.post("/api/v1/chat/stream", tags=["chat"])
    async def stream_chat(request: ChatRequest) -> StreamingResponse:
        async def event_stream() -> AsyncIterator[str]:
            async for event in services().agent.stream(request):
                yield _sse(event)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get(
        "/api/v1/device/actions/pending",
        response_model=DeviceActionView | None,
        tags=["device"],
    )
    async def pending_device_action(
        session_id: str,
        user_id: str,
    ) -> DeviceActionView | None:
        return await asyncio.to_thread(
            services().database.get_pending_device_action,
            session_id=session_id,
            user_id=user_id,
        )

    @app.get("/api/v1/sessions/{session_id}/messages", response_model=list[StoredMessage], tags=["chat"])
    async def session_messages(session_id: str) -> list[StoredMessage]:
        return await asyncio.to_thread(services().database.get_messages, session_id, 100)

    @app.get("/api/v1/demo/users", response_model=list[DemoUser], tags=["demo"])
    async def demo_users() -> list[DemoUser]:
        return await asyncio.to_thread(services().database.list_users)

    @app.get("/api/v1/demo/users/{user_id}/sessions", response_model=list[ChatSessionSummary], tags=["demo"])
    async def demo_sessions(user_id: str) -> list[ChatSessionSummary]:
        return await asyncio.to_thread(services().database.list_sessions, user_id)

    @app.get("/api/v1/demo/users/{user_id}/months", response_model=list[str], tags=["demo"])
    async def demo_months(user_id: str) -> list[str]:
        return await asyncio.to_thread(services().database.list_months, user_id)

    @app.post(
        ("/api/v1/device/actions/{action_id}/decision"),
        response_model=(DeviceActionDecisionResponse),
        tags=["device"],
    )
    async def decide_device_action(
        action_id: str,
        request: DeviceActionDecisionRequest,
    ) -> DeviceActionDecisionResponse:
        try:
            outcome = await services().device_control.decide(
                action_id=action_id,
                user_id=request.user_id,
                session_id=request.session_id,
                approve=(request.decision == "approve"),
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=404,
                detail=("Device action is not available for this request"),
            ) from exc

        if outcome.action is None:
            raise HTTPException(
                status_code=500,
                detail=("Device decision returned no action state"),
            )

        return DeviceActionDecisionResponse(
            message=outcome.message,
            action=outcome.action,
            result=outcome.result,
        )

    @app.post(
        "/api/v1/knowledge/documents",
        response_model=IngestResult,
        status_code=status.HTTP_201_CREATED,
        tags=["knowledge"],
    )
    async def upload_document(
        file: Annotated[UploadFile, File(description="UTF-8 TXT or text PDF")],
        x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    ) -> IngestResult:
        verify_admin(x_admin_token)
        filename = Path(file.filename or "").name
        if not filename or Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
            raise HTTPException(status_code=415, detail="Only .txt and .pdf files are supported")
        content = await file.read(services().settings.max_upload_bytes + 1)
        if len(content) > services().settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="File is larger than the configured upload limit")
        target = services().settings.upload_dir / filename
        target.write_bytes(content)
        try:
            result = await asyncio.to_thread(services().knowledge_base.ingest_path, target)
            services().retriever.invalidate()
            return result
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/v1/knowledge/documents/{document_id}", tags=["knowledge"])
    async def delete_document(
        document_id: str,
        x_admin_token: Annotated[str | None, Header(alias="X-Admin-Token")] = None,
    ) -> dict[str, Any]:
        verify_admin(x_admin_token)
        deleted = await asyncio.to_thread(services().knowledge_base.delete_document, document_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        services().retriever.invalidate()
        return {"deleted": True, "document_id": document_id}

    return app


app = create_app()
