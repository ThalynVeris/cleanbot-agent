from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from cleanbot.api.app import create_app
from cleanbot.core.config import Settings
from cleanbot.core.schemas import ChatEvent, IngestResult
from cleanbot.db.database import Database
from cleanbot.db.models import (
    DeviceActionName,
)
from cleanbot.workflow.device_control import (
    DeviceControlOutcome,
)


class FakeKnowledgeBase:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def count(self) -> int:
        return 7

    def ingest_path(self, path: Path) -> IngestResult:
        return IngestResult(
            document_id="uploaded-doc",
            filename=path.name,
            content_hash="abc123",
            chunk_count=1,
            status="created",
        )

    def delete_document(self, document_id: str) -> bool:
        if document_id != "uploaded-doc":
            return False
        self.deleted.append(document_id)
        return True


class FakeDeviceMCP:
    def __init__(self) -> None:
        self.healthy = True

    async def health(self) -> bool:
        return self.healthy


class FakeRetriever:
    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate(self) -> None:
        self.invalidations += 1


class FakeAgent:
    async def stream(self, request):
        yield ChatEvent(event="token", request_id="request-1", data={"text": "测试回答"})
        yield ChatEvent(event="done", request_id="request-1", data={"intent": "knowledge"})


class FakeDeviceControl:
    def __init__(
        self,
        database: Database,
    ) -> None:
        self.database = database
        self.calls: list[dict[str, object]] = []

    async def decide(
        self,
        *,
        action_id: str,
        user_id: str,
        session_id: str,
        approve: bool,
    ) -> DeviceControlOutcome:
        self.calls.append(
            {
                "action_id": action_id,
                "user_id": user_id,
                "session_id": session_id,
                "approve": approve,
            }
        )

        action = self.database.decide_device_action(
            action_id=action_id,
            user_id=user_id,
            session_id=session_id,
            approve=approve,
        )

        return DeviceControlOutcome(
            kind="answer",
            message=("操作已批准。" if approve else "操作已拒绝。"),
            action=action,
        )


class FakeContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings)
        self.knowledge_base = FakeKnowledgeBase()
        self.device_mcp = FakeDeviceMCP()
        self.retriever = FakeRetriever()
        self.agent = FakeAgent()
        self.device_control = FakeDeviceControl(self.database)

    def initialize(self) -> None:
        self.database.create_schema()
        self.database.seed_demo_data()


def test_health_demo_chat_and_admin_knowledge_endpoints(settings: Settings) -> None:
    container = FakeContainer(settings)
    app = create_app(container)  # type: ignore[arg-type]
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["chunks"] == 7
        assert health.json()["device_mcp"] == "ok"

        users = client.get("/api/v1/demo/users").json()
        assert len(users) == 10
        months = client.get("/api/v1/demo/users/1001/months").json()
        assert months[0] == "2025-12"

        stream = client.post(
            "/api/v1/chat/stream",
            json={
                "session_id": "api-session-01",
                "user_id": "1001",
                "message": "扫地机器人主刷不转",
            },
        )
        assert stream.status_code == 200
        assert "event: token" in stream.text
        assert "测试回答" in stream.text

        unauthorized = client.post("/api/v1/knowledge/documents", files={"file": ("manual.txt", b"hello")})
        assert unauthorized.status_code == 401
        uploaded = client.post(
            "/api/v1/knowledge/documents",
            headers={"X-Admin-Token": settings.admin_token},
            files={"file": ("manual.txt", "1. 主刷维护".encode())},
        )
        assert uploaded.status_code == 201
        assert uploaded.json()["document_id"] == "uploaded-doc"

        deleted = client.delete(
            "/api/v1/knowledge/documents/uploaded-doc",
            headers={"X-Admin-Token": settings.admin_token},
        )
        assert deleted.status_code == 200
        assert container.retriever.invalidations == 2


def test_list_user_sessions_endpoint(settings: Settings) -> None:
    container = FakeContainer(settings)
    app = create_app(container)
    with TestClient(app) as client:
        container.database.ensure_session("api-session-list", "1001")
        container.database.add_message(
            "api-session-list",
            "user",
            "主刷被宠物毛发缠住怎么办？",
        )
        response = client.get("/api/v1/demo/users/1001/sessions")

        assert response.status_code == 200
        assert len(response.json()) == 1
        assert response.json()[0]["id"] == "api-session-list"
        assert response.json()[0]["title"] == "主刷被宠物毛发缠住怎么办？"

        other_user = client.get("/api/v1/demo/users/1002/sessions")
        assert other_user.status_code == 200
        assert other_user.json() == []


def test_health_returns_503_when_device_mcp_is_unhealthy(
    settings: Settings,
) -> None:
    container = FakeContainer(settings)
    container.device_mcp.healthy = False
    app = create_app(container)  # type: ignore[arg-type]

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == ("Health check failed: RuntimeError")


def test_pending_and_device_decision_endpoints(
    settings: Settings,
) -> None:
    container = FakeContainer(settings)
    app = create_app(container)

    with TestClient(app) as client:
        container.database.ensure_session(
            "device-api-session",
            "1001",
        )

        pending = container.database.create_pending_device_action(
            session_id="device-api-session",
            user_id="1001",
            action_name=(DeviceActionName.START_CLEANING),
            idempotency_key=("device-api-request:start_cleaning"),
            checkpoint_thread_id=("device:device-api-session:request-001"),
        )

        restored = client.get(
            "/api/v1/device/actions/pending",
            params={
                "session_id": ("device-api-session"),
                "user_id": "1001",
            },
        )

        assert restored.status_code == 200
        assert restored.json()["id"] == pending.id
        assert restored.json()["status"] == ("pending")

        hidden = client.get(
            "/api/v1/device/actions/pending",
            params={
                "session_id": ("device-api-session"),
                "user_id": "1002",
            },
        )

        assert hidden.status_code == 200
        assert hidden.json() is None

        decision = client.post(
            (f"/api/v1/device/actions/{pending.id}/decision"),
            json={
                "user_id": "1001",
                "session_id": ("device-api-session"),
                "decision": "approve",
            },
        )

        assert decision.status_code == 200
        assert decision.json()["action"]["status"] == "approved"

        calls = container.device_control.calls

        assert len(calls) == 1
        assert calls[0]["approve"] is True


def test_device_decision_rejects_invalid_value(
    settings: Settings,
) -> None:
    container = FakeContainer(settings)
    app = create_app(container)

    with TestClient(app) as client:
        response = client.post(
            ("/api/v1/device/actions/missing-action/decision"),
            json={
                "user_id": "1001",
                "session_id": ("device-api-session"),
                "decision": "yes",
            },
        )

    assert response.status_code == 422
