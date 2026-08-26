from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, AIMessageChunk
from openai import APIConnectionError

from cleanbot.api.app import create_app
from cleanbot.core.config import Settings
from cleanbot.core.schemas import (
    ChatRequest,
    DeviceActionView,
    KnowledgeHit,
    WeatherResult,
)
from cleanbot.db.database import Database
from cleanbot.workflow.device_control import (
    DeviceControlOutcome,
)
from cleanbot.workflow.graph import CleanBotGraph
from cleanbot.workflow.service import AgentService


class FakeModel:
    def __init__(self) -> None:
        self.stream_calls = 0

    async def ainvoke(self, prompt):
        if "Classify the user's request" in prompt:
            return AIMessage(content='{"intent":"knowledge","reason":"cleaning robot follow-up"}')
        if "那木地板呢" in prompt and "不同地面" in prompt:
            return AIMessage(content="木地板环境扫拖机器人维护")
        return AIMessage(content="宠物家庭主刷毛发缠绕")

    async def astream(self, prompt):
        self.stream_calls += 1
        yield AIMessageChunk(content="请断电后清理主刷")
        yield AIMessageChunk(
            content="。[来源1]",
            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        )


class FailingStreamModel(FakeModel):
    async def astream(self, prompt):
        self.stream_calls += 1

        yield AIMessageChunk(content="请先断电，")

        raise APIConnectionError(
            request=httpx.Request(
                "POST",
                "https://example.invalid",
            )
        )


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def retrieve(self, query: str):
        self.queries.append(query)
        return [
            KnowledgeHit(
                document_id="doc-1",
                chunk_id="doc-1:001",
                source="维护保养.txt",
                section="主刷维护",
                content="断电后清理主刷缠绕的宠物毛发。",
                score=0.95,
            )
        ]


class FakeWeather:
    async def current(self, city: str):
        return WeatherResult(ok=False, city=city, error="offline")


class FakeDeviceControl:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def prepare(
        self,
        *,
        session_id: str,
        user_id: str,
        request_id: str,
        message: str,
    ) -> DeviceControlOutcome:
        self.calls.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "request_id": request_id,
                "message": message,
            }
        )

        return DeviceControlOutcome(
            kind="answer",
            message="模拟设备当前状态为 docked，剩余电量 85%。",
        )


class FakeApprovalDeviceControl(FakeDeviceControl):
    async def prepare(
        self,
        *,
        session_id: str,
        user_id: str,
        request_id: str,
        message: str,
    ) -> DeviceControlOutcome:
        self.calls.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "request_id": request_id,
                "message": message,
            }
        )

        now = datetime.now(timezone.utc)

        action = DeviceActionView(
            id="pending-action-001",
            user_id=user_id,
            device_id="device-1001",
            session_id=session_id,
            action="start_cleaning",
            status="pending",
            checkpoint_thread_id=(f"device:{session_id}:{request_id}"),
            approval_expires_at=(now + timedelta(minutes=30)),
            created_at=now,
        )

        return DeviceControlOutcome(
            kind="approval_required",
            message=("是否允许模拟设备执行 start_cleaning？"),
            action=action,
        )


def create_service(
    settings: Settings,
    model: FakeModel | None = None,
    device_control=None,
):
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()
    model = model or FakeModel()
    retriever = FakeRetriever()
    graph = CleanBotGraph(
        database=database,
        retriever=retriever,  # type: ignore[arg-type]
        weather=FakeWeather(),  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        settings=settings,
        device_control=device_control,
    )
    return AgentService(database, graph, model), database, retriever  # type: ignore[arg-type]


class StreamingApiContainer:
    def __init__(
        self,
        settings: Settings,
        agent: AgentService,
    ) -> None:
        self.settings = settings
        self.agent = agent

    def initialize(self) -> None:
        pass


async def collect(service: AgentService, request: ChatRequest):
    return [event async for event in service.stream(request)]


async def test_knowledge_stream_has_sources_and_persists_messages(settings: Settings) -> None:
    service, database, _ = create_service(settings)
    events = await collect(
        service,
        ChatRequest(
            session_id="session-knowledge",
            user_id="1001",
            message="扫地机器人主刷被宠物毛发缠绕如何处理？",
        ),
    )
    assert [event.event for event in events].count("source") == 1
    assert events[-1].event == "done"
    messages = database.get_messages("session-knowledge")
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[-1].sources[0].source == "维护保养.txt"


async def test_done_event_reports_generation_metrics(settings: Settings) -> None:
    service, _, _ = create_service(settings)
    events = await collect(
        service,
        ChatRequest(
            session_id="metrics-generated",
            user_id="1001",
            message="扫地机器人主刷不转怎么办？",
        ),
    )
    done = events[-1]
    assert done.event == "done"
    assert service.model.stream_calls == 1
    assert done.data["model_called"] is True
    assert done.data["first_token_ms"] >= 0
    assert done.data["latency_ms"] >= done.data["first_token_ms"]
    assert done.data["token_usage"] == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}


async def test_direct_answer_reports_no_generation_model_call(settings: Settings) -> None:
    service, _, _ = create_service(settings)
    events = await collect(
        service,
        ChatRequest(
            session_id="metrics-direct-answer",
            user_id="1001",
            message="你能做什么？",
        ),
    )
    text = "".join(event.data.get("text", "") for event in events if event.event == "token")
    done = events[-1]
    assert done.event == "done"
    assert service.model.stream_calls == 0
    assert done.data["model_called"] is False
    assert done.data["first_token_ms"] >= 0
    assert done.data["latency_ms"] >= done.data["first_token_ms"]
    assert "token_usage" not in done.data
    assert text == ("你好，我可以帮助你排查扫地机器人故障、查询使用维护知识，或生成指定月份的演示使用报告。")


async def test_follow_up_uses_history_to_rewrite_query(settings: Settings) -> None:
    service, _, retriever = create_service(settings)
    await collect(
        service,
        ChatRequest(
            session_id="session-followup",
            user_id="1001",
            message="扫地机器人主刷应该多久清理一次？",
        ),
    )
    await collect(
        service,
        ChatRequest(session_id="session-followup", user_id="1001", message="那宠物家庭呢？"),
    )
    assert retriever.queries[-1] == "宠物家庭主刷毛发缠绕"


async def test_short_follow_up_becomes_standalone_query(settings: Settings) -> None:
    service, _, retriever = create_service(settings)
    await collect(
        service,
        ChatRequest(
            session_id="session-floor-followup",
            user_id="1001",
            message="扫拖机器人在不同地面应该怎么维护？",
        ),
    )
    await collect(
        service,
        ChatRequest(
            session_id="session-floor-followup",
            user_id="1001",
            message="那木地板呢？",
        ),
    )
    assert retriever.queries[-1] == "木地板环境扫拖机器人维护"


async def test_report_uses_selected_month_and_missing_record_does_not_invent(settings: Settings) -> None:
    service, database, _ = create_service(settings)
    valid = await collect(
        service,
        ChatRequest(
            session_id="session-report1",
            user_id="1003",
            message="生成使用报告",
            month="2025-03",
        ),
    )
    assert valid[-1].event == "done"
    missing = await collect(
        service,
        ChatRequest(
            session_id="session-report2",
            user_id="1003",
            message="生成使用报告",
            month="2024-01",
        ),
    )
    text = "".join(event.data.get("text", "") for event in missing if event.event == "token")
    assert "不会生成推测报告" in text
    assert len(database.get_messages("session-report2")) == 2


async def test_explicit_message_month_overrides_ui_default(
    settings: Settings,
) -> None:
    service, _, _ = create_service(settings)

    events = await collect(
        service,
        ChatRequest(
            session_id="session-report-explicit",
            user_id="1001",
            message="生成用户 1001 在 2024-01 的使用报告",
            month="2025-12",
        ),
    )

    text = "".join(event.data.get("text", "") for event in events if event.event == "token")

    assert "2024-01" in text
    assert "不会生成推测报告" in text
    assert events[-1].data["model_called"] is False


async def test_ten_concurrent_sessions_do_not_mix(settings: Settings) -> None:
    service, database, _ = create_service(settings)
    requests = [
        ChatRequest(
            session_id=f"concurrent-{index:02d}",
            user_id=f"{1001 + index}",
            message="扫地机器人主刷不转怎么办？",
        )
        for index in range(10)
    ]
    results = await asyncio.gather(*(collect(service, request) for request in requests))
    assert all(events[-1].event == "done" for events in results)
    for index in range(10):
        messages = database.get_messages(f"concurrent-{index:02d}")
        assert len(messages) == 2
        assert all(message.session_id == f"concurrent-{index:02d}" for message in messages)


async def test_session_ownership_error_becomes_error_event(settings: Settings) -> None:
    service, database, _ = create_service(settings)
    database.ensure_session("shared-session", "1001")

    events = await collect(
        service,
        ChatRequest(
            session_id="shared-session",
            user_id="1002",
            message="继续刚才的问题",
        ),
    )
    assert [event.event for event in events] == ["error"]
    assert events[0].data["error_type"] == "SessionOwnershipError"
    assert database.get_messages("shared-session") == []


def test_model_stream_failure_returns_complete_error_event_and_persists_fallback(
    settings: Settings,
) -> None:
    failing_model = FailingStreamModel()
    service, database, _ = create_service(
        settings,
        model=failing_model,
    )
    container = StreamingApiContainer(
        settings,
        service,
    )
    app = create_app(
        container,  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "session_id": "stream-failure-session",
                "user_id": "1001",
                "message": "扫地机器人主刷不转怎么办？",
            },
        )

    assert response.status_code == 200
    assert "event: token" in response.text
    assert "请先断电" in response.text
    assert "event: error" in response.text
    assert '"error_type":"APIConnectionError"' in response.text
    assert "event: done" not in response.text
    assert response.text.endswith("\n\n")

    messages = database.get_messages("stream-failure-session")

    assert [message.role for message in messages] == [
        "user",
        "assistant",
    ]
    assert "服务暂时无法完成本次请求" in messages[-1].content
    assert "请先断电" not in messages[-1].content


def test_environment_city_prefers_explicit_message_city() -> None:
    assert CleanBotGraph._city_from_message("东京目前的天气怎么样", "上海") == "东京"
    assert CleanBotGraph._city_from_message("请问北京现在天气如何", "上海") == "北京"
    assert CleanBotGraph._city_from_message("今天的天气怎么样", "上海") == "上海"
    assert CleanBotGraph._city_from_message("根据我所在城市的实时天气给建议", "上海") == "上海"


async def test_device_request_enters_device_graph_branch(
    settings: Settings,
) -> None:
    device_control = FakeDeviceControl()

    service, _, _ = create_service(
        settings,
        device_control=device_control,
    )

    events = await collect(
        service,
        ChatRequest(
            session_id="device-status-session",
            user_id="1001",
            message="查看设备状态",
        ),
    )

    text = "".join(event.data.get("text", "") for event in events if event.event == "token")

    assert len(device_control.calls) == 1
    assert device_control.calls[0]["session_id"] == ("device-status-session")
    assert device_control.calls[0]["user_id"] == "1001"
    assert device_control.calls[0]["request_id"]
    assert events[-1].event == "done"
    assert events[-1].data["intent"] == "device"
    assert events[-1].data["model_called"] is False
    assert "docked" in text


async def test_device_write_returns_approval_event(
    settings: Settings,
) -> None:
    device_control = FakeApprovalDeviceControl()

    service, database, _ = create_service(
        settings,
        device_control=device_control,
    )

    events = await collect(
        service,
        ChatRequest(
            session_id="device-approval-session",
            user_id="1001",
            message="请开始清扫",
        ),
    )

    assert [event.event for event in events] == [
        "status",
        "status",
        "approval_required",
        "done",
    ]

    approval = events[-2]

    assert approval.data["message"] == ("是否允许模拟设备执行 start_cleaning？")
    assert approval.data["action"]["id"] == ("pending-action-001")
    assert approval.data["action"]["action"] == ("start_cleaning")
    assert approval.data["action"]["status"] == ("pending")

    done = events[-1]

    assert done.data["intent"] == "device"
    assert done.data["model_called"] is False
    assert done.data["pending_approval"] is True

    assert not any(event.event == "token" for event in events)

    messages = database.get_messages("device-approval-session")

    assert [message.role for message in messages] == [
        "user",
        "assistant",
    ]
    assert "start_cleaning" in (messages[-1].content)
