from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, AIMessageChunk

from cleanbot.core.config import Settings
from cleanbot.core.schemas import ChatRequest, KnowledgeHit, WeatherResult
from cleanbot.db.database import Database
from cleanbot.workflow.graph import CleanBotGraph
from cleanbot.workflow.service import AgentService


class FakeModel:
    async def ainvoke(self, prompt):
        if "Classify the user's request" in prompt:
            return AIMessage(content='{"intent":"knowledge","reason":"cleaning robot follow-up"}')
        return AIMessage(content="宠物家庭主刷毛发缠绕")

    async def astream(self, prompt):
        yield AIMessageChunk(content="请断电后清理主刷")
        yield AIMessageChunk(content="。[来源1]")


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


def create_service(settings: Settings):
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()
    model = FakeModel()
    retriever = FakeRetriever()
    graph = CleanBotGraph(
        database=database,
        retriever=retriever,  # type: ignore[arg-type]
        weather=FakeWeather(),  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        settings=settings,
    )
    return AgentService(database, graph, model), database, retriever  # type: ignore[arg-type]


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
