from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from cleanbot.core.config import Settings
from cleanbot.db.database import Database
from cleanbot.device_mcp.client import DeviceMCPClient
from cleanbot.device_mcp.server import create_device_mcp
from cleanbot.workflow.device_approval import (
    DeviceApprovalWorkflow,
)
from cleanbot.workflow.device_control import (
    DeviceControlService,
)


class FakeDeviceIntentModel:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return AIMessage(content=('{"operation":"start_cleaning","confidence":0.92}'))


async def test_device_control_uses_model_for_ambiguous_command(
    settings: Settings,
) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    model = FakeDeviceIntentModel()
    control = DeviceControlService(
        database=database,
        approval_workflow=DeviceApprovalWorkflow(InMemorySaver()),
        mcp_client=DeviceMCPClient(create_device_mcp(database)),
        model=model,  # type: ignore[arg-type]
    )

    outcome = await control.prepare(
        session_id="ambiguous-control-session",
        user_id="1001",
        request_id="ambiguous-request-001",
        message="让它去干活吧",
    )

    assert model.calls == 1
    assert outcome.kind == "approval_required"
    assert outcome.action is not None
    assert outcome.action.action == "start_cleaning"

    device = database.get_user_device_status("1001")
    assert device.status == "docked"


async def test_device_control_pauses_approves_and_executes(
    settings: Settings,
) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    mcp_server = create_device_mcp(database)
    control = DeviceControlService(
        database=database,
        approval_workflow=DeviceApprovalWorkflow(InMemorySaver()),
        mcp_client=DeviceMCPClient(mcp_server),
    )

    pending = await control.prepare(
        session_id="control-session",
        user_id="1001",
        request_id="request-control-001",
        message="请开始清扫",
    )

    assert pending.kind == "approval_required"
    assert pending.action is not None
    assert pending.action.status == "pending"

    stored_pending = database.get_pending_device_action(
        session_id="control-session",
        user_id="1001",
    )
    assert stored_pending is not None
    assert stored_pending.id == pending.action.id

    completed = await control.decide(
        action_id=pending.action.id,
        user_id="1001",
        session_id="control-session",
        approve=True,
    )

    assert completed.kind == "answer"
    assert completed.result is not None
    assert completed.result.ok is True
    assert completed.result.device_status == "cleaning"

    device = database.get_user_device_status("1001")
    assert device.status == "cleaning"
    assert completed.action is not None
    assert completed.action.status == "succeeded"
