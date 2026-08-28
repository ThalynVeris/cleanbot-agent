from __future__ import annotations

from datetime import timedelta

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from cleanbot.core.config import Settings
from cleanbot.db.database import Database
from cleanbot.db.models import (
    DeviceAction,
    utc_now,
)
from cleanbot.device_mcp.client import (
    DeviceMCPCallError,
    DeviceMCPClient,
)
from cleanbot.device_mcp.server import create_device_mcp
from cleanbot.workflow.device_approval import (
    DeviceApprovalWorkflow,
)
from cleanbot.workflow.device_control import (
    DeviceControlService,
)


class FailingDeviceMCPClient:
    async def execute(self, action):
        raise DeviceMCPCallError("internal transport details")


class FakeDeviceIntentModel:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        return AIMessage(content=('{"operation":"start_cleaning","confidence":0.92}'))


def create_device_control(
    settings: Settings,
) -> tuple[
    DeviceControlService,
    Database,
]:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    control = DeviceControlService(
        database=database,
        approval_workflow=(DeviceApprovalWorkflow(InMemorySaver())),
        mcp_client=DeviceMCPClient(create_device_mcp(database)),
    )

    return control, database


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


async def test_duplicate_approval_does_not_execute_twice(
    settings: Settings,
) -> None:
    control, database = create_device_control(settings)

    pending = await control.prepare(
        session_id="duplicate-approval-session",
        user_id="1001",
        request_id="duplicate-request-001",
        message="请开始清扫",
    )

    assert pending.action is not None

    first = await control.decide(
        action_id=pending.action.id,
        user_id="1001",
        session_id=("duplicate-approval-session"),
        approve=True,
    )

    repeated = await control.decide(
        action_id=pending.action.id,
        user_id="1001",
        session_id=("duplicate-approval-session"),
        approve=True,
    )

    assert first.result is not None
    assert repeated.result is not None

    assert first.result.idempotent_replay is False
    assert repeated.result.idempotent_replay is True

    assert repeated.action is not None
    assert repeated.action.status == ("succeeded")

    device = database.get_user_device_status("1001")
    assert device.status == "cleaning"


async def test_rejected_action_does_not_change_device(
    settings: Settings,
) -> None:
    control, database = create_device_control(settings)

    pending = await control.prepare(
        session_id="rejected-action-session",
        user_id="1001",
        request_id="rejected-request-001",
        message="请开始清扫",
    )

    assert pending.action is not None

    rejected = await control.decide(
        action_id=pending.action.id,
        user_id="1001",
        session_id="rejected-action-session",
        approve=False,
    )

    assert rejected.action is not None
    assert rejected.action.status == ("rejected")
    assert rejected.result is None
    assert "已拒绝" in rejected.message

    device = database.get_user_device_status("1001")
    assert device.status == "docked"


async def test_expired_approval_does_not_execute(
    settings: Settings,
) -> None:
    control, database = create_device_control(settings)

    pending = await control.prepare(
        session_id="expired-action-session",
        user_id="1001",
        request_id="expired-request-001",
        message="请开始清扫",
    )

    assert pending.action is not None

    with database.session() as db:
        stored = db.get(
            DeviceAction,
            pending.action.id,
        )

        assert stored is not None

        stored.approval_expires_at = utc_now() - timedelta(seconds=1)

    expired = await control.decide(
        action_id=pending.action.id,
        user_id="1001",
        session_id="expired-action-session",
        approve=True,
    )

    assert expired.action is not None
    assert expired.action.status == "expired"
    assert expired.result is None
    assert "已经过期" in expired.message

    device = database.get_user_device_status("1001")
    assert device.status == "docked"


async def test_another_user_cannot_decide_action(
    settings: Settings,
) -> None:
    control, database = create_device_control(settings)

    pending = await control.prepare(
        session_id="ownership-action-session",
        user_id="1001",
        request_id="ownership-request-001",
        message="请开始清扫",
    )

    assert pending.action is not None

    with pytest.raises(
        ValueError,
        match="not available",
    ):
        await control.decide(
            action_id=pending.action.id,
            user_id="1002",
            session_id=("ownership-action-session"),
            approve=True,
        )

    stored = database.get_device_action(
        action_id=pending.action.id,
        user_id="1001",
        session_id=("ownership-action-session"),
    )

    assert stored is not None
    assert stored.status == "pending"

    device = database.get_user_device_status("1001")
    assert device.status == "docked"


async def test_mcp_failure_marks_action_failed(
    settings: Settings,
) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    control = DeviceControlService(
        database=database,
        approval_workflow=(DeviceApprovalWorkflow(InMemorySaver())),
        mcp_client=(FailingDeviceMCPClient()),  # type: ignore[arg-type]
    )

    pending = await control.prepare(
        session_id="mcp-failure-session",
        user_id="1001",
        request_id="mcp-failure-request",
        message="请开始清扫",
    )

    assert pending.action is not None

    failed = await control.decide(
        action_id=pending.action.id,
        user_id="1001",
        session_id="mcp-failure-session",
        approve=True,
    )

    assert failed.action is not None
    assert failed.action.status == "failed"
    assert failed.action.error_type == ("DeviceMCPCallError")
    assert failed.result is None

    assert failed.message == ("设备服务暂时不可用，本次操作未执行。")
    assert "internal transport details" not in failed.message

    device = database.get_user_device_status("1001")
    assert device.status == "docked"
