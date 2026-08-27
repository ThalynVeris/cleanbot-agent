from __future__ import annotations

import json

import pytest
from mcp import Client

from cleanbot.core.config import Settings
from cleanbot.db.database import Database
from cleanbot.db.models import (
    DeviceAction,
    DeviceActionName,
    DeviceActionStatus,
    utc_now,
)
from cleanbot.device_mcp.client import (
    DeviceMCPCallError,
    DeviceMCPClient,
)
from cleanbot.device_mcp.server import create_device_mcp


def add_device_action(
    database: Database,
    *,
    action_id: str,
    action_name: DeviceActionName,
    status: DeviceActionStatus,
) -> None:
    database.ensure_session("device-write-session", "1001")

    with database.session() as db:
        db.add(
            DeviceAction(
                id=action_id,
                user_id="1001",
                device_id="demo-device-1001",
                session_id="device-write-session",
                action=action_name,
                idempotency_key=f"idempotency:{action_id}",
                checkpoint_thread_id="device-write-session",
                status=status,
                decided_at=(utc_now() if status is DeviceActionStatus.APPROVED else None),
            )
        )


@pytest.mark.asyncio
async def test_mcp_write_tool_requires_approved_action(
    settings: Settings,
) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    add_device_action(
        database,
        action_id="pending-start",
        action_name=DeviceActionName.START_CLEANING,
        status=DeviceActionStatus.PENDING,
    )

    server = create_device_mcp(database)

    async with Client(server) as client:
        result = await client.call_tool(
            "start_cleaning",
            {
                "action_id": "pending-start",
                "user_id": "1001",
                "device_id": "demo-device-1001",
            },
        )

        assert result.is_error is True
        assert "has not been approved" in result.content[0].text

    device = database.get_device_status(
        "1001",
        "demo-device-1001",
    )
    assert device.status == "docked"


@pytest.mark.asyncio
async def test_mcp_executes_approved_actions_idempotently(
    settings: Settings,
) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    add_device_action(
        database,
        action_id="approved-start",
        action_name=DeviceActionName.START_CLEANING,
        status=DeviceActionStatus.APPROVED,
    )

    server = create_device_mcp(database)

    async with Client(server) as client:
        first = await client.call_tool(
            "start_cleaning",
            {
                "action_id": "approved-start",
                "user_id": "1001",
                "device_id": "demo-device-1001",
            },
        )
        second = await client.call_tool(
            "start_cleaning",
            {
                "action_id": "approved-start",
                "user_id": "1001",
                "device_id": "demo-device-1001",
            },
        )

        assert first.is_error is False
        assert first.structured_content is not None
        assert first.structured_content["device_status"] == "cleaning"
        assert first.structured_content["idempotent_replay"] is False

        assert second.is_error is False
        assert second.structured_content is not None
        assert second.structured_content["device_status"] == "cleaning"
        assert second.structured_content["idempotent_replay"] is True

    device = database.get_device_status(
        "1001",
        "demo-device-1001",
    )
    assert device.status == "cleaning"


@pytest.mark.asyncio
async def test_mcp_discovers_and_calls_read_tools(
    settings: Settings,
) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    server = create_device_mcp(database)

    async with Client(server) as client:
        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools.tools}

        assert tool_names == {
            "get_device_status",
            "get_consumable_status",
            "start_cleaning",
            "pause_cleaning",
            "return_to_dock",
        }

        status = await client.call_tool(
            "get_device_status",
            {
                "user_id": "1001",
                "device_id": "demo-device-1001",
            },
        )
        consumable = await client.call_tool(
            "get_consumable_status",
            {
                "user_id": "1001",
                "device_id": "demo-device-1001",
            },
        )

        assert status.is_error is False
        assert status.structured_content is not None
        assert status.structured_content["status"] == "docked"
        assert status.structured_content["battery_percent"] == 100

        assert consumable.is_error is False
        assert consumable.structured_content is not None
        assert consumable.structured_content["consumable_percent"] == 100
        assert consumable.structured_content["replacement_recommended"] is False


@pytest.mark.asyncio
async def test_mcp_exposes_device_capability_resource(
    settings: Settings,
) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    server = create_device_mcp(database)

    async with Client(server) as client:
        templates = await client.list_resource_templates()

        assert len(templates.resource_templates) == 1
        assert templates.resource_templates[0].uri_template == "device://{device_id}/capabilities"

        result = await client.read_resource("device://demo-device-1001/capabilities")
        content = result.contents[0]
        capabilities = json.loads(content.text)

        assert capabilities["device_id"] == "demo-device-1001"
        assert capabilities["simulated"] is True
        assert capabilities["supported_actions"] == [
            "start_cleaning",
            "pause_cleaning",
            "return_to_dock",
        ]


@pytest.mark.asyncio
async def test_mcp_rejects_access_to_another_users_device(
    settings: Settings,
) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    server = create_device_mcp(database)

    async with Client(server) as client:
        result = await client.call_tool(
            "get_device_status",
            {
                "user_id": "1002",
                "device_id": "demo-device-1001",
            },
        )

        assert result.is_error is True
        assert result.structured_content is None
        assert "not available for this user" in result.content[0].text


@pytest.mark.asyncio
async def test_mcp_client_translates_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutClient:
        def __init__(
            self,
            *args,
            **kwargs,
        ) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(
            self,
            exc_type,
            exc,
            traceback,
        ) -> None:
            return None

        async def call_tool(
            self,
            tool_name,
            arguments,
        ):
            raise TimeoutError

    monkeypatch.setattr(
        "cleanbot.device_mcp.client.Client",
        TimeoutClient,
    )

    client = DeviceMCPClient(
        "http://device-mcp:8001/mcp",
        timeout_seconds=0.01,
    )

    with pytest.raises(
        DeviceMCPCallError,
        match="timed out",
    ):
        await client.get_device_status(
            "1001",
            "demo-device-1001",
        )
