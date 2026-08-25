from __future__ import annotations

import json

import pytest
from mcp import Client

from cleanbot.core.config import Settings
from cleanbot.db.database import Database
from cleanbot.device_mcp.server import create_device_mcp


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
