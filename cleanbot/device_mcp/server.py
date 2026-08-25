from __future__ import annotations

import json

from mcp.server import MCPServer
from starlette.applications import Starlette

from cleanbot.core.config import get_settings
from cleanbot.core.schemas import (
    ConsumableStatusView,
    DeviceStatusView,
)
from cleanbot.db.database import Database


def create_device_mcp(database: Database) -> MCPServer:
    server = MCPServer(
        name="CleanBot Device MCP",
        version="0.3.0",
        instructions=("Provides read and write operations for simulated CleanBot devices."),
    )

    @server.tool()
    def get_device_status(user_id: str, device_id: str) -> DeviceStatusView:
        """Read a simulated device's status and battery level."""
        return database.get_device_status(user_id, device_id)

    @server.tool()
    def get_consumable_status(user_id: str, device_id: str) -> ConsumableStatusView:
        """Read a simulated device's consumable remaining percentage."""
        return database.get_consumable_status(user_id, device_id)

    @server.resource("device://{device_id}/capabilities", mime_type="application/json")
    def get_capabilities(device_id: str) -> str:
        """Read the supported capabilities of a simulated device."""
        capabilities = database.get_device_capabilities(device_id)
        return json.dumps(capabilities.model_dump(mode="json"), ensure_ascii=False)

    return server


def create_app() -> Starlette:
    settings = get_settings()
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    server = create_device_mcp(database)

    return server.streamable_http_app(streamable_http_path="/mcp", stateless_http=True)
