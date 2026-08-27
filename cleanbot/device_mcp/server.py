from __future__ import annotations

import json

from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse

from cleanbot.core.config import get_settings
from cleanbot.core.schemas import (
    ConsumableStatusView,
    DeviceActionResult,
    DeviceStatusView,
)
from cleanbot.db.database import Database
from cleanbot.db.models import DeviceActionName

DEVICE_MCP_TRANSPORT_SECURITY = TransportSecuritySettings(
    allowed_hosts=[
        "device-mcp:8001",
        "127.0.0.1:*",
        "localhost:*",
    ],
)


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

    @server.tool()
    def start_cleaning(
        action_id: str,
        user_id: str,
        device_id: str,
    ) -> DeviceActionResult:
        """Start cleaning after validating an approved action."""
        return database.execute_device_action(
            action_id,
            user_id,
            device_id,
            DeviceActionName.START_CLEANING,
        )

    @server.tool()
    def pause_cleaning(
        action_id: str,
        user_id: str,
        device_id: str,
    ) -> DeviceActionResult:
        """Pause cleaning after validating an approved action."""
        return database.execute_device_action(
            action_id,
            user_id,
            device_id,
            DeviceActionName.PAUSE_CLEANING,
        )

    @server.tool()
    def return_to_dock(
        action_id: str,
        user_id: str,
        device_id: str,
    ) -> DeviceActionResult:
        """Return to dock after validating an approved action."""
        return database.execute_device_action(
            action_id,
            user_id,
            device_id,
            DeviceActionName.RETURN_TO_DOCK,
        )

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

    def health(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "service": "device-mcp",
                "users": len(database.list_users()),
            }
        )

    app = server.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,
        transport_security=DEVICE_MCP_TRANSPORT_SECURITY,
    )
    app.add_route("/health", health, methods=["GET"])
    return app
