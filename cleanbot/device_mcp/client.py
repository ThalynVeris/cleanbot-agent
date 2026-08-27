from __future__ import annotations

from typing import Any

import httpx
from mcp import Client

from cleanbot.core.schemas import (
    ConsumableStatusView,
    DeviceActionResult,
    DeviceActionView,
    DeviceStatusView,
)


class DeviceMCPCallError(RuntimeError):
    """Raised when the device MCP service returns an error."""


class DeviceMCPClient:
    def __init__(
        self,
        server: Any,
        timeout_seconds: float = 5,
    ) -> None:
        self.server = server
        self.timeout_seconds = timeout_seconds

    async def _call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            async with Client(
                self.server,
                read_timeout_seconds=(self.timeout_seconds),
            ) as client:
                result = await client.call_tool(
                    tool_name,
                    arguments,
                )
        except (
            TimeoutError,
            httpx.TimeoutException,
        ) as exc:
            raise DeviceMCPCallError("Device MCP call timed out") from exc
        except Exception as exc:
            raise DeviceMCPCallError("Device MCP transport unavailable") from exc

        if result.is_error:
            message = "Device MCP tool failed"

            if result.content:
                message = getattr(
                    result.content[0],
                    "text",
                    message,
                )

            raise DeviceMCPCallError(message)

        if result.structured_content is None:
            raise DeviceMCPCallError("Device MCP returned no structured result")

        return result.structured_content

    async def get_device_status(
        self,
        user_id: str,
        device_id: str,
    ) -> DeviceStatusView:
        data = await self._call(
            "get_device_status",
            {
                "user_id": user_id,
                "device_id": device_id,
            },
        )
        return DeviceStatusView.model_validate(data)

    async def get_consumable_status(
        self,
        user_id: str,
        device_id: str,
    ) -> ConsumableStatusView:
        data = await self._call(
            "get_consumable_status",
            {
                "user_id": user_id,
                "device_id": device_id,
            },
        )
        return ConsumableStatusView.model_validate(data)

    async def execute(
        self,
        action: DeviceActionView,
    ) -> DeviceActionResult:
        data = await self._call(
            action.action,
            {
                "action_id": action.id,
                "user_id": action.user_id,
                "device_id": action.device_id,
            },
        )
        return DeviceActionResult.model_validate(data)
