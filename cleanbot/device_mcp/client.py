from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

import httpx
import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

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
        token: str | None = None,
    ) -> None:
        self.server = server
        self.timeout_seconds = timeout_seconds
        self.token = token

    async def _call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            async with AsyncExitStack() as stack:
                transport = self.server

                if isinstance(self.server, str) and self.token:
                    http_client = await stack.enter_async_context(
                        httpx2.AsyncClient(
                            headers={
                                "Authorization": f"Bearer {self.token}",
                            },
                            timeout=self.timeout_seconds,
                        )
                    )
                    transport = streamable_http_client(
                        self.server,
                        http_client=http_client,
                    )

                client = await stack.enter_async_context(
                    Client(
                        transport,
                        read_timeout_seconds=self.timeout_seconds,
                    )
                )
                result = await client.call_tool(
                    tool_name,
                    arguments,
                )
        except (
            TimeoutError,
            httpx.TimeoutException,
            httpx2.TimeoutException,
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
