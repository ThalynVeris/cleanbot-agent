from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models.chat_models import (
    BaseChatModel,
)

from cleanbot.core.schemas import (
    DeviceActionResult,
    DeviceActionView,
    DeviceIntentDecision,
    DeviceOperation,
)
from cleanbot.db.database import Database
from cleanbot.db.models import DeviceActionName
from cleanbot.device_mcp.client import DeviceMCPClient
from cleanbot.workflow.device_approval import (
    DeviceApprovalWorkflow,
)


@dataclass(frozen=True, slots=True)
class DeviceControlOutcome:
    kind: Literal["answer", "approval_required"]
    message: str
    action: DeviceActionView | None = None
    result: DeviceActionResult | None = None


OPERATION_PHRASES = {
    DeviceOperation.READ_STATUS: (
        "设备状态",
        "当前状态",
        "机器人状态",
        "剩余电量",
        "还有多少电",
    ),
    DeviceOperation.READ_CONSUMABLE: (
        "耗材剩余",
        "耗材还剩",
        "耗材状态",
    ),
    DeviceOperation.START_CLEANING: (
        "开始清扫",
        "开始打扫",
        "启动清扫",
    ),
    DeviceOperation.PAUSE_CLEANING: (
        "暂停清扫",
        "暂停打扫",
    ),
    DeviceOperation.RETURN_TO_DOCK: (
        "返回充电座",
        "回到充电座",
        "立即回充",
        "开始回充",
    ),
}


OPERATION_ACTIONS = {
    DeviceOperation.START_CLEANING: (DeviceActionName.START_CLEANING),
    DeviceOperation.PAUSE_CLEANING: (DeviceActionName.PAUSE_CLEANING),
    DeviceOperation.RETURN_TO_DOCK: (DeviceActionName.RETURN_TO_DOCK),
}


class DeviceControlService:
    def __init__(
        self,
        database: Database,
        approval_workflow: DeviceApprovalWorkflow,
        mcp_client: DeviceMCPClient,
        model: BaseChatModel | None = None,
    ) -> None:
        self.database = database
        self.approval_workflow = approval_workflow
        self.mcp_client = mcp_client
        self.model = model

    @staticmethod
    def deterministic_operation(
        message: str,
    ) -> DeviceOperation | None:
        for operation, phrases in OPERATION_PHRASES.items():
            if any(phrase in message for phrase in phrases):
                return operation

        return None

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)

        if isinstance(content, str):
            return content

        return str(content)

    async def resolve_operation(
        self,
        message: str,
    ) -> DeviceOperation:
        direct = self.deterministic_operation(message)

        if direct is not None:
            return direct

        if self.model is None:
            return DeviceOperation.UNKNOWN

        try:
            response = await self.model.ainvoke(
                """Classify the user's intended operation for their
simulated cleaning robot.

Allowed operations:
- read_status
- read_consumable
- start_cleaning
- pause_cleaning
- return_to_dock
- unknown

Rules:
1. Questions about faults, maintenance or general product knowledge
   are unknown, not device commands.
2. Use a write operation only when the user is asking the current
   simulated device to perform that action.
3. Return only one JSON object:
{"operation":"start_cleaning","confidence":0.95}
4. Do not return reasoning or Markdown.

User message:
"""
                + message
            )

            content = self._message_text(response)
            start = content.find("{")
            end = content.rfind("}")

            if start >= 0 and end > start:
                content = content[start : end + 1]

            decision = DeviceIntentDecision.model_validate_json(content)

            if decision.confidence < 0.75:
                return DeviceOperation.UNKNOWN

            return decision.operation
        except Exception:
            return DeviceOperation.UNKNOWN

    async def prepare(
        self,
        *,
        session_id: str,
        user_id: str,
        request_id: str,
        message: str,
    ) -> DeviceControlOutcome:
        device = self.database.get_user_device_status(user_id)
        operation = await self.resolve_operation(message)

        if operation is DeviceOperation.UNKNOWN:
            return DeviceControlOutcome(
                kind="answer",
                message=(
                    "我没有准确理解你希望查询还是操作设备。"
                    "请明确说明查看状态、查看耗材、开始清扫、"
                    "暂停清扫或返回充电座。"
                ),
            )

        if operation is DeviceOperation.READ_CONSUMABLE:
            consumable = await self.mcp_client.get_consumable_status(
                user_id,
                device.device_id,
            )
            return DeviceControlOutcome(
                kind="answer",
                message=(
                    f"模拟设备耗材剩余 "
                    f"{consumable.consumable_percent}%"
                    f"{'，建议尽快更换。' if consumable.replacement_recommended else '。'}"
                ),
            )

        if operation is DeviceOperation.READ_STATUS:
            status = await self.mcp_client.get_device_status(
                user_id,
                device.device_id,
            )
            return DeviceControlOutcome(
                kind="answer",
                message=(f"模拟设备当前状态为 {status.status}，剩余电量 {status.battery_percent}%。"),
            )

        action_name = OPERATION_ACTIONS[operation]

        checkpoint_thread_id = f"device:{session_id}:{request_id}"
        pending = self.database.create_pending_device_action(
            session_id=session_id,
            user_id=user_id,
            action_name=action_name,
            idempotency_key=(f"{request_id}:{action_name.value}"),
            checkpoint_thread_id=(checkpoint_thread_id),
        )

        interrupted = await asyncio.to_thread(
            self.approval_workflow.start,
            {
                "action_id": pending.id,
                "user_id": pending.user_id,
                "session_id": pending.session_id,
                "device_id": pending.device_id,
                "action": pending.action,
            },
            checkpoint_thread_id,
        )

        if "__interrupt__" not in interrupted:
            raise RuntimeError("Device approval workflow did not interrupt")

        return DeviceControlOutcome(
            kind="approval_required",
            message=(f"是否允许模拟设备执行 {pending.action}？该审批将在 30 分钟后过期。"),
            action=pending,
        )

    async def decide(
        self,
        *,
        action_id: str,
        user_id: str,
        session_id: str,
        approve: bool,
    ) -> DeviceControlOutcome:
        action = self.database.get_device_action(
            action_id=action_id,
            user_id=user_id,
            session_id=session_id,
        )

        if action is None:
            raise ValueError("Device action is not available for this request")

        resumed = await asyncio.to_thread(
            self.approval_workflow.resume,
            action.checkpoint_thread_id,
            approve,
        )

        if (
            resumed.get("action_id") != action.id
            or resumed.get("user_id") != user_id
            or resumed.get("session_id") != session_id
        ):
            raise ValueError("Approval checkpoint does not match device action")

        graph_approved = bool(resumed.get("approved"))

        decided = self.database.decide_device_action(
            action_id=action.id,
            user_id=user_id,
            session_id=session_id,
            approve=graph_approved,
        )

        if decided.status == "rejected":
            return DeviceControlOutcome(
                kind="answer",
                message="已拒绝本次模拟设备操作，设备状态未改变。",
                action=decided,
            )

        if decided.status == "expired":
            return DeviceControlOutcome(
                kind="answer",
                message="本次设备操作审批已经过期，请重新发起。",
                action=decided,
            )

        if decided.status not in {
            "approved",
            "succeeded",
        }:
            return DeviceControlOutcome(
                kind="answer",
                message=(f"设备操作当前状态为 {decided.status}，无法执行。"),
                action=decided,
            )

        result = await self.mcp_client.execute(decided)
        final_action = self.database.get_device_action(
            action_id=decided.id,
            user_id=user_id,
            session_id=session_id,
        )

        if final_action is None:
            raise RuntimeError("Executed device action disappeared")
        return DeviceControlOutcome(
            kind="answer",
            message=(
                f"模拟设备操作执行成功，当前状态为 {result.device_status}。"
                if result.ok
                else f"模拟设备操作失败：{result.message}"
            ),
            action=final_action,
            result=result,
        )
