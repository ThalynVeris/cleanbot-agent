from __future__ import annotations

from typing import Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict, total=False):
    action_id: str
    user_id: str
    session_id: str
    device_id: str
    action: str
    approved: bool


class DeviceApprovalWorkflow:
    def __init__(self, checkpointer: BaseCheckpointSaver) -> None:
        graph = StateGraph(ApprovalState)

        graph.add_node("request_approval", self._request_approval)
        graph.add_edge(START, "request_approval")
        graph.add_edge("request_approval", END)

        self.compiled = graph.compile(checkpointer=checkpointer)

    @staticmethod
    def _request_approval(state: ApprovalState) -> dict[str, bool]:
        decision = interrupt(
            {
                "action_id": state["action_id"],
                "user_id": state["user_id"],
                "session_id": state["session_id"],
                "device_id": state["device_id"],
                "action": state["action"],
            }
        )

        if not isinstance(decision, dict) or "approve" not in decision:
            raise ValueError("Approval decision must contain approve")
        return {"approved": bool(decision["approve"])}

    @staticmethod
    def _config(thread_id: str) -> dict[str, dict[str, str]]:
        return {
            "configurable": {
                "thread_id": thread_id,
            }
        }

    def start(self, state: ApprovalState, thread_id: str) -> dict[str, Any]:
        return self.compiled.invoke(state, self._config(thread_id))

    def resume(
        self,
        thread_id: str,
        approve: bool,
    ) -> dict[str, Any]:
        return self.compiled.invoke(
            Command(
                resume={
                    "approve": approve,
                }
            ),
            self._config(thread_id),
        )
