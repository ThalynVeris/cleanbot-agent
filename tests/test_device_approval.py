from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from cleanbot.workflow.device_approval import (
    DeviceApprovalWorkflow,
)


def test_approval_graph_resumes_after_checkpointer_reopens(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "device-checkpoints.sqlite"
    thread_id = "device:approval-session"

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        workflow = DeviceApprovalWorkflow(checkpointer)

        interrupted = workflow.start(
            {
                "action_id": "action-001",
                "user_id": "1001",
                "session_id": "approval-session",
                "device_id": "demo-device-1001",
                "action": "start_cleaning",
            },
            thread_id,
        )

        assert "__interrupt__" in interrupted

        approval_request = interrupted["__interrupt__"][0].value

        assert approval_request["action_id"] == "action-001"
        assert approval_request["user_id"] == "1001"
        assert approval_request["action"] == "start_cleaning"

    with SqliteSaver.from_conn_string(str(checkpoint_path)) as reopened_checkpointer:
        reopened_workflow = DeviceApprovalWorkflow(reopened_checkpointer)

        resumed = reopened_workflow.resume(
            thread_id,
            approve=True,
        )

        assert resumed["action_id"] == "action-001"
        assert resumed["session_id"] == "approval-session"
        assert resumed["approved"] is True
        assert "__interrupt__" not in resumed
