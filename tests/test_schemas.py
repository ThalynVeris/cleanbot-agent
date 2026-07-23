from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from cleanbot.core.schemas import ChatSessionSummary


def test_chat_session_summary_validates_messages_count() -> None:
    now = datetime(2026, 7, 23, 10, 0, tzinfo=timezone.utc)

    summary = ChatSessionSummary(
        id="session-001",
        user_id="1001",
        title="主刷被宠物毛发缠住怎么办",
        message_count=2,
        created_at=now,
        updated_at=now,
    )

    assert summary.id == "session-001"
    assert summary.message_count == 2
    with pytest.raises(ValidationError):
        ChatSessionSummary(
            id="session-002",
            user_id="1001",
            title="非法会话",
            message_count=-1,
            created_at=now,
            updated_at=now,
        )
