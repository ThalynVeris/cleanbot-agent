from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from cleanbot.core.config import PROJECT_ROOT, Settings
from cleanbot.db.database import Database, SessionOwnershipError
from cleanbot.db.models import ChatSession, Message


def test_project_root_is_independent_of_current_working_directory() -> None:
    assert (PROJECT_ROOT / "cleanbot").is_dir()
    assert PROJECT_ROOT == Path(__file__).resolve().parents[1]


def test_seed_reports_and_messages_are_idempotent(settings: Settings) -> None:
    database = Database(settings)
    database.create_schema()
    first = database.seed_demo_data()
    second = database.seed_demo_data()

    assert first == (10, 120)
    assert second == (0, 120)
    assert len(database.list_users()) == 10
    assert database.list_months("1003")[0] == "2025-12"
    report = database.get_device_report("1003", "2025-03")
    assert report is not None
    assert "毛发清理:92%" in report.efficiency
    assert database.get_device_report("1003", "2024-01") is None

    database.ensure_session("session-0001", "1001")
    database.add_message("session-0001", "user", "第一条")
    database.add_message("session-0001", "assistant", "回复")
    assert [item.content for item in database.get_messages("session-0001")] == ["第一条", "回复"]


def test_session_cannot_move_between_users(settings: Settings) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()
    database.ensure_session("session-fixed", "1001")
    with pytest.raises(SessionOwnershipError, match="cannot be reassigned"):
        database.ensure_session("session-fixed", "1002")


def test_list_sessions_is_user_scoped_and_builds_summary(settings: Settings) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    database.ensure_session("session-list-1001", "1001")
    database.add_message("session-list-1001", "user", "主刷被宠物毛发缠住怎么办？")
    database.add_message("session-list-1001", "assistant", "请断电后清理主刷。")

    database.ensure_session("session-list-1002", "1002")
    database.add_message("session-list-1002", "user", "这是另一个用户的问题")

    sessions = database.list_sessions("1001")

    assert len(sessions) == 1
    assert sessions[0].id == "session-list-1001"
    assert sessions[0].user_id == "1001"
    assert sessions[0].title == "主刷被宠物毛发缠住怎么办？"
    assert sessions[0].message_count == 2


def test_new_message_moves_session_to_front(settings: Settings) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    database.ensure_session("session-older", "1001")
    database.ensure_session("session-newer", "1001")

    with database.session() as db:
        older = db.get(ChatSession, "session-older")
        newer = db.get(ChatSession, "session-newer")

        assert older is not None
        assert newer is not None

        older.updated_at = datetime(2020, 1, 1)
        newer.updated_at = datetime(2021, 1, 1)

    sessions = database.list_sessions("1001")
    assert [item.id for item in sessions] == ["session-newer", "session-older"]

    database.add_message(
        "session-older",
        "user",
        "这条消息应该让旧会话移动到最前面",
    )
    sessions = database.list_sessions("1001")
    assert [item.id for item in sessions] == ["session-older", "session-newer"]


def test_session_messages_relationship_and_delete_cascade(
    settings: Settings,
) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    database.ensure_session("session-relations", "1001")
    database.add_message(
        "session-relations",
        "user",
        "第一条消息",
    )
    database.add_message(
        "session-relations",
        "assistant",
        "第二条消息",
    )

    with database.session() as db:
        chat_session = db.scalar(
            select(ChatSession)
            .options(selectinload(ChatSession.messages))
            .where(ChatSession.id == "session-relations")
        )

        assert chat_session is not None
        assert chat_session.user_id == "1001"
        assert chat_session.user.id == "1001"
        assert [message.content for message in chat_session.messages] == [
            "第一条消息",
            "第二条消息",
        ]
        assert all(
            message.session is chat_session
            for message in chat_session.messages
        )

        db.delete(chat_session)

    with database.session() as db:
        deleted_session = db.get(ChatSession, "session-relations")
        remaining_messages = db.scalars(
            select(Message).where(
                Message.session_id == "session-relations"
            )
        ).all()

        assert deleted_session is None
        assert remaining_messages == []


def test_database_session_rolls_back_when_block_raises(settings: Settings) -> None:
    database = Database(settings)
    database.create_schema()
    database.seed_demo_data()

    with pytest.raises(RuntimeError, match="force rollback"):
        with database.session() as db:
            chat_session = ChatSession(id="rollback-session", user_id="1001")
            db.add(chat_session)
            db.flush()

            assert db.get(ChatSession, "rollback-session") is not None
            raise RuntimeError("force rollback")

    with database.session() as db:
        assert db.get(ChatSession, "rollback-session") is None
