from __future__ import annotations

from pathlib import Path

import pytest

from cleanbot.core.config import PROJECT_ROOT, Settings
from cleanbot.db.database import Database


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
    with pytest.raises(ValueError, match="cannot be reassigned"):
        database.ensure_session("session-fixed", "1002")
