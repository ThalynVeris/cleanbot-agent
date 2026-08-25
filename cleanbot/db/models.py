from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def approval_deadline() -> datetime:
    return utc_now() + timedelta(minutes=30)


class Base(DeclarativeBase):
    pass


class DeviceStatus(str, Enum):
    DOCKED = "docked"
    CLEANING = "cleaning"
    PAUSED = "paused"
    RETURNING_TO_DOCK = "returning_to_dock"


class DeviceActionName(str, Enum):
    START_CLEANING = "start_cleaning"
    PAUSE_CLEANING = "pause_cleaning"
    RETURN_TO_DOCK = "return_to_dock"


class DeviceActionStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(100))
    city: Mapped[str] = mapped_column(String(100), default="上海")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    sessions: Mapped[list[ChatSession]] = relationship(back_populates="user")
    records: Mapped[list[DeviceMonthlyRecord]] = relationship(back_populates="user")
    device: Mapped[Device | None] = relationship(
        back_populates="user",
        uselist=False,
    )
    device_actions: Mapped[list[DeviceAction]] = relationship(
        back_populates="user",
    )


class Device(Base):
    __tablename__ = "devices"
    __table_args__ = (
        UniqueConstraint("user_id", name="uq_device_user"),
        CheckConstraint(
            "battery_percent BETWEEN 0 AND 100",
            name="ck_device_battery_percent",
        ),
        CheckConstraint(
            "consumable_percent BETWEEN 0 AND 100",
            name="ck_device_consumable_percent",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    model: Mapped[str] = mapped_column(String(100))
    status: Mapped[DeviceStatus] = mapped_column(
        SqlEnum(
            DeviceStatus,
            values_callable=lambda members: [member.value for member in members],
            native_enum=False,
            create_constraint=True,
            name="device_status",
        ),
        default=DeviceStatus.DOCKED,
    )
    battery_percent: Mapped[int] = mapped_column(Integer, default=100)
    consumable_percent: Mapped[int] = mapped_column(Integer, default=100)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    user: Mapped[User] = relationship(back_populates="device")
    actions: Mapped[list[DeviceAction]] = relationship(
        back_populates="device",
    )


class DeviceMonthlyRecord(Base):
    __tablename__ = "device_monthly_records"
    __table_args__ = (UniqueConstraint("user_id", "month", name="uq_record_user_month"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    month: Mapped[str] = mapped_column(String(7), index=True)
    features: Mapped[str] = mapped_column(Text)
    efficiency: Mapped[str] = mapped_column(Text)
    consumables: Mapped[str] = mapped_column(Text)
    comparison: Mapped[str] = mapped_column(Text)

    user: Mapped[User] = relationship(back_populates="records")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    user: Mapped[User] = relationship(back_populates="sessions")
    messages: Mapped[list[Message]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Message.id"
    )
    device_actions: Mapped[list[DeviceAction]] = relationship(
        back_populates="session",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("chat_sessions.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    sources_json: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    session: Mapped[ChatSession] = relationship(back_populates="messages")


class DeviceAction(Base):
    __tablename__ = "device_actions"
    __table_args__ = (
        UniqueConstraint(
            "idempotency_key",
            name="uq_device_action_idempotency_key",
        ),
        Index(
            "ix_device_action_session_status",
            "session_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
    )
    device_id: Mapped[str] = mapped_column(
        ForeignKey("devices.id", ondelete="CASCADE"),
        index=True,
    )
    session_id: Mapped[str] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        index=True,
    )
    action: Mapped[DeviceActionName] = mapped_column(
        SqlEnum(
            DeviceActionName,
            values_callable=lambda members: [member.value for member in members],
            native_enum=False,
            create_constraint=True,
            name="device_action_name",
        )
    )
    arguments_json: Mapped[str] = mapped_column(Text, default="{}")
    result_json: Mapped[str] = mapped_column(Text, default="{}")
    idempotency_key: Mapped[str] = mapped_column(String(128))
    checkpoint_thread_id: Mapped[str] = mapped_column(
        String(128),
        index=True,
    )
    status: Mapped[DeviceActionStatus] = mapped_column(
        SqlEnum(
            DeviceActionStatus,
            values_callable=lambda members: [member.value for member in members],
            native_enum=False,
            create_constraint=True,
            name="device_action_status",
        ),
        default=DeviceActionStatus.PENDING,
    )
    approval_expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=approval_deadline,
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    executed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    user: Mapped[User] = relationship(back_populates="device_actions")
    device: Mapped[Device] = relationship(back_populates="actions")
    session: Mapped[ChatSession] = relationship(
        back_populates="device_actions",
    )


class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    source_path: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(32), default="ready")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
