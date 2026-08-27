from __future__ import annotations

import csv
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session, selectinload, sessionmaker

from cleanbot.core.config import Settings, get_settings
from cleanbot.core.schemas import (
    ChatSessionSummary,
    ConsumableStatusView,
    DemoUser,
    DeviceActionResult,
    DeviceActionView,
    DeviceCapabilitiesView,
    DeviceReport,
    DeviceStatusView,
    SourceRef,
    StoredMessage,
)
from cleanbot.db.models import (
    Base,
    ChatSession,
    Device,
    DeviceAction,
    DeviceActionName,
    DeviceActionStatus,
    DeviceMonthlyRecord,
    DeviceStatus,
    KnowledgeDocument,
    Message,
    User,
    utc_now,
)

DEMO_CITIES = {
    "1001": "上海",
    "1002": "杭州",
    "1003": "成都",
    "1004": "深圳",
    "1005": "北京",
    "1006": "苏州",
    "1007": "武汉",
    "1008": "广州",
    "1009": "南京",
    "1010": "西安",
}

DEVICE_TRANSITIONS = {
    DeviceActionName.START_CLEANING: (
        frozenset(
            {
                DeviceStatus.DOCKED,
                DeviceStatus.PAUSED,
            }
        ),
        DeviceStatus.CLEANING,
    ),
    DeviceActionName.PAUSE_CLEANING: (
        frozenset({DeviceStatus.CLEANING}),
        DeviceStatus.PAUSED,
    ),
    DeviceActionName.RETURN_TO_DOCK: (
        frozenset(
            {
                DeviceStatus.CLEANING,
                DeviceStatus.PAUSED,
            }
        ),
        DeviceStatus.RETURNING_TO_DOCK,
    ),
}


class SessionOwnershipError(ValueError):
    """Raised when a session is attempted to be reassigned to a different user."""


class DeviceOwnershipError(ValueError):
    """Raised when a user attempts to access another user's device."""


class DeviceActionApprovalError(ValueError):
    """Raised when a write action has no valid approval."""


class PendingDeviceActionError(ValueError):
    """Raised when a session already has a pending device action."""


def deadline_passed(deadline: datetime) -> bool:
    now = utc_now()

    if deadline.tzinfo is None:
        now = now.replace(tzinfo=None)

    return deadline <= now


def as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class Database:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        connect_args = {"check_same_thread": False} if self.settings.database_url.startswith("sqlite") else {}
        self.engine = create_engine(
            self.settings.database_url,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, class_=Session)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        db = self.session_factory()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def seed_demo_data(self, csv_path: Path | None = None) -> tuple[int, int]:
        csv_path = csv_path or self.settings.data_dir / "external" / "records.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Demo records not found: {csv_path}")

        users_created = 0
        records_upserted = 0
        processed_device_users: set[str] = set()
        with csv_path.open("r", encoding="utf-8", newline="") as handle, self.session() as db:
            reader = csv.DictReader(handle)
            for row in reader:
                user_id = row["用户ID"].strip()
                user = db.get(User, user_id)
                if user is None:
                    db.add(
                        User(
                            id=user_id,
                            display_name=f"演示用户 {user_id}",
                            city=DEMO_CITIES.get(user_id, "上海"),
                        )
                    )
                    db.flush()
                    users_created += 1
                if user_id not in processed_device_users:
                    device = db.scalar(select(Device).where(Device.user_id == user_id))
                    if device is None:
                        db.add(
                            Device(
                                id=f"demo-device-{user_id}",
                                user_id=user_id,
                                model="CleanBot X1（模拟设备）",
                                status=DeviceStatus.DOCKED,
                                battery_percent=100,
                                consumable_percent=100,
                            )
                        )
                    processed_device_users.add(user_id)
                record = db.scalar(
                    select(DeviceMonthlyRecord).where(
                        DeviceMonthlyRecord.user_id == user_id,
                        DeviceMonthlyRecord.month == row["时间"].strip(),
                    )
                )
                values = {
                    "features": row["特征"].replace("\\n", "\n").strip(),
                    "efficiency": row["清洁效率"].replace("\\n", "\n").strip(),
                    "consumables": row["耗材"].replace("\\n", "\n").strip(),
                    "comparison": row["对比"].replace("\\n", "\n").strip(),
                }
                if record is None:
                    db.add(
                        DeviceMonthlyRecord(
                            user_id=user_id,
                            month=row["时间"].strip(),
                            **values,
                        )
                    )
                else:
                    for key, value in values.items():
                        setattr(record, key, value)
                records_upserted += 1
        return users_created, records_upserted

    def list_users(self) -> list[DemoUser]:
        with self.session() as db:
            users = db.scalars(select(User).order_by(User.id)).all()
            return [DemoUser(id=user.id, display_name=user.display_name, city=user.city) for user in users]

    def get_user(self, user_id: str) -> DemoUser | None:
        with self.session() as db:
            user = db.get(User, user_id)
            if user is None:
                return None
            return DemoUser(id=user.id, display_name=user.display_name, city=user.city)

    @staticmethod
    def _owned_device(
        db: Session,
        user_id: str,
        device_id: str,
    ) -> Device:
        device = db.scalar(
            select(Device).where(
                Device.id == device_id,
                Device.user_id == user_id,
            )
        )

        if device is None:
            raise DeviceOwnershipError("Device is not available for this user")

        return device

    def get_user_device_status(
        self,
        user_id: str,
    ) -> DeviceStatusView:
        with self.session() as db:
            device = db.scalar(select(Device).where(Device.user_id == user_id))

            if device is None:
                raise DeviceOwnershipError("Device is not available for this user")

            device_id = device.id

        return self.get_device_status(
            user_id,
            device_id,
        )

    def get_device_status(
        self,
        user_id: str,
        device_id: str,
    ) -> DeviceStatusView:
        with self.session() as db:
            device = self._owned_device(db, user_id, device_id)

            return DeviceStatusView(
                device_id=device.id,
                user_id=device.user_id,
                model=device.model,
                status=device.status.value,
                battery_percent=device.battery_percent,
                simulated=True,
            )

    def get_consumable_status(
        self,
        user_id: str,
        device_id: str,
    ) -> ConsumableStatusView:
        with self.session() as db:
            device = self._owned_device(db, user_id, device_id)

            return ConsumableStatusView(
                device_id=device.id,
                user_id=device.user_id,
                consumable_percent=device.consumable_percent,
                replacement_recommended=(device.consumable_percent <= 10),
                simulated=True,
            )

    def get_device_capabilities(
        self,
        device_id: str,
    ) -> DeviceCapabilitiesView:
        with self.session() as db:
            device = db.get(Device, device_id)

            if device is None:
                raise ValueError("Unknown device")

            return DeviceCapabilitiesView(
                device_id=device.id,
                model=device.model,
                supported_actions=[
                    "start_cleaning",
                    "pause_cleaning",
                    "return_to_dock",
                ],
                readable_properties=[
                    "status",
                    "battery_percent",
                    "consumable_percent",
                ],
                simulated=True,
            )

    def create_pending_device_action(
        self,
        *,
        session_id: str,
        user_id: str,
        action_name: DeviceActionName,
        idempotency_key: str,
        checkpoint_thread_id: str,
    ) -> DeviceActionView:
        self.ensure_session(session_id, user_id)

        with self.session() as db:
            chat_session = db.scalar(
                select(ChatSession)
                .where(
                    ChatSession.id == session_id,
                    ChatSession.user_id == user_id,
                )
                .with_for_update()
            )

            if chat_session is None:
                raise SessionOwnershipError("Session is not available for this user")

            existing = db.scalar(select(DeviceAction).where(DeviceAction.idempotency_key == idempotency_key))

            if existing is not None:
                if (
                    existing.user_id != user_id
                    or existing.session_id != session_id
                    or existing.action is not action_name
                ):
                    raise DeviceActionApprovalError("Idempotency key belongs to another request")

                return self._device_action_schema(existing)

            pending = db.scalar(
                select(DeviceAction).where(
                    DeviceAction.session_id == session_id,
                    DeviceAction.status == DeviceActionStatus.PENDING,
                )
            )

            if pending is not None and deadline_passed(pending.approval_expires_at):
                pending.status = DeviceActionStatus.EXPIRED
                pending.error_type = "ApprovalExpiredError"
                pending = None

            if pending is not None:
                raise PendingDeviceActionError("Session already has a pending device action")

            device = db.scalar(select(Device).where(Device.user_id == user_id))

            if device is None:
                raise DeviceOwnershipError("Device is not available for this user")

            action = DeviceAction(
                id=str(uuid.uuid4()),
                user_id=user_id,
                device_id=device.id,
                session_id=session_id,
                action=action_name,
                idempotency_key=idempotency_key,
                checkpoint_thread_id=checkpoint_thread_id,
                status=DeviceActionStatus.PENDING,
            )
            db.add(action)
            db.flush()
            db.refresh(action)

            return self._device_action_schema(action)

    def decide_device_action(
        self,
        *,
        action_id: str,
        user_id: str,
        session_id: str,
        approve: bool,
    ) -> DeviceActionView:
        with self.session() as db:
            action = db.scalar(select(DeviceAction).where(DeviceAction.id == action_id).with_for_update())

            if action is None or action.user_id != user_id or action.session_id != session_id:
                raise DeviceActionApprovalError("Device action is not available for this request")

            if action.status is DeviceActionStatus.PENDING:
                if deadline_passed(action.approval_expires_at):
                    action.status = DeviceActionStatus.EXPIRED
                    action.error_type = "ApprovalExpiredError"
                else:
                    action.status = DeviceActionStatus.APPROVED if approve else DeviceActionStatus.REJECTED
                    action.decided_at = utc_now()

            return self._device_action_schema(action)

    def fail_device_action(
        self,
        *,
        action_id: str,
        user_id: str,
        session_id: str,
        error_type: str,
    ) -> DeviceActionView:
        with self.session() as db:
            action = db.scalar(
                select(DeviceAction)
                .where(
                    DeviceAction.id == action_id,
                    DeviceAction.user_id == user_id,
                    DeviceAction.session_id == session_id,
                )
                .with_for_update()
            )

            if action is None:
                raise DeviceActionApprovalError("Device action is not available for this request")

            if action.status is DeviceActionStatus.APPROVED:
                action.status = DeviceActionStatus.FAILED
                action.error_type = error_type
                action.executed_at = utc_now()

            return self._device_action_schema(action)

    def get_device_action(
        self,
        *,
        action_id: str,
        user_id: str,
        session_id: str,
    ) -> DeviceActionView | None:
        with self.session() as db:
            action = db.scalar(
                select(DeviceAction).where(
                    DeviceAction.id == action_id,
                    DeviceAction.user_id == user_id,
                    DeviceAction.session_id == session_id,
                )
            )

            if action is None:
                return None

            return self._device_action_schema(action)

    def get_pending_device_action(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> DeviceActionView | None:
        with self.session() as db:
            action = db.scalar(
                select(DeviceAction)
                .where(
                    DeviceAction.session_id == session_id,
                    DeviceAction.user_id == user_id,
                    DeviceAction.status == DeviceActionStatus.PENDING,
                )
                .order_by(DeviceAction.created_at.desc())
            )

            if action is None:
                return None

            if deadline_passed(action.approval_expires_at):
                action.status = DeviceActionStatus.EXPIRED
                action.error_type = "ApprovalExpiredError"
                return None

            return self._device_action_schema(action)

    def execute_device_action(
        self,
        action_id: str,
        user_id: str,
        device_id: str,
        expected_action: DeviceActionName,
    ) -> DeviceActionResult:
        approval_error: DeviceActionApprovalError | None = None
        result: DeviceActionResult | None = None

        with self.session() as db:
            action = db.scalar(select(DeviceAction).where(DeviceAction.id == action_id).with_for_update())

            if (
                action is None
                or action.user_id != user_id
                or action.device_id != device_id
                or action.action is not expected_action
            ):
                raise DeviceActionApprovalError("Device action is not available for this request")

            if action.status is DeviceActionStatus.SUCCEEDED:
                stored = DeviceActionResult.model_validate_json(action.result_json)
                return stored.model_copy(update={"idempotent_replay": True})

            if action.status is not DeviceActionStatus.APPROVED:
                raise DeviceActionApprovalError("Device action has not been approved")

            now = utc_now()

            if deadline_passed(action.approval_expires_at):
                action.status = DeviceActionStatus.EXPIRED
                action.error_type = "ApprovalExpiredError"
                approval_error = DeviceActionApprovalError("Device action approval has expired")
            else:
                device = self._owned_device(
                    db,
                    user_id,
                    device_id,
                )
                allowed_statuses, target_status = DEVICE_TRANSITIONS[expected_action]

                if device.status not in allowed_statuses:
                    action.status = DeviceActionStatus.FAILED
                    action.error_type = "InvalidDeviceTransitionError"
                    action.executed_at = now

                    result = DeviceActionResult(
                        ok=False,
                        action_id=action.id,
                        device_id=device.id,
                        action=action.action.value,
                        action_status="failed",
                        device_status=device.status.value,
                        error_type=action.error_type,
                        message=(
                            f"Cannot execute {action.action.value} while device is {device.status.value}"
                        ),
                    )
                else:
                    device.status = target_status
                    device.updated_at = now
                    action.status = DeviceActionStatus.SUCCEEDED
                    action.executed_at = now
                    action.error_type = None

                    result = DeviceActionResult(
                        ok=True,
                        action_id=action.id,
                        device_id=device.id,
                        action=action.action.value,
                        action_status="succeeded",
                        device_status=device.status.value,
                        message="Simulated device action completed",
                    )

                action.result_json = result.model_dump_json()

        if approval_error is not None:
            raise approval_error

        if result is None:
            raise RuntimeError("Device action produced no result")

        return result

    def list_months(self, user_id: str) -> list[str]:
        with self.session() as db:
            return list(
                db.scalars(
                    select(DeviceMonthlyRecord.month)
                    .where(DeviceMonthlyRecord.user_id == user_id)
                    .order_by(DeviceMonthlyRecord.month.desc())
                ).all()
            )

    def get_device_report(self, user_id: str, month: str) -> DeviceReport | None:
        with self.session() as db:
            record = db.scalar(
                select(DeviceMonthlyRecord).where(
                    DeviceMonthlyRecord.user_id == user_id,
                    DeviceMonthlyRecord.month == month,
                )
            )
            if record is None:
                return None
            return DeviceReport(
                user_id=record.user_id,
                month=record.month,
                features=record.features,
                efficiency=record.efficiency,
                consumables=record.consumables,
                comparison=record.comparison,
            )

    def ensure_session(self, session_id: str, user_id: str) -> None:
        with self.session() as db:
            user = db.get(User, user_id)
            if user is None:
                raise ValueError(f"Unknown demo user: {user_id}")
            chat_session = db.get(ChatSession, session_id)
            if chat_session is None:
                db.add(ChatSession(id=session_id, user_id=user_id))
            elif chat_session.user_id != user_id:
                raise SessionOwnershipError("A session cannot be reassigned to another user")

    def list_sessions(self, user_id: str) -> list[ChatSessionSummary]:
        with self.session() as db:
            rows = db.scalars(
                select(ChatSession)
                .options(selectinload(ChatSession.messages))
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.updated_at.desc())
            ).all()

            summaries: list[ChatSessionSummary] = []

            for chat_session in rows:
                title = "新会话"

                for message in chat_session.messages:
                    if message.role == "user":
                        title = message.content[:40]
                        break

                summaries.append(
                    ChatSessionSummary(
                        id=chat_session.id,
                        user_id=chat_session.user_id,
                        title=title,
                        message_count=len(chat_session.messages),
                        created_at=chat_session.created_at,
                        updated_at=chat_session.updated_at,
                    )
                )

            return summaries

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[SourceRef] | None = None,
    ) -> StoredMessage:
        encoded_sources = json.dumps(
            [source.model_dump(mode="json") for source in (sources or [])], ensure_ascii=False
        )
        with self.session() as db:
            chat_session = db.get(ChatSession, session_id)

            if chat_session is None:
                raise ValueError(f"Unknown chat session: {session_id}")

            message = Message(
                session_id=session_id,
                role=role,
                content=content,
                sources_json=encoded_sources,
            )
            db.add(message)
            chat_session.updated_at = utc_now()
            db.flush()
            db.refresh(message)
            return self._message_schema(message)

    def get_messages(self, session_id: str, limit: int = 50) -> list[StoredMessage]:
        with self.session() as db:
            rows = list(
                db.scalars(
                    select(Message)
                    .where(Message.session_id == session_id)
                    .order_by(Message.id.desc())
                    .limit(limit)
                ).all()
            )
            rows.reverse()
            return [self._message_schema(row) for row in rows]

    @staticmethod
    def _message_schema(message: Message) -> StoredMessage:
        try:
            sources = [SourceRef.model_validate(item) for item in json.loads(message.sources_json)]
        except (json.JSONDecodeError, TypeError, ValueError):
            sources = []
        return StoredMessage(
            id=message.id,
            session_id=message.session_id,
            role=message.role,  # type: ignore[arg-type]
            content=message.content,
            sources=sources,
            created_at=message.created_at,
        )

    @staticmethod
    def _device_action_schema(
        action: DeviceAction,
    ) -> DeviceActionView:
        return DeviceActionView(
            id=action.id,
            user_id=action.user_id,
            device_id=action.device_id,
            session_id=action.session_id,
            action=action.action.value,
            status=action.status.value,
            checkpoint_thread_id=action.checkpoint_thread_id,
            approval_expires_at=as_utc(action.approval_expires_at),
            decided_at=as_utc(action.decided_at),
            executed_at=as_utc(action.executed_at),
            error_type=action.error_type,
            created_at=as_utc(action.created_at),
        )

    def get_knowledge_document(self, document_id: str) -> KnowledgeDocument | None:
        with self.session() as db:
            row = db.get(KnowledgeDocument, document_id)
            if row is not None:
                db.expunge(row)
            return row

    def get_knowledge_document_by_filename(self, filename: str) -> KnowledgeDocument | None:
        with self.session() as db:
            row = db.scalar(select(KnowledgeDocument).where(KnowledgeDocument.filename == filename))
            if row is not None:
                db.expunge(row)
            return row

    def upsert_knowledge_document(
        self,
        document_id: str,
        filename: str,
        source_path: str,
        content_hash: str,
        chunk_count: int,
    ) -> None:
        with self.session() as db:
            row = db.get(KnowledgeDocument, document_id)
            if row is None:
                db.add(
                    KnowledgeDocument(
                        id=document_id,
                        filename=filename,
                        source_path=source_path,
                        content_hash=content_hash,
                        chunk_count=chunk_count,
                        status="ready",
                    )
                )
            else:
                row.source_path = source_path
                row.content_hash = content_hash
                row.chunk_count = chunk_count
                row.status = "ready"

    def delete_knowledge_document(self, document_id: str) -> bool:
        with self.session() as db:
            result = db.execute(delete(KnowledgeDocument).where(KnowledgeDocument.id == document_id))
            return bool(result.rowcount)

    def knowledge_document_count(self) -> int:
        with self.session() as db:
            return len(db.scalars(select(KnowledgeDocument.id)).all())
