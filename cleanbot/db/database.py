from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session, sessionmaker

from cleanbot.core.config import Settings, get_settings
from cleanbot.core.schemas import DemoUser, DeviceReport, SourceRef, StoredMessage
from cleanbot.db.models import (
    Base,
    ChatSession,
    DeviceMonthlyRecord,
    KnowledgeDocument,
    Message,
    User,
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


class SessionOwnershipError(ValueError):
    """Raised when a session is attempted to be reassigned to a different user."""

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
            message = Message(
                session_id=session_id,
                role=role,
                content=content,
                sources_json=encoded_sources,
            )
            db.add(message)
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
