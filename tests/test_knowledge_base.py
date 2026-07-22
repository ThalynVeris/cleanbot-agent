from __future__ import annotations

from pathlib import Path
from typing import Any

from cleanbot.core.config import Settings
from cleanbot.db.database import Database
from cleanbot.rag.knowledge_base import KnowledgeBase


class FakeCollection:
    def __init__(self, store: FakeStore) -> None:
        self.store = store

    def count(self) -> int:
        return len(self.store.documents)


class FakeStore:
    def __init__(self) -> None:
        self.documents: dict[str, Any] = {}
        self._collection = FakeCollection(self)

    def add_documents(self, documents, ids) -> None:
        self.documents.update(dict(zip(ids, documents, strict=True)))

    def get(self, where=None, include=None):
        selected = [
            (chunk_id, document)
            for chunk_id, document in self.documents.items()
            if not where or all(document.metadata.get(key) == value for key, value in where.items())
        ]
        return {
            "ids": [item[0] for item in selected],
            "documents": [item[1].page_content for item in selected],
            "metadatas": [item[1].metadata for item in selected],
        }

    def delete(self, ids) -> None:
        for chunk_id in ids:
            self.documents.pop(chunk_id, None)


def test_ingest_update_idempotency_and_delete(settings: Settings, tmp_path: Path) -> None:
    database = Database(settings)
    database.create_schema()
    store = FakeStore()
    knowledge = KnowledgeBase(database, settings, store=store)  # type: ignore[arg-type]
    source = tmp_path / "手册.txt"
    source.write_text("1. 主刷被毛发缠绕；请清理主刷。\n\n2. 尘盒满了；请清空尘盒。", encoding="utf-8")

    created = knowledge.ingest_path(source)
    unchanged = knowledge.ingest_path(source)
    assert created.status == "created"
    assert unchanged.status == "unchanged"
    assert knowledge.count() == 2

    source.write_text("1. 主刷被毛发缠绕；请断电后清理。", encoding="utf-8")
    updated = knowledge.ingest_path(source)
    assert updated.status == "updated"
    assert knowledge.count() == 1
    assert "断电" in knowledge.all_chunks()[0].content

    assert knowledge.delete_document(created.document_id) is True
    assert knowledge.delete_document(created.document_id) is False
    assert knowledge.count() == 0
