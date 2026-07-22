from __future__ import annotations

import hashlib
from pathlib import Path
from threading import RLock
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from pypdf import PdfReader

from cleanbot.core.config import Settings, get_settings
from cleanbot.core.models import get_embedding_model
from cleanbot.core.schemas import IngestResult, KnowledgeHit
from cleanbot.db.database import Database
from cleanbot.rag.chunking import TextChunk, split_structured_text

ALLOWED_SUFFIXES = {".txt", ".pdf"}
INDEX_VERSION = "structured-chunks-v2"


class KnowledgeBase:
    def __init__(
        self,
        database: Database,
        settings: Settings | None = None,
        embedding_function: Any | None = None,
        store: Chroma | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.database = database
        self._lock = RLock()
        self.store = store or Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=embedding_function or get_embedding_model(),
            persist_directory=str(self.settings.vector_dir),
        )

    @staticmethod
    def document_id_for(filename: str) -> str:
        return hashlib.sha256(filename.casefold().encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _load_chunks(self, path: Path) -> list[TextChunk]:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return split_structured_text(path.read_text(encoding="utf-8"))
        if suffix == ".pdf":
            chunks: list[TextChunk] = []
            reader = PdfReader(str(path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                for chunk in split_structured_text(text):
                    chunks.append(TextChunk(text=chunk.text, section=f"第 {page_number} 页｜{chunk.section}"))
            return chunks
        raise ValueError(f"Unsupported knowledge file type: {suffix}")

    def ingest_path(self, path: Path) -> IngestResult:
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            raise ValueError("Only .txt and .pdf knowledge files are supported")

        filename = path.name
        document_id = self.document_id_for(filename)
        source_hash = self.file_hash(path)
        content_hash = hashlib.sha256(f"{source_hash}:{INDEX_VERSION}".encode()).hexdigest()
        existing = self.database.get_knowledge_document(document_id)
        if existing is not None and existing.content_hash == content_hash:
            return IngestResult(
                document_id=document_id,
                filename=filename,
                content_hash=content_hash,
                chunk_count=existing.chunk_count,
                status="unchanged",
            )

        chunks = self._load_chunks(path)
        if not chunks:
            raise ValueError(f"No readable content found in {filename}")

        documents: list[Document] = []
        chunk_ids: list[str] = []
        for index, chunk in enumerate(chunks):
            chunk_hash = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"{document_id}:{index:04d}:{chunk_hash}"
            metadata: dict[str, str | int | float | bool] = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "source": filename,
                "section": chunk.section,
                "content_hash": content_hash,
            }
            if chunk.section.startswith("第 ") and " 页" in chunk.section:
                try:
                    metadata["page"] = int(chunk.section.split()[1])
                except (IndexError, ValueError):
                    pass
            documents.append(Document(page_content=chunk.text, metadata=metadata))
            chunk_ids.append(chunk_id)

        status = "updated" if existing is not None else "created"
        with self._lock:
            if existing is not None:
                self._delete_vectors(document_id)
            self.store.add_documents(documents, ids=chunk_ids)
            self.database.upsert_knowledge_document(
                document_id=document_id,
                filename=filename,
                source_path=str(path),
                content_hash=content_hash,
                chunk_count=len(documents),
            )

        return IngestResult(
            document_id=document_id,
            filename=filename,
            content_hash=content_hash,
            chunk_count=len(documents),
            status=status,
        )

    def ingest_directory(self, directory: Path | None = None) -> list[IngestResult]:
        directory = (directory or self.settings.data_dir).resolve()
        paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in ALLOWED_SUFFIXES)
        return [self.ingest_path(path) for path in paths]

    def _delete_vectors(self, document_id: str) -> None:
        result = self.store.get(where={"document_id": document_id}, include=[])
        ids = result.get("ids", [])
        if ids:
            self.store.delete(ids=ids)

    def delete_document(self, document_id: str) -> bool:
        existing = self.database.get_knowledge_document(document_id)
        if existing is None:
            return False
        with self._lock:
            self._delete_vectors(document_id)
            self.database.delete_knowledge_document(document_id)
        return True

    def count(self) -> int:
        return int(self.store._collection.count())  # Chroma exposes collection count only on its client.

    def all_chunks(self) -> list[KnowledgeHit]:
        result = self.store.get(include=["documents", "metadatas"])
        hits: list[KnowledgeHit] = []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        ids = result.get("ids") or []
        for chunk_id, content, metadata in zip(ids, documents, metadatas, strict=False):
            metadata = metadata or {}
            hits.append(self._to_hit(content or "", metadata, str(chunk_id)))
        return hits

    def dense_search(self, query: str, k: int) -> list[KnowledgeHit]:
        if self.count() == 0:
            return []
        rows = self.store.similarity_search_with_relevance_scores(query, k=k)
        hits: list[KnowledgeHit] = []
        for document, score in rows:
            hit = self._to_hit(document.page_content, document.metadata)
            hit.dense_score = max(0.0, min(1.0, float(score)))
            hit.score = hit.dense_score
            hits.append(hit)
        return hits

    @staticmethod
    def _to_hit(content: str, metadata: dict[str, Any], fallback_id: str = "") -> KnowledgeHit:
        return KnowledgeHit(
            document_id=str(metadata.get("document_id", "unknown")),
            chunk_id=str(metadata.get("chunk_id", fallback_id)),
            source=str(metadata.get("source", "unknown")),
            page=int(metadata["page"]) if metadata.get("page") is not None else None,
            section=str(metadata["section"]) if metadata.get("section") else None,
            content=content,
        )
