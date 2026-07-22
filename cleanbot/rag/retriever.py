from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import RLock

import dashscope
from rank_bm25 import BM25Okapi

from cleanbot.core.config import Settings, get_settings
from cleanbot.core.logging import get_logger
from cleanbot.core.schemas import KnowledgeHit
from cleanbot.rag.chunking import tokenize_for_bm25
from cleanbot.rag.knowledge_base import KnowledgeBase

logger = get_logger(__name__)


@dataclass(slots=True)
class _SparseIndex:
    count: int
    chunks: list[KnowledgeHit]
    bm25: BM25Okapi


class HybridRetriever:
    def __init__(self, knowledge_base: KnowledgeBase, settings: Settings | None = None) -> None:
        self.knowledge_base = knowledge_base
        self.settings = settings or get_settings()
        self._index: _SparseIndex | None = None
        self._index_lock = RLock()

    def invalidate(self) -> None:
        with self._index_lock:
            self._index = None

    def _sparse_index(self) -> _SparseIndex | None:
        count = self.knowledge_base.count()
        with self._index_lock:
            if self._index is not None and self._index.count == count:
                return self._index
            chunks = self.knowledge_base.all_chunks()
            if not chunks:
                self._index = None
                return None
            tokenized = [tokenize_for_bm25(chunk.content) or ["空"] for chunk in chunks]
            self._index = _SparseIndex(count=count, chunks=chunks, bm25=BM25Okapi(tokenized))
            return self._index

    def sparse_search(self, query: str, k: int | None = None) -> list[KnowledgeHit]:
        index = self._sparse_index()
        if index is None:
            return []
        k = k or self.settings.sparse_top_k
        scores = index.bm25.get_scores(tokenize_for_bm25(query) or [query])
        ranked = sorted(range(len(scores)), key=lambda item: float(scores[item]), reverse=True)[:k]
        max_score = max((float(scores[item]) for item in ranked), default=0.0)
        hits: list[KnowledgeHit] = []
        for item in ranked:
            if float(scores[item]) <= 0:
                continue
            hit = index.chunks[item].model_copy(deep=True)
            hit.sparse_score = float(scores[item]) / max_score if max_score else 0.0
            hit.score = hit.sparse_score
            hits.append(hit)
        return hits

    async def baseline(self, query: str, k: int = 3) -> list[KnowledgeHit]:
        return await asyncio.to_thread(self.knowledge_base.dense_search, query, k)

    async def retrieve(self, query: str) -> list[KnowledgeHit]:
        dense_task = asyncio.to_thread(self.knowledge_base.dense_search, query, self.settings.dense_top_k)
        sparse_task = asyncio.to_thread(self.sparse_search, query, self.settings.sparse_top_k)
        dense, sparse = await asyncio.gather(dense_task, sparse_task)
        fused = self.fuse_rrf(dense, sparse)
        if not fused:
            return []

        ranked = fused
        if self.settings.enable_rerank and self.settings.dashscope_api_key:
            try:
                ranked = await self._rerank(query, fused[: max(self.settings.rerank_top_n * 4, 12)])
            except Exception as exc:  # Provider failure must not take down question answering.
                logger.warning(
                    "rerank_failed_falling_back_to_rrf",
                    extra={"context": {"error_type": type(exc).__name__}},
                )

        filtered = [hit for hit in ranked if hit.score >= self.settings.min_retrieval_score]
        return filtered[: self.settings.answer_top_n]

    @staticmethod
    def fuse_rrf(
        dense: list[KnowledgeHit], sparse: list[KnowledgeHit], rank_constant: int = 60
    ) -> list[KnowledgeHit]:
        candidates: dict[str, KnowledgeHit] = {}
        fusion_scores: dict[str, float] = {}
        for result_set in (dense, sparse):
            for rank, hit in enumerate(result_set, start=1):
                if hit.chunk_id not in candidates:
                    candidates[hit.chunk_id] = hit.model_copy(deep=True)
                candidate = candidates[hit.chunk_id]
                candidate.dense_score = max(candidate.dense_score, hit.dense_score)
                candidate.sparse_score = max(candidate.sparse_score, hit.sparse_score)
                fusion_scores[hit.chunk_id] = fusion_scores.get(hit.chunk_id, 0.0) + 1.0 / (
                    rank_constant + rank
                )

        if not fusion_scores:
            return []
        max_fusion = max(fusion_scores.values())
        for chunk_id, candidate in candidates.items():
            candidate.fusion_score = fusion_scores[chunk_id]
            candidate.score = candidate.fusion_score / max_fusion if max_fusion else 0.0
        return sorted(candidates.values(), key=lambda hit: hit.fusion_score, reverse=True)

    async def _rerank(self, query: str, candidates: list[KnowledgeHit]) -> list[KnowledgeHit]:
        if not candidates:
            return []

        def call_provider():
            return dashscope.TextReRank.call(
                model=self.settings.rerank_model_name,
                query=query,
                documents=[candidate.content for candidate in candidates],
                top_n=min(self.settings.rerank_top_n, len(candidates)),
                return_documents=False,
                instruct="Retrieve passages that directly answer the cleaning-robot support question.",
                api_key=self.settings.dashscope_api_key,
            )

        response = await asyncio.to_thread(call_provider)
        status_code = getattr(response, "status_code", 200)
        if status_code != 200:
            raise RuntimeError(f"Rerank provider returned status {status_code}")
        output = getattr(response, "output", None) or response.get("output", {})
        results = output.get("results", []) if isinstance(output, dict) else getattr(output, "results", [])
        reranked: list[KnowledgeHit] = []
        for result in results:
            index = result.get("index") if isinstance(result, dict) else getattr(result, "index", None)
            score = (
                result.get("relevance_score")
                if isinstance(result, dict)
                else getattr(result, "relevance_score", None)
            )
            if index is None or not 0 <= int(index) < len(candidates):
                continue
            hit = candidates[int(index)].model_copy(deep=True)
            hit.rerank_score = float(score or 0.0)
            hit.score = max(0.0, min(1.0, hit.rerank_score))
            reranked.append(hit)
        if not reranked:
            raise RuntimeError("Rerank provider returned no valid results")
        return reranked
