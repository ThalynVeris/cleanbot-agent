from __future__ import annotations

from dataclasses import replace

import dashscope
import pytest

from cleanbot.core.config import Settings
from cleanbot.core.schemas import KnowledgeHit
from cleanbot.rag.retriever import HybridRetriever


def hit(chunk_id: str, content: str, dense: float = 0.0, sparse: float = 0.0) -> KnowledgeHit:
    return KnowledgeHit(
        document_id="doc",
        chunk_id=chunk_id,
        source="manual.txt",
        content=content,
        dense_score=dense,
        sparse_score=sparse,
        score=max(dense, sparse),
    )


class FakeKnowledgeBase:
    def __init__(self) -> None:
        self.chunks = [
            hit("a", "主刷毛发缠绕，需要断电清理"),
            hit("b", "水箱漏水时检查密封圈"),
            hit("c", "电池鼓包应停止使用"),
        ]

    def count(self) -> int:
        return len(self.chunks)

    def all_chunks(self):
        return [item.model_copy(deep=True) for item in self.chunks]

    def dense_search(self, query: str, k: int):
        return [hit("b", self.chunks[1].content, dense=0.8), hit("a", self.chunks[0].content, dense=0.7)][:k]


class AgreeingKnowledgeBase(FakeKnowledgeBase):
    def dense_search(self, query: str, k: int) -> list[KnowledgeHit]:
        return [hit("a", self.chunks[0].content, dense=0.9), hit("b", self.chunks[1].content, dense=0.8)][:k]


class RerankPolicySettings:
    def __init__(self, settings: Settings, rerank_policy: str):
        self.settings = settings
        self.rerank_policy = rerank_policy

    def __getattr__(self, name: str) -> object:
        return getattr(self.settings, name)


def test_rrf_combines_dense_and_sparse_rankings() -> None:
    dense = [hit("a", "A", dense=0.9), hit("b", "B", dense=0.7)]
    sparse = [hit("b", "B", sparse=1.0), hit("c", "C", sparse=0.5)]
    fused = HybridRetriever.fuse_rrf(dense, sparse)
    assert fused[0].chunk_id == "b"
    assert fused[0].dense_score == 0.7
    assert fused[0].sparse_score == 1.0
    assert fused[0].score == 1.0


def test_rrf_uses_rank_positions_and_rewards_shared_results() -> None:
    dense = [
        hit("a", "A", dense=0.99),
        hit("b", "B", dense=0.10),
        hit("c", "C", dense=0.01),
    ]
    sparse = [
        hit("b", "B", sparse=0.80),
        hit("d", "D", sparse=0.70),
        hit("a", "A", sparse=0.60),
    ]

    fused = HybridRetriever.fuse_rrf(
        dense,
        sparse,
    )

    assert [item.chunk_id for item in fused] == [
        "b",
        "a",
        "d",
        "c",
    ]

    by_id = {item.chunk_id: item for item in fused}

    expected_b = 1 / 62 + 1 / 61
    expected_a = 1 / 61 + 1 / 63

    assert by_id["b"].fusion_score == pytest.approx(expected_b)
    assert by_id["a"].fusion_score == pytest.approx(expected_a)
    assert by_id["b"].score == 1.0
    assert by_id["a"].score == pytest.approx(expected_a / expected_b)


async def test_hybrid_retrieval_works_without_rerank(settings: Settings) -> None:
    retriever = HybridRetriever(FakeKnowledgeBase(), settings)  # type: ignore[arg-type]
    results = await retriever.retrieve("主刷毛发")
    assert results
    assert results[0].chunk_id == "a"
    assert results[0].sparse_score > 0


@pytest.mark.asyncio
async def test_rerank_provider_failure_falls_back_to_rrf(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    rerank_settings = replace(
        settings,
        enable_rerank=True,
        dashscope_api_key="test-api-key",
    )
    retriever = HybridRetriever(
        knowledge_base=FakeKnowledgeBase(),
        settings=rerank_settings,
    )

    def raise_provider_error(**kwargs: object) -> None:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        dashscope.TextReRank,
        "call",
        raise_provider_error,
    )

    results = await retriever.retrieve("主刷毛发")

    assert [hit.chunk_id for hit in results] == ["a", "b"]
    assert all(hit.rerank_score is None for hit in results)
    assert results[0].fusion_score > results[1].fusion_score
    assert "rerank_failed_falling_back_to_rrf" in caplog.text


@pytest.mark.asyncio
async def test_disagreement_policy_skips_rerank_when_retrievers_agree(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_settings = replace(
        settings,
        enable_rerank=True,
        dashscope_api_key="test-api-key",
    )
    policy_settings = RerankPolicySettings(
        enabled_settings,
        rerank_policy="disagreement",
    )
    retriever = HybridRetriever(
        AgreeingKnowledgeBase(),  # type: ignore[arg-type]
        policy_settings,  # type: ignore[arg-type]
    )
    rerank_calls = 0

    async def count_rerank(
        query: str,
        candidates: list[KnowledgeHit],
    ) -> list[KnowledgeHit]:
        nonlocal rerank_calls
        rerank_calls += 1
        return candidates

    monkeypatch.setattr(
        retriever,
        "_rerank",
        count_rerank,
    )

    results = await retriever.retrieve("主刷毛发")

    assert results
    assert results[0].chunk_id == "a"
    assert rerank_calls == 0


@pytest.mark.asyncio
async def test_disagreement_policy_calls_rerank_when_retrievers_disagree(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enabled_settings = replace(
        settings,
        enable_rerank=True,
        dashscope_api_key="test-api-key",
    )
    policy_settings = RerankPolicySettings(
        enabled_settings,
        rerank_policy="disagreement",
    )
    retriever = HybridRetriever(
        FakeKnowledgeBase(),  # type: ignore[arg-type]
        policy_settings,  # type: ignore[arg-type]
    )
    rerank_calls = 0

    async def count_rerank(
        query: str,
        candidates: list[KnowledgeHit],
    ) -> list[KnowledgeHit]:
        nonlocal rerank_calls
        rerank_calls += 1
        return candidates

    monkeypatch.setattr(
        retriever,
        "_rerank",
        count_rerank,
    )

    results = await retriever.retrieve("主刷毛发")

    assert results
    assert rerank_calls == 1
