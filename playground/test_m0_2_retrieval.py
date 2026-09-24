from __future__ import annotations

import pytest

from cleanbot.core.schemas import KnowledgeHit
from cleanbot.rag.chunking import tokenize_for_bm25 as reference_tokenize
from cleanbot.rag.retriever import HybridRetriever
from playground.m0_2_retrieval import fuse_rrf, tokenize_for_bm25


def hit(chunk_id: str, dense: float = 0.0, sparse: float = 0.0) -> KnowledgeHit:
    return KnowledgeHit(
        document_id="doc",
        chunk_id=chunk_id,
        source="manual.txt",
        content=f"content {chunk_id}",
        dense_score=dense,
        sparse_score=sparse,
        score=max(dense, sparse),
    )


@pytest.mark.parametrize(
    "text",
    ["主刷缠绕", "HEPA滤网 2500Pa 吸力", "", "扫地机器人，找不到充电座？", "A1 型号"],
)
def test_tokenizer_matches_reference(text: str) -> None:
    assert tokenize_for_bm25(text) == reference_tokenize(text)


@pytest.mark.parametrize(
    ("dense", "sparse"),
    [
        ([hit("a", dense=0.9), hit("b", dense=0.7)], [hit("b", sparse=1.0), hit("c", sparse=0.5)]),
        ([hit("a", dense=0.9)], []),
        ([], []),
        ([hit(c, dense=0.5) for c in "abcdef"], [hit(c, sparse=0.5) for c in "fedcba"]),
    ],
)
def test_rrf_matches_reference(dense: list[KnowledgeHit], sparse: list[KnowledgeHit]) -> None:
    mine = fuse_rrf(dense, sparse)
    reference = HybridRetriever.fuse_rrf(dense, sparse)
    assert [h.model_dump() for h in mine] == [h.model_dump() for h in reference]


def test_rrf_does_not_mutate_inputs() -> None:
    dense = [hit("a", dense=0.9)]
    sparse = [hit("a", sparse=0.8)]
    fuse_rrf(dense, sparse)
    assert dense[0].sparse_score == 0.0
    assert sparse[0].dense_score == 0.0
