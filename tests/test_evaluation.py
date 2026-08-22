from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from cleanbot.core.config import Settings
from cleanbot.core.schemas import Intent, KnowledgeHit
from cleanbot.evaluation.runner import EvaluationItem, EvaluationRunner


def test_evaluation_relevance_and_metrics(tmp_path: Path) -> None:
    item = EvaluationItem(
        id="q1",
        category="test",
        query="主刷",
        expected_intent="knowledge",
        expected_contains=["毛发缠绕"],
    )
    hits = [
        KnowledgeHit(
            document_id="doc",
            chunk_id="chunk",
            source="manual.txt",
            content="处理水箱",
            score=1,
        ),
        KnowledgeHit(
            document_id="doc",
            chunk_id="chunk2",
            source="manual.txt",
            content="清理主刷毛发缠绕",
            score=0.8,
        ),
    ]
    row = EvaluationRunner._retrieval_row(item, hits, 0.01)
    summary = EvaluationRunner._summarize_retrieval([row])
    assert row["first_relevant_rank"] == 2
    assert summary["hit_at_3"] == 1.0
    assert summary["mrr_at_5"] == 0.5


def test_answer_sample_is_stratified_by_category() -> None:
    items = [
        EvaluationItem(
            id=f"{category}-{index}",
            category=category,
            query="test",
            expected_intent="knowledge",
        )
        for category in ("exact", "paraphrase", "maintenance")
        for index in range(2)
    ]
    sampled = EvaluationRunner._stratified_sample(items, 4)
    assert [item.id for item in sampled] == [
        "exact-0",
        "paraphrase-0",
        "maintenance-0",
        "exact-1",
    ]


class FakeRetriever:
    async def baseline(self, query: str, k: int):
        return [self._hit()]

    async def retrieve(self, query: str):
        return [self._hit()]

    @staticmethod
    def _hit() -> KnowledgeHit:
        return KnowledgeHit(
            document_id="doc",
            chunk_id="chunk",
            source="manual.txt",
            content="主刷毛发缠绕时，应断电清理。",
            score=0.9,
        )


class FakeRouter:
    def deterministic(self, query: str):
        return Intent.KNOWLEDGE if "主刷" in query else Intent.OUT_OF_SCOPE

    async def classify(self, query: str):
        return self.deterministic(query)


class FakeModel:
    async def ainvoke(self, prompt: str):
        if "Return only one JSON object" in prompt:
            return AIMessage(
                content=('{"faithfulness":1,"relevance":1,"unsupported_claims":[],"reason":"supported"}')
            )
        return AIMessage(
            content="请断电清理主刷。[来源1]",
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )


async def test_full_runner_includes_answer_metrics_and_cost_counters(
    settings: Settings, tmp_path: Path
) -> None:
    dataset = tmp_path / "questions.jsonl"
    records = [
        {
            "id": "q1",
            "category": "test",
            "query": "主刷毛发怎么清理",
            "expected_intent": "knowledge",
            "expected_contains": ["毛发缠绕"],
        },
        {
            "id": "q2",
            "category": "test",
            "query": "股票预测",
            "expected_intent": "out_of_scope",
            "expected_contains": [],
        },
    ]
    dataset.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")
    runner = EvaluationRunner(
        FakeRetriever(),  # type: ignore[arg-type]
        FakeRouter(),  # type: ignore[arg-type]
        FakeModel(),  # type: ignore[arg-type]
        settings,
    )
    result = await runner.run(dataset, answer_sample_size=1)
    assert result["baseline"]["hit_at_3"] == 1.0
    assert result["routing"]["accuracy"] == 1.0
    assert result["answers"]["faithfulness"] == 1.0
    assert result["answers"]["citation_correctness"] == 1.0
    assert result["cost_counters"]["reported_generation_tokens"] == 15

    json_output = tmp_path / "result.json"
    markdown_output = tmp_path / "result.md"
    runner.save(result, json_output, markdown_output)
    assert "答案级抽样评测" in markdown_output.read_text(encoding="utf-8")


async def test_report_records_reproducible_configuration(
    settings: Settings,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "reproducible.jsonl"
    record = {
        "id": "q1",
        "category": "test",
        "query": "主刷毛发怎么清理",
        "expected_intent": "knowledge",
        "expected_contains": ["毛发缠绕"],
    }
    dataset.write_text(
        json.dumps(record, ensure_ascii=False),
        encoding="utf-8",
    )

    runner = EvaluationRunner(
        FakeRetriever(),  # type: ignore[arg-type]
        FakeRouter(),  # type: ignore[arg-type]
        None,
        settings,
    )
    result = await runner.run(dataset)

    assert result["dataset"] == str(dataset)
    assert result["models"]["chat"] == settings.chat_model_name
    assert result["models"]["embedding"] == settings.embedding_model_name
    assert result["retrieval_config"] == {
        "collection_name": settings.collection_name,
        "enable_rerank": settings.enable_rerank,
        "dense_top_k": settings.dense_top_k,
        "sparse_top_k": settings.sparse_top_k,
        "rerank_top_n": settings.rerank_top_n,
        "answer_top_n": settings.answer_top_n,
        "min_retrieval_score": settings.min_retrieval_score,
    }

    json_output = tmp_path / "result.json"
    markdown_output = tmp_path / "result.md"
    runner.save(result, json_output, markdown_output)

    report = markdown_output.read_text(encoding="utf-8")
    assert f"数据集：`{dataset}`" in report
    assert f"Dense Top-K：`{settings.dense_top_k}`" in report
    assert f"最低检索分数：`{settings.min_retrieval_score}`" in report
