from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from cleanbot.core.config import Settings, get_settings
from cleanbot.core.schemas import Intent, KnowledgeHit
from cleanbot.rag.retriever import HybridRetriever
from cleanbot.workflow.router import IntentRouter


class EvaluationItem(BaseModel):
    id: str
    category: str
    query: str
    expected_intent: Intent
    expected_contains: list[str] = Field(default_factory=list)


class AnswerJudge(BaseModel):
    faithfulness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list)
    reason: str = Field(description="Brief evidence-based reason, not chain-of-thought")


class EvaluationRunner:
    def __init__(
        self,
        retriever: HybridRetriever,
        router: IntentRouter,
        model: BaseChatModel | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.retriever = retriever
        self.router = router
        self.model = model
        self.settings = settings or get_settings()

    @staticmethod
    def load_dataset(path: Path) -> list[EvaluationItem]:
        items: list[EvaluationItem] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    items.append(EvaluationItem.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"Invalid evaluation item at line {line_number}") from exc
        return items

    async def run(
        self, dataset_path: Path, limit: int | None = None, answer_sample_size: int = 0
    ) -> dict[str, Any]:
        items = self.load_dataset(dataset_path)
        if limit:
            items = items[:limit]
        baseline_rows: list[dict[str, Any]] = []
        optimized_rows: list[dict[str, Any]] = []
        routing_rows: list[dict[str, Any]] = []
        rerank_calls_before = int(getattr(self.retriever, "rerank_calls", 0))
        for item in items:
            route_started = time.perf_counter()
            predicted = await self.router.classify(item.query)
            routing_rows.append(
                {
                    "id": item.id,
                    "expected": item.expected_intent.value,
                    "predicted": predicted.value,
                    "correct": predicted == item.expected_intent,
                    "latency_ms": round((time.perf_counter() - route_started) * 1000, 2),
                }
            )
            if item.expected_intent != Intent.KNOWLEDGE:
                continue

            baseline_started = time.perf_counter()
            baseline_hits = await self.retriever.baseline(item.query, k=5)
            baseline_rows.append(
                self._retrieval_row(item, baseline_hits, time.perf_counter() - baseline_started)
            )

            optimized_started = time.perf_counter()
            optimized_hits = await self.retriever.retrieve(item.query)
            optimized_rows.append(
                self._retrieval_row(item, optimized_hits, time.perf_counter() - optimized_started)
            )

        knowledge_items = [item for item in items if item.expected_intent == Intent.KNOWLEDGE]
        answer_evaluation = await self._evaluate_answers(
            self._stratified_sample(knowledge_items, answer_sample_size)
        )
        rerank_calls_after = int(getattr(self.retriever, "rerank_calls", 0))
        actual_rerank_calls = max(
            0,
            rerank_calls_after - rerank_calls_before,
        )
        routing_model_calls = sum(self.router.deterministic(item.query) is None for item in items)
        return {
            "dataset": str(dataset_path),
            "items": len(items),
            "knowledge_items": len(baseline_rows),
            "models": {
                "chat": self.settings.chat_model_name,
                "embedding": self.settings.embedding_model_name,
                "rerank": (self.settings.rerank_model_name if self.settings.enable_rerank else None),
            },
            "retrieval_config": {
                "collection_name": self.settings.collection_name,
                "enable_rerank": self.settings.enable_rerank,
                "rerank_policy": self.settings.rerank_policy,
                "dense_top_k": self.settings.dense_top_k,
                "sparse_top_k": self.settings.sparse_top_k,
                "rerank_top_n": self.settings.rerank_top_n,
                "answer_top_n": self.settings.answer_top_n,
                "min_retrieval_score": self.settings.min_retrieval_score,
            },
            "baseline": self._summarize_retrieval(baseline_rows),
            "optimized": self._summarize_retrieval(optimized_rows),
            "routing": {
                "accuracy": self._mean([float(row["correct"]) for row in routing_rows]),
                "mean_latency_ms": self._mean([row["latency_ms"] for row in routing_rows]),
                "rows": routing_rows,
            },
            "answers": answer_evaluation,
            "cost_counters": {
                "embedding_calls": len(baseline_rows) * 2,
                "rerank_calls": actual_rerank_calls,
                "routing_model_calls": routing_model_calls,
                "answer_model_calls": answer_evaluation.get("successful_samples", 0),
                "judge_model_calls": answer_evaluation.get("successful_samples", 0),
                "reported_generation_tokens": answer_evaluation.get("reported_tokens", 0),
                "note": (
                    "Counters are reproducible; exact currency cost depends on provider price at run time."
                ),
            },
        }

    async def _evaluate_answers(self, items: list[EvaluationItem]) -> dict[str, Any]:
        if not items:
            return {"samples": 0, "successful_samples": 0, "rows": []}
        if self.model is None:
            return {
                "samples": len(items),
                "successful_samples": 0,
                "rows": [],
                "error": "No model configured for answer evaluation",
            }

        rows: list[dict[str, Any]] = []
        reported_tokens = 0
        for item in items:
            try:
                hits = await self.retriever.retrieve(item.query)
                context = "\n\n".join(
                    f"[来源{index}] {hit.content}" for index, hit in enumerate(hits, start=1)
                )
                generation_started = time.perf_counter()
                response = await self.model.ainvoke(
                    f"""仅根据资料回答扫地机器人问题。
每个包含信息的段落或列表项末尾都标注至少一个 [来源N]，开头直接结论也必须引用；
资料不足时明确拒答，不依据常识补充未引用内容，不输出内部推理。

问题：{item.query}
<references>
{context}
</references>"""
                )
                answer = self._message_text(response)
                usage = self._usage_metadata(response)
                generation_ms = round((time.perf_counter() - generation_started) * 1000, 2)

                judged_response = await self.model.ainvoke(
                    f"""Evaluate an answer using only the supplied references.
Faithfulness is the proportion of answer claims supported by references.
Relevance measures how directly the answer addresses the question.
List unsupported claims. Scores must be between 0 and 1.
Return only one JSON object with exactly these fields:
{{"faithfulness": 0.0, "relevance": 0.0, "unsupported_claims": [], "reason": "brief reason"}}
Do not use Markdown fences and do not include hidden reasoning.

Question: {item.query}
References:
{context}

Answer:
{answer}"""
                )
                judge_usage = self._usage_metadata(judged_response)
                generation_tokens = int(usage.get("total_tokens", 0) or 0)
                judge_tokens = int(judge_usage.get("total_tokens", 0) or 0)
                reported_tokens += generation_tokens + judge_tokens
                result = self._parse_judge(self._message_text(judged_response))
                citations = [int(value) for value in re.findall(r"\[来源(\d+)]", answer)]
                citation_correct = bool(citations) and all(1 <= value <= len(hits) for value in citations)
                rows.append(
                    {
                        "id": item.id,
                        "category": item.category,
                        "answer": answer,
                        "faithfulness": result.faithfulness,
                        "relevance": result.relevance,
                        "citation_correct": citation_correct,
                        "unsupported_claims": result.unsupported_claims,
                        "judge_reason": result.reason,
                        "generation_latency_ms": generation_ms,
                        "generation_tokens": generation_tokens,
                        "judge_tokens": judge_tokens,
                    }
                )
            except Exception as exc:
                rows.append({"id": item.id, "error": type(exc).__name__})

        successful = [row for row in rows if "error" not in row]
        return {
            "samples": len(items),
            "successful_samples": len(successful),
            "faithfulness": self._mean([row["faithfulness"] for row in successful]),
            "relevance": self._mean([row["relevance"] for row in successful]),
            "citation_correctness": self._mean([float(row["citation_correct"]) for row in successful]),
            "mean_generation_latency_ms": self._mean([row["generation_latency_ms"] for row in successful]),
            "reported_tokens": reported_tokens,
            "rows": rows,
            "limitation": (
                "The generator and judge use the configured provider; manual review is still required."
            ),
        }

    @classmethod
    def _retrieval_row(
        cls, item: EvaluationItem, hits: list[KnowledgeHit], elapsed_seconds: float
    ) -> dict[str, Any]:
        relevant_ranks = [
            rank for rank, hit in enumerate(hits, start=1) if cls._is_relevant(hit, item.expected_contains)
        ]
        first_rank = min(relevant_ranks) if relevant_ranks else None
        valid_citations = all(
            hit.document_id != "unknown" and bool(hit.chunk_id) and hit.source != "unknown" for hit in hits
        )
        return {
            "id": item.id,
            "category": item.category,
            "query": item.query,
            "first_relevant_rank": first_rank,
            "hit_at_3": bool(first_rank and first_rank <= 3),
            "reciprocal_rank_at_5": 1 / first_rank if first_rank and first_rank <= 5 else 0.0,
            "citation_fields_valid": valid_citations,
            "latency_ms": round(elapsed_seconds * 1000, 2),
            "hits": [
                {
                    "source": hit.source,
                    "chunk_id": hit.chunk_id,
                    "score": round(hit.score, 6),
                    "relevant": cls._is_relevant(hit, item.expected_contains),
                }
                for hit in hits
            ],
        }

    @staticmethod
    def _is_relevant(hit: KnowledgeHit, expected_contains: list[str]) -> bool:
        normalized = hit.content.casefold().replace(" ", "")
        return any(term.casefold().replace(" ", "") in normalized for term in expected_contains)

    @classmethod
    def _summarize_retrieval(cls, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "hit_at_3": cls._mean([float(row["hit_at_3"]) for row in rows]),
            "mrr_at_5": cls._mean([row["reciprocal_rank_at_5"] for row in rows]),
            "citation_field_validity": cls._mean([float(row["citation_fields_valid"]) for row in rows]),
            "mean_latency_ms": cls._mean([row["latency_ms"] for row in rows]),
            "p95_latency_ms": cls._percentile([row["latency_ms"] for row in rows], 0.95),
            "rows": rows,
        }

    @staticmethod
    def _mean(values: list[float]) -> float:
        return round(statistics.fmean(values), 4) if values else 0.0

    @staticmethod
    def _percentile(values: list[float], quantile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * quantile))))
        return round(ordered[index], 2)

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(item.get("text", "") if isinstance(item, dict) else str(item) for item in content)
        return str(content)

    @staticmethod
    def _usage_metadata(message: Any) -> dict[str, Any]:
        direct = getattr(message, "usage_metadata", None)
        if isinstance(direct, dict) and direct:
            return direct
        response_metadata = getattr(message, "response_metadata", None) or {}
        if isinstance(response_metadata, dict):
            for key in ("token_usage", "usage"):
                nested = response_metadata.get(key)
                if isinstance(nested, dict):
                    return nested
        return {}

    @staticmethod
    def _stratified_sample(items: list[EvaluationItem], size: int) -> list[EvaluationItem]:
        if size <= 0:
            return []
        grouped: dict[str, list[EvaluationItem]] = {}
        for item in items:
            grouped.setdefault(item.category, []).append(item)

        selected: list[EvaluationItem] = []
        round_index = 0
        while len(selected) < size:
            added = False
            for group in grouped.values():
                if round_index < len(group):
                    selected.append(group[round_index])
                    added = True
                    if len(selected) == size:
                        return selected
            if not added:
                break
            round_index += 1
        return selected

    @staticmethod
    def _parse_judge(text: str) -> AnswerJudge:
        stripped = text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?})\s*```", stripped, re.DOTALL)
        if fenced:
            stripped = fenced.group(1)
        else:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start >= 0 and end > start:
                stripped = stripped[start : end + 1]
        return AnswerJudge.model_validate_json(stripped)

    @staticmethod
    def save(result: dict[str, Any], json_path: Path, markdown_path: Path) -> None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        baseline = result["baseline"]
        optimized = result["optimized"]
        routing = result["routing"]
        costs = result["cost_counters"]
        retrieval_config = result["retrieval_config"]
        answers = result.get("answers", {})
        answer_section = ""
        if answers.get("samples"):
            answer_section = f"""
## 答案级抽样评测

- 抽样数/成功数：{answers["samples"]} / {answers["successful_samples"]}
- LLM Judge 忠实度：{answers.get("faithfulness", 0):.2%}
- LLM Judge 回答相关性：{answers.get("relevance", 0):.2%}
- 引用编号正确率：{answers.get("citation_correctness", 0):.2%}
- 平均生成延迟：{answers.get("mean_generation_latency_ms", 0):.2f} ms
- 供应商返回的答案与 Judge Token 合计：{answers.get("reported_tokens", 0)}

Judge 与生成器使用同一供应商，可能存在同源偏差；该结果必须结合人工抽查，不能表述为“答案准确率”。
"""
        markdown_path.write_text(
            f"""# CleanBot 检索评测报告

> 本报告由 `python -m cleanbot.evaluation` 基于固定数据集生成，不包含人工美化数字。

| 指标 | 向量基线 | 混合检索 + Rerank |
|---|---:|---:|
| Hit@3 | {baseline["hit_at_3"]:.2%} | {optimized["hit_at_3"]:.2%} |
| MRR@5 | {baseline["mrr_at_5"]:.4f} | {optimized["mrr_at_5"]:.4f} |
| 引用字段完整率 | {baseline["citation_field_validity"]:.2%} | {optimized["citation_field_validity"]:.2%} |
| 平均检索延迟 | {baseline["mean_latency_ms"]:.2f} ms | {optimized["mean_latency_ms"]:.2f} ms |
| P95 检索延迟 | {baseline["p95_latency_ms"]:.2f} ms | {optimized["p95_latency_ms"]:.2f} ms |

- 数据集：`{result["dataset"]}`。
- 数据集条目：{result["items"]}，其中知识检索题：{result["knowledge_items"]}。
- 路由准确率：{routing["accuracy"]:.2%}。
- Chat：`{result["models"]["chat"]}`；Embedding：`{result["models"]["embedding"]}`；Rerank：`{result["models"]["rerank"]}`。
- 向量集合：`{retrieval_config["collection_name"]}`；Rerank 开启：`{retrieval_config["enable_rerank"]}`；Rerank 策略：`{retrieval_config["rerank_policy"]}`。
- Dense Top-K：`{retrieval_config["dense_top_k"]}`；Sparse Top-K：`{retrieval_config["sparse_top_k"]}`；Rerank Top-N：`{retrieval_config["rerank_top_n"]}`。
- Answer Top-N：`{retrieval_config["answer_top_n"]}`；最低检索分数：`{retrieval_config["min_retrieval_score"]}`。
- Embedding 调用：{costs["embedding_calls"]}；Rerank 调用：{costs["rerank_calls"]}。
- 路由模型调用：{costs["routing_model_calls"]}。
- 答案模型调用：{costs["answer_model_calls"]}；Judge 调用：{costs["judge_model_calls"]}。
- 货币成本依赖运行时供应商价格，报告只记录可复现的调用次数和供应商返回 Token，不使用过期单价推算。
{answer_section}
""",
            encoding="utf-8",
        )
