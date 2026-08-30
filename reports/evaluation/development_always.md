# CleanBot 检索评测报告

> 本报告由 `python -m cleanbot.evaluation` 基于固定数据集生成，不包含人工美化数字。

| 指标 | 向量基线 | 混合检索 + Rerank |
|---|---:|---:|
| Hit@3 | 98.00% | 100.00% |
| MRR@5 | 0.9233 | 0.9367 |
| 引用字段完整率 | 100.00% | 100.00% |
| 平均检索延迟 | 419.88 ms | 770.11 ms |
| P95 检索延迟 | 519.63 ms | 993.72 ms |

- 数据集：`evaluation/questions.jsonl`。
- 数据集条目：60，其中知识检索题：50。
- 路由准确率：100.00%。
- Chat：`qwen3-max`；Embedding：`text-embedding-v4`；Rerank：`qwen3-rerank`。
- 向量集合：`cleanbot_knowledge_v2`；Rerank 开启：`True`；Rerank 策略：`always`。
- Dense Top-K：`20`；Sparse Top-K：`20`；Rerank Top-N：`5`。
- Answer Top-N：`4`；最低检索分数：`0.1`。
- Embedding 调用：100；Rerank 调用：50。
- 路由模型调用：14。
- 答案模型调用：0；Judge 调用：0。
- 货币成本依赖运行时供应商价格，报告只记录可复现的调用次数和供应商返回 Token，不使用过期单价推算。
