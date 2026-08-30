# CleanBot 检索评测报告

> 本报告由 `python -m cleanbot.evaluation` 基于固定数据集生成，不包含人工美化数字。

| 指标 | 向量基线 | 混合检索 + Rerank |
|---|---:|---:|
| Hit@3 | 62.50% | 81.25% |
| MRR@5 | 0.5625 | 0.7865 |
| 引用字段完整率 | 100.00% | 100.00% |
| 平均检索延迟 | 382.88 ms | 637.93 ms |
| P95 检索延迟 | 407.80 ms | 802.97 ms |

- 数据集：`evaluation/heldout.jsonl`。
- 数据集条目：20，其中知识检索题：16。
- 路由准确率：95.00%。
- Chat：`qwen3-max`；Embedding：`text-embedding-v4`；Rerank：`qwen3-rerank`。
- 向量集合：`cleanbot_knowledge_v2`；Rerank 开启：`True`；Rerank 策略：`disagreement`。
- Dense Top-K：`20`；Sparse Top-K：`20`；Rerank Top-N：`5`。
- Answer Top-N：`4`；最低检索分数：`0.1`。
- Embedding 调用：32；Rerank 调用：11。
- 路由模型调用：10。
- 答案模型调用：0；Judge 调用：0。
- 货币成本依赖运行时供应商价格，报告只记录可复现的调用次数和供应商返回 Token，不使用过期单价推算。
