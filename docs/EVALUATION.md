# CleanBot 定量评测报告

本文档汇总仓库中三次正式实验的真实结果，并说明数据划分、指标口径、策略选择和限制。原始 Markdown 报告保存在 `reports/evaluation/`，生成过程中不手工修改指标。

## 1. 评测目标

评测回答以下问题：

1. Hybrid RAG 是否比 Dense Baseline 更容易在前几名找到人工标注的相关切片？
2. 条件 Rerank 能否在基本保持检索质量的同时减少外部重排调用和延迟？
3. 来源字段、意图路由和状态隔离是否满足工程契约？

## 2. 数据与实验配置

| 配置项 | 验证集 | 冻结测试集 |
|---|---:|---:|
| 数据文件 | `evaluation/questions.jsonl` | `evaluation/heldout.jsonl` |
| 总题数 | 60 | 20 |
| 知识检索题 | 50 | 16 |
| 路由/超范围题 | 10 | 4 |
| 用途 | 策略选择与参数调试 | 独立检查泛化效果 |
| 是否参与逐题调参 | 是 | 否 |

冻结测试集覆盖口语/错别字、隐式表达、型号与数字、路由和超范围问题。项目没有根据这些问题更新任何模型参数，因此这里使用“验证集”和“冻结测试集”，不将其描述为模型训练集。

| 检索配置 | 固定值 |
|---|---|
| Chat Model | `qwen3-max` |
| Embedding | `text-embedding-v4` |
| Rerank | `qwen3-rerank` |
| Vector Collection | `cleanbot_knowledge_v2` |
| Dense Top-K | 20 |
| Sparse Top-K | 20 |
| Rerank Top-N | 5 |
| Answer Top-N | 4 |
| 最低检索分数 | 0.10 |
| 知识库规模 | 6 份文档、913 个结构化切片 |

### 指标口径

- **Hit@3**：相关切片是否出现在前三名，衡量召回是否覆盖正确证据。
- **MRR@5**：第一个相关切片在前五名中的倒数排名均值，更关注正确证据是否靠前。
- **引用字段映射正确率**：返回结果能否映射到有效的文档、切片、来源、章节/页码字段；它不等价于回答正确率。
- **平均/P95 检索延迟**：单次运行中的端到端检索耗时，不包含最终答案生成。
- **意图识别准确率**：全部知识题和路由题的业务意图分类结果。

## 3. 验证集：Rerank 策略选择

两个策略使用同一份 60 条验证集：

- `always`：每条知识检索题都调用 Rerank。
- `disagreement`：Dense 与 BM25 第一名不一致时才调用 Rerank，否则保留 RRF 结果。

| 策略 | Hit@3 | MRR@5 | 平均检索延迟 | P95 检索延迟 | Rerank 调用 |
|---|---:|---:|---:|---:|---:|
| Always Rerank | 100.00% | 0.9367 | 770.11 ms | 993.72 ms | 50 |
| Conditional Rerank | 100.00% | 0.9800 | 560.09 ms | 794.37 ms | 24 |

条件策略相较全量 Rerank：

| 变化 | 结果 |
|---|---:|
| Rerank 调用 | `50 → 24`，减少 **52%** |
| 平均检索延迟 | 降低约 **27.3%** |
| P95 检索延迟 | 降低约 **20.1%** |
| Hit@3 | 持平，均为 **100%** |
| MRR@5 | `0.9367 → 0.9800`，提升 **0.0433** |

因此当前默认策略选择 `disagreement`。这项结论只表示它在本验证集上取得了更好的质量/调用次数平衡；不同实验在不同时间调用外部 API，绝对延迟会受到网络和供应商负载影响。

原始结果：[Always Rerank](../reports/evaluation/development_always.md) · [Conditional Rerank](../reports/evaluation/development_disagreement.md)

## 4. 冻结测试集：Dense 与最终方案

最终策略确定后，在 20 条冻结测试集上进行一次独立实验：

| 方案 | Hit@3 | MRR@5 | 引用字段映射正确率 | 平均检索延迟 | P95 检索延迟 |
|---|---:|---:|---:|---:|---:|
| Dense Baseline | 62.50% | 0.5625 | 100.00% | 382.88 ms | 407.80 ms |
| Hybrid + Conditional Rerank | 81.25% | 0.7865 | 100.00% | 637.93 ms | 802.97 ms |

| 变化 | 结果 |
|---|---:|
| Hit@3 | 提升 **18.75 个百分点** |
| MRR@5 | 提升 **0.2240** |
| 平均检索延迟 | 增加 **255.05 ms** |
| P95 检索延迟 | 增加 **395.17 ms** |
| Rerank 调用 | 16 条知识题中调用 **11 次** |
| 全量意图识别准确率 | **95%（19/20）** |

结果表明混合检索显著改善了口语化和隐式表达下的证据排序，但引入稀疏召回、融合和外部 Rerank 后延迟增加。冻结测试集 Hit@3 为 **81.25%**，没有达到预设的 85% 目标，因此项目不将该目标表述为已经完成。

原始结果：[冻结测试集报告](../reports/evaluation/heldout_final.md)

## 5. 回答级抽样验证

另一次开发阶段实验从验证集中分层抽取 10 条知识题，检查完整的检索、生成与引用链路：

| 项目 | 结果 |
|---|---:|
| 执行成功 | 10 / 10 |
| Judge 忠实度通过 | 10 / 10 |
| Judge 回答相关性通过 | 10 / 10 |
| 引用编号格式正确 | 10 / 10 |
| 平均生成延迟 | 2019.79 ms |
| 生成与 Judge Token 合计 | 8116 |

生成器和 Judge 使用同一供应商模型，可能存在同源偏差；样本量也只有 10 条。因此这些数字只作为 Prompt 和引用链路的辅助检查，不能表述为“回答准确率 100%”。

## 6. 工程质量验证

| 检查项 | 实测结果 |
|---|---:|
| 自动化测试 | 71 项通过 |
| 核心模块覆盖率 | 87% |
| CI 覆盖率门槛 | 80% |
| 并发隔离 | 10 个模拟模型会话，无串话和未处理异常 |
| 知识库 | 6 份文档，913 个切片 |
| Docker Compose | `web + api + device-mcp + postgres` 四个服务完成验收 |

测试覆盖配置、数据库关系、会话所有权、路由、天气失败降级、结构化切分、文档幂等更新删除、Dense/BM25/RRF/Rerank、SSE 异常完成、MCP 协议、设备所有权、审批过期/拒绝/重复批准与重启恢复。

## 7. 复现命令

先完成数据库初始化和知识库导入，并在固定模型与参数下运行：

```bash
python -m cleanbot.cli init-db
python -m cleanbot.cli ingest

ENABLE_RERANK=true RERANK_POLICY=always python -m cleanbot.evaluation \
  --dataset evaluation/questions.jsonl \
  --answer-sample-size 0 \
  --json-output reports/evaluation/development_always.json \
  --markdown-output reports/evaluation/development_always.md

ENABLE_RERANK=true RERANK_POLICY=disagreement python -m cleanbot.evaluation \
  --dataset evaluation/questions.jsonl \
  --answer-sample-size 0 \
  --json-output reports/evaluation/development_disagreement.json \
  --markdown-output reports/evaluation/development_disagreement.md

ENABLE_RERANK=true RERANK_POLICY=disagreement python -m cleanbot.evaluation \
  --dataset evaluation/heldout.jsonl \
  --answer-sample-size 0 \
  --json-output reports/evaluation/heldout_final.json \
  --markdown-output reports/evaluation/heldout_final.md
```

JSON 输出包含逐题结果和模型响应，默认被 `.gitignore` 排除；公开仓库只保留不含完整回答内容的正式 Markdown 汇总。

## 8. 结论与限制

- Hybrid RAG 的主要收益是提高相关证据进入前列的概率，不代表生成答案必然正确。
- 条件 Rerank 在验证集上减少了 52% 的外部重排调用，并保持或改善检索指标，因此作为当前默认策略。
- 冻结测试集暴露了泛化不足和延迟成本，后续应优先扩充自然口语、错别字和隐式表达样本，而不是根据现有冻结题逐题修改。
- 外部 API 的单次延迟不能用作生产 SLA；若评估吞吐，需要在固定网络、并发、模型配额和多次重复实验下重新测量。
- 当前评测没有人工专家大规模审核、跨供应商 Judge 或在线用户反馈，不能声称生产级答案质量。
