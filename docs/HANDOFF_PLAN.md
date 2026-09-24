# CleanBot 拓展计划交接文档

> 用途：交接给接手的 AI 助手，说明项目现状、已完成工作、已知问题和后续路线。教学方式由接手方自行决定。

## 1. 背景与目标

- **学习者目标**：以 SAP 类企业的 AI/LLM 应用开发岗位为求职目标（先申请第二段实习）；目前正在第一段实习中，时间充裕。
- **学习者基础**：能读懂本仓库全部代码，但独立手写能力一般。更看重懂原理、读懂 AI 写的代码、用 AI 修改代码的能力。
- **项目目标**：把 CleanBot 从"单实例可评测原型"推进为"有数据支撑、对接 SAP 生态的企业售后 Agent 平台"，并产出简历量化点与面试素材。
- **工作分支**：`claude/tender-allen-86vhds`。

## 2. 项目现状速览

CleanBot 是扫地/扫拖机器人智能客服 Agent，核心代码约 3,900 行 Python，另有 72 项测试，覆盖率约 87%。

| 层 | 技术 | 位置 |
|---|---|---|
| 编排 | LangGraph 状态图；6 个意图节点（knowledge/report/environment/device/smalltalk/out_of_scope） | `cleanbot/workflow/graph.py`、`router.py` |
| 人工审批 | LangGraph `interrupt()` + `Command(resume)`，SQLite Checkpointer | `workflow/device_approval.py`、`device_control.py` |
| 模型 | 阿里云百炼：`qwen3-max`（经 langchain-openai 兼容接口调用）、`text-embedding-v4`、`qwen3-rerank` | `core/models.py` |
| RAG | Chroma 嵌入式向量库 + BM25（中文单字/双字切词）+ RRF 融合 + 条件 Rerank | `cleanbot/rag/` |
| 工具协议 | 独立 Device MCP 服务（Streamable HTTP），5 个 Tool + 1 个 Resource | `cleanbot/device_mcp/` |
| API / 前端 | FastAPI + SSE + Pydantic v2；Streamlit | `api/app.py`、`app.py` |
| 数据 | SQLAlchemy 2.0；Docker 下用 PostgreSQL 16，本地用 SQLite | `cleanbot/db/` |
| 工程 | Docker Compose（web/api/device-mcp/postgres）、GitHub Actions（Ruff + pytest，覆盖率门槛 80%） | — |
| 评测 | 60 条验证集、20 条冻结集；冻结集 Hit@3 = 81.25%（未达 85% 目标） | `evaluation/`、`docs/EVALUATION.md` |

详细架构见 `docs/ARCHITECTURE.md`、`docs/EVALUATION.md`。

## 3. 已完成的工作（已推送）

| 提交 | 内容 |
|---|---|
| M0.1 修复 | `RERANK_POLICY` 默认值由 `always` 改为评测选定的 `disagreement`（`cleanbot/core/config.py`、`.env.example`），新增回归测试 `test_rerank_policy_defaults_to_evaluated_disagreement_strategy`（`tests/test_config_and_database.py`） |
| 路线图 | `docs/learning/README.md`：JD 证据、技术栈优先级、模块单元进度勾选表 |
| 练习区 | `playground/`：M0.2 手敲练习存根 `m0_2_retrieval.py`（`tokenize_for_bm25`、`fuse_rrf`）+ 对比测试 `test_m0_2_retrieval.py`。存根尚未由学习者完成，因此该测试当前失败属预期；`playground/` 不在 CI 的 ruff 和覆盖率范围内 |

**进度**：M0.1 完成；M0.2（检索链路）已讲解过一遍，学习者评价效果不佳，建议接手方重新讲授；M0.3 未开始。

## 4. 已发现、尚未修复的问题（可作为后续单元素材）

1. **拒答阈值在 RRF 路径上失效**（`cleanbot/rag/retriever.py`）：不调用 Rerank 时，分数是 `fusion_score / max_fusion`。在 Top-20+20、k=60 的设置下，最低值约为 (1/80)/(2/61) ≈ 0.38，所以 `MIN_RETRIEVAL_SCORE=0.10` 永远过滤不掉结果。只要知识库非空，知识问答总会输出 4 条"证据"，"证据不足就拒答"几乎不会触发。计划在 M1.3 改为绝对分数阈值（原始向量相似度或 Rerank 分数）。
2. **BM25 缓存失效判断只比较切片数量**（`_sparse_index`）：API 的上传和删除会主动调用 `invalidate()`，但另一个进程（如 CLI 的 `ingest`）改动向量库且总块数不变时，缓存会过期却检测不到。
3. **`_as_bool` 静默吞掉非法值**（`core/config.py`）：若改为严格校验，需注意 Docker Compose 传入空字符串 `""` 的情况，应回退到默认值而非报错。
4. **路由和设备操作的短语表重复**：`workflow/router.py` 与 `workflow/device_control.py` 各维护一套（计划在 M1.1 合并）。
5. 追问改写靠正则触发（`graph._rewrite_query`）；演示身份无真实鉴权；审批状态存 SQLite、向量库是嵌入式 Chroma，导致只能单实例部署。

## 5. 招聘要求依据（2026 年 SAP 公开 JD 摘要）

| 岗位 | 关键要求 |
|---|---|
| SAP China iXp AI Engineer Intern（上海） | Python；RAG + **知识图谱**；LangChain/LlamaIndex、PyTorch/HF；FastAPI/REST、Git；加分：Agentic AI 项目、能独立部署 AI 功能 |
| SAP STAR（上海/成都/大连/北京） | CS 相关专业、英语沟通、把 AI 用于企业软件场景的想法 |
| SAP Junior AI Engineer | Python + **TypeScript/Node**；工具调用、结构化输出、Agent 控制循环；技能路由、HITL、状态机；任一 Agent 框架（LangGraph 等）；容器 + K8s 基础（BTP/Kyma）；BTP/CAP；幻觉、工具误用、失控循环、信任边界；**AI 系统评测设计** |
| SAP AI Developer | LLM、Embedding、向量库、RAG、**MCP**、Agentic AI；Docker/K8s/CI/CD；SAP AI Core、Joule 加分 |
| Frontend & Agent Engineer（Joule Work Desktop） | **TypeScript + React**；Agent Harness：技能选择、工具使用、MCP 连接、安全 |
| Junior AI Quality Engineer | AI 质量与评测专岗 |

SAP 官方样例 [`SAP-samples/joule-a2a-agent-toolkit`](https://github.com/SAP-samples/joule-a2a-agent-toolkit) 通过 **A2A** 把代码实现的 Agent 接入 Joule，部署在 BTP Cloud Foundry，支持 TypeScript（Express/CAP）和 Python（LangGraph + SAP GenAI Hub）。这是 M4–M6 的对标形态。

**已与学习者达成的结论**：

- Agent 框架可以互换，面试考的是概念（工具调用、结构化输出、控制循环、HITL、状态机、评测、故障模式）。LangGraph 是 JD 点名最多、SAP 样例也在用的框架，保留不换。
- SAP 平台经验（BTP、CAP、GenAI Hub、Joule/A2A）是差异化加分项。
- TypeScript 有明确需求（CAP、A2A Agent、React 聊天界面）。
- 技术选型以招聘要求为准，不必顾虑复用现有代码的成本。

## 6. 技术栈优先级

| 优先级 | 技术 |
|---|---|
| **P0 必须** | Python 异步、FastAPI、LangGraph（状态机、interrupt、子图、Supervisor）、结构化输出/Function Calling、RAG、MCP、Agent 评测、Docker、Git/CI |
| **P1 强加分** | A2A、SAP Generative AI Hub / SAP Cloud SDK for AI、BTP Cloud Foundry、CAP（TypeScript）+ OData v4、TypeScript/Node、知识图谱/GraphRAG、Kubernetes 基础、Agent 安全护栏 |
| **P2 加分** | React + TypeScript（Next.js）、Fiori Elements、HANA Cloud Vector Engine、OpenTelemetry/Langfuse、OAuth2/JWT |
| 暂缓 | 模型微调、Redis 缓存、Grafana、Keycloak |

## 7. 模块与单元路线（按顺序推进）

### M0 吃透现有代码（P0）
- [x] M0.1 修复 Rerank 默认策略；建立路线图
- [ ] M0.2 检索链路：Dense、BM25、RRF、条件 Rerank（`cleanbot/rag/*`；核心：`fuse_rrf`、`tokenize_for_bm25`，练习见 `playground/`）
- [ ] M0.3 编排链路：LangGraph 状态图、interrupt/resume 审批、SSE（`workflow/*`、`api/app.py`）

### M1 Agent 设计模式硬化（P0）
- [ ] M1.1 结构化输出路由：合并意图与设备操作识别，规则降为兜底
- [ ] M1.2 技能路由：六类意图抽象为可注册 Skill（描述 + 输入 Schema + 处理节点）
- [ ] M1.3 故障模式防护：失控循环（最大步数、超时）、工具误用（Schema 校验、白名单）、信任边界、幻觉（引用校验器 + 修复第 4 节问题 1）
- [ ] M1.4 多轮澄清与长期记忆（LangGraph Store）

### M2 AI 评测工程（P0）
- [ ] M2.1 验证集扩充（口语、错别字、隐式表达），按类别出指标
- [ ] M2.2 回答级评测：RAGAS/DeepEval + 非同源 Judge
- [ ] M2.3 Agent 轨迹评测（工具调用序列、审批是否被绕过）
- [ ] M2.4 红队集（Prompt 注入、越权、诱导工具误用）
- [ ] M2.5 CI 评测门禁

### M3 RAG + 知识图谱（P0/P1）
- [ ] M3.1 查询归一化 + jieba 领域分词 + 多查询/HyDE
- [ ] M3.2 父子块、FAQ 索引、切片元数据
- [ ] M3.3 故障知识图谱：抽取"部件—现象—原因—处理步骤"（Neo4j 或 PostgreSQL）
- [ ] M3.4 GraphRAG，与纯向量方案做评测对比
- [ ] M3.5 冻结集里程碑（目标 Hit@3 ≥ 85%，只跑一次，如实记录）

### M4 多智能体 + MCP + A2A（P1）
- [ ] M4.1 故障诊断 Supervisor 子图：知识/图谱 Agent、设备状态 Agent（复用 `DeviceMCPClient`）、历史 Agent（复用 `Database.get_device_report`）；写操作仍走现有审批流；新增 `Intent.DIAGNOSIS`
- [ ] M4.2 单 Agent vs 多智能体评测（准确率、Token、延迟）
- [ ] M4.3 MCP 加固：鉴权、工具权限分级、审计
- [ ] M4.4 A2A Server（Agent Card、任务生命周期、流式），参照 joule-a2a-agent-toolkit
- [ ] M4.5 A2A Client："售后工单 Agent"调用 CleanBot

### M5 TypeScript + CAP（P1）
- [ ] M5.1 TypeScript/Node 基础（类型、泛型、async、Zod）
- [ ] M5.2 CAP（TypeScript）售后服务单/备件申请服务：CDS、OData v4、单测
- [ ] M5.3 CleanBot 新增 MCP 工具，调用 CAP 创建服务单（走审批）：对话 → 诊断 → 审批 → 服务单
- [ ] M5.4 Fiori Elements 服务单管理界面
- [ ] M5.5（可选）TypeScript A2A Agent

### M6 SAP AI 平台与 BTP（P1）
- [ ] M6.1 模型层多 provider 抽象（DashScope / SAP GenAI Hub）
- [ ] M6.2 SAP Cloud SDK for AI 接入 Generative AI Hub（先核实 BTP 试用账号是否提供 AI Core）
- [ ] M6.3 CAP 服务与 A2A Agent 部署到 BTP Cloud Foundry（试用账号）
- [ ] M6.4 视权限在 Joule 注册 A2A Agent；不可用时做本地演示并写集成文档
- [ ] M6.5（可选）HANA Cloud Vector Engine 与 pgvector 对比

### M7 云原生与生产化（P0/P1）
- [ ] M7.1 PostgreSQL Checkpointer + pgvector + Alembic
- [ ] M7.2 Docker 多阶段构建；kind + Helm 多副本，验证审批跨实例恢复
- [ ] M7.3 GitHub Actions CD + 评测门禁
- [ ] M7.4 OAuth2/JWT + RBAC（对照 BTP XSUAA 概念）
- [ ] M7.5 OpenTelemetry + Langfuse

### M8 React 前端（P2）
- [ ] M8.1 React + TS；自定义 hook `useChatStream`（解析 SSE）
- [ ] M8.2 Next.js 聊天页：流式消息、来源卡片、审批卡片、诊断步骤可视化
- [ ] M8.3 OpenAPI → TS 类型生成；评测结果管理页
- [ ] M8.4 Playwright E2E（问答 → 审批 → 服务单）

### M9 求职交付
- [ ] M9.1 README 重写为"企业售后 Agent 平台"叙事，架构图含 A2A/Joule、CAP、BTP；ADR
- [ ] M9.2 简历 bullet 与 JD 关键词对齐
- [ ] M9.3 模拟面试：项目深挖、Agent 故障模式、系统设计（SAP 式跨系统售后/采购 Agent）、AI 代码审查、英文项目介绍

## 8. 环境与验证

- CI 用 Python 3.10。当前容器是 Python 3.11，系统 pip 与 Debian 自带的 PyYAML/PyJWT 冲突，需使用虚拟环境：

  ```bash
  python -m venv .venv
  .venv/bin/pip install -r requirements.lock
  .venv/bin/pip install --no-deps -e .
  ```

  `.venv/` 已被 `.gitignore` 忽略。

- 每个单元必须通过：

  ```bash
  python -m ruff check cleanbot tests app.py
  python -m pytest --cov=cleanbot --cov-fail-under=80
  ```

- 涉及检索、路由、诊断的单元，额外运行评测（需要 `DASHSCOPE_API_KEY`）：

  ```bash
  ENABLE_RERANK=true RERANK_POLICY=disagreement python -m cleanbot.evaluation \
    --dataset evaluation/questions.jsonl --answer-sample-size 0 \
    --markdown-output reports/evaluation/<unit>.md
  ```

- 冻结集 `evaluation/heldout.jsonl` 只在里程碑各跑一次，结果如实记录，不按冻结题逐题调参。
- 端到端最终验收：对话 → 诊断多智能体 → 审批 → MCP → CAP OData 服务单；外部 A2A Client 能调用 CleanBot。
