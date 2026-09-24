# CleanBot × SAP 求职学习路线

本目录记录围绕 CleanBot 的「边学边做」路线与进度。讲解、习题与答案都在与 Claude 的对话中进行；仓库只保留项目代码、测试和 `playground/` 手敲练习。

## 1. 招聘要求依据（2026 年 SAP 公开 JD 摘要）

| 岗位 | 关键要求 |
|---|---|
| SAP China iXp AI Engineer Intern（上海） | Python；RAG 与**知识图谱**基础；LangChain/LlamaIndex、PyTorch/Hugging Face；FastAPI/REST、Git；加分：RAG/Agentic AI 项目、能独立部署 AI 功能 |
| SAP STAR（上海/成都/大连/北京） | CS 相关专业、英语沟通、把 AI 用于企业软件场景的想法；项目涉及 ERP、云原生与企业 AI |
| SAP Junior AI Engineer | Python + **TypeScript/Node**；工具调用、结构化输出、Agent 控制循环；技能路由、Human-in-the-loop、状态机；任一 Agent 框架（LangGraph 等）；容器与 K8s 基础（BTP/Kyma）；BTP/CAP；幻觉、工具误用、失控循环、信任边界；**AI 系统评测设计** |
| SAP AI Developer | LLM、Embedding、向量库、RAG、**MCP**、Agentic AI；REST/微服务/Docker/K8s/CI/CD；SAP AI Core、Joule 加分 |
| Frontend & Agent Engineer（Joule Work Desktop） | **TypeScript + React**；Agent Harness 的技能选择、工具使用、MCP 连接与安全 |
| Junior AI Quality Engineer | AI 质量与评测专岗 |

SAP 官方样例 [`SAP-samples/joule-a2a-agent-toolkit`](https://github.com/SAP-samples/joule-a2a-agent-toolkit) 通过 **A2A** 把代码实现的 Agent 接入 Joule，部署在 BTP Cloud Foundry，支持 TypeScript（Express/CAP）与 Python（LangGraph + SAP GenAI Hub）——这是本项目 M4–M6 的对标形态。

**结论**：Agent 框架可以互换，面试考的是概念（工具调用、结构化输出、控制循环、HITL、状态机、评测、故障模式）；LangGraph 是点名最多的框架，保留。SAP 平台经验（BTP、CAP、GenAI Hub、Joule/A2A）是差异化加分项；TypeScript 有明确需求。

## 2. 技术栈优先级

| 优先级 | 技术 |
|---|---|
| **P0 必须** | Python 异步、FastAPI、LangGraph（状态机、interrupt、子图、Supervisor）、结构化输出、RAG、MCP、Agent 评测、Docker、Git/CI |
| **P1 强加分** | A2A、SAP Generative AI Hub / SAP Cloud SDK for AI、BTP Cloud Foundry、CAP（TypeScript）+ OData、TypeScript/Node、知识图谱 / GraphRAG、Kubernetes 基础、Agent 安全护栏 |
| **P2 加分** | React + TypeScript、Fiori Elements、HANA Cloud Vector Engine、OpenTelemetry/Langfuse、OAuth2/JWT |
| 暂缓 | 模型微调、Redis 缓存、Grafana、Keycloak |

## 3. 代码标注规则

| 标注 | 做法 |
|---|---|
| 🟢 复制 | 粘贴运行，能一句话讲清作用 |
| 🟡 精读 | 逐行读懂；回答对话中的「为什么」（附答案）；用 AI 改一处并按清单审查 diff |
| 🔴 对照手敲 | 面试高频核心 20–60 行，对着答案抄一遍并写注释，跑通测试 |

**AI diff 审查清单**：① 是否只改了需求相关的地方；② 是否符合周围代码风格；③ 失败分支与边界值；④ 是否有测试且测试真的覆盖了改动；⑤ 是否引入新依赖、密钥或破坏接口。

## 4. 单元进度

每个单元交付：完整代码 + 测试（讲解、习题与答案在对话中给出）。完成标准：测试通过、🔴 已手敲。

### M0 吃透现有代码（P0）
- [x] M0.1 修复 Rerank 默认策略与文档不一致；建立学习体系
- [ ] M0.2 检索链路精读：Dense、BM25、RRF、条件 Rerank
- [ ] M0.3 编排链路精读：LangGraph 状态图、interrupt/resume、SSE

### M1 Agent 设计模式硬化（P0）
- [ ] M1.1 结构化输出路由（合并意图与设备操作识别）
- [ ] M1.2 技能路由（可注册 Skill）
- [ ] M1.3 故障模式防护：失控循环、工具误用、信任边界、幻觉
- [ ] M1.4 多轮澄清与长期记忆

### M2 AI 评测工程（P0）
- [ ] M2.1 验证集扩充与分类别指标
- [ ] M2.2 回答级评测（RAGAS/DeepEval，非同源 Judge）
- [ ] M2.3 Agent 轨迹评测
- [ ] M2.4 红队集
- [ ] M2.5 CI 评测门禁

### M3 RAG + 知识图谱（P0/P1）
- [ ] M3.1 查询归一化、jieba 领域分词、多查询/HyDE
- [ ] M3.2 父子块、FAQ 索引、切片元数据
- [ ] M3.3 故障知识图谱抽取
- [ ] M3.4 GraphRAG 与评测对比
- [ ] M3.5 冻结集里程碑（目标 Hit@3 ≥ 85%）

### M4 多智能体 + MCP + A2A（P1）
- [ ] M4.1 故障诊断 Supervisor 子图
- [ ] M4.2 单 Agent vs 多智能体评测
- [ ] M4.3 MCP 加固（鉴权、权限分级、审计）
- [ ] M4.4 A2A Server（Agent Card、任务生命周期）
- [ ] M4.5 A2A Client：售后工单 Agent

### M5 TypeScript + CAP（P1）
- [ ] M5.1 TypeScript/Node 速成
- [ ] M5.2 CAP 售后服务单服务（CDS、OData v4）
- [ ] M5.3 CleanBot 通过 MCP 创建服务单（带审批）
- [ ] M5.4 Fiori Elements 管理界面
- [ ] M5.5（可选）TypeScript A2A Agent

### M6 SAP AI 平台与 BTP（P1）
- [ ] M6.1 模型层多 provider 抽象
- [ ] M6.2 SAP Generative AI Hub 接入
- [ ] M6.3 BTP Cloud Foundry 部署
- [ ] M6.4 Joule 注册或 A2A 集成演示
- [ ] M6.5（可选）HANA Cloud Vector Engine

### M7 云原生与生产化（P0/P1）
- [ ] M7.1 PostgreSQL Checkpointer + pgvector + Alembic
- [ ] M7.2 Kubernetes（kind）+ Helm 多副本
- [ ] M7.3 GitHub Actions CD 与评测门禁
- [ ] M7.4 OAuth2/JWT + RBAC
- [ ] M7.5 OpenTelemetry + Langfuse

### M8 React 前端（P2）
- [ ] M8.1 React + TS 与 `useChatStream`
- [ ] M8.2 Next.js 聊天、来源、审批、诊断可视化
- [ ] M8.3 OpenAPI → TS 类型与评测管理页
- [ ] M8.4 Playwright E2E

### M9 求职交付
- [ ] M9.1 README 与架构叙事、ADR
- [ ] M9.2 简历与 JD 关键词对齐
- [ ] M9.3 模拟面试（中英文）
