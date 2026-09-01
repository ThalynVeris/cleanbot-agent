# CleanBot 系统架构设计

本文档描述当前仓库中已经实现并可通过 Docker Compose 复现的架构。系统采用单 API 实例、独立设备 MCP 服务、PostgreSQL 和本地持久化 Chroma；图中不包含尚未实现的基础设施。

## 1. 系统上下文

```mermaid
flowchart TB
    User["演示用户"]
    Admin["知识库管理员"]
    CleanBot["CleanBot 智能设备客服"]
    Model["Alibaba Cloud Model Studio<br/>Chat / Embedding / Rerank"]
    Weather["Open-Meteo<br/>地理编码与实时天气"]

    User -->|知识问答、月报、环境建议、设备操作| CleanBot
    CleanBot -->|SSE 回答、引用、审批卡片| User
    Admin -->|管理员令牌保护的文档上传与删除| CleanBot
    CleanBot -->|模型推理与检索模型调用| Model
    CleanBot -->|异步 HTTP 请求| Weather
```

系统边界：用户与设备均为演示身份和模拟数据；管理员接口使用共享令牌；模型和天气为外部服务。

## 2. 容器与部署

```mermaid
flowchart LR
    Browser["Browser"]

    subgraph Compose["Docker Compose"]
        Web["web<br/>Streamlit :8501"]
        API["api<br/>FastAPI :8000"]
        MCP["device-mcp<br/>MCP Streamable HTTP :8001"]
        PG[("postgres<br/>PostgreSQL 16")]
        CV[("chroma_data<br/>Chroma 持久化目录")]
        CP[("checkpoint_data<br/>SQLite Checkpointer")]
    end

    Model["Model Studio API"]
    Weather["Open-Meteo API"]
    Docs["只读领域文档目录"]

    Browser -->|HTTP| Web
    Web -->|HTTP + SSE| API
    API --> PG
    API --> CV
    API --> CP
    API -->|MCP Streamable HTTP| MCP
    MCP --> PG
    Docs --> API
    API --> Model
    API --> Weather
```

| 容器 | 职责 | 持久化/依赖 |
|---|---|---|
| `web` | 用户、会话选择；流式内容、来源和审批卡片展示 | 无业务状态，依赖 API |
| `api` | API 契约、Agent 编排、RAG、天气与审批恢复 | PostgreSQL、Chroma 卷、Checkpoint 卷 |
| `device-mcp` | 暴露设备 Tool/Resource，校验调用凭证和批准状态 | PostgreSQL |
| `postgres` | 用户、会话、消息、月报、设备、操作审计与文档登记 | `postgres_data` 卷 |

Chroma 当前作为 API 容器内的嵌入式存储使用 `chroma_data` 卷，并不是独立 Chroma Server。设备审批图状态存入独立 SQLite Checkpoint 卷。

## 3. 应用组件与 LangGraph 工作流

```mermaid
flowchart TB
    Endpoint["FastAPI chat/stream"] --> Agent["AgentService"]
    Agent -->|保存用户消息| DB[("Database")]
    Agent --> Graph["CleanBotGraph"]
    Graph --> Load["load_context"]
    Load --> Classify["classify"]

    Classify --> Knowledge["knowledge"]
    Classify --> Report["report"]
    Classify --> Environment["environment"]
    Classify --> Device["device"]
    Classify --> Smalltalk["smalltalk"]
    Classify --> OOS["out_of_scope"]

    Knowledge --> Retriever["HybridRetriever"]
    Retriever --> Dense["Dense Top-K"]
    Retriever --> BM25["中文 BM25 Top-K"]
    Dense --> RRF["RRF Fusion"]
    BM25 --> RRF
    RRF --> Policy{"Top-1 是否一致"}
    Policy -->|不一致| Rerank["qwen3-rerank"]
    Policy -->|一致| Evidence["证据阈值"]
    Rerank --> Evidence

    Report --> DB
    Report --> Retriever
    Environment --> Weather["WeatherClient"]
    Environment --> Retriever
    Device --> Control["DeviceControlService"]
    Control --> Approval["DeviceApprovalWorkflow"]
    Approval --> MCPClient["DeviceMCPClient"]
    MCPClient --> MCPServer["Device MCP Service"]

    Evidence --> Prompt["带编号来源的回答 Prompt"]
    Report --> Prompt
    Environment --> Prompt
    Prompt --> Model["Qwen Chat Model"]
    Smalltalk --> Direct["固定边界回答"]
    OOS --> Direct
    Model --> Agent
    Direct --> Agent
    Agent -->|保存助手消息与来源| DB
    Agent -->|status/source/token/done/error| Endpoint
```

六类意图均由显式节点结束，不使用无限制的通用工具循环。能力咨询和明确设备短语优先走确定性规则；模糊表达才调用结构化模型分类，并在低置信度或异常时返回受控结果。

## 4. 知识问答时序

```mermaid
sequenceDiagram
    actor U as 用户
    participant W as Streamlit
    participant A as FastAPI / AgentService
    participant D as Database
    participant G as LangGraph
    participant R as HybridRetriever
    participant C as Chroma
    participant RR as Rerank API
    participant M as Chat Model

    U->>W: 提交问题
    W->>A: POST /api/v1/chat/stream
    A->>D: 校验会话所有权并保存用户消息
    A-->>W: status(routing)
    A->>G: 执行状态图
    G->>D: 加载最近会话消息
    G->>M: 短追问时改写为独立检索问题
    G->>R: retrieve(query)
    par Dense 召回
        R->>C: similarity search
    and BM25 召回
        R->>C: 读取结构化切片缓存
        R->>R: 中文 unigram/bigram BM25
    end
    R->>R: RRF 名次融合
    alt Dense 与 BM25 第一名不一致
        R->>RR: 对候选集重排
        RR-->>R: relevance scores
    else 第一名一致或 Rerank 失败
        R->>R: 保留 RRF 排序
    end
    R-->>G: 结构化 KnowledgeHit 列表
    alt 证据不足
        G-->>A: 明确拒答/请求补充信息
    else 证据充足
        G->>M: 参考资料 + 引用约束
        M-->>A: 流式 Token
    end
    A-->>W: source / token / done
    A->>D: 保存回答、来源与时间
    W-->>U: 展示回答与来源卡片
```

失败原则：查询改写失败时使用用户原问题；Rerank 失败时回退到 RRF；知识证据不足时不调用模型补写事实。

## 5. 设备写操作审批时序

```mermaid
sequenceDiagram
    actor U as 用户
    participant W as Streamlit
    participant A as FastAPI
    participant G as LangGraph
    participant DC as DeviceControl
    participant DB as PostgreSQL
    participant CP as SQLite Checkpointer
    participant MCP as Device MCP

    U->>W: 开始清扫
    W->>A: chat/stream
    A->>G: device 分支
    G->>DC: prepare(session, user, request)
    DC->>DB: 校验设备所有权与当前待审批操作
    DC->>DB: 创建 pending DeviceAction 与幂等键
    DC->>CP: 保存图状态并 interrupt
    DC-->>A: approval_required + action_id
    A-->>W: SSE approval_required
    W-->>U: 展示批准/拒绝卡片

    U->>W: 点击批准
    W->>A: POST /device/actions/{id}/decision
    A->>DC: decide(action_id, user, session, approve)
    DC->>DB: 校验所有权、状态和 30 分钟有效期
    DC->>CP: Command(resume=approve)
    DC->>DB: 标记 approved
    DC->>MCP: start_cleaning(action_id, approved credential)
    MCP->>DB: 再次校验批准记录与幂等状态
    MCP->>DB: 更新设备和审计结果
    MCP-->>DC: DeviceActionResult
    DC-->>A: 已执行或已有结果
    A-->>W: decision response
    W-->>U: 展示设备新状态
```

拒绝不会调用 MCP 写工具；重复批准直接返回已有结果；过期、其他用户、其他会话或缺少有效批准的写操作均被拒绝。

## 6. 数据模型

```mermaid
erDiagram
    USER ||--o{ CHAT_SESSION : owns
    CHAT_SESSION ||--o{ MESSAGE : contains
    USER ||--o{ DEVICE_MONTHLY_RECORD : has
    USER ||--o| DEVICE : owns
    USER ||--o{ DEVICE_ACTION : requests
    CHAT_SESSION ||--o{ DEVICE_ACTION : contains
    DEVICE ||--o{ DEVICE_ACTION : executes

    USER {
        string id PK
        string display_name
        string city
        datetime created_at
    }
    CHAT_SESSION {
        string id PK
        string user_id FK
        datetime created_at
        datetime updated_at
    }
    MESSAGE {
        int id PK
        string session_id FK
        string role
        text content
        text sources_json
        datetime created_at
    }
    DEVICE_MONTHLY_RECORD {
        int id PK
        string user_id FK
        string month
        text features
        text efficiency
        text consumables
        text comparison
    }
    DEVICE {
        string id PK
        string user_id FK
        string model
        string status
        int battery_percent
        int consumable_percent
    }
    DEVICE_ACTION {
        string id PK
        string user_id FK
        string device_id FK
        string session_id FK
        string action
        string idempotency_key UK
        string checkpoint_thread_id
        string status
        datetime approval_expires_at
        string error_type
    }
    KNOWLEDGE_DOCUMENT {
        string id PK
        string filename UK
        string content_hash
        int chunk_count
        string status
    }
```

`KnowledgeDocument` 记录文档生命周期；向量切片及其 `document_id/chunk_id/source/page/section/content_hash` 元数据保存在 Chroma 中，因此未与关系表建立数据库外键。

## 7. 工程保障与边界

| 关注点 | 当前机制 |
|---|---|
| 会话隔离 | `session_id` 首次绑定用户；其他用户复用时抛出所有权错误 |
| 持久化 | PostgreSQL 保存业务状态，Chroma 卷保存向量，SQLite Checkpointer 保存审批图状态 |
| 幂等 | 文档内容哈希避免重复索引；设备操作使用唯一幂等键和已有结果回放 |
| 超时与降级 | 天气、模型和 MCP 配置超时；Rerank 回退 RRF；失败不返回伪造天气或设备结果 |
| API 边界 | Pydantic 校验输入输出；管理员令牌保护知识库写接口；MCP 使用独立共享令牌 |
| 可观测性 | 请求 ID、意图、来源数、首 Token、总耗时、模型调用、Token 用量和错误类型 |
| 自动化质量 | CI 使用 Python 3.10，执行 Ruff、全量 pytest，并设置 80% 覆盖率门槛 |

当前原型不提供企业 OAuth、真实设备云接入、多 API 副本、分布式向量数据库或生产级高可用。若扩展为多副本部署，需先将 SQLite Checkpointer 和嵌入式 Chroma 替换为共享服务，并完善鉴权、限流、密钥管理与监控告警。
