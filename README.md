# SciCoreAgent

面向复杂科研研究任务的多智能体协作与可审计执行系统。

> 项目状态：科研工程原型。仓库已经实现并测试了多智能体编排、证据追踪、回放评测、上下文治理和三层记忆等核心链路；它仍需要真实用户数据验证、生产级安全加固和长期运行观测，不应被描述为“已经替代科研人员的全自动科学家”。

仓库名称为 **SciCoreAgent**。为保持已有安装和脚本兼容，当前 Python 包名及命令仍使用 **BiocoreagentV2.0 / biocoreagent-v2**；后续如更名会提供迁移期，不会静默破坏旧接口。

## 为什么做这个项目

科研 Agent 的难点不只是生成代码。复杂研究任务通常涉及脆弱的 Python/R 环境、大文件、长时间运行脚本、文献证据、多阶段依赖和失败恢复。如果只有一个自由工具循环，很容易发生重复读取、盲目重试、未验证却声称完成，或无法回答“这个结果由当时哪一版代码生成”。

SciCoreAgent 在模型与工具之间加入可执行治理层，使任务具备以下性质：

- **可编排**：Supervisor 将长任务拆成 DAG，由角色化子 Agent 并发执行。
- **可恢复**：状态机、checkpoint、动态 replan 和三级降级保留有效进展。
- **可审计**：Trace、代码快照、Git 现场、环境与产物血缘形成证据链。
- **可回放**：历史运行可抽取 replay case，由当前 runtime 生成 fresh trace 并评分。
- **可沉淀**：RAG、Skill、Wiki 和 DREAM 后台审查积累有证据的成功经验。
- **可控记忆**：会话态、工作记忆和长期记忆分层保存，可检索并回跳原文。
- **科研导向**：确定性分析路由、能力注册、受控 fallback 和结果 verifier 降低误报。

## 系统架构

~~~mermaid
flowchart TD
    U["用户任务"] --> CLI["CLI / Session Runtime"]
    CLI --> AR["科研任务路由与 Capability Registry"]
    CLI --> AL["ReAct AgentLoop"]
    AL --> TM["Tool Manager<br/>Schema · Policy · Approval · Workspace"]
    AL --> SUP["中心化 Supervisor"]
    SUP --> DAG["DAG Scheduler<br/>asyncio + Semaphore"]
    DAG --> W["Role Workers<br/>explorer · planner · executor · verifier · bio_worker"]
    W --> CI["Completion Inbox<br/>结果回注主会话"]

    AL --> CM["四层上下文压缩"]
    AL --> MEM["三层记忆"]
    MEM --> S1["SQLite 会话态"]
    MEM --> S2["本地 User / Project 工作记忆"]
    MEM --> S3["Postgres 蒸馏长期记忆<br/>Dense + BM25 + RRF"]

    AL --> KS["RAG + Skill + Wiki"]
    KS --> DR["DREAM 异步审查与版本迭代"]

    TM --> ES["Canonical Event Sink"]
    W --> ES
    ES --> TR["Trace + Evidence Index"]
    TM --> CAS["CAS 精确代码快照"]
    TM --> ART["Artifact Registry<br/>Deliverable Bundle"]
    TR --> RP["Replay Case"]
    RP --> FR["Current Runtime Fresh Run"]
    FR --> VF["Verifier + Scorecard"]
~~~

### 多智能体编排

- 中心化 Supervisor 管理父子会话隔离、异步执行、状态查询、日志、产物列表与结果回注。
- DAG 调度器使用 asyncio、Semaphore 和同步 Worker 线程桥接；依赖满足的节点并发执行。
- 9 个核心运行态描述正常生命周期，同时保留 cancelled 和 blocked 异常终态。
- 三级降级分别处理单 Worker 超时、批量失败触发动态 replan、全局预算耗尽后的证据合成。
- 线程 Worker 采用协作式超时隔离；需要硬终止时可使用进程 Worker。系统不声称能够强杀任意 Python 线程。

详细设计见 [多智能体设计](docs/multiagent_design.md) 和 [运行时状态机](docs/runtime_state_machine.md)。

### 评测、证据与回放

一次运行的核心证据链为：

~~~text
交付物
  → 精确生成代码快照（SHA-256 内容寻址对象）
  → 输入文件哈希与参数
  → 执行命令、工作目录和退出状态
  → Python / R / 系统环境
  → Trace 事件区间
  → Git commit + dirty patch
~~~

Canonical Trace 记录脱敏后的用户输入、模型/工具事件、文件影响、审批、异常恢复、产物和唯一终态。重要产物通过 Deliverable Bundle 绑定 lineage、environment、code manifest 与 verifier 结果。历史 trace 只能抽取 draft case；fixture 和 verifier 明确后才能作为 ready case重新驱动当前 runtime。

详细设计见 [审计系统](docs/audit_system.md) 和 [Trace / Replay 架构](docs/trace_replay_architecture.md)。

### 上下文压缩与记忆

上下文治理依次执行预算截断、冗余读取裁剪、旧工具结果结构化精缩、全局阈值自适应裁剪，并将压缩行为写入 prompt metadata 和 trace。

三层记忆分别承担：

1. SQLite 会话态：消息、不可变 chunk、蒸馏 outbox 与 JSON 人类可读镜像。
2. 本地 User / Project 工作记忆：独立 SQLite 存储，并镜像 Markdown。
3. 可选 Postgres 长期记忆：保存蒸馏事实、作用域和证据；使用 SentenceTransformer Dense、BM25 与 RRF 混合召回，并通过 session://...#sha256 回跳原消息。

未配置 Postgres 时，系统回退到本地 SQLite FTS5/BM25 + Dense Hash。当前 Postgres embedding 保存为 JSONB，在作用域过滤后的候选集上计算精确余弦；若所用镜像含 `vector` 扩展（如 `pgvector/pgvector:pg15`），`initialize()` 会幂等启用 `embedding vector(256)` + HNSW 并在检索时用 `embedding <=> query` 逼近 dense——**该 pgvector ANN 为可选项，默认仍是 JSONB，且未在仓库 CI 内验证（无 DSN 时测试跳过）**。

语义嵌入为**双档可选**：默认是确定性、离线、可逐字节复现的哈希嵌入（`dense_hash_fallback`，见 [pico/features/postgres_memory.py](pico/features/postgres_memory.py)）；只有安装 `memory-postgres` 附加包并设置 `BIOCOREAGENT_EMBEDDING_BACKEND=sentence-transformers`（见 [`.env.example`](.env.example)）时才启用真实 SentenceTransformer 语义向量，否则检索永远可本地复现，不依赖外部模型或网络。

详细设计见 [记忆架构](docs/memory_architecture.md)。

### 知识与技能沉淀

- 用户显式请求可同步保存 Skill / Wiki。
- DREAM 后台流程按 Detect、Review、Extract、Apply、Monitor 五阶段持久化执行。
- 只有存在工具证据的成功任务才参与学习；同类观察累计达到阈值后生成带 draft 标记的 Skill。
- 更新前归档旧版本，并在 Wiki 中保留来源证据。自动沉淀用于形成候选知识，不能替代人工科学审查。

详细设计见 [知识沉淀](docs/knowledge_sedimentation.md)。

## 代码架构

本项目是**一个 harness，三个 Python 包**：`biocoreagent/` 负责把 `pico/` 的通用 agent 内核与 `corecoder/` 的科研/生信域能力拼装成 CLI、REPL 与 benchmark 入口。控制平面（Pico）与域平面（CoreCoder）在 `BioPico` 处合流，领域工具只能经由 Pico 的单一工具闸口执行，无法绕过它另行执行。

~~~text
用户 / CLI
    |
    v
biocoreagent/                   拼装与审计层
  ├── cli / __main__            主入口（biocoreagent-v2）
  ├── runtime.py                BioPico = Pico + 领域/编排能力
  ├── orchestrator.py           AsyncMultiAgentOrchestrator
  ├── audit / evidence          AuditTrail + SHA-256 内容寻址证据
  ├── replay / replay_cli       回放、评分卡、source 重建
  └── workflow_ir / capabilities / domain
    |
    +-- pico/                   通用 agent 内核（控制平面）
    |     AgentLoop · ContextManager（四层压缩）· ToolExecutor（单一工具闸口）
    |     LayeredMemory（三层记忆）· TaskState/Checkpoint · RunStore · 模型适配
    |
    +-- corecoder/              科研/生信域（域平面）
          transcriptome（能力契约/注册表）· tools/bio（生信工具链）
          knowledge_base（RAG+Skill+Wiki）· mcp_client · skill_router
          eval/bixbench（benchmark 适配 + 确定性解析器）
~~~

### 包结构与职责

| 包 | 关键模块 | 职责 |
|---|---|---|
| `pico/` | `agent_loop.py`、`runtime.py`、`context_manager.py`、`tool_executor.py`、`task_state.py`、`checkpoint.py`、`session_store.py`、`run_store.py`、`features/` | 可复用通用 agent 内核：驱动「感知→决策→行动→记录」主循环、按预算组 prompt 并做四层上下文压缩、提供统一工具执行闸口（校验/审批/重复检测/workspace diff）、`ask()` 生命周期状态机与断点恢复、SQLite 会话态与单次运行审计工件落盘、三层记忆（会话/本地/Postgres）与科研知识图谱、模型后端适配与确定性 FakeModel |
| `corecoder/` | `tools/bio/`、`transcriptome/`、`knowledge_base.py`、`skills.py`、`wiki.py`、`bio/rag_store.py`、`sedimentation.py`、`skill_router.py`、`mcp_client.py`、`policy.py`、`remote.py`、`eval/` | 科研/生信域扩展：生信工具链（协议/证据/DESeq2/R桥/count matrix/workflow）、转录组能力契约与注册表 `CAPABILITIES`、统一 RAG+Skill+Wiki 检索、DREAM 后台审查沉淀、三层技能路由、外部 MCP 客户端、命令/工作区安全策略、BixBench 适配与确定性回答解析器 |
| `biocoreagent/` | `cli.py`、`runtime.py`、`orchestrator.py`、`audit.py`、`evidence.py`、`replay.py`、`workflow_ir.py`、`capabilities.py`、`domain.py`、`analysis_router.py` | 拼装与审计层：`BioPico` 注入角色工具策略/审计/证据/审批/领域编排，`AsyncMultiAgentOrchestrator` 管理多 Agent 生命周期，`AuditTrail` 追加式 JSONL + 脱敏，`EvidenceManager`/内容寻址存储形成证据链，回放与评分卡，工作流 IR 与能力注册表，领域/MCP 适配，分析路由/答案校验/降级/导出 |

### 命令行入口

| 命令 | 入口模块 | 用途 |
|---|---|---|
| `biocoreagent-v2` / `biocoreagent` | `biocoreagent.cli:main` | 交互式 / 一次性科研任务主入口 |
| `biocore-replay` | `biocoreagent.replay_cli:main` | 解释产物、回放、重建源代码、评审 case |
| `biocore-bixbench` | `corecoder.eval.bixbench:main` | BixBench benchmark 运行与评分 |
| `biocore-mcp` | `corecoder.mcp_server:main` | 把内置工具外露为 stdio MCP server |
| `biocore-doctor` | `corecoder.bio_cli:main` | 文献综述/协议采集/系统状态诊断 |

### 运行态目录与关键文档

- 运行态默认落在 `.biocoreagent/`（审计/证据/多 Agent/回放）与 `.pico/`（记忆/会话），两者均被 Git 忽略。
- 模块级细节与固化边界：[架构](docs/architecture.md) · [审计系统](docs/audit_system.md) · [Trace/Replay](docs/trace_replay_architecture.md) · [多智能体设计](docs/multiagent_design.md) · [记忆架构](docs/memory_architecture.md) · [知识沉淀](docs/knowledge_sedimentation.md) · [运行时状态机](docs/runtime_state_machine.md)

## 已验证结果与边界

下表是仓库当前保存的工程资格测试结果，不是未经限定的生产效果宣传。

| 项目 | 结果 | 测试边界与证据 |
|---|---:|---|
| 当前全量测试（非 slow） | 301 passed，12 skipped，1 deselected | 仓库内可复跑：`python -m pytest -q -m "not slow" --basetemp $(mktemp -d)`（Windows 下需一个干净 basetemp 绕过系统临时目录权限）；12 个跳过均为条件性集成测试（7 个 live-MCP、2 个 pgvector、3 个 Postgres，无 DSN/标志时跳过）；1 个 deselected 为 slow 的 100 条 Replay 套件，单独跑 `-m slow`；命令见“评测与测试” |
| 固定 Harness 回归 | 14/14 通过；预算合规与 artifact verifier 均为 100% | FakeModelClient 确定性任务集；[原始 JSON](artifacts/harness-regression-v2.json) |
| Replay Smoke 套件 | 100 条工程回归用例通过率 100%，安全硬门槛全过 | 由生成器固化的 `benchmarks/replay_cases/engineering/`（100 个 case）；生成器为唯一真值来源；复现命令见“评测与测试” |
| 长上下文压缩 | 15 组：平均输入 26,928 字符、平均输出 1,795 字符；**各组**压缩比（1−out/in）平均 92.95%、最高 95.23% | 当前请求、最新证据保留率及预算合规率均为 100%；92.95% 是**各组比值**的平均，不是用「1−均值比 1795/26928」得出的 93.33%（若那样算约为 93.33%）；保留率/预算指上下文裁剪正确性，不是 LLM 答案正确率；[证据](artifacts/resume-metrics-v1.json) |
| 多 Agent 调度 | 8 Worker、并发 4 相对串行约 2.40× 端到端加速 | 250 ms 合成 Worker，包含状态、审计和 Artifact I/O；不能外推为真实模型任务固定加速比；[证据](artifacts/resume-metrics-v1.json) |
| 长期记忆召回 | HitRate@10 100%，MRR@10 1.000，scope leakage 0 | 真实 Postgres + all-MiniLM-L6-v2 上的 200 条冻结合成命名实体资格集；阈值为 99.4% / 0.67；不代表真实用户或开放域问答；[原始 JSON](artifacts/memory-recall-synthetic-v1.json) |
| Research Graph | 8 个冻结策划场景，资格分 100/100 | 仅验证策划工程场景，不代表开放域抽取效果；[原始 JSON](artifacts/research-graph-curated-v1.json) |
| 知识能力单元（C4/C1/C3） | 统一知识库`cross_search`/`cross_enhance`/`suggest_skill_from_wiki`、语义嵌入双档、会话语义切块共 19 项单元测试 | 全部纯离线可跑，不依赖模型、网络或 Postgres；`strategy="semantic"` 与 `sentence-transformers` 均为可选禁用无副作用；命令见“评测与测试” |
| BixBench Verified-50 | 初版 22/50（44%）→ 最佳 38/50（76%），+32pp | 两次历史仓库外 run（本地 direct verifier；未运行官方外部 LLM judge）。当前环境无 HF token，仓库内不可复跑；原始报告在仓库外。口径见 [工作流与边界](docs/verified50_benchmark_workflow.md#历史出分与可复现性说明) |

完整口径、复现命令和禁止使用的夸大表述见 [工程证据说明](docs/metrics/biocoreagent-resume-evidence.md) 与 [目标完成审计](docs/resume/goal_completion_audit.md)。

## 快速开始

### 环境要求

- Python 3.10+
- Windows PowerShell 或兼容 Shell
- 至少一个模型后端：DeepSeek、OpenAI-compatible、Anthropic-compatible 或 Ollama
- Docker（仅在需要本地 Postgres 记忆服务时使用）

### 安装

~~~powershell
git clone https://github.com/ttsdj/SciCoreAgent.git
cd SciCoreAgent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

### 配置模型

~~~powershell
Copy-Item .env.example .env
notepad .env
~~~

密钥只写入本机 .env；该文件已被 Git 忽略。示例：

~~~env
BIOCOREAGENT_PROVIDER=deepseek
BIOCOREAGENT_DEEPSEEK_API_KEY=replace-me
BIOCOREAGENT_DEEPSEEK_API_BASE=https://api.deepseek.com/anthropic
BIOCOREAGENT_DEEPSEEK_MODEL=deepseek-chat
~~~

也可通过 --provider 选择 ollama、openai 或 anthropic。请根据对应服务填写模型名与 base URL。

### 运行任务

交互模式：

~~~powershell
biocoreagent-v2 --cwd D:\path\to\project --approval ask
~~~

一次性科研任务：

~~~powershell
biocoreagent-v2 --cwd D:\path\to\project --approval ask "检查 counts.tsv，并制定可验证的差异分析计划"
~~~

查看完整参数：

~~~powershell
biocoreagent-v2 --help
~~~

默认审批策略为 ask。auto 会减少交互但不会取消策略校验；never 会拒绝需要人工批准的高风险动作。

## 可选 Postgres 长期记忆

安装可选依赖并启动仅绑定本机的开发数据库：

~~~powershell
python -m pip install -e ".[memory-postgres]"
docker compose --profile memory up -d memory-postgres
~~~

在本机 .env 中配置：

~~~env
BIOCOREAGENT_MEMORY_POSTGRES_DSN=postgresql://biocoreagent:biocoreagent_dev_only@127.0.0.1:54329/biocoreagent_memory
BIOCOREAGENT_MEMORY_USER_SCOPE=local-user
BIOCOREAGENT_EMBEDDING_BACKEND=sentence-transformers
BIOCOREAGENT_EMBEDDING_MODEL=all-MiniLM-L6-v2
~~~

compose 文件中的密码只用于本机开发。生产环境必须使用外部密钥、访问控制、备份与迁移方案。

## 审计与回放示例

解释一个交付物：

~~~powershell
biocore-replay artifact explain path\to\artifact.csv
~~~

查看运行在某事件时捕获的源代码：

~~~powershell
biocore-replay source show .biocoreagent\runs\RUN_ID --at EVENT_ID
~~~

将当时代码重建到新的空目录（不会覆盖当前项目）：

~~~powershell
biocore-replay source materialize .biocoreagent\runs\RUN_ID --at EVENT_ID --target D:\empty\reconstructed
~~~

从历史运行抽取 replay case，并使用当前 runtime 生成 fresh trace：

~~~powershell
biocore-replay case extract .biocoreagent\runs\RUN_ID --case-id my-case
biocore-replay run benchmarks\replay_cases\my-case\case.json
~~~

查看全部子命令：

~~~powershell
biocore-replay --help
~~~

## 评测与测试

发布前安全检查：

~~~powershell
python scripts\check_repo_safety.py
~~~

全量测试：

~~~powershell
python -m pytest -q --basetemp C:\tmp\pytest-scicoreagent
~~~

固定 Harness 回归、上下文压缩和调度指标：

~~~powershell
python -c "from pico.evaluation.evaluator import run_harness_regression_v2; import json; print(json.dumps(run_harness_regression_v2()['summary'], indent=2))"
python scripts\benchmark_resume_metrics.py
~~~

100 条确定性 Replay Smoke 套件（生成器是唯一真值来源，改 case 先改生成器再重跑）：

~~~powershell
python scripts\generate_replay_case_suite.py          # 重建 benchmarks/replay_cases/engineering 下的 100 个 case
biocore-replay run benchmarks\replay_cases\engineering --output-root .biocoreagent\replay_runs\smoke100
~~~

上面 `run` 对目录套件会自动检测「每条 case 自带 fake_outputs.json」并走 `run_suite_deterministic`，逐条用其自身的确定性输出驱动工具、真的在 workspace 产生产物并被 verifier 检查；scorecard 的 `passed == 100` 字面成立。相应测试 `tests/test_replay_smoke_100.py`（`--run slow` 才跑全套）。

知识能力强化（C4 统一知识库 / C1 语义嵌入双档 / C3 语义切块）的单元测试，纯离线可跑：

~~~powershell
python -m pytest tests\test_knowledge_base_unified.py tests\test_embedding_provider.py tests\test_session_chunks_semantic.py -q
~~~

`biocoreagent\replay.py`、`pico\session_store.py` 的改动均以「默认行为不变」为铁律：`create_chunks` 默认 `strategy="fixed"` 逐字节保持原有打包逻辑；`embedding_provider_from_environment` 默认仍是确定性哈希嵌入。设 `BIOCOREAGENT_EMBEDDING_BACKEND=sentence-transformers` 或传 `strategy="semantic"` 仅显式启用时才改变行为。

长期记忆召回与知识图谱评测：

~~~powershell
python scripts\evaluate_research_graph.py benchmarks\research_graph\curated-v1\cases.jsonl --output artifacts\research-graph-curated-v1.json
python scripts\evaluate_memory_recall.py benchmarks\memory_recall\synthetic-v1\frozen.jsonl --corpus benchmarks\memory_recall\synthetic-v1\corpus.jsonl --output artifacts\memory-recall-synthetic-v1.json
~~~

`evaluate_research_graph.py` 无外部服务依赖，可用仓库自带场景直接跑；`evaluate_memory_recall.py` 需配置 Postgres 长期记忆（见“可选 Postgres 长期记忆”）。注意仓库有两套记忆召回数字（本地 40 条 / Postgres 200 条，语义不同），口径见 [声明证据对照表](docs/claim_evidence_ledger.md)。

可选 pgvector（仅在有 `vector` 扩展的镜像上验证；无 DSN 时测试跳过）：

~~~powershell
docker compose --profile memory up -d memory-postgres
set BIOCOREAGENT_MEMORY_POSTGRES_DSN=postgresql://biocoreagent:biocoreagent_dev_only@127.0.0.1:54329/biocoreagent_memory
python -m pytest tests\test_pgvector_migration.py tests\test_postgres_memory_integration.py -q
~~~

内置 BixBench-compatible smoke（不等同于官方完整数据集）：

~~~powershell
python -m corecoder.eval.bixbench --output-dir .biocoreagent\bixbench_runs\smoke
~~~

大型 benchmark 数据、原始生物数据和运行输出应放在仓库外，不应提交到 Git。

## 目录结构

~~~text
biocoreagent/       统一 runtime、多 Agent Supervisor、审计、证据和 replay
pico/               轻量 AgentLoop、上下文、会话态、记忆与评测内核
corecoder/          科研工具、能力路由、Skill/Wiki 沉淀及 benchmark adapter
benchmarks/         固定工程资格集、fixture 与 replay case
tests/              单元、集成、回放及确定性回归测试
artifacts/          可公开复核的小型指标证据
docs/               架构、边界、复现说明与设计记录
scripts/            安全检查、benchmark 和记忆评测脚本
~~~

运行态默认写入 .biocoreagent/ 或 .pico/，两者都被 Git 忽略。

## 安全与数据治理

- .env、数据库、日志、缓存、原始测序数据和常见大型生物格式默认忽略。
- 工具路径解析受 --cwd 工作区约束，高风险调用遵循审批策略。
- Trace 入库前执行敏感信息脱敏；大型输入默认保存路径和哈希，而非复制数据。
- CAS 通过 SHA-256 去重并保留精确代码版本；首期提供完整性验证，不宣称数字签名级不可抵赖。
- Replay 默认面向隔离 fixture；不要把生产密钥或患者数据放入 case。
- 在受监管或临床场景使用前，必须另行完成权限、隐私、合规和科学验证。

运行 [仓库安全检查](scripts/check_repo_safety.py) 可发现疑似密钥、大文件和不应提交的运行态路径。

## 当前限制

- 尚未完成真实用户跨会话记忆数据集的人工标注；目前公开结果来自冻结合成资格集。
- Postgres 检索默认 JSONB 精确余弦；pgvector ANN（`embedding vector(256)` + HNSW，`<=>` 召回）为**可选**，需 `pgvector/pgvector` 镜像；其大规模候选集性能尚未在仓库 CI 内验证（无 DSN 时 `tests/test_pgvector_migration.py` 整体跳过）。
- 真实模型回放受模型漂移、网络和费用影响；提交级回归主要使用确定性 FakeModel。
- 第三方科研后端不可用时会降级或诊断失败；系统不会把降级结果伪装成完整成功。
- 自动生成的 Skill 仍需人工审查，不能直接视为科学事实。
- 当前以 Windows 本地开发验证为主，Linux、容器化和长时间生产运行仍需扩大测试覆盖。

## License

[MIT License](LICENSE)
