# StudyPilot

StudyPilot 是一个面向大学期末冲刺场景的跨端学习 Agent。系统不按课件目录机械讲解，而是在考试范围、剩余时间和学生掌握状态的约束下，持续选择当前最值得学习的内容，并通过 PC 与 QQ 共享同一教学会话。

项目关注四个工程问题：

1. 如何将分散的课件、笔记、试题与解析转为可追溯的知识块；
2. 如何在有限时间内生成可解释、可动态调整的复习计划；
3. 如何让 Agent 在讲解、出题、等待作答和重规划之间可靠暂停与恢复；
4. 如何避免重复消息和跨端并发更新破坏教学状态。

## 技术栈

- Python 3.11+
- FastAPI / Pydantic
- LangGraph / SQLite Checkpoint
- SQLite
- BM25 / BGE Embedding / RRF
- MCP Python SDK
- QQ 官方 Bot API
- Pytest

## 核心链路

```text
课件、往年题、作业 + 考试目标 + 用户补充说明
        ↓
解析为原文块 → LLM 分批分析考点、考试证据与前置假设
        ↓
LLM 制定初始计划（代码检查预算、引用与依赖）
        ↓
学习 Agent 决策 → 检索 / 诊断 / 讲解 / 练习 / 提问 / 修改计划
        ↑                         ↓
更新掌握证据与剩余时间 ← 评估学生作答 / 获取补充信息
```

用户不必录入考点分值或前置依赖。LLM 根据资料和反馈决定学习行动，LangGraph 承载执行、暂停与恢复；教学不强制每轮同时生成讲解和题目。PC 与 QQ 使用同一会话。

## 关键设计

### 1. 证据驱动的资料分析与模型规划

资料按批次覆盖，成功的分析结果按输入指纹保存，失败或重启后复用。模型输出考点、时间与收益估计、候选依赖，并区分资料事实、推测和未知信息，保留原文块或用户陈述引用。重复资料不增加考试频率，答案解析不当作另一份试卷。

模型制定包含“必学 / 争取 / 暂缓”的计划，服务端校验考点归属、前置关系和分钟预算。原有依赖包贪心规划器只提供预算参考，不决定所有教学动作。无往年题时仍可依靠课件、作业和用户说明规划，但分布是推测，不保证考试提分。

### 2. 可追溯的课程资料层

资料层将“文件内容”与“资料来源”分离：

- `ContentBlob` 按 SHA-256 内容指纹去重；
- `SourceAsset` 独立保存来源、展示名称、可信度和解析状态；
- 相同内容只解析一次，但不会丢失不同渠道的来源信息；
- `DocumentBlock` 保存课程、章节、块序号、来源和原文哈希；
- 解析结果会生成一份可阅读的 `课程资料.generated.md`，但检索和引用始终基于原始块，不用整理结果覆盖证据层。

Markdown 与纯文本由本地 parser 处理；PDF 与 Office 文件通过外部文档解析 MCP 接入。MCP 不可用、响应为空、输出截断、路径越权或工具超时时均显式失败，不会写入空知识块。

### 3. Knowledge Window 与引用校验

检索层支持中文 BM25、BGE Dense 和固定参数的 RRF。Knowledge Window 受到块数量与字符预算双重限制，只向教学节点注入当前问题所需的原文。

模型返回的引用必须通过持久化证据校验：系统核对 `block_id`、`source_id`、课程、定位字段和原文内容，防止调用方或模型使用不属于窗口的文本伪造引用。

### 4. 可暂停、可恢复的学习 Agent

```text
bootstrap → decide → execute → decide
                       │
                 waiting_answer
                       │
                    observe → decide
```

模型选择检索、诊断、讲解、练习、澄清或调整计划等行动；代码执行并校验。诊断发现前置缺口时可先补基础，保留原目标，随后回到原题；诊断、讲解和练习是独立动作。每轮最多八个行动，重复无进展或连续非法行动会暂停，而不虚报完成。

等待用户输入使用 interrupt，SQLite Checkpoint 保存分析、计划、待回答问题、作答证据和返回目标。明确评分规则裁决掌握状态，模型的猜测不能直接标为已掌握。失败节点可续跑；已处理消息与回执共同降低恢复后重复评分。

真实模型必须配置，产品不会默认使用 fake。测试通过显式注入可控模型验证决策路径。服务仍面向单进程，不能声称实现多实例共享锁或模型调用 exactly-once。

### 5. 跨端一致性

PC 与 QQ 共用同一个 `TeachingSession`。QQ 入口将官方 C2C 文本事件转换为平台无关的 `CanonicalMessage`，再进入同一 application service。

系统使用三层保护：

- `message_id` + answer receipt 保证重复投递不重复评分；
- `expected_version` 乐观锁拒绝过期更新；
- 单进程 Session 锁保证同一会话串行处理。

QQ 官方 Bot 的 C2C 文本收发链路已完成手工联调；当前不支持群聊、频道、图片、主动消息或多进程互斥。

## 检索评测

仓库包含一组冻结的概率论检索评测：九份章节笔记作为语料，30 条人工核对 query（15 条术语查询、15 条语义改写）作为查询。答案、真题解析和人工金标准正文不进入语料。

| 检索方案 | Recall@5 |
| --- | ---: |
| 正文 BM25 | 22/30 = 73.3% |
| Title-aware BM25（title weight = 2） | 26/30 = 86.7% |
| BGE Dense | 23/30 = 76.7% |
| RRF Hybrid（k = 60） | 25/30 = 83.3% |

结果表明，标题字段加权修复了当前测试集中的主要召回缺口；Dense 和 Hybrid 已接通，但没有超过最强单路结果，因此项目不声称“混合检索必然提升”。详细实验条件和逐题分析见 [`evaluation/`](evaluation/)。

该评测只衡量小规模检索召回，不代表教学质量或考试提分效果。

## 目录结构

```text
src/studypilot/
├── api.py                         # FastAPI 入口与本地学习页面
├── application/
│   ├── source_import.py           # 内容寻址与资料导入
│   ├── parsing.py                 # 本地文档解析边界
│   ├── external_document_mcp.py   # 外部文档解析 MCP adapter
│   ├── retrieval.py               # 检索、Knowledge Window 与引用校验
│   ├── teaching.py                # LangGraph 学习 Agent 执行与恢复
│   ├── learning_agent.py          # 模型分析、决策合约与证据/计划校验
│   ├── teaching_service.py        # Session、幂等与串行边界
│   ├── llm.py                     # OpenAI-compatible provider
│   ├── channel.py                 # 平台无关消息路由
│   └── qq.py                      # QQ 官方 C2C adapter
├── domain/                        # 领域模型、规划器与状态规则
└── infrastructure/
    └── sqlite_repository.py       # SQLite 仓储
tests/                             # 单元与集成测试
evaluation/                        # 冻结数据、结果与实验记录
```

## 本地运行

```powershell
git clone https://github.com/vanyony/StudyPilot.git
cd StudyPilot
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m uvicorn studypilot.api:app --reload
```

打开：

- 学习入口：`http://127.0.0.1:8000/study`
- OpenAPI 文档：`http://127.0.0.1:8000/docs`

运行测试：

```powershell
python -m pytest
```

测试覆盖资料分析、模型规划、前置诊断与返回、分析缓存续跑、非法行动、引用校验、检索、消息幂等、乐观锁和跨端会话。离线测试不代替真实模型语义质量验证。

## 可选能力

### BGE Embedding

```powershell
python -m pip install -e ".[dev,embeddings]"
```

### 外部文档解析 MCP

```powershell
$env:STUDYPILOT_DOCUMENT_MCP_URL = "http://127.0.0.1:3001/mcp"
$env:STUDYPILOT_DOCUMENT_MCP_TIMEOUT_SECONDS = "300"
```

MCP server 需要暴露 Office/PDF 解析工具；具体工具名可通过 `STUDYPILOT_DOCUMENT_MCP_*_TOOL` 环境变量覆盖。

### LLM Provider

```powershell
$env:STUDYPILOT_LLM_API_KEY = "..."
$env:STUDYPILOT_LLM_BASE_URL = "https://example.com/v1"
$env:STUDYPILOT_LLM_MODEL = "model-name"
```

未配置 provider 时，开始学习会明确提示配置缺失；资料导入等基础功能仍可使用。

### QQ 官方 Bot

```powershell
python -m pip install -e ".[dev,qq]"
$env:STUDYPILOT_QQ_APP_ID = "..."
$env:STUDYPILOT_QQ_APP_SECRET = "..."
```

凭据只从环境变量读取，不写入数据库或日志。QQ 用户需要由本地 operator 显式绑定到 `learner_id / course_id / session_id`。

## 当前边界

- 当前以单机、单进程运行模型为主；Session 锁不提供跨进程互斥；
- SQLite 适合当前个人学习场景，尚未实现 PostgreSQL / Redis / MQ 部署；
- 文档 OCR 与 Office/PDF 解析依赖外部 MCP，本仓库不实现识别模型；
- 检索评测规模较小，尚未建立教学效果或提分效果评测；
- LLM 输出始终视为不可信输入，需要经过 Schema、rubric 和引用白名单校验；
- QQ 当前仅支持已绑定用户的 C2C 私聊文本消息。

## 安全说明

仓库不包含 API Key、QQ App Secret、个人课程原文、运行数据库或本地数据目录。所有外部凭据均通过环境变量注入。

## 学习模型配置

设置 `STUDYPILOT_LLM_API_KEY`、`STUDYPILOT_LLM_MODEL`，按需设置 `STUDYPILOT_LLM_BASE_URL`。学习 Agent 与答案评价复用 OpenAI-compatible SDK 的 JSON 调用和错误处理。未配置时开始学习会显式失败，资料上传等基础功能仍可使用。

学习页面上传 Markdown/Text 自动解析；PDF/Office 使用已注入的外部文档 MCP，没有配置时保留文件并说明尚未解析。资料解析完成后填写考试目标与可用分钟，在“开始复习”补充老师范围或个人情况，无需填写考点清单。手动考点编辑只供可选修正。
