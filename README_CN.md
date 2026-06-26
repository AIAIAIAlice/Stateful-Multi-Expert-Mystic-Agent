# YHJ Agent

英文文档：[README.md](README.md)

YHJ Agent 是一个基于 LangGraph 思路构建的多轮 Agent 执行系统，面向个性化咨询场景。项目将输入理解、任务规划、确定性工具调用、RAG 检索、专家 Agent 协作、质量评估和报告生成组织为可追踪的执行流程。

## 核心功能

- 多轮会话状态管理：维护单轮任务状态和跨轮会话上下文。
- 意图识别与执行规划：根据用户输入生成可执行的节点计划。
- 工具调用治理：控制每轮允许、必须或禁止调用的工具。
- RAG 检索：结合知识库、元数据和检索器为回答提供证据。
- 确定性计算：通过符号计算工具处理八字排盘等结构化任务。
- 专家协作与评估：并行调用专业节点，并通过 critic/revision 流程提升输出质量。
- 可视化演示：包含静态 Web 演示、Streamlit 前端和 React 架构查看器。

## 目录结构

```text
.
|-- api/                  # 本地 HTTP API 与静态前端服务
|-- frontend/             # Streamlit 前端和静态演示页面
|-- architecture-viewer/  # React + Vite 架构查看器
|-- src/yhj_agent/        # Agent 核心源码
|-- tests/                # 单元测试
|-- docs/                 # 项目文档
|-- data/knowledge/       # 轻量知识规则
`-- data/metadata/        # 数据集元信息
```

## 环境要求

- Python 3.11 或更高版本。
- uv，或其他兼容 `pyproject.toml` 的 Python 包管理工具。
- Node.js 和 npm，用于运行 `architecture-viewer`。

## 安装

```powershell
uv sync
```

如果不使用 uv，也可以在虚拟环境中根据 `pyproject.toml` 安装依赖。

## 配置

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

主要配置项：

- `MIMO_API_KEY`：MiMo LLM API key。
- `MIMO_OPENAI_BASE_URL`：OpenAI-compatible API 地址。
- `USE_MOCK_LLM`：是否使用离线 mock 模式，默认适合本地演示。
- `EMBEDDING_API_KEY`：DashScope embedding API key。
- `CHROMA_DIR`：本地 ChromaDB 向量库路径。
- `JINA_API_KEY`：Jina reranker API key。
- `LANGSMITH_*`：LangSmith tracing 相关配置。

请不要提交真实 `.env` 文件或任何密钥。

## 运行后端与演示

启动本地 API 和静态演示页面：

```powershell
python api\main.py
```

默认地址：

- Web 页面：`http://127.0.0.1:8001`
- 健康检查：`http://127.0.0.1:8001/api/health`

启动 Streamlit 前端：

```powershell
streamlit run frontend\app.py
```

Streamlit 前端默认访问本地 API：`http://127.0.0.1:8001`。

## 架构查看器

```powershell
cd architecture-viewer
npm install
npm run dev
```

构建生产版本：

```powershell
npm run build
```

## 测试

```powershell
python -m pytest tests/ -v
```

## 数据与隐私

本 GitHub 版本只保留轻量元数据和知识规则。以下内容默认不应上传：

- 真实 `.env`、API key、证书和私钥。
- SQLite checkpoint、profile 数据库和 WAL/SHM 文件。
- ChromaDB 向量库、本地索引、embedding 批处理输入和生成数据。
- 日志、虚拟环境、缓存、前端依赖和构建产物。

如需复现实验数据，请在本地重新生成，并确保不会把敏感数据提交到仓库。
