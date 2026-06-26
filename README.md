# Stateful-Multi-Expert-Mystic-Agent


Chinese documentation: [README_CN.md](README_CN.md)

Stateful-Multi-Expert-Mystic-Agent is a multi-turn agent execution system inspired by LangGraph-style state orchestration. It is designed for personalized consultation workflows and coordinates input normalization, task planning, deterministic tool calls, RAG retrieval, specialist-agent collaboration, quality evaluation, and report generation in a traceable flow.

## Key Features

- Multi-turn state management for per-turn task state and cross-turn session context.
- Intent routing and execution planning based on user input.
- Tool-use policy controls for allowed, required, and blocked tool calls.
- RAG retrieval over local knowledge, metadata, and retriever components.
- Deterministic symbolic calculation for structured tasks such as Bazi chart calculation.
- Specialist collaboration with critic/revision steps for output quality.
- Demo interfaces including a static web UI, a Streamlit frontend, and a React architecture viewer.

## Project Layout

```text
.
|-- api/                  # Local HTTP API and static frontend server
|-- frontend/             # Streamlit frontend and static demo assets
|-- architecture-viewer/  # React + Vite architecture viewer
|-- src/yhj_agent/        # Core agent source code
|-- tests/                # Unit tests
|-- docs/                 # Project documentation
|-- data/knowledge/       # Lightweight knowledge rules
`-- data/metadata/        # Dataset metadata
```

## Requirements

- Python 3.11 or newer
- uv, or another Python package manager compatible with `pyproject.toml`
- Node.js and npm for `architecture-viewer`

## Installation

```powershell
uv sync
```

If you do not use uv, create a virtual environment and install dependencies from `pyproject.toml`.

## Configuration

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Important settings:

- `MIMO_API_KEY`: MiMo LLM API key.
- `MIMO_OPENAI_BASE_URL`: OpenAI-compatible API endpoint.
- `USE_MOCK_LLM`: Enables offline mock mode, useful for local demos.
- `EMBEDDING_API_KEY`: DashScope embedding API key.
- `CHROMA_DIR`: Local ChromaDB vector-store path.
- `JINA_API_KEY`: Jina reranker API key.
- `LANGSMITH_*`: LangSmith tracing settings.

Do not commit real `.env` files or secrets.

## Run the Backend and Demo

Start the local API and static demo page:

```powershell
python api\main.py
```

Default URLs:

- Web page: `http://127.0.0.1:8001`
- Health check: `http://127.0.0.1:8001/api/health`

Start the Streamlit frontend:

```powershell
streamlit run frontend\app.py
```

The Streamlit frontend expects the local API at `http://127.0.0.1:8001`.

## Architecture Viewer

```powershell
cd architecture-viewer
npm install
npm run dev
```

Build for production:

```powershell
npm run build
```

## Tests

```powershell
python -m pytest tests/ -v
```

## Data and Privacy

This GitHub-ready version keeps only lightweight metadata and knowledge rules. The following should not be committed:

- Real `.env` files, API keys, certificates, or private keys.
- SQLite checkpoint/profile databases and WAL/SHM files.
- ChromaDB vector stores, local indexes, embedding batch inputs, and generated data.
- Logs, virtual environments, caches, frontend dependencies, and build artifacts.

Regenerate experiment data locally when needed, and verify that sensitive data is not committed.
