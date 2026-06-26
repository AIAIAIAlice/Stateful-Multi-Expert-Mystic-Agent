from collections.abc import Mapping
from pathlib import Path
from typing import Literal
import os

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from yhj_agent.common.errors import LLMAuthError


class MimoConfig(BaseModel):
    """MiMo LLM provider 配置。

    API key 只从环境变量或本地 .env 读取，不能写入代码和文档。
    """

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr | None = None
    api_keys: list[SecretStr] = Field(default_factory=list)
    openai_base_url: str = "https://token-plan-sgp.xiaomimimo.com/v1"
    anthropic_base_url: str = "https://token-plan-sgp.xiaomimimo.com/anthropic"
    default_model: str = "mimo-v2.5-pro"
    light_model: str = "mimo-v2.5"
    timeout_seconds: float = Field(default=180.0, gt=0)
    auth_header_mode: Literal["bearer", "api-key"] = "bearer"
    max_attempts: int = Field(default=3, ge=1)
    retry_base_delay_ms: int = Field(default=500, ge=0)
    retry_max_delay_ms: int = Field(default=2000, ge=0)

    @field_validator("openai_base_url", "anthropic_base_url")
    @classmethod
    def strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    def require_api_key(self) -> str:
        """返回默认 key，仅供兼容旧调用方。"""

        return self.require_api_keys()[0]

    def require_api_keys(self) -> list[str]:
        """返回可用 MiMo key 列表，按配置顺序最多使用前三个。"""

        values = [
            secret.get_secret_value().strip()
            for secret in self.api_keys
            if secret.get_secret_value().strip()
        ]
        if not values and self.api_key is not None:
            fallback = self.api_key.get_secret_value().strip()
            if fallback:
                values = [fallback]

        if not values:
            raise LLMAuthError("MIMO_API_KEY is required for MiMo LLM calls.")
        return values[:3]


def load_dotenv_values(dotenv_path: str | Path | None = None) -> dict[str, str]:
    """轻量读取 .env，避免为了一个简历项目引入额外配置框架。"""

    path = Path(dotenv_path or ".env")
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    return values


def get_mimo_config(
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path | None = None,
) -> MimoConfig:
    """从环境变量和 .env 生成 MiMo 配置。

    环境变量优先级高于 .env，方便本地临时覆盖。
    """

    dotenv_values = load_dotenv_values(dotenv_path)
    active_env = env or os.environ

    def read(name: str, default: str | None = None) -> str | None:
        return active_env.get(name) or dotenv_values.get(name) or default

    timeout_raw = read("MIMO_TIMEOUT_SECONDS", "180")
    timeout_seconds = float(timeout_raw) if timeout_raw else 60.0
    api_keys_raw = read("MIMO_API_KEYS", "")
    api_keys = [part.strip() for part in (api_keys_raw or "").split(",") if part.strip()][:3]
    primary_api_key = api_keys[0] if api_keys else read("MIMO_API_KEY")

    return MimoConfig(
        api_key=primary_api_key,
        api_keys=api_keys,
        openai_base_url=read(
            "MIMO_OPENAI_BASE_URL",
            "https://token-plan-sgp.xiaomimimo.com/v1",
        ),
        anthropic_base_url=read(
            "MIMO_ANTHROPIC_BASE_URL",
            "https://token-plan-sgp.xiaomimimo.com/anthropic",
        ),
        default_model=read("MIMO_DEFAULT_MODEL", "mimo-v2.5-pro"),
        light_model=read("MIMO_LIGHT_MODEL", "mimo-v2.5"),
        timeout_seconds=timeout_seconds,
        auth_header_mode=read("MIMO_AUTH_HEADER_MODE", "bearer"),
        max_attempts=int(read("MIMO_MAX_ATTEMPTS", "3") or "3"),
        retry_base_delay_ms=int(read("MIMO_RETRY_BASE_DELAY_MS", "500") or "500"),
        retry_max_delay_ms=int(read("MIMO_RETRY_MAX_DELAY_MS", "2000") or "2000"),
    )


def _read_env(name: str, default: str | None = None) -> str | None:
    """从环境变量或 .env 文件读取配置值。
    优先级：环境变量 > .env 文件 > 默认值。
    """
    value = os.environ.get(name)
    if value:
        return value
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, raw_value = line.split("=", 1)
            if key.strip() == name:
                return raw_value.strip().strip('"').strip("'")
    return default


# Jina Reranker 配置
JINA_API_KEY = _read_env("JINA_API_KEY", "")
JINA_RERANKER_MODEL = _read_env("JINA_RERANKER_MODEL", "jina-reranker-v2-base-multilingual")
JINA_RERANKER_URL = _read_env("JINA_RERANKER_URL", "https://api.jina.ai/v1/rerank")

# LangSmith Tracing 配置
LANGSMITH_API_KEY = _read_env("LANGSMITH_API_KEY", "")
LANGSMITH_TRACING = _read_env("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_ENDPOINT = _read_env("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_PROJECT = _read_env("LANGSMITH_PROJECT", "yhj-agent")


def setup_langsmith() -> None:
    """配置 LangSmith 环境变量（需在 LangGraph 图编译前调用）。"""
    if LANGSMITH_TRACING and LANGSMITH_API_KEY:
        os.environ["LANGSMITH_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = LANGSMITH_ENDPOINT
        os.environ["LANGSMITH_PROJECT"] = LANGSMITH_PROJECT
