from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class LLMMessage(BaseModel):
    """项目内部统一 message schema。"""

    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]]
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class LLMRequest(BaseModel):
    """上层模块只构造该请求结构，不直接依赖 MiMo 原始协议。"""

    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[LLMMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """MiMo 原始响应转换后的统一响应结构。"""

    model_config = ConfigDict(extra="forbid")

    model: str
    content: str
    reasoning_content: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    raw_response: dict[str, Any] | None = None

