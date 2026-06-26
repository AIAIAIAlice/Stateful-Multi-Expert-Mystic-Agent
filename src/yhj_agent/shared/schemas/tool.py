from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ToolCallRequest(BaseModel):
    """确定性工具统一输入。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolCallResult(BaseModel):
    """确定性工具统一输出。"""

    model_config = ConfigDict(extra="forbid")

    tool_name: str
    output: dict[str, Any] = Field(default_factory=dict)
    artifact_id: str | None = None
    error: str | None = None

