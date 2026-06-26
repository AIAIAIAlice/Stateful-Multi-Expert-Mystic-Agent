from typing import Any

from pydantic import BaseModel, Field


class NodeResult(BaseModel):
    """LangGraph 节点统一返回结构。

    节点只声明本节点负责的 state updates，不直接修改完整 state。
    """

    updates: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
    trace: dict[str, Any] = Field(default_factory=dict)

