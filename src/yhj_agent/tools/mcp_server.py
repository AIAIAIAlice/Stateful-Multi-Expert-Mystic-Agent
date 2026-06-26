"""MCP 工具服务器 — 封装确定性计算工具。"""
from __future__ import annotations

from typing import Any

from yhj_agent.tools.bazi_calculator import BaziCalculator


class MCPServer:
    """轻量 MCP 工具服务器。

    只暴露确定性符号计算，不承载推理。
    """

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {
            "bazi_calculator": BaziCalculator(),
        }

    def call(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """调用指定工具。"""
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        return self.tools[tool_name].execute(params)

    def list_tools(self) -> list[str]:
        """列出所有可用工具。"""
        return sorted(self.tools.keys())
