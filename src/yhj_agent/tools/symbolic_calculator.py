from datetime import datetime
from typing import Any

from yhj_agent.shared.schemas.tool import ToolCallRequest, ToolCallResult


class SymbolicCalculator:
    """确定性结构化计算工具。

    该工具只做字段标准化和规则化提取，不生成自然语言建议。
    """

    name = "SymbolicCalculator"

    def run(self, request: ToolCallRequest) -> ToolCallResult:
        args = request.arguments
        birth_date = str(args.get("birth_date", "")).strip()
        birth_time = str(args.get("birth_time", "")).strip()
        birth_place = str(args.get("birth_place", "")).strip()

        output: dict[str, Any] = {
            "birth_date": birth_date,
            "birth_time": birth_time,
            "birth_place": birth_place,
            "calendar": self._parse_date(birth_date),
            "symbolic_structure": self._build_symbolic_structure(birth_date, birth_time, birth_place),
        }

        return ToolCallResult(
            tool_name=self.name,
            output=output,
            artifact_id="symbolic_outputs",
        )

    def _parse_date(self, birth_date: str) -> dict[str, Any]:
        for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
            try:
                parsed = datetime.strptime(birth_date, fmt)
                return {
                    "year": parsed.year,
                    "month": parsed.month,
                    "day": parsed.day,
                    "source_format": fmt,
                }
            except ValueError:
                continue
        return {"raw": birth_date, "source_format": "unknown"}

    def _build_symbolic_structure(self, birth_date: str, birth_time: str, birth_place: str) -> dict[str, Any]:
        # 简历项目不实现真实命理推算，只输出可解释、可复用的结构化占位结果。
        return {
            "time_bucket": self._time_bucket(birth_time),
            "place_region": birth_place or "unknown",
            "calculation_scope": "demo_structured_symbolic_features",
            "source_fields": ["birth_date", "birth_time", "birth_place"],
        }

    def _time_bucket(self, birth_time: str) -> str:
        if any(token in birth_time for token in ("上午", "早", "8", "9", "10", "11")):
            return "morning"
        if any(token in birth_time for token in ("下午", "13", "14", "15", "16", "17")):
            return "afternoon"
        if any(token in birth_time for token in ("晚上", "夜", "18", "19", "20", "21", "22", "23")):
            return "evening"
        return "unknown"

