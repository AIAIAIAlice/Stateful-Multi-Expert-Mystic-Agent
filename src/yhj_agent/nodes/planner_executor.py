"""Node4: Planner-Executor — 确定性规则引擎。

根据 IntentRouter 的分类结果，通过确定性规则引擎构建最小执行子图。
"""
from __future__ import annotations

from typing import Any

from yhj_agent.shared.schemas.planning import PlannerExecutorOutput


# intent → 执行计划映射表
INTENT_MAP: dict[str, dict[str, list[str]]] = {
    "career_guidance_bazi": {
        "mcp_tools": ["bazi_calculator"],
        "rag_targets": ["symbolic_rules", "psychology_support", "safety_policy"],
        "specialists": ["symbolic_interpreter", "psychology_support", "practical_advisor"],
    },
    "career_guidance_bazi_followup": {
        "mcp_tools": [],
        "rag_targets": [],
        "specialists": ["practical_advisor"],
    },
    "health_outlook": {
        "mcp_tools": ["bazi_calculator"],
        "rag_targets": ["symbolic_rules", "safety_policy"],
        "specialists": ["symbolic_interpreter", "psychology_support"],
    },
    "relationship_analysis": {
        "mcp_tools": ["bazi_calculator"],
        "rag_targets": ["symbolic_rules", "psychology_support", "safety_policy"],
        "specialists": ["symbolic_interpreter", "psychology_support", "practical_advisor"],
    },
    "finance_consultation": {
        "mcp_tools": ["bazi_calculator"],
        "rag_targets": ["symbolic_rules", "safety_policy"],
        "specialists": ["symbolic_interpreter", "practical_advisor"],
    },
}


class PlannerExecutor:
    """确定性规则引擎：intent → 执行模式 → 子图实例化。"""

    def plan(
        self,
        consultation_intent: str,
        turn_type: str,
        risk_level: int,
        normalized_input: dict[str, Any] | None = None,
    ) -> PlannerExecutorOutput:
        """生成执行计划。

        Args:
            consultation_intent: 意图分类结果
            turn_type: 轮次类型
            risk_level: 风险等级 (1-10)
            normalized_input: 当前标准化输入

        Returns:
            PlannerExecutorOutput
        """
        # 1. 执行模式决策
        execution_mode = self._decide_execution_mode(turn_type, risk_level, normalized_input)

        # 2. 查表确定 MCP/RAG/Specialist
        intent_config = INTENT_MAP.get(consultation_intent, INTENT_MAP.get("career_guidance_bazi", {}))

        # 3. 构建 active_nodes
        active_nodes = self._build_active_nodes(execution_mode, intent_config)

        return PlannerExecutorOutput(
            execution_mode=execution_mode,
            mcp_calls=self._build_mcp_calls(intent_config.get("mcp_tools", []), normalized_input),
            rag_targets=intent_config.get("rag_targets", []),
            specialist_targets=intent_config.get("specialists", []),
            active_nodes=active_nodes,
        )

    def _decide_execution_mode(
        self,
        turn_type: str,
        risk_level: int,
        normalized_input: dict[str, Any] | None,
    ) -> str:
        """执行模式决策。"""
        if turn_type == "follow_up_question":
            return "explanation_only"
        if turn_type == "format_refinement":
            question = (normalized_input or {}).get("question", "")
            style_keywords = ("诗意", "口语化", "正式", "轻松", "深刻", "通俗", "专业", "幽默", "温柔", "文艺", "改写风格", "换个风格")
            if any(kw in question for kw in style_keywords):
                return "style_only"
            return "format_only"
        if turn_type == "safety_intervention" or risk_level >= 9:
            return "safe_completion"
        return "full_execution"

    def _build_active_nodes(
        self,
        execution_mode: str,
        intent_config: dict[str, list[str]],
    ) -> list[str]:
        """构建最小执行子图节点列表。"""
        if execution_mode == "explanation_only":
            return ["explanation"]
        if execution_mode in ("format_only", "style_only"):
            return ["report_compressor"]
        if execution_mode == "safe_completion":
            return ["safe_completion"]

        nodes = []
        if intent_config.get("mcp_tools"):
            nodes.append("symbolic_calculator")
        if intent_config.get("rag_targets"):
            nodes.extend(["domain_rag"])
        if intent_config.get("specialists"):
            nodes.extend(["specialist_subgraph", "conflict_debate"])
        nodes.extend(["synthesis", "critic", "report_generator"])
        return nodes

    def _build_mcp_calls(
        self,
        mcp_tools: list[str],
        normalized_input: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """构建 MCP 工具调用列表。"""
        calls = []
        for tool in mcp_tools:
            if tool == "bazi_calculator" and normalized_input:
                calls.append({
                    "tool": "bazi_calculator",
                    "params": {
                        "birth_date": f"{int(normalized_input.get('birth_year', 0) or 0)}-{int(normalized_input.get('birth_month', 0) or 0):02d}-{int(normalized_input.get('birth_day', 0) or 0):02d}",
                        "birth_time": normalized_input.get("birth_hour", ""),
                        "gender": normalized_input.get("gender", "male"),
                    },
                })
        return calls

