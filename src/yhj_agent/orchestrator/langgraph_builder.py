"""LangGraph StateGraph 编排器。

使用 LangGraph 的 StateGraph 构建完整的 Agent 执行图。
支持 add_node / add_edge / add_conditional_edges / interrupt / Checkpointer。
"""
from __future__ import annotations

import re
from typing import Any

from yhj_agent.common.config import setup_langsmith
from yhj_agent.orchestrator.state_graph_builder import build_state_graph
from yhj_agent.state.workflow_state import WorkflowState, create_empty_state

def _to_dicts(items):
    """将 Pydantic 对象列表转为 dict 列表。"""
    if not items:
        return items
    result = []
    for item in items:
        if hasattr(item, "model_dump"):
            result.append(item.model_dump())
        elif isinstance(item, dict):
            result.append(item)
    return result or items


class LangGraphOrchestrator:
    """LangGraph StateGraph 编排器。

    使用 build_state_graph() 构建完整的 LangGraph 图，
    支持 add_node / add_edge / add_conditional_edges / interrupt / Checkpointer。
    """

    def __init__(self, use_checkpointer: bool = True) -> None:
        # 配置 LangSmith 环境变量（需在图编译前）
        setup_langsmith()

        graph = build_state_graph()
        if use_checkpointer:
            from yhj_agent.orchestrator.checkpointer import get_checkpointer
            self.app = graph.compile(checkpointer=get_checkpointer())
        else:
            self.app = graph.compile()

    def run_turn(self, session_id: str, user_input: str, user_id: str = "") -> dict:
        """执行单轮对话。

        Args:
            session_id: 会话 ID
            user_input: 用户输入
            user_id: 用户 ID

        Returns:
            完整的执行结果，或 interrupt 响应
        """
        # 初始化状态
        state = create_empty_state()
        state["user_input"] = user_input
        state["session_id"] = session_id
        state["user_id"] = user_id or session_id

        # 尝试从 checkpointer 加载上一轮状态，继承关键上下文
        config = {"configurable": {"thread_id": session_id}}
        try:
            prev = self.app.get_state(config)
            if prev and prev.values:
                prev_vals = prev.values if isinstance(prev.values, dict) else {}
                # 继承上一轮的 normalized_input（含出生信息）供纠错/主题切换复用
                if prev_vals.get("normalized_input"):
                    state["normalized_input"] = prev_vals["normalized_input"]
                    state["previous_normalized_input"] = prev_vals["normalized_input"]
                # 继承上一轮的 final_report 供追问/格式调整参考
                if prev_vals.get("final_report"):
                    state["final_report"] = prev_vals["final_report"]
                # Inherit consultation_request so intent_router can detect follow-up/explanation
                if prev_vals.get("consultation_request"):
                    state["consultation_request"] = prev_vals["consultation_request"]
        except Exception:
            pass  # checkpointer 无历史或查询失败不影响主流程

        # 执行图（可能触发 interrupt）
        run_config = {
            "configurable": config["configurable"],
            "metadata": {"session_id": session_id, "thread_id": session_id, "user_id": user_id or session_id},
            "run_name": f"turn-{session_id}",
        }
        try:
            result = self.app.invoke(state, run_config)
        except Exception as exc:
            # 检测是否为 LangGraph interrupt（兼容多种异常类型）
            exc_type = type(exc).__name__
            exc_str = str(exc).lower()
            if ("interrupt" in exc_type.lower() or "interrupt" in exc_str
                    or hasattr(exc, "value")):
                interrupt_data = self._extract_interrupt_data(exc, session_id)
                if interrupt_data:
                    return interrupt_data
            raise

        # 检测 invoke 返回值是否为 interrupt payload（而非完整 state）
        if self._is_interrupt_payload(result):
            return {
                "interrupted": True,
                "session_id": session_id,
                "question": result.get("question", ""),
                "missing_fields": result.get("missing_fields", []),
            }

        # LangGraph v0.2+ interrupt 返回格式（__interrupt__ 键）
        if isinstance(result, dict) and result.get("__interrupt__"):
            interrupt_info = result["__interrupt__"]
            if isinstance(interrupt_info, (list, tuple)) and interrupt_info:
                interrupt_info = interrupt_info[0]
            q = ""
            if hasattr(interrupt_info, "value") and isinstance(interrupt_info.value, dict):
                q = interrupt_info.value.get("question", "")
            elif isinstance(interrupt_info, dict):
                q = interrupt_info.get("question", "")
            return {
                "interrupted": True,
                "session_id": session_id,
                "question": q,
                "missing_fields": [],
            }

        return self._build_response(result, session_id)

    def resume_turn(self, session_id: str, answer: str) -> dict:
        """恢复被 interrupt 的对话。

        Args:
            session_id: 会话 ID
            answer: 用户对澄清问题的回答

        Returns:
            完整的执行结果
        """
        try:
            from langgraph.types import Command
        except ImportError:
            return {"error": "当前 LangGraph 版本不支持 interrupt/resume，请升级到 v0.2+"}

        config = {
            "configurable": {"thread_id": session_id},
            "metadata": {"session_id": session_id, "thread_id": session_id},
            "run_name": f"resume-{session_id}",
        }

        try:
            result = self.app.invoke(Command(resume=answer), config)
        except Exception as exc:
            # checkpointer 状态丢失 → 降级为新任务执行
            exc_str = str(exc).lower()
            if "no checkpoint" in exc_str or "not found" in exc_str or "no state" in exc_str:
                return self.run_turn(session_id, answer)
            return {"error": f"恢复执行失败：{type(exc).__name__}: {exc}"}

        # 同样检测是否再次 interrupt
        if self._is_interrupt_payload(result):
            return {
                "session_id": session_id,
                "question": result.get("question", ""),
                "missing_fields": result.get("missing_fields", []),
            }

        # LangGraph v0.2+ interrupt 返回格式（__interrupt__ 键）
        if isinstance(result, dict) and result.get("__interrupt__"):
            interrupt_info = result["__interrupt__"]
            if isinstance(interrupt_info, (list, tuple)) and interrupt_info:
                interrupt_info = interrupt_info[0]
            q = ""
            if hasattr(interrupt_info, "value") and isinstance(interrupt_info.value, dict):
                q = interrupt_info.value.get("question", "")
            elif isinstance(interrupt_info, dict):
                q = interrupt_info.get("question", "")
            return {
                "interrupted": True,
                "session_id": session_id,
                "question": q,
                "missing_fields": [],
            }

        return self._build_response(result, session_id)

    def get_session(self, session_id: str) -> dict:
        """获取 session 信息（兼容 API）。"""
        return {"session_id": session_id, "status": "active"}

    def _build_response(self, state: WorkflowState, session_id: str) -> dict:
        """构建响应。"""
        final_report = state.get("final_report", {})
        if isinstance(final_report, dict):
            report_text = final_report.get("report_text", "")
        else:
            report_text = str(final_report)

        return {
                "session_id": session_id,
                "final_report": report_text,
                "final_report_detail": final_report,
                "turn_type": state.get("turn_type"),
                "consultation_request": state.get("consultation_request"),
                "risk_level": state.get("risk_level"),
                "symbolic_result": state.get("symbolic_result"),
                "evaluation": state.get("evaluation"),
                "specialist_outputs": state.get("specialist_outputs"),
                "debate_output": state.get("debate_output"),
                "synthesis": state.get("synthesis"),
                "execution_plan": state.get("execution_plan"),
                "user_profile": state.get("user_profile"),
                "relevant_memories": state.get("relevant_memories", []),
                "domain_rag_result": _to_dicts(state.get("domain_rag_result")),
                "trace": state.get("trace", []),
                "explanation_output": state.get("explanation_output"),
                "compressed_report": state.get("compressed_report"),
                "safe_completion_output": state.get("safe_completion_output"),
                "memory_write_result": state.get("memory_write_result"),
                "active_nodes": (state.get("execution_plan", {}) or {}).get("active_nodes", []),
            }

    @staticmethod
    def _extract_interrupt_data(exc: Exception, session_id: str) -> dict | None:
        """从异常中提取 interrupt 数据。"""
        exc_type = type(exc).__name__
        # LangGraph 的 interrupt 可能抛出 NodeInterrupt 或 GraphInterrupt
        if "Interrupt" not in exc_type and "interrupt" not in str(exc).lower():
            return None

        # 尝试从异常属性中提取 payload
        payload = getattr(exc, "value", None) or getattr(exc, "payload", None)
        if isinstance(payload, dict):
            return {
                "interrupted": True,
                "session_id": session_id,
                "question": payload.get("question", ""),
                "missing_fields": payload.get("missing_fields", []),
            }

        # 降级：返回通用 interrupt 响应
        return {
            "interrupted": True,
            "session_id": session_id,
            "question": "请补充缺失信息后继续",
            "missing_fields": [],
        }

    @staticmethod
    def _is_interrupt_payload(result: Any) -> bool:
        """检测 invoke 返回值是否为 interrupt payload。"""
        if not isinstance(result, dict):
            return False
        # interrupt payload 包含 type="clarification" 或 question + missing_fields
        return result.get("type") == "clarification" or (
            "question" in result and "missing_fields" in result
        )

