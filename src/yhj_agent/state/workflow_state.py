"""LangGraph WorkflowState 定义。

目标架构的 TypedDict 状态，用于 LangGraph StateGraph 编排。
"""
from typing import Any, Optional


class WorkflowState(dict):
    """LangGraph 兼容的工作流状态。

    使用 dict 子类而非 TypedDict，以支持 LangGraph 的 partial update 机制。
    所有字段均为可选，节点按需读写。
    """

    # === 用户输入 ===
    user_input: str
    user_id: str
    turn_type: str
    clarification_answer: str
    pending_clarification: bool

    # === Node1: InputNormalizer ===
    normalized_input: dict
    previous_normalized_input: dict

    # === Node2: IntentRouter ===
    consultation_request: dict
    route_decision: dict
    risk_level: int
    needs_clarification: bool
    clarification_count: int
    degraded_clarification: bool

    # === Node3: ProfileMemoryReader ===
    user_profile: dict
    relevant_memories: list

    # === Node4: Planner-Executor ===
    execution_plan: dict
    node_selection: dict

    # === Node5: SymbolicCalculator ===
    symbolic_result: dict

    # === Node6: DomainRAG ===
    domain_rag_result: dict
    pruned_context: dict

    # === Node7: SpecialistSubgraph ===
    specialist_outputs: dict

    # === Node7.5: ConflictDebateNode ===
    debate_output: dict

    # === Node8: SynthesisNode ===
    synthesis: dict

    # === Node9: CriticEvaluator ===
    evaluation: dict
    retry_count: int
    best_result: Optional[dict]
    is_degraded: bool
    degradation_log: Optional[dict]

    # === Node10: ReportGenerator ===
    final_report: dict

    # === Node11: MemoryWriter ===
    memory_write_result: dict

    # === Node12: ExplanationNode ===
    explanation_output: dict

    # === Node13: ReportCompressor ===
    compressed_report: dict

    # === Node14: SafeCompletionNode ===
    safe_completion_output: dict

    # === Node15: ClarificationNode ===
    clarification_output: dict


def create_empty_state() -> WorkflowState:
    """创建空的初始状态。"""
    return WorkflowState(
        user_input="",
        user_id="",
        turn_type="",
        clarification_answer="",
        pending_clarification=False,
        normalized_input={},
        previous_normalized_input={},
        consultation_request={},
        route_decision={},
        risk_level=4,
        needs_clarification=False,
        clarification_count=0,
        degraded_clarification=False,
        user_profile={},
        relevant_memories=[],
        execution_plan={},
        node_selection={},
        symbolic_result={},
        domain_rag_result={},
        pruned_context={},
        specialist_outputs={},
        debate_output={"debate_occurred": False},
        synthesis={},
        evaluation={},
        retry_count=0,
        best_result=None,
        is_degraded=False,
        degradation_log=None,
        final_report={},
        memory_write_result={},
        explanation_output={},
        compressed_report={},
        safe_completion_output={},
        clarification_output={},
    )


def merge_state(base: WorkflowState, updates: dict) -> WorkflowState:
    """合并状态更新（LangGraph 风格的 partial update）。"""
    merged = WorkflowState(base)
    merged.update(updates)
    return merged


def agent_state_to_workflow(agent_state: object) -> WorkflowState:
    """将旧 AgentState (Pydantic) 转换为 WorkflowState (dict)。

    用于兼容旧 orchestrator 的输出。
    """
    if hasattr(agent_state, "model_dump"):
        data = agent_state.model_dump()
    else:
        data = dict(agent_state) if agent_state else {}

    state = create_empty_state()
    field_mapping = {
        "raw_user_input": "user_input",
        "normalized_input": "normalized_input",
        "turn_type": "turn_type",
        "intent": "consultation_request",
        "risk_level": "risk_level",
        "missing_fields": "normalized_input.missing_fields",
        "symbolic_outputs": "symbolic_result",
        "retrieved_evidence": "domain_rag_result.evidence",
        "specialist_outputs": "specialist_outputs",
        "draft_report": "synthesis.synthesis_text",
        "final_report": "final_report.report_text",
        "critic_result": "evaluation",
        "revision_plan": "execution_plan",
    }

    for old_key, new_key in field_mapping.items():
        value = data.get(old_key)
        if value is not None:
            if "." in new_key:
                parts = new_key.split(".")
                if parts[0] not in state or not isinstance(state[parts[0]], dict):
                    state[parts[0]] = {}
                state[parts[0]][parts[1]] = value
            else:
                state[new_key] = value

    return state


def workflow_to_agent_state(workflow_state: WorkflowState) -> dict:
    """将 WorkflowState 转换为旧 AgentState 兼容格式。

    用于旧测试和 API 的向后兼容。
    """
    return {
        "raw_user_input": workflow_state.get("user_input", ""),
        "normalized_input": workflow_state.get("user_input", ""),
        "turn_type": workflow_state.get("turn_type"),
        "intent": workflow_state.get("consultation_request", {}).get("consultation_intent"),
        "risk_level": workflow_state.get("risk_level", 4),
        "symbolic_outputs": workflow_state.get("symbolic_result", {}),
        "retrieved_evidence": workflow_state.get("domain_rag_result", {}).get("evidence", []),
        "specialist_outputs": workflow_state.get("specialist_outputs", {}),
        "final_report": workflow_state.get("final_report", {}).get("report_text", ""),
        "critic_result": workflow_state.get("evaluation", {}),
        "selected_nodes": workflow_state.get("execution_plan", {}).get("active_nodes", []),
    }
