"""LangGraph StateGraph 编排器。

使用 LangGraph 的 StateGraph 构建完整的 Agent 执行图。
支持 add_node / add_edge / add_conditional_edges / interrupt / Checkpointer。
"""
from __future__ import annotations

import asyncio
from typing import Any

from langgraph.graph import END, StateGraph

from yhj_agent.agents.specialist_llm import SpecialistSubgraphLLM
from yhj_agent.common.prompt_loader import PromptLoader
from yhj_agent.evaluators.critic_evaluator_llm import CriticEvaluatorLLM
from yhj_agent.memory.memory_writer_enhanced import MemoryWriterEnhanced
from yhj_agent.nodes.clarification_node import ClarificationNode
from yhj_agent.nodes.conflict_debate_node import ConflictDebateNode
from yhj_agent.nodes.explanation_node import ExplanationNode
from yhj_agent.nodes.input_normalizer import InputNormalizer
from yhj_agent.nodes.planner_executor import PlannerExecutor
from yhj_agent.nodes.profile_memory_reader import ProfileMemoryReader
from yhj_agent.nodes.report_compressor import ReportCompressor
from yhj_agent.nodes.report_generator_llm import ReportGeneratorLLM
from yhj_agent.nodes.synthesis_node import SynthesisNode
from yhj_agent.rag.hybrid_retriever import HybridRetriever
from yhj_agent.rag.retriever import ChromaRagRetriever
from yhj_agent.rag.query_refiner import QueryRefiner
from yhj_agent.routers.intent_router_llm import IntentRouterLLM
from yhj_agent.state.workflow_state import WorkflowState, create_empty_state, merge_state
from yhj_agent.tools.mcp_server import MCPServer



def _to_dict(obj):
    """Convert Pydantic model to dict, pass through dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj

def build_state_graph() -> StateGraph:
    """构建完整的 LangGraph StateGraph。"""

    # ── 节点函数 ─────────────────────────────────────────────

    def input_normalizer_node(state: WorkflowState) -> dict:
        """Node1: 输入归一化。"""
        normalizer = InputNormalizer()
        raw = _parse_raw_input(state.get("user_input", ""))

        prev_norm = state.get("normalized_input") or {}
        if prev_norm and not raw.get("birth_year"):
            for field in ("birth_year", "birth_month", "birth_day", "birth_hour", "birth_location", "gender"):
                if not raw.get(field) and prev_norm.get(field):
                    raw[field] = prev_norm[field]

        result = normalizer.normalize(raw)
        return {
            "normalized_input": result.model_dump(),
            "previous_normalized_input": prev_norm or state.get("previous_normalized_input"),
            "user_id": str(raw.get("user_id", "") or state.get("user_id", "")),
            "turn_type": "",
            "retry_count": 0,
            "best_result": None,
            "is_degraded": False,
        }

    def intent_router_node(state: WorkflowState) -> dict:
        """Node2: 意图路由 (LLM)。

        Step 1.5: 跨 session 上下文恢复 — 如果 consultation_request 为空（新 session），
        从 ChromaDB user_memory 查询该 user_id 最近的 session_metadata。
        """
        # Step 1.5: 跨 session 上下文恢复
        last_session_metadata = state.get("last_session_metadata")
        if not state.get("consultation_request") and not last_session_metadata:
            user_id = state.get("user_id", "")
            if user_id:
                try:
                    from yhj_agent.rag.retriever import ChromaRagRetriever
                    retriever = ChromaRagRetriever()
                    if retriever.client:
                        coll = retriever.client.get_or_create_collection("user_memory")
                        results = coll.query(
                            query_texts=["session_metadata"],
                            n_results=5,
                            where={"$and": [
                                {"user_id": user_id},
                                {"type": "session_metadata"},
                            ]},
                        )
                        if results and results.get("metadatas") and results["metadatas"][0]:
                            # Sort by timestamp descending to get the most recent
                            metas = results["metadatas"][0]
                            metas_sorted = sorted(
                                metas,
                                key=lambda m: m.get("timestamp", ""),
                                reverse=True,
                            )
                            meta = metas_sorted[0]
                            import json
                            content_val = meta.get("content", "")
                            if content_val:
                                last_session_metadata = json.loads(content_val)
                except Exception:
                    pass

        # Step 2: LLM 意图路由
        router = IntentRouterLLM()
        result = router.route(
            state.get("normalized_input", {}),
            question=state.get("user_input", ""),
            previous_state=state if state.get("consultation_request") else None,
            last_session_metadata=last_session_metadata,
        )

        turn_type = result.turn_type

        # 澄清恢复后，强制走 new_task + full_execution（不走 follow_up/explanation）
        clarification_resolved = state.get("clarification_resolved", False)
        is_clarify_resume = clarification_resolved or (
            state.get("clarification_count", 0) > 0
            and state.get("clarification_answer", "")
            and turn_type in ("follow_up_question", "correction", "missing_field")
        )
        if is_clarify_resume:
            turn_type = "new_task"
            result.needs_clarification = False

        
        # 防御：missing_fields 为空时，不允许 LLM 擅自开启澄清
        if not state.get("normalized_input", {}).get("missing_fields"):
            result.needs_clarification = False
            result.clarification_fields = []
        if turn_type == "new_task" and state.get("normalized_input", {}).get("missing_fields"):
            # 安全优先：高风险时不强制走澄清
            if result.risk_level < 9 and "self_harm_risk" not in (result.safety_flags or []):
                turn_type = "missing_field"
                result.needs_clarification = True

        updates = {
            "turn_type": turn_type,
            "consultation_request": result.model_dump(),
            "risk_level": result.risk_level,
            "needs_clarification": result.needs_clarification,
        }
        if last_session_metadata:
            updates["last_session_metadata"] = last_session_metadata

        return updates

    def profile_memory_reader_node(state: WorkflowState) -> dict:
        """Node3: 用户画像 + 长期记忆读取。"""
        reader = ProfileMemoryReader()
        user_id = state.get("user_id", "")
        topics = state.get("consultation_request", {}).get("topics", [])
        result = reader.read(user_id, topics)
        return {
            "user_profile": result["user_profile"],
            "relevant_memories": result["relevant_memories"],
        }

    def planner_executor_node(state: WorkflowState) -> dict:
        """Node4: 执行模式决策 + 子图实例化。"""
        planner = PlannerExecutor()
        req = state.get("consultation_request", {})
        result = planner.plan(
            consultation_intent=req.get("consultation_intent", ""),
            turn_type=state.get("turn_type", ""),
            risk_level=state.get("risk_level", 4),
            normalized_input=state.get("normalized_input"),
        )
        return {
            "execution_plan": result.model_dump(),
            "execution_mode": result.execution_mode,
        }

    def symbolic_calculator_node(state: WorkflowState) -> dict:
        """Node5: 符号计算 (MCP)。"""
        planner_output = state.get("execution_plan", {})
        mcp_calls = planner_output.get("mcp_calls", [])
        if not mcp_calls:
            return {"symbolic_result": {}}
        mcp = MCPServer()
        call = mcp_calls[0]
        result = mcp.call(call["tool"], call.get("params", {}))
        if isinstance(result, dict) and "result" in result:
            merged = dict(result["result"])
            merged["computation_time_ms"] = result.get("computation_time_ms", 0)
            merged["confidence"] = result.get("confidence", "")
            result = merged
        # Circuit breaker: detect bazi calculation errors
        symbolic_error = False
        if isinstance(result, dict) and result.get("confidence") == "error":
            symbolic_error = True
        return {"symbolic_result": result, "symbolic_error": symbolic_error}

    def domain_rag_node(state: WorkflowState) -> dict:
        """Node6: 领域 RAG 检索。"""
        planner_output = state.get("execution_plan", {})
        rag_targets = planner_output.get("rag_targets", [])
        consultation_req = state.get("consultation_request", {})
        query = consultation_req.get("retrieval_query", "")
        if not query or not rag_targets:
            return {"domain_rag_result": {}}
        retriever = HybridRetriever(chroma_retriever=ChromaRagRetriever())
        refiner = QueryRefiner()
        sym_result = state.get("symbolic_result", {})
        refined_query = refiner.refine(query, sym_result if isinstance(sym_result, dict) else None)
        queries = refined_query if isinstance(refined_query, list) else [refined_query]
        all_results = []
        for rq in queries:
            q = rq.query if hasattr(rq, 'query') else str(rq)
            all_results.extend(retriever.retrieve(q, rag_targets=rag_targets, top_k=3))
        deduped = {}
        for item in all_results:
            existing = deduped.get(item.doc_id)
            if existing is None or item.score > existing.score:
                deduped[item.doc_id] = item
        results = sorted(deduped.values(), key=lambda item: item.score, reverse=True)[:3]
        pruned_context = {
            "chunks": [
                {
                    "content": item.text,
                    "title": item.title,
                    "score": item.score,
                    "source_name": item.source_name,
                    "source_url": item.source_url,
                    "doc_id": item.doc_id,
                }
                for item in results
            ]
        }
        return {"domain_rag_result": results, "pruned_context": pruned_context}

    def specialist_subgraph_node(state: WorkflowState) -> dict:
        """Node7: 专家子图 (LLM)。"""
        specialist = SpecialistSubgraphLLM()
        planner_output = state.get("execution_plan", {})
        specialist_targets = planner_output.get("specialist_targets", [])
        req = state.get("consultation_request", {})
        symbolic_result = state.get("symbolic_result", {})
        pruned_context = state.get("pruned_context", {})
        user_profile = state.get("user_profile", {})
        results = asyncio.run(specialist.run(
            specialist_targets=specialist_targets,
            consultation_request=req,
            symbolic_result=symbolic_result,
            pruned_context=pruned_context,
            question=state.get("user_input", ""),
            normalized_input=state.get("normalized_input", {}),
            user_profile=user_profile,
        ))
        # Convert SpecialistOutput Pydantic models to dicts for downstream compatibility
        normalized = {}
        for role, output in results.items():
            if hasattr(output, "model_dump"):
                normalized[role] = output.model_dump()
            elif isinstance(output, dict):
                normalized[role] = output
            else:
                normalized[role] = {}
        return {"specialist_outputs": normalized}

    def conflict_debate_node(state: WorkflowState) -> dict:
        """Node7.5: 冲突辩论。"""
        node = ConflictDebateNode()
        specialist_outputs = state.get("specialist_outputs", {})
        result = node.debate(
            specialist_outputs,
            symbolic_result=state.get("symbolic_result"),
            pruned_context=state.get("domain_rag_result"),
            revision_feedback=state.get("revision_feedback", ""),
            user_question=state.get("user_input", ""),
        )
        debate = result.model_dump() if hasattr(result, "model_dump") else result
        return {"debate_output": debate}

    def synthesis_node(state: WorkflowState) -> dict:
        """Node8: 综合 (LLM)。"""
        node = SynthesisNode()
        specialist_outputs = state.get("specialist_outputs", {})
        debate_output = state.get("debate_output", {})
        result = asyncio.run(node.synthesize(
            specialist_outputs=specialist_outputs,
            debate_output=debate_output,
        ))
        return {"synthesis": _to_dict(result)}

    def critic_evaluator_node(state: WorkflowState) -> dict:
        """Node9: 评审 (LLM)。"""
        evaluator = CriticEvaluatorLLM()
        synthesis = state.get("synthesis", {})
        specialist_outputs = state.get("specialist_outputs", {})
        normalized_input = state.get("normalized_input", {})
        result = evaluator.evaluate(
            synthesis=synthesis,
            question=state.get("user_input", ""),
            specialist_outputs=specialist_outputs,
        )
        return {"evaluation": result.model_dump(), "retry_count": state.get("retry_count", 0) + 1}

    def report_generator_node(state: WorkflowState) -> dict:
        """Node10: 报告生成 (LLM)。"""
        generator = ReportGeneratorLLM()
        synthesis = state.get("synthesis", {})
        evaluation = state.get("evaluation", {})
        specialist_outputs = state.get("specialist_outputs", {})
        req = state.get("consultation_request", {})
        normalized_input = state.get("normalized_input", {})
        user_profile = state.get("user_profile", {})
        result = generator.generate(
            synthesis=synthesis,
            specialist_outputs=specialist_outputs,
            evaluation=evaluation,
            output_style=req.get("response_style", "深刻且实际"),
            user_profile=user_profile,
            raw_input=state.get("user_input", ""),
            focus_question=state.get("user_input", ""),
            presentation_mode="prose",
        )
        # Store both internal (for audit) and api (for frontend compat)
        internal_dict = _to_dict(result)
        api_dict = _to_dict(result.api)
        return {"final_report": api_dict, "final_report_internal": internal_dict}

    def memory_writer_node(state: WorkflowState) -> dict:
        """Node11: 记忆写入。"""
        writer = MemoryWriterEnhanced()
        user_id = state.get("user_id", "")
        session_id = state.get("session_id", "")
        req = state.get("consultation_request", {})
        final_report = state.get("final_report", {})
        normalized_input = state.get("normalized_input", {})
        result = writer.write(
            user_id=user_id,
            session_id=session_id,
            specialist_outputs=state.get("specialist_outputs", {}),
            evaluation=state.get("evaluation", {}),
            consultation_type=req.get("consultation_type", ""),
            consultation_intent=req.get("consultation_intent", ""),
            question=state.get("user_input", ""),
            final_report=final_report,
        )
        return {"memory_write_result": result}

    def explanation_node(state: WorkflowState) -> dict:
        """Node12: 解释上一轮报告。"""
        node = ExplanationNode()
        req = state.get("consultation_request", {})
        prev_report = state.get("final_report", {})
        normalized_input = state.get("normalized_input", {})
        user_profile = state.get("user_profile", {})
        relevant_memories = state.get("relevant_memories", [])

        # 提取问题和知识水平
        question = normalized_input.get("question", "")
        knowledge_level = user_profile.get("knowledge_level", "beginner")

        result = node.explain(
            question=state.get("user_input", ""),
            final_report=prev_report,
            consultation_request=req,
            relevant_memories=relevant_memories,
            knowledge_level=knowledge_level,
        )

        # 转换为 final_report 格式
        final_report = {
            "report_text": result.explanation_text,
            "disclaimer": prev_report.get("disclaimer", "") if isinstance(prev_report, dict) else "",
            "referenced_concepts": result.referenced_concepts,
        }

        return {"explanation_output": result.model_dump(), "final_report": final_report}

    def report_compressor_node(state: WorkflowState) -> dict:
        """Node13: report compress / format / style rewrite."""
        node = ReportCompressor()
        prev_report = state.get("final_report", {})
        normalized_input = state.get("normalized_input", {})
        execution_plan = state.get("execution_plan", {})
        mode = execution_plan.get("execution_mode", "format_only")

        if isinstance(prev_report, dict):
            report_text = prev_report.get("report_text", "")
        else:
            report_text = str(prev_report)

        question = state.get("user_input", "")
        output_style = state.get("consultation_request", {}).get("response_style", "")

        if mode == "style_only":
            result = node.restyle(report_text=report_text, target_style=output_style or question)
        else:
            format_type = "short"
            if "详细" in question or "完整" in question:
                format_type = "medium"
            elif "要点" in question or "条" in question or "bullet" in question.lower():
                format_type = "bullet"
            result = node.compress(report_text=report_text, format_type=format_type)

        final_report = {
            "report_text": result.compressed_report,
            "disclaimer": prev_report.get("disclaimer", "") if isinstance(prev_report, dict) else "",
            "overall_score": prev_report.get("overall_score", 0.0) if isinstance(prev_report, dict) else 0.0,
            "safety_score": prev_report.get("safety_score", 0.0) if isinstance(prev_report, dict) else 0.0,
            "compression_ratio": result.compression_ratio,
            "format_used": result.format_used,
        }

        return {"final_report": final_report}
    def safe_completion_node(state: WorkflowState) -> dict:
        """Node14: 安全降级回复。"""
        from yhj_agent.common.prompt_loader import PromptLoader as PL
        loader = PL()
        user_text = loader.render(
            "safe_completion/user.j2",
            question=state.get("user_input", ""),
            risk_level=state.get("risk_level", 8),
            safety_flags=state.get("consultation_request", {}).get("safety_flags", []),
        )
        return {"safe_completion_output": {"text": user_text}, "final_report": {"report_text": user_text}}

    def clarification_node(state: WorkflowState) -> dict:
        """Node15: 澄清提问（支持 LangGraph interrupt/resume）。"""
        from langgraph.types import interrupt

        node = ClarificationNode()
        result = node.process(
            missing_fields=state.get("normalized_input", {}).get("missing_fields", []),
            normalized_input=state.get("normalized_input", {}),
            clarification_answer=state.get("clarification_answer", ""),
            clarification_count=state.get("clarification_count", 0),
        )
        updates: dict[str, Any] = {
            "needs_clarification": result.needs_interrupt,
            "clarification_count": state.get("clarification_count", 0) + 1,
        }
        if result.merged_input:
            updates["normalized_input"] = result.merged_input
        if result.degraded:
            updates["degraded_clarification"] = True

        if result.needs_interrupt:
            answer = interrupt({
                "type": "clarification",
                "question": result.clarification_question,
                "missing_fields": state.get("normalized_input", {}).get("missing_fields", []),
            })
            if answer:
                # 使用 updates 中已部分更新的 normalized_input（而非旧 state）
                current_norm = updates.get("normalized_input", state.get("normalized_input", {}))
                merged = node._merge_answer(
                    current_norm,
                    str(answer),
                    current_norm.get("missing_fields", []),
                )
                updates["normalized_input"] = merged
                updates["needs_clarification"] = False
                updates["turn_type"] = "clarification_resolved"
                updates["clarification_answer"] = str(answer)

        return updates

    # ── 条件路由函数 ──────────────────────────────────────────

    def intent_router_routing(state: WorkflowState) -> str:
        """IntentRouter 路由分发。"""
        turn_type = state.get("turn_type", "")
        # 安全优先：高风险/安全干预 优先于 澄清
        if state.get("risk_level", 4) >= 9 or turn_type == "safety_intervention":
            return "safe_completion"
        if state.get("needs_clarification"):
            return "clarification"
        if turn_type == "follow_up_question":
            return "planner_executor"
        if turn_type in ("correction", "topic_switch", "format_refinement"):
            return "planner_executor"
        return "profile_memory_reader"

    def planner_routing(state: WorkflowState) -> list[str] | str:
        """Planner-Executor 分发。full_execution 模式返回 list 触发并行 fan-out。

        优先从 execution_plan.execution_mode 读取（planner_executor_node 本轮写入），
        其次从顶层 execution_mode 读取（可能被 checkpointer 恢复为旧值），最后默认 full_execution。
        """
        plan = state.get("execution_plan", {})
        mode = plan.get("execution_mode") or state.get("execution_mode") or "full_execution"
        if mode == "full_execution":
            return ["symbolic_calculator", "domain_rag"]
        mapping = {
            "explanation_only": "explanation",
            "format_only": "report_compressor",
            "style_only": "report_compressor",
            "safe_completion": "safe_completion",
        }
        return mapping.get(mode, "safe_completion")

    def critic_routing(state: WorkflowState) -> str:
        """CriticEvaluator revision routing (simplified to 2 outputs)."""
        evaluation = state.get("evaluation", {})
        need_revision = evaluation.get("need_revision", False)
        if not need_revision:
            return "report_generator"
        retry_count = state.get("retry_count", 0)
        if retry_count >= 3:
            return "report_generator"
        return "conflict_debate"

    def should_continue_to_specialist(state: WorkflowState) -> str:
        """RAG/符号计算完成后判断是否需要专家子图。"""
        # Circuit breaker: skip specialists if symbolic calculation failed
        if state.get("symbolic_error"):
            return "synthesis"
        planner_output = state.get("execution_plan", {})
        specialist_targets = planner_output.get("specialist_targets", [])
        if specialist_targets:
            return "specialist_subgraph"
        return "synthesis"

    # ── 图构建 ──────────────────────────────────────────────


    def missing_field_check(state: WorkflowState) -> str:
        norm = state.get("normalized_input", {})
        missing = norm.get("missing_fields", [])
        critical = {"birth_year", "birth_month", "birth_day"}
        has_critical = all(norm.get(f) for f in critical)
        if has_critical and set(missing) <= {"birth_location", "birth_hour"}:
            return "intent_router"
        if missing and not has_critical:
            return "clarification"
        return "intent_router"

    graph = StateGraph(WorkflowState)

    graph.add_node("input_normalizer", input_normalizer_node)
    graph.add_node("intent_router", intent_router_node)
    graph.add_node("profile_memory_reader", profile_memory_reader_node)
    graph.add_node("planner_executor", planner_executor_node)
    graph.add_node("symbolic_calculator", symbolic_calculator_node)
    graph.add_node("domain_rag", domain_rag_node)
    graph.add_node("specialist_subgraph", specialist_subgraph_node)
    graph.add_node("conflict_debate", conflict_debate_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("critic_evaluator", critic_evaluator_node)
    graph.add_node("report_generator", report_generator_node)
    graph.add_node("memory_writer", memory_writer_node)
    graph.add_node("explanation", explanation_node)
    graph.add_node("report_compressor", report_compressor_node)
    graph.add_node("safe_completion", safe_completion_node)
    graph.add_node("clarification", clarification_node)

    graph.set_entry_point("input_normalizer")

    graph.add_conditional_edges("input_normalizer", missing_field_check, {
        "intent_router": "intent_router",
        "clarification": "clarification",
    })
    graph.add_edge("profile_memory_reader", "planner_executor")
    graph.add_edge("specialist_subgraph", "conflict_debate")
    graph.add_edge("conflict_debate", "synthesis")
    graph.add_edge("synthesis", "critic_evaluator")

    graph.add_edge("report_generator", "memory_writer")
    graph.add_edge("explanation", "memory_writer")
    graph.add_edge("report_compressor", "memory_writer")
    graph.add_edge("safe_completion", "memory_writer")
    graph.add_edge("memory_writer", END)

    # 条件边：IntentRouter 路由
    graph.add_conditional_edges("intent_router", intent_router_routing, {
        "profile_memory_reader": "profile_memory_reader",
        "planner_executor": "planner_executor",
        "safe_completion": "safe_completion",
        "clarification": "clarification",
    })

    # 条件边：Planner-Executor 分发（full_execution 触发 fan-out）
    graph.add_conditional_edges("planner_executor", planner_routing, {
        "symbolic_calculator": "symbolic_calculator",
        "domain_rag": "domain_rag",
        "explanation": "explanation",
        "report_compressor": "report_compressor",
        "report_generator": "report_generator",
        "safe_completion": "safe_completion",
    })

    # 条件边：CriticEvaluator 修订（简化为 2 出口）
    graph.add_conditional_edges("critic_evaluator", critic_routing, {
        "report_generator": "report_generator",
        "conflict_debate": "conflict_debate",
    })

    # Clarification → IntentRouter（有答案时，重新评估意图）
    graph.add_conditional_edges("clarification", lambda s: "intent_router" if not s.get("needs_clarification") else END, {
        "intent_router": "intent_router",
        END: END,
    })

    # DomainRAG → SpecialistSubgraph 或 Synthesis
    graph.add_conditional_edges("domain_rag", should_continue_to_specialist, {
        "specialist_subgraph": "specialist_subgraph",
        "synthesis": "synthesis",
    })

    # SymbolicCalculator → SpecialistSubgraph 或 Synthesis（与 DomainRAG 并行 fan-out 后汇合）
    graph.add_conditional_edges("symbolic_calculator", should_continue_to_specialist, {
        "specialist_subgraph": "specialist_subgraph",
        "synthesis": "synthesis",
    })

    return graph

def _parse_raw_input(user_input: str) -> dict[str, Any]:
    """解析用户输入为结构化数据。"""
    import re
    result: dict[str, Any] = {"question": user_input}

    date_match = re.search(r"(\d{4})\s*[年\-/\.]\s*(\d{1,2})\s*[月\-/\.]\s*(\d{1,2})", user_input)
    if date_match:
        result["birth_year"] = date_match.group(1)
        result["birth_month"] = date_match.group(2)
        result["birth_day"] = date_match.group(3)

    time_match = re.search(r"(凌晨|早上|上午|中午|下午|晚上|夜里)?\s*(\d{1,2})\s*[点时]", user_input)
    if time_match:
        result["birth_hour"] = f"{time_match.group(1) or ''}{time_match.group(2)}点"
    else:
        for kw in ("子时", "丑时", "寅时", "卯时", "辰时", "巳时", "午时", "未时", "申时", "酉时", "戌时", "亥时"):
            if kw in user_input:
                result["birth_hour"] = kw
                break

    for city in ("北京", "上海", "广州", "深圳", "成都", "重庆", "杭州", "武汉", "南京", "西安",
                 "郑州", "长沙", "合肥", "福州", "济南", "青岛", "大连", "沈阳", "哈尔滨", "长春",
                 "太原", "石家庄", "昆明", "贵阳", "南宁", "兰州", "银川", "西宁", "乌鲁木齐",
                 "呼和浩特", "拉萨", "海口", "三亚", "厦门", "珠海", "佛山", "东莞", "惠州",
                 "中山", "无锡", "苏州", "宁波", "温州", "常州", "南通", "徐州", "嘉兴",
                 "绍兴", "金华", "台州", "泉州", "漳州", "南昌", "九江", "赣州", "洛阳",
                 "开封", "许昌", "新乡", "焦作", "平顶山", "安阳", "信阳", "南阳", "商丘",
                 "周口", "驻马店", "漯河", "濮阳", "鹤壁", "三门峡", "济源"):
        if city in user_input:
            result["birth_location"] = city
            break

    if re.search(r'男伴侣|男伴|男方', user_input):
        result["gender"] = "female"
    elif re.search(r'女伴侣|女伴|女方', user_input):
        result["gender"] = "male"
    elif "男" in user_input:
        result["gender"] = "male"
    elif "女" in user_input:
        result["gender"] = "female"

    return result

