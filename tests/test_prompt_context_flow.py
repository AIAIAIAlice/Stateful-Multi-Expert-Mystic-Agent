from __future__ import annotations

import pytest
from dataclasses import dataclass

from yhj_agent.common.prompt_loader import PromptLoader
from yhj_agent.orchestrator.state_graph_builder import build_state_graph
from yhj_agent.routers.intent_router_llm import IntentRouterLLM
from yhj_agent.shared.schemas.evaluation import (
    CriticEvaluation,
    DebateOutput,
    FinalReportAPI,
    FinalReportInternal,
    SynthesisResult,
)
from yhj_agent.shared.schemas.llm import LLMResponse
from yhj_agent.shared.schemas.rag import EvidenceItem
from yhj_agent.shared.schemas.routing import IntentRouterOutput, UserProfile


QUESTION = (
    "我想做一个新的事业决策咨询。\n"
    "出生信息：1993年8月15日，巳时，男，杭州。\n"
    "咨询目标：结合八字看未来3年事业发展方向与节奏，帮我判断是继续深耕当前岗位，还是今年开始准备跳槽/转岗。\n"
    "期望风格：深刻且实际。\n"
    "问题：请从运势节奏、关键年份、行动顺序、风险点四个角度，给出可执行建议，并说明哪些判断是高置信、哪些只是趋势参考。"
)


class CaptureLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests = []
        self.config = RaisingLLM._Config()

    def chat(self, request):
        self.requests.append(request)
        return LLMResponse(model="test-model", content=self.content)

    def chat_structured(self, request, response_model):
        self.requests.append(request)
        return response_model.model_validate_json(self.content)


class RaisingLLM:
    class _Config:
        light_model = "test-model"

    def __init__(self) -> None:
        self.config = self._Config()

    def chat(self, request):
        raise RuntimeError("llm unavailable")

    def chat_structured(self, request, response_model):
        raise RuntimeError("llm unavailable")


@dataclass
class FakeProfileMemoryReader:
    def read(self, user_id: str, topics: list[str]):
        return {
            "user_profile": UserProfile().model_dump(),
            "relevant_memories": [],
        }


@dataclass
class FakeMCPServer:
    def call(self, tool: str, params: dict):
        return {
            "day_master": "戊",
            "day_master_strength": "weak",
            "favorable_elements": ["土", "火"],
            "unfavorable_elements": ["金", "水"],
            "current_cycle": {
                "heavenly_stem": "丁",
                "earthly_branch": "巳",
                "start_age": 24,
                "end_age": 33,
            },
            "current_year": {
                "heavenly_stem": "丙",
                "earthly_branch": "午",
                "year": 2026,
            },
            "confidence": "deterministic",
        }


class FakeHybridRetriever:
    def __init__(self, chroma_retriever=None) -> None:
        self.chroma_retriever = chroma_retriever

    def retrieve(self, query: str, rag_targets=None, top_k: int = 3):
        return [
            EvidenceItem(
                doc_id="career-1",
                title="career_doc",
                text="未来3年事业发展建议，包含跳槽转岗节奏与风险控制。",
                source_name="symbolic_rules_index",
                source_url="",
                score=0.91,
                metadata={"source": "career"},
            ),
            EvidenceItem(
                doc_id="career-2",
                title="career_doc_2",
                text="关键年份与行动顺序建议。",
                source_name="symbolic_rules_index",
                source_url="",
                score=0.87,
                metadata={"source": "career"},
            ),
        ]


class FakeSpecialistSubgraphLLM:
    last_kwargs = None

    async def run(self, **kwargs):
        FakeSpecialistSubgraphLLM.last_kwargs = kwargs
        return {
            "symbolic_interpreter": {
                "agent_name": "symbolic_interpreter_agent",
                "claims": ["命理侧认为当前阶段宜稳中求进。"],
                "confidence": 0.7,
                "content": {"interpretation_text": "命理侧认为当前阶段宜稳中求进。"},
            },
            "psychology_support": {
                "agent_name": "psychology_support_agent",
                "claims": ["先降低重大决策压力。"],
                "confidence": 0.7,
                "content": {"support_text": "先降低重大决策压力。"},
            },
            "practical_advisor": {
                "agent_name": "practical_advisor_agent",
                "claims": ["先做低风险探索，再做转岗判断。"],
                "confidence": 0.7,
                "content": {"key_suggestions": ["先做低风险探索，再做转岗判断。"]},
            },
        }


@dataclass
class FakeConflictDebateNode:
    def debate(self, *args, **kwargs):
        return DebateOutput(debate_occurred=False)


@dataclass
class FakeSynthesisNode:
    async def synthesize(self, specialist_outputs, debate_output=None):
        return SynthesisResult(
            synthesis_text="综合分析：先以低风险探索验证转岗方向。",
            main_themes=["事业决策", "低风险探索"],
            expert_consensus=True,
            confidence=0.8,
            consistency_score=0.8,
        )


@dataclass
class FakeCriticEvaluatorLLM:
    def evaluate(self, synthesis, question: str, specialist_outputs):
        return CriticEvaluation(
            evidence_score=4.2,
            safety_score=4.3,
            practicality_score=4.1,
            balance_score=4.0,
            actionability_score=4.2,
            overall_score=4.2,
            need_revision=False,
        )


@dataclass
class FakeReportGeneratorLLM:
    def generate(self, **kwargs):
        return FinalReportInternal(
            api=FinalReportAPI(
                summary="先以低风险探索验证方向。",
                report_text="先以低风险探索验证方向。",
                confidence="medium",
            ),
            raw_input=kwargs.get("raw_input", ""),
            focus_question=kwargs.get("focus_question", ""),
            synthesis=kwargs.get("synthesis", {}),
            evaluation=kwargs.get("evaluation", {}),
            evidence=kwargs.get("specialist_outputs", {}),
            overall_score=4.2,
            safety_score=4.3,
            generation_timestamp="2026-06-14T00:00:00+00:00",
        )


@dataclass
class FakeMemoryWriterEnhanced:
    def write(self, **kwargs):
        return {"status": "success", "written": 0, "skipped_duplicate": 0}


def test_router_uses_current_turn_question_in_prompt():
    llm = CaptureLLM(
        """{
            "turn_type": "new_task",
            "consultation_type": "career",
            "topics": ["career development", "job change"],
            "retrieval_query": "事业发展 跳槽 转岗 风险点",
            "tags": ["career", "bazi_analysis"],
            "consultation_intent": "career_guidance_bazi",
            "metaphysical_subsystem": "bazi",
            "risk_level": 3,
            "safety_flags": [],
            "needs_clarification": false,
            "clarification_fields": [],
            "suggested_route": "normal_planning",
            "response_style": "深刻且实际"
        }"""
    )
    router = IntentRouterLLM(llm_client=llm, prompt_loader=PromptLoader())

    result = router.route(
        normalized_input={
            "birth_year": 1993,
            "birth_month": 8,
            "birth_day": 15,
            "birth_hour": "巳时",
            "birth_location": "杭州",
            "gender": "male",
            "output_style": "深刻且实际",
            "missing_fields": [],
        },
        question=QUESTION,
        previous_state={"user_input": ""},
    )

    assert result.consultation_intent == "career_guidance_bazi"
    rendered_user_prompt = llm.requests[0].messages[1].content
    assert QUESTION in rendered_user_prompt


def test_router_raises_when_llm_unavailable():
    router = IntentRouterLLM(llm_client=RaisingLLM(), prompt_loader=PromptLoader())

    with pytest.raises(RuntimeError):
        router.route(
            normalized_input={
                "birth_year": 1993,
                "birth_month": 8,
                "birth_day": 15,
                "birth_hour": "巳时",
                "birth_location": "杭州",
                "gender": "male",
                "output_style": "深刻且实际",
                "missing_fields": [],
            },
            question=QUESTION,
            previous_state={"user_input": ""},
        )


def test_graph_builds_pruned_context_chunks_and_passes_user_input(monkeypatch):
    import yhj_agent.orchestrator.state_graph_builder as builder

    class FakeRouter:
        def route(self, normalized_input, question: str, previous_state=None, last_session_metadata=None):
            return IntentRouterOutput(
                turn_type="new_task",
                consultation_type="career",
                topics=["career development", "job change"],
                retrieval_query="事业发展 跳槽 转岗 关键年份 风险点",
                tags=["career", "bazi_analysis"],
                consultation_intent="career_guidance_bazi",
                metaphysical_subsystem="bazi",
                risk_level=3,
                safety_flags=[],
                needs_clarification=False,
                clarification_fields=[],
                suggested_route="normal_planning",
                response_style="深刻且实际",
            )

    monkeypatch.setattr(builder, "IntentRouterLLM", FakeRouter)
    monkeypatch.setattr(builder, "ProfileMemoryReader", FakeProfileMemoryReader)
    monkeypatch.setattr(builder, "MCPServer", FakeMCPServer)
    monkeypatch.setattr(builder, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(builder, "ChromaRagRetriever", lambda: object())
    monkeypatch.setattr(builder, "SpecialistSubgraphLLM", FakeSpecialistSubgraphLLM)
    monkeypatch.setattr(builder, "ConflictDebateNode", FakeConflictDebateNode)
    monkeypatch.setattr(builder, "SynthesisNode", FakeSynthesisNode)
    monkeypatch.setattr(builder, "CriticEvaluatorLLM", FakeCriticEvaluatorLLM)
    monkeypatch.setattr(builder, "ReportGeneratorLLM", FakeReportGeneratorLLM)
    monkeypatch.setattr(builder, "MemoryWriterEnhanced", FakeMemoryWriterEnhanced)

    graph = build_state_graph().compile()
    result = graph.invoke({"user_input": QUESTION, "user_id": "session-dfbe5ddc"})

    specialist_kwargs = FakeSpecialistSubgraphLLM.last_kwargs
    assert specialist_kwargs is not None
    assert specialist_kwargs["normalized_input"].get("question") is None
    assert specialist_kwargs["question"] == QUESTION
    assert specialist_kwargs["pruned_context"]["chunks"]
    assert specialist_kwargs["pruned_context"]["chunks"][0]["content"] == "未来3年事业发展建议，包含跳槽转岗节奏与风险控制。"
    assert result["pruned_context"]["chunks"][0]["title"] == "career_doc"
    assert "事业发展" in result["consultation_request"]["retrieval_query"]


def test_graph_raises_when_router_cannot_recover(monkeypatch):
    import yhj_agent.orchestrator.state_graph_builder as builder

    monkeypatch.setattr(builder, "ProfileMemoryReader", FakeProfileMemoryReader)
    monkeypatch.setattr(builder, "MCPServer", FakeMCPServer)
    monkeypatch.setattr(builder, "HybridRetriever", FakeHybridRetriever)
    monkeypatch.setattr(builder, "ChromaRagRetriever", lambda: object())
    monkeypatch.setattr(builder, "SpecialistSubgraphLLM", FakeSpecialistSubgraphLLM)
    monkeypatch.setattr(builder, "ConflictDebateNode", FakeConflictDebateNode)
    monkeypatch.setattr(builder, "SynthesisNode", FakeSynthesisNode)
    monkeypatch.setattr(builder, "CriticEvaluatorLLM", FakeCriticEvaluatorLLM)
    monkeypatch.setattr(builder, "ReportGeneratorLLM", FakeReportGeneratorLLM)
    monkeypatch.setattr(builder, "MemoryWriterEnhanced", FakeMemoryWriterEnhanced)
    monkeypatch.setattr(builder, "IntentRouterLLM", lambda: IntentRouterLLM(llm_client=RaisingLLM(), prompt_loader=PromptLoader()))

    graph = build_state_graph().compile()
    with pytest.raises(RuntimeError):
        graph.invoke({"user_input": QUESTION, "user_id": "session-dfbe5ddc"})
