"""LLM-driven specialist agents and concurrent specialist subgraph."""
from __future__ import annotations

import asyncio
from typing import Any

from langsmith import Client as LangSmithClient
from pydantic import BaseModel

from yhj_agent.common.prompt_loader import PromptLoader
from yhj_agent.services.llm_client import MimoLLMClient
from yhj_agent.shared.schemas.agent import (
    PracticalAdvisorOutput,
    PsychologySupportOutput,
    SpecialistOutput,
    SymbolicInterpreterOutput,
)
from yhj_agent.shared.schemas.llm import LLMMessage, LLMRequest


SPECIALIST_CONFIGS = {
    "symbolic_interpreter": {
        "system_template": "specialists/symbolic_interpreter/system.j2",
        "user_template": "specialists/symbolic_interpreter/user.j2",
        "output_schema": SymbolicInterpreterOutput,
        "context_keys": ["symbolic_result", "pruned_context", "relevant_memories", "question", "metaphysical_subsystem"],
    },
    "psychology_support": {
        "system_template": "specialists/psychology_support/system.j2",
        "user_template": "specialists/psychology_support/user.j2",
        "output_schema": PsychologySupportOutput,
        "context_keys": ["pruned_context", "relevant_memories", "question"],
    },
    "practical_advisor": {
        "system_template": "specialists/practical_advisor/system.j2",
        "user_template": "specialists/practical_advisor/user.j2",
        "output_schema": PracticalAdvisorOutput,
        "context_keys": ["symbolic_result", "pruned_context", "question"],
    },
}


class SpecialistAgentLLM:
    """Single specialist agent."""

    def __init__(
        self,
        role: str,
        llm_client: MimoLLMClient | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        if role not in SPECIALIST_CONFIGS:
            raise ValueError(f"Unknown specialist role: {role}")

        self.role = role
        self.config = SPECIALIST_CONFIGS[role]
        self.llm = llm_client or MimoLLMClient()
        self.prompts = prompt_loader or PromptLoader()

    def run_sync(self, context: dict[str, Any], user_profile: dict[str, Any] | None = None) -> SpecialistOutput:
        """Execute one specialist synchronously."""
        template_vars = {key: context.get(key) for key in self.config["context_keys"]}
        template_vars["question"] = context.get("question", "")
        template_vars["knowledge_level"] = (user_profile or {}).get("knowledge_level", "beginner")
        template_vars["preferred_style"] = (user_profile or {}).get("preferred_style", "深刻且实际")
        template_vars["current_year"] = __import__("datetime").date.today().year

        system = self.prompts.render(self.config["system_template"], **template_vars)
        user = self.prompts.render(self.config["user_template"], **template_vars)
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            model=self.llm.config.light_model,
            temperature=0.7,
        )
        result = self.llm.chat_structured(request, self.config["output_schema"])

        return SpecialistOutput(
            agent_name=f"{self.role}_agent",
            content=result.model_dump(),
            claims=self._extract_claims(result),
            confidence=result.confidence if hasattr(result, "confidence") else 0.7,
            risk_flags=[],
            evidence_refs=self._extract_evidence_refs(result),
        )

    async def run(self, context: dict[str, Any], user_profile: dict[str, Any] | None = None) -> SpecialistOutput:
        return self.run_sync(context, user_profile)

    def _extract_claims(self, result: BaseModel) -> list[str]:
        if isinstance(result, SymbolicInterpreterOutput):
            return [result.interpretation_text] if result.interpretation_text else []
        if isinstance(result, PsychologySupportOutput):
            return [result.support_text] if result.support_text else []
        if isinstance(result, PracticalAdvisorOutput):
            return result.key_suggestions or []
        return []

    def _extract_evidence_refs(self, result: BaseModel) -> list[str]:
        if isinstance(result, SymbolicInterpreterOutput):
            return [f.get("evidence_ref", "") for f in result.key_findings if isinstance(f, dict)]
        return []


class SpecialistSubgraphLLM:
    """Execute the three specialist agents concurrently."""

    def __init__(
        self,
        llm_client: MimoLLMClient | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self.llm = llm_client or MimoLLMClient()
        self.prompts = prompt_loader or PromptLoader()

    async def run(
        self,
        context_slices: dict[str, Any] | None = None,
        user_profile: dict[str, Any] | None = None,
        parent_run_id=None,
        **kwargs,
    ) -> dict[str, SpecialistOutput]:
        if context_slices is None:
            context_slices = {}
            req = kwargs.get("consultation_request", {})
            sym = kwargs.get("symbolic_result", {})
            pruned_context = kwargs.get("pruned_context", {}) or {}
            question = kwargs.get("question", "")
            for role in kwargs.get("specialist_targets", list(SPECIALIST_CONFIGS.keys())):
                context_slices[role] = {
                    "payload": {
                        "symbolic_result": sym,
                        "pruned_context": pruned_context,
                        "relevant_memories": [],
                        "question": question,
                        "metaphysical_subsystem": req.get("metaphysical_subsystem", "bazi"),
                    }
                }

        agents = {role: SpecialistAgentLLM(role, self.llm, self.prompts) for role in SPECIALIST_CONFIGS}

        def build_context(role: str) -> dict[str, Any]:
            slice_data = context_slices.get(role, {})
            if isinstance(slice_data, dict) and "payload" in slice_data:
                return slice_data["payload"]
            return slice_data

        try:
            ls_client = LangSmithClient()
        except Exception:
            ls_client = None

        async def _run_with_trace(role: str) -> SpecialistOutput:
            agent = agents[role]
            ctx = build_context(role)
            run_id = None
            if ls_client:
                try:
                    run = ls_client.create_run(
                        name=role,
                        run_type="chain",
                        inputs={"role": role, "context_keys": list(ctx.keys())},
                        parent_run_id=parent_run_id,
                        project_name="yhj-agent-trace-7paths",
                    )
                    run_id = run.id
                except Exception:
                    pass
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, agent.run_sync, ctx, user_profile)
                if ls_client and run_id:
                    try:
                        ls_client.update_run(
                            run_id,
                            outputs={
                                "role": role,
                                "claims": result.claims if hasattr(result, "claims") else [],
                                "confidence": result.confidence if hasattr(result, "confidence") else 0,
                                "risk_flags": result.risk_flags if hasattr(result, "risk_flags") else [],
                            },
                        )
                    except Exception:
                        pass
                return result
            except Exception as exc:
                if ls_client and run_id:
                    try:
                        ls_client.update_run(run_id, error=str(exc))
                    except Exception:
                        pass
                raise

        results = await asyncio.gather(
            _run_with_trace("symbolic_interpreter"),
            _run_with_trace("psychology_support"),
            _run_with_trace("practical_advisor"),
        )
        return {
            "symbolic_interpreter": results[0],
            "psychology_support": results[1],
            "practical_advisor": results[2],
        }
