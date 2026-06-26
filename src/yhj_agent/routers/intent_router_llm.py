"""Node2: IntentRouter - LLM-driven intent routing."""
from __future__ import annotations

from typing import Any

from yhj_agent.common.prompt_loader import PromptLoader
from yhj_agent.services.llm_client import MimoLLMClient
from yhj_agent.shared.schemas.llm import LLMMessage, LLMRequest
from yhj_agent.shared.schemas.routing import IntentRouterOutput


class IntentRouterLLM:
    """LLM-driven intent router."""

    def __init__(
        self,
        llm_client: MimoLLMClient | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self.llm = llm_client or MimoLLMClient()
        self.prompts = prompt_loader or PromptLoader()

    def route(
        self,
        normalized_input: dict[str, Any],
        question: str,
        previous_state: dict[str, Any] | None = None,
        last_session_metadata: dict[str, Any] | None = None,
    ) -> IntentRouterOutput:
        """执行意图路由。"""
        template_vars = {
            "question": question,
            "birth_year": normalized_input.get("birth_year", 0),
            "birth_month": normalized_input.get("birth_month", 0),
            "birth_day": normalized_input.get("birth_day", 0),
            "birth_hour": normalized_input.get("birth_hour", ""),
            "birth_location": normalized_input.get("birth_location", ""),
            "gender": normalized_input.get("gender", ""),
            "consultation_goal": (previous_state or {}).get("consultation_request", {}).get("consultation_intent", ""),
            "output_style": normalized_input.get("output_style", "深刻且实际"),
            "missing_fields": normalized_input.get("missing_fields", []),
            "consultation_request": previous_state.get("consultation_request", {}) if previous_state else {},
            "final_report": previous_state.get("final_report", {}) if previous_state else {},
            "last_session_metadata": last_session_metadata or {},
        }

        system = self.prompts.render("intent_router/system.j2")
        user = self.prompts.render("intent_router/user.j2", **template_vars)
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            model=self.llm.config.light_model,
            temperature=0.1,
        )
        return self.llm.chat_structured(request, IntentRouterOutput)
