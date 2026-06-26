"""Node10: ReportGenerator - LLM-driven final report generation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from yhj_agent.common.prompt_loader import PromptLoader
from yhj_agent.services.llm_client import MimoLLMClient
from yhj_agent.shared.schemas.evaluation import FinalReport, FinalReportAPI, FinalReportInternal
from yhj_agent.shared.schemas.llm import LLMMessage, LLMRequest


class ReportGeneratorLLM:
    """LLM-driven final report generator."""

    def __init__(
        self,
        llm_client: MimoLLMClient | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self.llm = llm_client or MimoLLMClient()
        self.prompts = prompt_loader or PromptLoader()

    def generate(
        self,
        synthesis: dict[str, Any],
        specialist_outputs: dict[str, Any],
        evaluation: dict[str, Any],
        output_style: str = "深刻且实际",
        user_profile: dict[str, Any] | None = None,
        raw_input: str = "",
        focus_question: str = "",
        presentation_mode: str = "prose",
        special_format_requests: list[str] | None = None,
    ) -> FinalReportInternal:
        """Generate the final frontend-facing report payload."""
        current_year = __import__("datetime").date.today().year
        system = self.prompts.render(
            "report_generator/system.j2",
            current_year=current_year,
            output_style=output_style,
            user_profile=user_profile or {"knowledge_level": "beginner"},
            evaluation=evaluation,
            presentation_mode=presentation_mode,
        )
        user = self.prompts.render(
            "report_generator/user.j2",
            synthesis=synthesis,
            specialists=specialist_outputs,
            output_style=output_style,
            user_profile=user_profile or {"knowledge_level": "beginner"},
            evaluation=evaluation,
            raw_input=raw_input or focus_question,
            focus_question=focus_question,
            presentation_mode=presentation_mode,
            special_format_requests=special_format_requests or [],
        )
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            model=self.llm.config.light_model,
            temperature=0.3,
        )
        api_report = self.llm.chat_structured(request, FinalReportAPI)

        return FinalReportInternal(
            api=api_report,
            raw_input=raw_input,
            focus_question=focus_question,
            output_style=output_style,
            presentation_mode=presentation_mode,
            evidence={
                "symbolic": specialist_outputs.get("symbolic_interpreter", {}),
                "psychology": specialist_outputs.get("psychology_support", {}),
                "practical": specialist_outputs.get("practical_advisor", {}),
            },
            synthesis=synthesis,
            evaluation=evaluation,
            overall_score=evaluation.get("overall_score", 0.0),
            safety_score=evaluation.get("safety_score", 0.0),
            generation_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def generate_compat(
        self,
        synthesis: dict[str, Any],
        specialist_outputs: dict[str, Any],
        evaluation: dict[str, Any],
        output_style: str = "深刻且实际",
        user_profile: dict[str, Any] | None = None,
        raw_input: str = "",
        focus_question: str = "",
        presentation_mode: str = "prose",
        special_format_requests: list[str] | None = None,
    ) -> FinalReport:
        """Compatibility wrapper returning the legacy FinalReport shape."""
        internal = self.generate(
            synthesis=synthesis,
            specialist_outputs=specialist_outputs,
            evaluation=evaluation,
            output_style=output_style,
            user_profile=user_profile,
            raw_input=raw_input,
            focus_question=focus_question,
            presentation_mode=presentation_mode,
            special_format_requests=special_format_requests,
        )
        api = internal.api
        return FinalReport(
            report_text=api.report_text,
            report_sections=[section.title for section in api.sections],
            disclaimer=api.disclaimer,
            overall_score=internal.overall_score,
            safety_score=internal.safety_score,
            confidence=api.confidence,
            generation_timestamp=internal.generation_timestamp,
        )
