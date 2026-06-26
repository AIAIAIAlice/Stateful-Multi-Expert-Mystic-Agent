"""Node7.5: ConflictDebateNode - multi-agent conflict debate.

Uses AutoGen GroupChat for expert conflict negotiation.
"""
from __future__ import annotations

import os
from typing import Any

from yhj_agent.common.config import get_mimo_config
from yhj_agent.shared.schemas.evaluation import (
    DebateOutput,
    DebateResolution,
    SpecialistUpdate,
)


class ConflictDebateNode:
    """Multi-agent conflict debate node."""

    def __init__(self) -> None:
        mimo_config = get_mimo_config()
        try:
            api_key = mimo_config.require_api_key()
        except Exception:
            api_key = ""
        self._llm_config = {
            "config_list": [{
                "model": os.getenv("MIMO_DEFAULT_MODEL", mimo_config.default_model),
                "api_base": os.getenv("MIMO_OPENAI_BASE_URL", mimo_config.openai_base_url),
                "api_key": api_key,
            }],
            "temperature": 0.7,
            "timeout": 120,
        }

    @staticmethod
    def _normalize_outputs(specialist_outputs: dict[str, Any]) -> dict[str, dict]:
        """Convert SpecialistOutput Pydantic models to plain dicts."""
        normalized: dict[str, dict] = {}
        for role, output in specialist_outputs.items():
            if hasattr(output, "model_dump"):
                normalized[role] = output.model_dump()
            elif isinstance(output, dict):
                normalized[role] = output
            else:
                normalized[role] = {}
        return normalized

    def debate(
        self,
        specialist_outputs: dict[str, Any],
        symbolic_result: dict[str, Any] | None = None,
        pruned_context: dict[str, Any] | None = None,
        revision_feedback: str = "",
        user_question: str = "",
    ) -> DebateOutput:
        """Execute conflict debate.

        Args:
            specialist_outputs: specialist outputs dict
            symbolic_result: symbolic calculation result (for grounding)
            pruned_context: pruned context (for grounding)
            revision_feedback: CriticEvaluator revision feedback
        Returns:
            DebateOutput
        """
        # 0. Normalize Pydantic models to dicts
        specialist_outputs = self._normalize_outputs(specialist_outputs)

        # 1. Rule pre-screening (force debate when revision_feedback or explicit user request)
        force_keywords = ("辩论", "冲突", "不一致", "争议", "debate", "conflict")
        user_wants_debate = any(kw in user_question for kw in force_keywords)
        if not revision_feedback and not user_wants_debate and not self._should_debate(specialist_outputs):
            return DebateOutput(debate_occurred=False)

        # 2. Try AutoGen debate
        try:
            return self._autogen_debate(specialist_outputs, symbolic_result, pruned_context, revision_feedback)
        except Exception:
            # Fallback: lightweight debate
            return self._lightweight_debate(specialist_outputs, revision_feedback)

    def _should_debate(self, specialist_outputs: dict[str, Any]) -> bool:
        """Rule pre-screening: whether debate is needed."""
        si = specialist_outputs.get("symbolic_interpreter", {})
        ps = specialist_outputs.get("psychology_support", {})
        pa = specialist_outputs.get("practical_advisor", {})

        si_claims = si.get("claims", [])
        pa_claims = pa.get("claims", [])
        ps_claims = ps.get("claims", [])

        # Rule 1: metaphysical recommendation vs practical restraint
        si_action = any(
            w in " ".join(si_claims)
            for w in ("\u5efa\u8bae", "\u9002\u5408", "\u53ef\u4ee5")
        )
        pa_restraint = any(
            w in " ".join(pa_claims)
            for w in ("\u4e0d\u5efa\u8bae", "\u6682\u7f13", "\u8c28\u614e")
        )
        if si_action and pa_restraint:
            return True

        # Rule 2: psychology support negates metaphysical basis
        ps_negate = any("\u4e0d\u9700\u8981" in c for c in ps_claims)
        si_need = any("\u9700\u8981" in c for c in si_claims)
        if ps_negate and si_need:
            return True

        # Rule 3: any agent confidence too low
        confidences = [
            self._parse_confidence(si.get("confidence", "medium")),
            self._parse_confidence(ps.get("confidence", "medium")),
            self._parse_confidence(pa.get("confidence", "medium")),
        ]
        if min(confidences) < 0.6:
            return True

        return False

    def _parse_confidence(self, confidence: str | float) -> float:
        """Parse confidence value."""
        if isinstance(confidence, (int, float)):
            return float(confidence)
        mapping = {"high": 0.9, "medium": 0.7, "low": 0.4}
        return mapping.get(str(confidence).lower(), 0.5)

    def _autogen_debate(
        self,
        specialist_outputs: dict[str, Any],
        symbolic_result: dict[str, Any] | None,
        pruned_context: dict[str, Any] | None,
        revision_feedback: str,
    ) -> DebateOutput:
        """Use AutoGen GroupChat for debate."""
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.conditions import MaxMessageTermination
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        from autogen_ext.models.openai._model_info import ModelInfo

        # Build model client from env config
        model_info = ModelInfo(
            vision=False, function_calling=False, json_output=True,
            family="other", structured_output=True,
        )
        model_client = OpenAIChatCompletionClient(
            model=self._llm_config["config_list"][0]["model"],
            base_url=self._llm_config["config_list"][0]["api_base"],
            api_key=self._llm_config["config_list"][0]["api_key"],
            model_info=model_info,
            temperature=self._llm_config.get("temperature", 0.7),

        )


        # Build agents
        si = AssistantAgent(
            "symbolic_interpreter",
            system_message="You are a senior metaphysical analyst. Defend your interpretation in the debate based on calculation results. Cite specific data. Respond in Chinese. End your final message with TERMINATE.",
            model_client=model_client,
        )
        ps = AssistantAgent(
            "psychology_support",
            system_message="You are a psychological counseling support assistant. Validate claims from a psychology perspective in the debate. Highlight unsupported predictions. Respond in Chinese. End your final message with TERMINATE.",
            model_client=model_client,
        )
        pa = AssistantAgent(
            "practical_advisor",
            system_message="You are a practical life advisor. Challenge impractical suggestions in the debate. Advocate for evidence-based decisions. Respond in Chinese. End your final message with TERMINATE.",
            model_client=model_client,
        )

        conflict_summary = self._build_conflict_summary(specialist_outputs, revision_feedback)
        prompt = (
            f"The following expert outputs have conflicts:\n\n{conflict_summary}\n\n"
            "Please debate and reach consensus. Each expert should state their position, "
            "identify disagreements, and propose compromises. "
            "Finally, output a JSON summary: {\"consensus_reached\": bool, \"resolution_summary\": str, \"synthesis_guidance\": str}"
        )

        team = RoundRobinGroupChat(
            participants=[si, ps, pa],
            max_turns=6,
            termination_condition=MaxMessageTermination(max_messages=10),
        )

        import asyncio
        result = asyncio.run(team.run(task=prompt))

        # Convert TaskResult messages to the format expected by _extract_debate_result
        messages = []
        for msg in result.messages:
            messages.append({"name": msg.source, "content": msg.content})

        return self._extract_debate_result(messages, specialist_outputs)

    def _lightweight_debate(
        self,
        specialist_outputs: dict[str, Any],
        revision_feedback: str,
    ) -> DebateOutput:
        """Lightweight fallback debate (rule-based)."""
        specialist_updates: dict[str, SpecialistUpdate] = {}
        for role, output in specialist_outputs.items():
            claims = output.get("claims", [])
            specialist_updates[role] = SpecialistUpdate(
                role=role,
                original_position="; ".join(claims[:2]),
                revised_position="; ".join(claims[:2]),
                concessions=[],
                maintained_points=claims,
                revised_confidence=self._parse_confidence(output.get("confidence", "medium")),
            )

        return DebateOutput(
            debate_occurred=True,
            conflicts=[],
            rounds_taken=1,
            resolution=DebateResolution(
                consensus_reached=True,
                resolution_summary="Lightweight debate completed, experts maintain original positions." if revision_feedback else "No substantive conflict.",
                synthesis_guidance="Integrate all perspectives, present balanced view.",
            ),
            specialist_updates=specialist_updates,
        )

    def _build_conflict_summary(self, specialist_outputs: dict[str, Any], revision_feedback: str) -> str:
        """Build conflict summary."""
        lines = ["=== Expert Output Summary ==="]
        for role, output in specialist_outputs.items():
            claims = output.get("claims", [])
            lines.append(f"{role}: {'; '.join(claims[:2])}")

        if revision_feedback:
            lines.append(f"\n=== CriticEvaluator Feedback ===\n{revision_feedback}")

        return "\n".join(lines)

    def _extract_debate_result(self, messages: list, specialist_outputs: dict[str, Any]) -> DebateOutput:
        """Extract debate result from AutoGen message history."""
        import json
        import re

        # Extract final messages per role
        role_messages: dict[str, str] = {}
        for msg in messages:
            sender = msg.get("name", "") if isinstance(msg, dict) else getattr(msg, "name", "")
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if sender in ("symbolic_interpreter", "psychology_support", "practical_advisor") and content:
                role_messages[sender] = content

        # Try to parse JSON from last message
        last_content = ""
        for msg in reversed(messages):
            content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            if content and len(content) > 50:
                last_content = content
                break

        consensus_reached = True
        resolution_summary = "Debate completed, experts have stated positions."
        synthesis_guidance = "Integrate all perspectives, present balanced view."

        if last_content:
            try:
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', last_content, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    consensus_reached = parsed.get("consensus_reached", True)
                    resolution_summary = parsed.get("resolution_summary", resolution_summary)
                    synthesis_guidance = parsed.get("synthesis_guidance", synthesis_guidance)
            except (json.JSONDecodeError, AttributeError):
                if "\u5171\u8bc6" in last_content or "\u4e00\u81f4" in last_content:
                    consensus_reached = True
                elif "\u5206\u6b67" in last_content or "\u4e0d\u540c\u610f" in last_content:
                    consensus_reached = False

        # Build specialist_updates
        specialist_updates = {}
        for role in specialist_outputs:
            original = " ".join(specialist_outputs[role].get("claims", []))
            revised = role_messages.get(role, original)

            concessions = []
            maintained = specialist_outputs[role].get("claims", [])

            if role in role_messages:
                msg = role_messages[role]
                if "\u540c\u610f" in msg or "\u8ba4\u53ef" in msg:
                    concessions.append("Acknowledged other experts' partial views.")
                if "\u575a\u6301" in msg or "\u4ecd\u7136\u8ba4\u4e3a" in msg:
                    maintained = specialist_outputs[role].get("claims", [])

            specialist_updates[role] = SpecialistUpdate(
                role=role,
                original_position=original[:200],
                revised_position=revised[:200],
                concessions=concessions,
                maintained_points=maintained,
                revised_confidence=self._parse_confidence(specialist_outputs[role].get("confidence", "medium")),
            )

        return DebateOutput(
            debate_occurred=True,
            conflicts=[],
            rounds_taken=min(len(messages) // 3, 3),
            resolution=DebateResolution(
                consensus_reached=consensus_reached,
                resolution_summary=resolution_summary[:300],
                synthesis_guidance=synthesis_guidance[:300],
            ),
            specialist_updates=specialist_updates,
        )
