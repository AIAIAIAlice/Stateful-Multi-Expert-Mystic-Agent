"""Node9: CriticEvaluator - LLM-driven quality evaluation."""
from __future__ import annotations

import re
from typing import Any

from yhj_agent.common.prompt_loader import PromptLoader
from yhj_agent.services.llm_client import MimoLLMClient
from yhj_agent.shared.schemas.evaluation import CriticEvaluation
from yhj_agent.shared.schemas.llm import LLMMessage, LLMRequest


class CriticEvaluatorLLM:
    """LLM-driven quality evaluator."""

    def __init__(
        self,
        llm_client: MimoLLMClient | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self.llm = llm_client or MimoLLMClient()
        self.prompts = prompt_loader or PromptLoader()

    def evaluate(
        self,
        synthesis: dict[str, Any],
        question: str = "",
        retry_count: int = 0,
        max_retries: int = 2,
        specialist_outputs: dict[str, Any] | None = None,
        domain_rag_result: dict[str, Any] | None = None,
    ) -> CriticEvaluation:
        specialist_outputs = self._normalize_outputs(specialist_outputs)
        evidence_validation = self._validate_evidence_refs(specialist_outputs, domain_rag_result)

        system = self.prompts.render("critic/system.j2")
        user = self.prompts.render(
            "critic/user.j2",
            synthesis=synthesis,
            question=question,
            retry_count=retry_count,
            max_retries=max_retries,
            evidence_validation=evidence_validation,
        )
        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            model=self.llm.config.light_model,
            temperature=0.1,
        )
        result = self.llm.chat_structured(request, CriticEvaluation)

        if result.need_revision:
            result.patch_target = self._determine_patch_target(result)
        if retry_count >= max_retries:
            result.patch_target = "max_revision"
            result.need_revision = False
        return result

    def _determine_patch_target(self, result: CriticEvaluation) -> str:
        if result.overall_score < 3.0:
            return "over_confident"
        if result.safety_score < 4.0:
            return "unsafe_advice"
        if result.evidence_score < 4.0:
            return "missing_evidence"
        if result.balance_score < 4.0:
            return "contradiction"
        if result.practicality_score < 4.0:
            return "too_long"
        if result.actionability_score < 4.0:
            return "format_error"
        return "none"

    @staticmethod
    def _normalize_outputs(specialist_outputs):
        if not specialist_outputs:
            return {}
        normalized = {}
        for role, output in specialist_outputs.items():
            if hasattr(output, "model_dump"):
                normalized[role] = output.model_dump()
            elif isinstance(output, dict):
                normalized[role] = output
            else:
                normalized[role] = {}
        return normalized

    def _validate_evidence_refs(
        self,
        specialist_outputs: dict[str, Any] | None,
        domain_rag_result: dict[str, Any] | None,
    ) -> list[dict[str, str]]:
        if not specialist_outputs:
            return []

        validation_results = []
        rag_chunk_ids: set[str] = set()
        if domain_rag_result:
            for chunk in domain_rag_result.get("retrieved_chunks", []):
                if isinstance(chunk, dict):
                    rag_chunk_ids.add(chunk.get("chunk_id", ""))

        for role, output in specialist_outputs.items():
            evidence_refs = output.get("evidence_refs", [])
            claims = output.get("claims", [])

            for ref in evidence_refs:
                if not ref:
                    continue

                status = "valid"
                issue = ""
                if ref.startswith("field:"):
                    field_path = ref[6:]
                    if not re.match(r"^symbolic_result\.\w+(\.\w+)*$", field_path):
                        status = "invalid"
                        issue = f"无效的 field 路径格式: {field_path}"
                elif ref.startswith("doc:"):
                    parts = ref.split(":")
                    if len(parts) != 3:
                        status = "invalid"
                        issue = f"无效的 doc 引用格式: {ref}"
                    else:
                        chunk_id = parts[2]
                        if rag_chunk_ids and chunk_id not in rag_chunk_ids:
                            status = "missing"
                            issue = f"RAG 结果中不存在 chunk_id: {chunk_id}"
                else:
                    status = "invalid"
                    issue = f"未知的 evidence_ref 前缀: {ref}"

                validation_results.append(
                    {
                        "role": role,
                        "claim": claims[0] if claims else "",
                        "ref": ref,
                        "status": status,
                        "issue": issue,
                    }
                )

        return validation_results
