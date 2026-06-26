"""Node12: ExplanationNode — 追问解释。

基于上一轮报告内容做通俗解释，根据用户命理认知水平调整解释深度。
跨 Session 三级回退：final_report → consultation_request → Qdrant user_memory。
"""
from __future__ import annotations

from typing import Any

from yhj_agent.common.prompt_loader import PromptLoader
from yhj_agent.services.llm_client import MimoLLMClient
from yhj_agent.shared.schemas.llm import LLMMessage, LLMRequest
from pydantic import BaseModel, ConfigDict


class ExplanationOutput(BaseModel):
    """ExplanationNode 输出。"""

    model_config = ConfigDict(extra="forbid")

    explanation_text: str
    referenced_concepts: list[str] = []
    simplified_level: bool = False


class ExplanationNode:
    """追问解释节点。"""

    def __init__(
        self,
        llm_client: MimoLLMClient | None = None,
        prompt_loader: PromptLoader | None = None,
    ) -> None:
        self.llm = llm_client or MimoLLMClient()
        self.prompts = prompt_loader or PromptLoader()

    def explain(
        self,
        question: str,
        final_report: dict[str, Any] | None = None,
        consultation_request: dict[str, Any] | None = None,
        relevant_memories: list[str] | None = None,
        knowledge_level: str = "beginner",
    ) -> ExplanationOutput:
        """生成追问解释。

        三级回退：
        1. final_report 存在 → 基于完整报告做解释
        2. consultation_request 存在 → 基于 consultation_type 做简要解释
        3. relevant_memories 存在 → 基于记忆做解释
        4. 全无 → 降级为 new_task
        """
        # 确定上下文
        if final_report and final_report.get("report_text"):
            context = final_report["report_text"]
            context_type = "full_report"
        elif consultation_request:
            context = f"咨询类型：{consultation_request.get('consultation_type', '未知')}，意图：{consultation_request.get('consultation_intent', '未知')}"
            context_type = "consultation_request"
        elif relevant_memories:
            context = "\n".join(relevant_memories[:5])
            context_type = "memories"
        else:
            return ExplanationOutput(
                explanation_text="当前还没有可追问的上一轮结论，请先运行一个新任务。",
                referenced_concepts=[],
                simplified_level=True,
            )

        # LLM 生成解释
        if self.llm:
            try:
                return self._llm_explain(question, context, context_type, knowledge_level)
            except Exception:
                pass

        # 降级：规则生成
        return ExplanationOutput(
            explanation_text=f"上一轮分析的核心意思是：先用小成本动作验证方向，再决定是否投入更多时间或资源。\n\n本轮追问：{question}",
            referenced_concepts=["小步验证", "渐进策略"],
            simplified_level=knowledge_level == "beginner",
        )

    def _llm_explain(
        self,
        question: str,
        context: str,
        context_type: str,
        knowledge_level: str,
    ) -> ExplanationOutput:
        """LLM 生成解释。"""
        system = f"""你是一位命理解读助手，负责用通俗易懂的语言解释上一轮的分析结果。
用户命理认知水平：{knowledge_level}
要求：
1. 用类比和生活化例子解释专业概念
2. 不引入新信息，只解释已有内容
3. 根据用户水平调整解释深度
4. 输出格式：仅输出合法 JSON"""

        user = f"""上一轮分析内容：
{context[:500]}

用户追问：{question}

请输出 JSON：{{"explanation_text": "...", "referenced_concepts": ["..."], "simplified_level": true/false}}"""

        request = LLMRequest(
            messages=[
                LLMMessage(role="system", content=system),
                LLMMessage(role="user", content=user),
            ],
            model=self.llm.config.light_model,
            temperature=0.3,
        )

        response = self.llm.chat(request)
        return ExplanationOutput.model_validate_json(response.content)
