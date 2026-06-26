from typing import Any

from yhj_agent.shared.schemas.rag import EvidenceItem, RefinedQuery


class QueryRefiner:
    """根据用户输入、工具输出和粗检索证据生成 refined queries。"""

    def refine(
        self,
        user_question: str,
        symbolic_outputs: dict[str, Any] | None = None,
        coarse_evidence: list[EvidenceItem] | None = None,
        user_context: dict[str, Any] | None = None,
    ) -> list[RefinedQuery]:
        queries = [
            RefinedQuery(
                query=user_question,
                reason="original user question",
            )
        ]
        if any(keyword in user_question for keyword in ("事业", "职业", "工作", "转行")):
            queries.append(
                RefinedQuery(
                    query=f"{user_question} career planning skills transition uncertainty",
                    reason="add English career retrieval terms for public datasets",
                )
            )

        symbolic_outputs = symbolic_outputs or {}
        if symbolic_outputs:
            symbolic_terms = " ".join(str(value) for value in symbolic_outputs.values() if value)
            if symbolic_terms:
                queries.append(
                    RefinedQuery(
                        query=f"{user_question} {symbolic_terms}",
                        reason="include symbolic outputs for grounded retrieval",
                    )
                )

        user_context = user_context or {}
        if user_context.get("target_role"):
            queries.append(
                RefinedQuery(
                    query=f"{user_question} {user_context['target_role']} skills career path",
                    reason="include target role from user context",
                )
            )

        coarse_evidence = coarse_evidence or []
        if coarse_evidence:
            top_titles = " ".join(evidence.title for evidence in coarse_evidence[:2])
            queries.append(
                RefinedQuery(
                    query=f"{user_question} {top_titles}",
                    reason="expand using coarse evidence titles",
                )
            )

        return self._dedupe_queries(queries)

    def _dedupe_queries(self, queries: list[RefinedQuery]) -> list[RefinedQuery]:
        seen: set[str] = set()
        deduped: list[RefinedQuery] = []
        for query in queries:
            if query.query in seen:
                continue
            seen.add(query.query)
            deduped.append(query)
        return deduped
