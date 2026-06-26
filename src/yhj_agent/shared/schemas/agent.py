import json

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContextSlice(BaseModel):
    """为单个 specialist 裁剪后的上下文。"""

    model_config = ConfigDict(extra="forbid")

    role: str
    user_question: str
    payload: dict = Field(default_factory=dict)
    token_estimate: int = 0


class SpecialistOutput(BaseModel):
    """Specialist agent 统一输出。

    content 保留各专家的原始结构化输出（interpretation_text / support_text / key_suggestions 等），
    claims 保留提取的关键结论（用于向后兼容），confidence 为 float 类型。
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    content: dict = Field(default_factory=dict)
    claims: list[str] = Field(default_factory=list)
    confidence: float = 0.7
    risk_flags: list[str] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)


class SymbolicInterpreterOutput(BaseModel):
    """命理解读 Agent LLM 输出。

    兼容 LLM 返回 interpretation_text / analysis / summary 字段。
    """

    model_config = ConfigDict(extra="allow")

    interpretation_text: str = ""
    key_findings: list[dict] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: dict) -> dict:
        """兼容 LLM 返回 analysis/summary 替代 interpretation_text。"""
        if isinstance(data, dict):
            if "interpretation_text" not in data:
                # 尝试从 analysis 或 summary 字段获取
                for alt_field in ["analysis", "summary", "overall_analysis", "professional_analysis"]:
                    if alt_field in data:
                        data["interpretation_text"] = data.pop(alt_field)
                        break
            # interpretation_text: dict → str（LLM 可能返回结构化对象）
            it = data.get("interpretation_text")
            if isinstance(it, dict):
                # 尝试提取文本内容
                data["interpretation_text"] = it.get("text", it.get("content", str(it)))
            elif isinstance(it, list):
                data["interpretation_text"] = str(it)

            # If interpretation_text is still empty, try to build it from professional_analysis
            it_check = data.get("interpretation_text", "")
            pa = data.get("professional_analysis")
            if not it_check and pa:
                if isinstance(pa, dict):
                    parts = []
                    for k, v in pa.items():
                        if isinstance(v, str) and len(v) > 10:
                            parts.append(v)
                        elif isinstance(v, list):
                            for item in v:
                                if isinstance(item, dict):
                                    for fk in ("claim", "key_findings", "analysis", "text", "content"):
                                        if item.get(fk):
                                            parts.append(str(item[fk]))
                                            break
                                elif isinstance(item, str) and len(item) > 5:
                                    parts.append(item)
                    if parts:
                        data["interpretation_text"] = "\n".join(parts)
                elif isinstance(pa, str):
                    data["interpretation_text"] = pa

            
            # If interpretation_text is still empty, build it from key_findings claims
            if not data.get("interpretation_text") and data.get("key_findings"):
                kf_claims = []
                for item in data["key_findings"]:
                    if isinstance(item, dict) and item.get("claim"):
                        kf_claims.append(item["claim"])
                    elif isinstance(item, str):
                        kf_claims.append(item)
                if kf_claims:
                    data["interpretation_text"] = "\n".join(kf_claims)

# key_findings: list[str] → list[dict]
            kf = data.get("key_findings")
            if isinstance(kf, list):
                data["key_findings"] = [
                    {"claim": item} if isinstance(item, str) else item
                    for item in kf
                ]
            # 移除额外字段避免验证错误
            for extra in ["recommendations", "overall_analysis"]:
                data.pop(extra, None)
        return data


class PsychologySupportOutput(BaseModel):
    """心理支持 Agent LLM 输出。

    兼容 LLM 返回 support_text / analysis / summary 字段。
    """

    model_config = ConfigDict(extra="allow")

    support_text: str = ""
    cognitive_reframes: list[str] = Field(default_factory=list)
    emotion_tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: dict) -> dict:
        """兼容 LLM 返回 analysis/summary 替代 support_text。"""
        if isinstance(data, dict):
            if "support_text" not in data:
                for alt_field in ["analysis", "summary", "overall_analysis", "response"]:
                    if alt_field in data:
                        data["support_text"] = data.pop(alt_field)
                        break
            for extra in ["recommendations", "overall_analysis"]:
                data.pop(extra, None)
            # support_text: dict/list -> str (LLM may return nested structures)
            st = data.get("support_text")
            if isinstance(st, dict):
                data["support_text"] = st.get("text", st.get("content", st.get("acknowledgment", json.dumps(st, ensure_ascii=False))))
            elif isinstance(st, list):
                data["support_text"] = "\n".join(str(item) for item in st)
        return data


class PracticalAdvisorOutput(BaseModel):
    """实用建议 Agent LLM 输出。

    兼容 LLM 返回 key_suggestions / suggestions / advice 字段。
    """

    model_config = ConfigDict(extra="allow")

    key_suggestions: list[str] = Field(default_factory=list)
    risk_warnings: list[str] = Field(default_factory=list)
    action_items: list[dict] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_fields(cls, data: dict) -> dict:
        """兼容 LLM 返回 suggestions/advice 替代 key_suggestions。"""
        if isinstance(data, dict):
            # 兼容替代字段名
            if "key_suggestions" not in data:
                for alt_field in ["suggestions", "advice", "recommendations"]:
                    if alt_field in data:
                        data["key_suggestions"] = data.pop(alt_field)
                        break
            # key_suggestions: str / list[dict] → list[str]
            ks = data.get("key_suggestions")
            if isinstance(ks, str):
                data["key_suggestions"] = [ks]
            elif isinstance(ks, dict):
                vals = []
                for v in ks.values():
                    if isinstance(v, list):
                        vals.extend(str(item) for item in v)
                    else:
                        vals.append(str(v))
                data["key_suggestions"] = vals
            elif isinstance(ks, list):
                data["key_suggestions"] = [
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in ks
                ]
            # risk_warnings: str → list[str]
            rw = data.get("risk_warnings")
            if isinstance(rw, str):
                data["risk_warnings"] = [rw]
            elif isinstance(rw, dict):
                vals = []
                for v in rw.values():
                    if isinstance(v, list):
                        vals.extend(str(item) for item in v)
                    else:
                        vals.append(str(v))
                data["risk_warnings"] = vals
            elif isinstance(rw, list):
                data["risk_warnings"] = [
                    item.get("text", str(item)) if isinstance(item, dict) else str(item)
                    for item in rw
                ]
            # action_items: list[str] → list[dict]
            ai = data.get("action_items")
            if isinstance(ai, dict):
                ai_list = []
                for v in ai.values():
                    if isinstance(v, list):
                        ai_list.extend(v)
                    else:
                        ai_list.append(v)
                data["action_items"] = [
                    {"action": item} if isinstance(item, str) else item
                    for item in ai_list
                ]
            elif isinstance(ai, list):
                data["action_items"] = [
                    {"action": item} if isinstance(item, str) else item
                    for item in ai
                ]
            for extra in ["recommendations"]:
                data.pop(extra, None)
        return data

