from pydantic import BaseModel, ConfigDict, Field


class IntentRouterOutput(BaseModel):
    """IntentRouter LLM 输出（目标架构）。"""

    model_config = ConfigDict(extra="forbid")

    turn_type: str
    turn_signals: list[str] = Field(default_factory=list)
    consultation_type: str
    topics: list[str] = Field(default_factory=list)
    retrieval_query: str = ""
    tags: list[str] = Field(default_factory=list)
    consultation_intent: str = ""
    metaphysical_subsystem: str = "bazi"
    risk_level: int = Field(default=4, ge=1, le=10)
    safety_flags: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_fields: list[str] = Field(default_factory=list)
    suggested_route: str = "normal_planning"
    response_style: str = "深刻且实际"


class UserProfile(BaseModel):
    """用户画像。"""

    model_config = ConfigDict(extra="forbid")

    knowledge_level: str = "beginner"
    metaphysical_familiarity: str = ""
    preferred_style: str = "balanced"
    personality_traits: list[str] = Field(default_factory=list)
    past_consultation_types: list[str] = Field(default_factory=list)
    sensitivity_flags: list[str] = Field(default_factory=list)


class MemoryReadOutput(BaseModel):
    """ProfileMemoryReader 输出。"""

    model_config = ConfigDict(extra="forbid")

    user_profile: UserProfile = Field(default_factory=UserProfile)
    relevant_memories: list[str] = Field(default_factory=list)
