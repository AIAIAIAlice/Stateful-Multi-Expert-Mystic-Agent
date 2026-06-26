from pydantic import BaseModel, ConfigDict, Field

from yhj_agent.common.types import RevisionFailureType


class CriticResult(BaseModel):
    """CriticEvaluator 输出。"""

    model_config = ConfigDict(extra="forbid")

    passed: bool
    failure_type: RevisionFailureType = RevisionFailureType.PASS
    affected_sections: list[str] = Field(default_factory=list)
    target_nodes: list[str] = Field(default_factory=list)
    revision_instruction: str = ""


class RevisionPlan(BaseModel):
    """RevisionRouter 输出。"""

    model_config = ConfigDict(extra="forbid")

    failure_type: RevisionFailureType
    root_cause: str
    revision_target: str
    revision_action: str
    then: str
    stop_condition: str = "max_revision_count <= 1"


class CriticEvaluation(BaseModel):
    """CriticEvaluator LLM 输出（目标架构 5 维评分）。"""

    model_config = ConfigDict(extra="allow")

    evidence_score: float = Field(default=4.0, ge=1.0, le=5.0)
    safety_score: float = Field(default=4.0, ge=1.0, le=5.0)
    practicality_score: float = Field(default=4.0, ge=1.0, le=5.0)
    balance_score: float = Field(default=4.0, ge=1.0, le=5.0)
    actionability_score: float = Field(default=4.0, ge=1.0, le=5.0)
    overall_score: float = Field(default=4.0, ge=1.0, le=5.0)
    need_revision: bool = False
    issues: list[str] = Field(default_factory=list)
    revision_feedback: str = ""
    patch_target: str = "none"



class SynthesisSection(BaseModel):
    """Synthesis ????????????"""

    model_config = ConfigDict(extra="allow")

    summary: str = ""
    key_claims: list[str] = Field(default_factory=list)
    key_actions: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list)
    decision_direction: str = ""
    open_questions: list[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    """SynthesisNode 输出。"""

    model_config = ConfigDict(extra="allow")

    synthesis_text: str = ""
    main_themes: list[str] = Field(default_factory=list)
    expert_consensus: bool = True
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    consistency_score: float = Field(default=1.0, ge=0.0, le=1.0)
    sections: dict[str, SynthesisSection] = Field(default_factory=dict)
    unresolved_conflicts: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    """?????????"""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    text: str = ""


class ActionPlanItem(BaseModel):
    """????????????"""

    model_config = ConfigDict(extra="forbid")

    title: str = ""
    text: str = ""


class VisualBlock(BaseModel):
    """??????????/??????"""

    model_config = ConfigDict(extra="allow")

    type: str = "ascii_bar"  # ascii_bar | timeline | decision_matrix | text
    title: str = ""
    lines: list[str] = Field(default_factory=list)
    items: list[dict] = Field(default_factory=list)


class FinalReportAPI(BaseModel):
    """?????????????????"""

    model_config = ConfigDict(extra="forbid")

    summary: str = ""
    report_text: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    action_plan: list[ActionPlanItem] = Field(default_factory=list)
    visual_blocks: list[VisualBlock] = Field(default_factory=list)
    disclaimer: str = "?????????????????????????????????????????????????????????????"
    confidence: str = "high"


class FinalReportInternal(BaseModel):
    """????????????state/LangSmith/debug??"""

    model_config = ConfigDict(extra="allow")

    api: FinalReportAPI = Field(default_factory=FinalReportAPI)
    raw_input: str = ""
    focus_question: str = ""
    output_style: str = ""
    presentation_mode: str = "prose"
    evidence: dict = Field(default_factory=dict)
    synthesis: dict = Field(default_factory=dict)
    evaluation: dict = Field(default_factory=dict)
    overall_score: float = 0.0
    safety_score: float = 0.0
    generation_timestamp: str = ""


class FinalReport(BaseModel):
    """ReportGenerator ??????????"""

    model_config = ConfigDict(extra="allow")

    report_text: str = ""
    report_sections: list[str] = Field(default_factory=list)
    disclaimer: str = "?????????????????????????????????????????????????????????????"
    overall_score: float = 0.0
    safety_score: float = 0.0
    confidence: str = "high"
    generation_timestamp: str = ""


class DebateResolution(BaseModel):
    """辩论共识结果。"""

    model_config = ConfigDict(extra="forbid")

    consensus_reached: bool = False
    resolution_summary: str = ""
    adopted_positions: dict[str, list[str]] = Field(default_factory=dict)
    rejected_positions: dict[str, list[str]] = Field(default_factory=dict)
    synthesis_guidance: str = ""


class SpecialistUpdate(BaseModel):
    """辩论后专家立场更新。"""

    model_config = ConfigDict(extra="forbid")

    role: str
    original_position: str = ""
    revised_position: str = ""
    concessions: list[str] = Field(default_factory=list)
    maintained_points: list[str] = Field(default_factory=list)
    revised_confidence: float = 0.5


class ConflictPoint(BaseModel):
    """辩论冲突点。"""

    model_config = ConfigDict(extra="forbid")

    conflict_id: str = ""
    description: str = ""
    parties: list[str] = Field(default_factory=list)
    severity: str = "medium"  # "high" | "medium" | "low"


class DebateOutput(BaseModel):
    """ConflictDebateNode 输出。"""

    model_config = ConfigDict(extra="forbid")

    debate_occurred: bool = False
    conflicts: list[ConflictPoint] = Field(default_factory=list)
    rounds_taken: int = 0
    resolution: DebateResolution = Field(default_factory=DebateResolution)
    specialist_updates: dict[str, SpecialistUpdate] = Field(default_factory=dict)


class SafeCompletionOutput(BaseModel):
    """SafeCompletionNode 输出。"""

    model_config = ConfigDict(extra="forbid")

    safe_text: str
    boundary_statement: str = ""
    referral: str = ""
    tone: str = "warm_cautious"

