from pydantic import BaseModel, ConfigDict, Field

from yhj_agent.common.types import PlanType


class TaskPlan(BaseModel):
    """TaskPlanner / DeltaPlanner 的统一计划结构。"""

    model_config = ConfigDict(extra="forbid")

    plan_type: PlanType
    goal: str
    required_capabilities: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    rerun_artifacts: list[str] = Field(default_factory=list)
    reuse_artifacts: list[str] = Field(default_factory=list)
    reason: str


class ToolPolicy(BaseModel):
    """ToolUsePolicy 的结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    allowed_tools: list[str] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    blocked_tools: list[str] = Field(default_factory=list)
    reused_artifacts: list[str] = Field(default_factory=list)
    allowed_nodes: list[str] = Field(default_factory=list)
    blocked_nodes: list[str] = Field(default_factory=list)
    reason: str


class NodeSelection(BaseModel):
    """NodeSelector 的结构化输出。"""

    model_config = ConfigDict(extra="forbid")

    selected_nodes: list[str] = Field(default_factory=list)
    skipped_nodes: list[str] = Field(default_factory=list)
    reason: str


class PlannerExecutorOutput(BaseModel):
    """Planner-Executor 输出（目标架构）。"""

    model_config = ConfigDict(extra="forbid")

    execution_mode: str = "full_execution"  # full_execution | explanation_only | format_only | style_only | safe_completion
    mcp_calls: list[dict] = Field(default_factory=list)
    rag_targets: list[str] = Field(default_factory=list)
    specialist_targets: list[str] = Field(default_factory=list)
    active_nodes: list[str] = Field(default_factory=list)


class BaziResult(BaseModel):
    """八字排盘结果。"""

    model_config = ConfigDict(extra="forbid")

    four_pillars: dict = Field(default_factory=dict)  # {year, month, day, hour} 各含 {heavenly_stem, earthly_branch}
    five_elements: dict = Field(default_factory=dict)  # {wood, fire, earth, metal, water}
    day_master: str = ""
    day_master_strength: str = ""  # "strong" | "weak"
    favorable_elements: list[str] = Field(default_factory=list)
    unfavorable_elements: list[str] = Field(default_factory=list)
    ten_gods: dict = Field(default_factory=dict)
    major_cycles: list[dict] = Field(default_factory=list)
    current_cycle: dict = Field(default_factory=dict)
    current_year: dict = Field(default_factory=dict)
    confidence: str = "deterministic"


class SymbolicOutput(BaseModel):
    """符号计算输出。"""

    model_config = ConfigDict(extra="forbid")

    subsystem: str = "bazi"  # "bazi" | "ziwei" | "yijing"
    result: BaziResult = Field(default_factory=BaziResult)
    confidence: str = "deterministic"
    computation_time_ms: int = 0


