from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from yhj_agent.common.types import ArtifactStatus, RiskLevel, TurnType


def now_utc() -> datetime:
    """统一生成 UTC 时间，避免各模块自己处理时间格式。"""

    return datetime.now(UTC)


class StrictStateModel(BaseModel):
    """状态模型基类，禁止未登记字段悄悄进入核心 state。"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class StateDeltaItem(StrictStateModel):
    """用户纠错或主题变化导致的单个字段变更。"""

    field_name: str
    old_value: Any = None
    new_value: Any = None
    delta_type: str = "correction"


class ArtifactRecord(StrictStateModel):
    """可缓存、可失效、可复用的中间产物记录。"""

    artifact_id: str
    artifact_type: str
    producer_node: str
    value: Any
    consumed_fields: list[str] = Field(default_factory=list)
    produced_fields: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    status: ArtifactStatus = ArtifactStatus.VALID
    invalidated_reason: str | None = None
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)


class ArtifactState(StrictStateModel):
    """本 session 内可复用 artifact 的统一容器。"""

    artifacts: dict[str, ArtifactRecord] = Field(default_factory=dict)

    def get_valid_artifacts(self) -> dict[str, ArtifactRecord]:
        """只返回仍然有效的 artifact，供复用策略读取。"""

        return {
            artifact_id: artifact
            for artifact_id, artifact in self.artifacts.items()
            if artifact.status == ArtifactStatus.VALID
        }


class TraceEvent(StrictStateModel):
    """单个节点的执行轨迹。"""

    node_name: str
    input_summary: str = ""
    output_summary: str = ""
    latency_ms: int | None = None
    token_input_estimate: int | None = None
    token_output_estimate: int | None = None
    cache_hit: bool = False
    selected_reason: str | None = None
    skipped_reason: str | None = None
    revision_count: int = 0
    error_type: str | None = None
    created_at: datetime = Field(default_factory=now_utc)


class TraceState(StrictStateModel):
    """当前 turn 的 tracing 容器。"""

    events: list[TraceEvent] = Field(default_factory=list)


class AgentState(StrictStateModel):
    """单轮 graph run 内部状态。

    AgentState 不直接持久化；节点只能返回 updates，由 orchestrator 统一合并。
    """

    raw_user_input: str = ""
    normalized_input: str = ""
    turn_type: TurnType | None = None
    intent: str | None = None
    risk_level: RiskLevel = RiskLevel.NORMAL
    missing_fields: list[str] = Field(default_factory=list)
    uncertainty_result: dict[str, Any] = Field(default_factory=dict)
    state_delta: list[StateDeltaItem] = Field(default_factory=list)
    invalidated_artifacts: list[str] = Field(default_factory=list)
    reusable_artifacts: list[str] = Field(default_factory=list)
    task_plan: dict[str, Any] = Field(default_factory=dict)
    tool_policy: dict[str, Any] = Field(default_factory=dict)
    selected_nodes: list[str] = Field(default_factory=list)
    symbolic_outputs: dict[str, Any] = Field(default_factory=dict)
    retrieved_evidence: list[dict[str, Any]] = Field(default_factory=list)
    evidence_map: dict[str, Any] = Field(default_factory=dict)
    context_slices: dict[str, Any] = Field(default_factory=dict)
    specialist_outputs: dict[str, Any] = Field(default_factory=dict)
    draft_report: str | None = None
    critic_result: dict[str, Any] = Field(default_factory=dict)
    revision_plan: dict[str, Any] = Field(default_factory=dict)
    final_report: str | None = None
    errors: list[dict[str, Any]] = Field(default_factory=list)
    trace_metadata: dict[str, Any] = Field(default_factory=dict)


class SessionState(StrictStateModel):
    """跨轮会话状态，只能由 SessionStateReader / Writer 读写。"""

    session_id: str
    turn_id: int = 0
    conversation_summary: str = ""
    current_topic: str | None = None
    last_intent: str | None = None
    last_final_report: str | None = None
    pending_flow: dict[str, Any] | None = None
    symbolic_outputs: dict[str, Any] = Field(default_factory=dict)
    retrieved_evidence: list[dict[str, Any]] = Field(default_factory=list)
    reusable_artifacts: list[str] = Field(default_factory=list)
    invalidated_artifacts: list[str] = Field(default_factory=list)
    artifact_dependency_map: dict[str, list[str]] = Field(default_factory=dict)
    user_preferences: dict[str, Any] = Field(default_factory=dict)
    artifact_state: ArtifactState = Field(default_factory=ArtifactState)
    updated_at: datetime = Field(default_factory=now_utc)


class MemoryState(StrictStateModel):
    """长期记忆状态，只保存稳定、非敏感、已确认的信息。"""

    session_summary: str = ""
    stable_user_preferences: dict[str, Any] = Field(default_factory=dict)
    broad_topic_interests: list[str] = Field(default_factory=list)
    source_turn_id: int | None = None
    created_at: datetime = Field(default_factory=now_utc)
    updated_at: datetime = Field(default_factory=now_utc)

