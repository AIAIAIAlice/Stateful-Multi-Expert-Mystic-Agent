from enum import StrEnum


class TurnType(StrEnum):
    """用户本轮输入类型，由 TurnRouter 统一判断。"""

    NEW_TASK = "new_task"
    FOLLOW_UP_QUESTION = "follow_up_question"
    CLARIFICATION_ANSWER = "clarification_answer"
    CORRECTION = "correction"
    FORMAT_REFINEMENT = "format_refinement"
    TOPIC_SWITCH = "topic_switch"
    SAFETY_INTERVENTION = "safety_intervention"


class RiskLevel(StrEnum):
    """请求风险等级，由 SafetyGate 或相关 router 统一输出。"""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class IntentType(StrEnum):
    """用户意图类型，由 IntentRouter 统一判断。"""

    CAREER_DECISION_SUPPORT = "career_decision_support"
    RELATIONSHIP_GUIDANCE = "relationship_guidance"
    SYMBOLIC_CONSULTATION = "symbolic_consultation"
    EXPLANATION = "explanation"
    FORMAT_REFINEMENT = "format_refinement"
    GENERAL_QA = "general_qa"
    UNKNOWN = "unknown"


class PlanType(StrEnum):
    """规划类型，区分完整任务规划和局部增量规划。"""

    FULL_TASK_PLAN = "full_task_plan"
    DELTA_PLAN = "delta_plan"


class ArtifactStatus(StrEnum):
    """artifact 当前是否可复用。"""

    VALID = "valid"
    INVALID = "invalid"


class MemorySensitivity(StrEnum):
    """长期记忆写入前的敏感性分类。"""

    PUBLIC = "public"
    PREFERENCE = "preference"
    SENSITIVE = "sensitive"
    TEMPORARY = "temporary"


class RevisionFailureType(StrEnum):
    """CriticEvaluator 输出的失败类型。"""

    PASS = "pass"
    MISSING_EVIDENCE = "missing_evidence"
    CONTRADICTION = "contradiction"
    UNSAFE_ADVICE = "unsafe_advice"
    TOO_LONG = "too_long"
    FORMAT_ERROR = "format_error"
    OVER_CONFIDENT = "over_confident"
    MAX_REVISION = "max_revision"
