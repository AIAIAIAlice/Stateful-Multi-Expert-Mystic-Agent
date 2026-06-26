class YHJAgentError(Exception):
    """项目统一异常基类。"""


class RecoverableError(YHJAgentError):
    """可恢复错误，例如 RAG 暂时不可用或外部工具超时。"""


class FatalWorkflowError(YHJAgentError):
    """当前 graph run 无法继续的错误。"""


class StateValidationError(YHJAgentError):
    """状态更新不符合 schema 或写入边界。"""


class LLMAuthError(FatalWorkflowError):
    """LLM API key 缺失、无效或权限不足。"""


class LLMRateLimitError(RecoverableError):
    """LLM provider 返回限速或额度不足。"""


class LLMTimeoutError(RecoverableError):
    """LLM 请求超时。"""


class LLMProviderError(RecoverableError):
    """LLM provider 返回非预期错误。"""


class LLMResponseValidationError(RecoverableError):
    """LLM 响应无法转换为项目统一 schema。"""

