from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RagDocument(BaseModel):
    """RAG 内部统一文档结构。

    不同来源数据必须先清洗成这个结构，再进入索引和检索。
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    text: str
    source_name: str
    source_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    """Retriever 返回的证据片段。"""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    title: str
    text: str
    source_name: str
    source_url: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class RefinedQuery(BaseModel):
    """QueryRefiner 输出的细化查询。"""

    model_config = ConfigDict(extra="forbid")

    query: str
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)

