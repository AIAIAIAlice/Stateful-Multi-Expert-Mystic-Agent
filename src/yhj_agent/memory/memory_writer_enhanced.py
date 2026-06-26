"""Node11: MemoryWriter — 增强版长期记忆写入。

规则提取 specialist outputs → ChromaDB user_memory 去重写入。
session_metadata 写入 + SQLite user_profile 更新。
"""
from __future__ import annotations

import json
from typing import Any

from yhj_agent.memory.user_profile_store import UserProfileStore


# 敏感信息过滤
SENSITIVE_KEYS = {"birth_date", "birth_time", "birth_place", "health", "relationship_detail"}


class MemoryWriterEnhanced:
    """增强版长期记忆写入。"""

    def __init__(
        self,
        profile_store: UserProfileStore | None = None,
        chroma_client: Any = None,
        embedding_fn: Any = None,
    ) -> None:
        self.profile_store = profile_store or UserProfileStore()
        self.chroma_client = chroma_client
        self.embedding_fn = embedding_fn

    def write(
        self,
        user_id: str,
        session_id: str,
        specialist_outputs: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        consultation_type: str = "",
        consultation_intent: str = "",
        question: str = "",
        final_report: dict[str, Any] | None = None,
        output_style: str = "",
        **kwargs,
    ) -> dict[str, Any]:

        if not consultation_type:
            req = kwargs.get('consultation_request', {})
            consultation_type = req.get('consultation_type', '')
            consultation_intent = req.get('consultation_intent', '')
        if not question:
            ni = kwargs.get('normalized_input', {})
            question = ni.get('question', '')
        """执行记忆写入。

        Returns:
            {"written": int, "skipped": int, "status": str}
        """
        # 1. 规则提取记忆
        memories = self._extract_memories(specialist_outputs, evaluation)

        # 2. 敏感信息过滤
        memories = [m for m in memories if not self._is_sensitive(m)]

        # 3. 置信度过滤
        overall_score = evaluation.get("overall_score", 0)
        if overall_score < 3.0:
            memories = []  # 低质量不写入

        written = 0
        skipped = 0

        # 4. 写入 ChromaDB user_memory
        if self.chroma_client and memories:
            for memory in memories:
                result = self._write_to_chroma(user_id, memory)
                if result:
                    written += 1
                else:
                    skipped += 1

        # 5. 写入 session_metadata
        if self.chroma_client and final_report:
            self._write_session_metadata(
                user_id, consultation_intent, consultation_type, final_report
            )

        # 6. 更新 SQLite user_profile
        if consultation_type:
            self.profile_store.record_consultation(user_id, consultation_type, output_style)

        return {
            "user_id": user_id,
            "total_extracted": len(memories),
            "written": written,
            "skipped_duplicate": skipped,
            "status": "success",
        }

    def _extract_memories(
        self,
        specialist_outputs: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """从 specialist outputs 规则提取记忆。"""
        memories: list[dict[str, Any]] = []

        # 根据 evaluation.overall_score 确定 confidence 等级
        overall_score = evaluation.get("overall_score", 0)
        if overall_score >= 4.0:
            confidence = "high"
        elif overall_score >= 3.0:
            confidence = "medium"
        else:
            confidence = "low"

        # symbolic_interpreter → key_findings
        si = specialist_outputs.get("symbolic_interpreter", {})
        for finding in si.get("content", {}).get("key_findings", []):
            if isinstance(finding, dict) and finding.get("claim"):
                memories.append({
                    "type": "preference_update",
                    "content": finding["claim"],
                    "topic": "symbolic",
                    "confidence": confidence,
                })

        # psychology_support → emotion_tags
        ps = specialist_outputs.get("psychology_support", {})
        for tag in ps.get("content", {}).get("emotion_tags", []):
            memories.append({
                "type": "behavior_note",
                "content": f"用户情绪标签：{tag}",
                "topic": "emotion",
                "confidence": confidence,
            })

        # practical_advisor → key_suggestions
        pa = specialist_outputs.get("practical_advisor", {})
        for suggestion in pa.get("content", {}).get("key_suggestions", []):
            memories.append({
                "type": "behavior_note",
                "content": suggestion,
                "topic": "decision",
                "confidence": confidence,
            })

        return memories

    def _is_sensitive(self, memory: dict[str, Any]) -> bool:
        """检查是否包含敏感信息。"""
        content = memory.get("content", "").lower()
        for key in SENSITIVE_KEYS:
            if key in content:
                return True
        return False

    def _write_to_chroma(self, user_id: str, memory: dict[str, Any]) -> bool:
        """写入 ChromaDB user_memory，带去重检查。"""
        try:
            coll = self.chroma_client.get_or_create_collection("user_memory")
        except Exception:
            return False

        content = memory.get("content", "")
        if not content:
            return False

        # 去重检查
        if self.embedding_fn:
            try:
                query_embedding = self.embedding_fn(content)
                existing = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=1,
                    where={"user_id": user_id},
                    include=["distances"],
                )
                if existing.get("distances") and existing["distances"][0]:
                    similarity = 1.0 / (1.0 + existing["distances"][0][0])
                    if similarity > 0.95:
                        return False  # 重复，跳过
            except Exception:
                pass

        # 写入
        doc_id = f"{user_id}_{memory['type']}_{hash(content) % 100000}"
        embedding = self.embedding_fn(content) if self.embedding_fn else None

        try:
            coll.upsert(
                ids=[doc_id],
                documents=[content],
                embeddings=[embedding] if embedding else None,
                metadatas=[{
                    "user_id": user_id,
                    "type": memory.get("type", "behavior_note"),
                    "topic": memory.get("topic", ""),
                    "confidence": "high",
                }],
            )
            return True
        except Exception:
            return False

    def _write_session_metadata(
        self,
        user_id: str,
        consultation_intent: str,
        consultation_type: str,
        final_report: dict[str, Any],
    ) -> None:
        """写入 session_metadata 记录。"""
        if not self.chroma_client:
            return

        try:
            coll = self.chroma_client.get_or_create_collection("user_memory")
        except Exception:
            return

        metadata_content = json.dumps({
            "last_consultation_intent": consultation_intent,
            "last_consultation_type": consultation_type,
            "last_report_summary": final_report.get("report_text", "")[:300],
        }, ensure_ascii=False)

        doc_id = f"{user_id}_session_metadata"
        embedding = self.embedding_fn(metadata_content) if self.embedding_fn else None

        try:
            coll.upsert(
                ids=[doc_id],
                documents=[metadata_content],
                embeddings=[embedding] if embedding else None,
                metadatas=[{
                    "user_id": user_id,
                    "type": "session_metadata",
                    "topic": "session",
                    "confidence": "high",
                }],
            )
        except Exception:
            pass
