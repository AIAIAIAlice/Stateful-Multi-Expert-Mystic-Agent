"""Node3: ProfileMemoryReader — 用户画像 + 长期记忆读取。

从 SQLite user_profile 查询结构化画像，
从 ChromaDB user_memory 检索相关长期记忆。
"""
from __future__ import annotations

from typing import Any

from yhj_agent.memory.user_profile_store import UserProfileStore


class ProfileMemoryReader:
    """读取用户画像和相关长期记忆。"""

    def __init__(
        self,
        profile_store: UserProfileStore | None = None,
        chroma_client: Any = None,
        embedding_fn: Any = None,
    ) -> None:
        self.profile_store = profile_store or UserProfileStore()
        self.chroma_client = chroma_client
        self.embedding_fn = embedding_fn

    def read(
        self,
        user_id: str,
        topics: list[str] | None = None,
        consultation_type: str = "",
    ) -> dict[str, Any]:
        """读取用户画像和相关记忆。

        Args:
            user_id: 用户 ID
            topics: 当前话题列表
            consultation_type: 咨询类型

        Returns:
            {"user_profile": {...}, "relevant_memories": [...]}
        """
        # 1. 查询 SQLite user_profile
        profile = self.profile_store.get(user_id)

        # 2. 查询 ChromaDB user_memory
        memories = self._search_memories(user_id, topics or [])

        return {
            "user_profile": profile,
            "relevant_memories": memories,
        }

    def _search_memories(self, user_id: str, topics: list[str]) -> list[str]:
        """从 ChromaDB user_memory 检索相关记忆。"""
        if not self.chroma_client:
            return []

        try:
            coll = self.chroma_client.get_collection("user_memory")
        except Exception:
            return []

        if coll.count() == 0:
            return []

        # 构建查询文本
        query_text = " ".join(topics) if topics else user_id
        if not query_text:
            return []

        # 获取 embedding
        if self.embedding_fn:
            query_embedding = self.embedding_fn(query_text)
        else:
            return []

        # 查询
        try:
            result = coll.query(
                query_embeddings=[query_embedding],
                n_results=10,
                where={"user_id": user_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        if not result.get("ids") or not result["ids"][0]:
            return []

        # 按 similarity_threshold=0.75 过滤
        memories: list[str] = []
        for i, distance in enumerate(result["distances"][0]):
            similarity = 1.0 / (1.0 + distance)
            if similarity >= 0.75:
                doc = result["documents"][0][i]
                memories.append(doc)

        return memories

    def get_session_metadata(self, user_id: str) -> dict[str, Any] | None:
        """获取最近的 session_metadata（跨 session 上下文恢复）。"""
        if not self.chroma_client:
            return None

        try:
            coll = self.chroma_client.get_collection("user_memory")
        except Exception:
            return None

        if coll.count() == 0:
            return None

        try:
            result = coll.get(
                where={"$and": [{"user_id": user_id}, {"type": "session_metadata"}]},
                include=["documents", "metadatas"],
                limit=1,
            )
        except Exception:
            return None

        if not result.get("documents"):
            return None

        import json
        try:
            return json.loads(result["documents"][0])
        except (json.JSONDecodeError, IndexError):
            return None
