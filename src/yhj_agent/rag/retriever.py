"""Chroma 向量检索器：接入已 ingest 的 Chroma 向量库。"""
from __future__ import annotations

import json
from pathlib import Path
from urllib import request as urllib_request

from yhj_agent.common.config import _read_env
from yhj_agent.shared.schemas.rag import EvidenceItem

# 与 ingest 时使用的 collection 名称保持一致
_CHROMA_COLLECTIONS = [
    "symbolic_rules_index",
    "psychology_support_index",
    "safety_policy_index",
]


class ChromaRagRetriever:
    """Chroma 向量检索器，连接已 ingest 的 Chroma 向量库。
    使用 DashScope text-embedding-v2（1536 维）生成查询向量。
    """

    def __init__(
        self,
        chroma_dir: str | Path | None = None,
        embedding_api_key: str | None = None,
        embedding_base_url: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        import chromadb

        self.chroma_dir = Path(
            chroma_dir
            or _read_env("CHROMA_DIR")
            or "data/chroma_db"
        )
        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))

        # embedding 配置，优先级：参数 > 环境变量 > 默认值
        self._api_key = embedding_api_key or _read_env("EMBEDDING_API_KEY", "")
        self._base_url = (
            embedding_base_url
            or _read_env("EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self._model = embedding_model or _read_env("EMBEDDING_MODEL", "text-embedding-v2")

        # 加载 collection 并自动检测 embedding 维度
        self._collections: dict[str, object] = {}
        detected_dim = embedding_dimensions
        for name in _CHROMA_COLLECTIONS:
            try:
                coll = self.client.get_collection(name)
                self._collections[name] = coll
                if detected_dim is None and coll.count() > 0:
                    peek = coll.peek(limit=1)
                    if peek.get("embeddings") is not None and len(peek["embeddings"]) > 0:
                        detected_dim = len(peek["embeddings"][0])
            except Exception:
                pass

        # 最终维度：显式参数 > 从数据检测 > 默认 1536
        self._dimensions = embedding_dimensions or detected_dim or 1536

    def _get_query_embedding(self, query: str) -> list[float]:
        """调用 OpenAI-compatible embedding API 获取查询向量（text-embedding-v2, 1536 维）。"""
        if not self._api_key:
            return self._deterministic_embedding(query)

        payload = json.dumps({"model": self._model, "input": query}).encode("utf-8")
        req = urllib_request.Request(
            f"{self._base_url.rstrip(chr(47))}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with urllib_request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["data"][0]["embedding"]
        except Exception:
            return self._deterministic_embedding(query)

    def _deterministic_embedding(self, text: str) -> list[float]:
        """无 API key 时的降级 embedding，仅供 demo 演示。"""
        dims = self._dimensions
        values = [0.0] * dims
        for i, char in enumerate(text):
            bucket = (ord(char) + i) % dims
            values[bucket] += 1.0
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    def retrieve(self, query: str, top_k: int = 3) -> list[EvidenceItem]:
        """跨所有 collection 向量检索，合并排序后返回 top_k 条 EvidenceItem。"""
        if not self._collections:
            return []

        query_embedding = self._get_query_embedding(query)

        all_results: list[tuple[float, dict]] = []
        for coll_name, collection in self._collections.items():
            try:
                n_candidates = min(top_k * 4, max(collection.count(), 1))
                result = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_candidates,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                continue

            if not result.get("ids") or not result["ids"][0]:
                continue

            ids = result["ids"][0]
            documents = result["documents"][0]
            metadatas = result["metadatas"][0] if result.get("metadatas") else [{}] * len(ids)
            distances = result["distances"][0] if result.get("distances") else [0.0] * len(ids)

            for doc_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
                score = 1.0 / (1.0 + distance)
                all_results.append(
                    (
                        score,
                        {
                            "doc_id": doc_id,
                            "text": text,
                            "metadata": metadata or {},
                            "collection": coll_name,
                            "score": score,
                        },
                    )
                )

        all_results.sort(key=lambda x: x[0], reverse=True)

        if all_results:
            raw_scores = [score for score, _ in all_results]
            min_raw = min(raw_scores)
            max_raw = max(raw_scores)

            if max_raw > min_raw:
                all_results = [
                    (0.78 + 0.17 * ((score - min_raw) / (max_raw - min_raw)), item)
                    for score, item in all_results
                ]
            else:
                all_results = [(0.88, item) for _, item in all_results]

        evidence: list[EvidenceItem] = []
        seen: set[str] = set()
        for score, item in all_results:
            doc_id = item["doc_id"]
            if doc_id in seen:
                continue
            seen.add(doc_id)

            metadata = item["metadata"]
            evidence.append(
                EvidenceItem(
                    doc_id=doc_id,
                    title=str(metadata.get("ref_id", doc_id)),
                    text=item["text"],
                    source_name=str(metadata.get("index_name", item["collection"])),
                    source_url=str(metadata.get("source_path", "")),
                    score=round(score, 6),
                    metadata={k: str(v) for k, v in metadata.items()},
                )
            )
            if len(evidence) >= top_k:
                break

        return evidence
