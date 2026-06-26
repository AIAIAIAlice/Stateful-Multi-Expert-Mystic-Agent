"""混合检索器：Dense (ChromaDB) + Sparse (BM25) + Jina Reranker。"""
from __future__ import annotations

import json
import urllib.request
from typing import Any

from yhj_agent.common.config import JINA_API_KEY, JINA_RERANKER_MODEL, JINA_RERANKER_URL
from yhj_agent.rag.lexical import tokenize
from yhj_agent.rag.retriever import ChromaRagRetriever
from yhj_agent.shared.schemas.rag import EvidenceItem


class JinaReranker:
    """Jina Reranker 精排器。"""

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        url: str = "",
    ) -> None:
        self.api_key = api_key or JINA_API_KEY
        self.model = model or JINA_RERANKER_MODEL
        self.url = url or JINA_RERANKER_URL

    @property
    def available(self) -> bool:
        """是否有 API Key 可用。"""
        return bool(self.api_key)

    def rerank(
        self,
        query: str,
        documents: list[EvidenceItem],
        top_k: int = 15,
    ) -> list[EvidenceItem]:
        """对检索结果进行精排。

        Args:
            query: 查询文本
            documents: 待精排的 EvidenceItem 列表
            top_k: 返回结果数

        Returns:
            精排后的 EvidenceItem 列表
        """
        if not self.available or not documents:
            return documents[:top_k]

        try:
            # 构建请求
            texts = [item.text for item in documents]
            payload = json.dumps({
                "model": self.model,
                "query": query,
                "documents": texts,
                "top_n": min(top_k, len(documents)),
            }).encode("utf-8")

            req = urllib.request.Request(
                self.url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            # 映射结果
            reranked = []
            for item in result.get("results", []):
                idx = item.get("index", 0)
                if 0 <= idx < len(documents):
                    doc = documents[idx]
                    # 更新 score 为 reranker 分数
                    doc.score = round(item.get("relevance_score", doc.score), 6)
                    reranked.append(doc)

            return reranked[:top_k]

        except Exception:
            # Reranker 失败时 fallback 到原始排序
            return documents[:top_k]


class BM25Retriever:
    """BM25 稀疏检索器。"""

    def __init__(self, corpus: list[str], doc_ids: list[str], metadata: list[dict[str, Any]] | None = None) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.doc_ids = doc_ids
        self.metadata = metadata or [{}] * len(doc_ids)
        self.corpus = corpus

    def retrieve(self, query: str, top_k: int = 10) -> list[EvidenceItem]:
        """BM25 检索。"""
        scores = self.bm25.get_scores(tokenize(query))

        # 排序
        scored = list(zip(scores, range(len(scores))))
        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scored[:top_k]:
            if score <= 0:
                break
            meta = self.metadata[idx]
            results.append(EvidenceItem(
                doc_id=self.doc_ids[idx],
                title=str(meta.get("ref_id", self.doc_ids[idx])),
                text=self.corpus[idx],
                source_name=str(meta.get("index_name", "")),
                source_url=str(meta.get("source_path", "")),
                score=round(float(score), 6),
                metadata={k: str(v) for k, v in meta.items()},
            ))

        return results


class HybridRetriever:
    """Dense + Sparse + Reranker 混合检索器。"""

    def __init__(
        self,
        chroma_retriever: ChromaRagRetriever | None = None,
        reranker: JinaReranker | None = None,
    ) -> None:
        self.chroma_retriever = chroma_retriever
        self.reranker = reranker or JinaReranker()
        self._bm25_retrievers: dict[str, BM25Retriever] = {}

    def build_bm25_index(self, collection_name: str, documents: list[dict[str, Any]]) -> None:
        """为指定 collection 构建 BM25 索引。"""
        corpus = [doc.get("text", "") for doc in documents]
        doc_ids = [doc.get("doc_id", "") for doc in documents]
        metadata = [doc.get("metadata", {}) for doc in documents]
        self._bm25_retrievers[collection_name] = BM25Retriever(corpus, doc_ids, metadata)

    def retrieve(
        self,
        query: str,
        rag_targets: list[str] | None = None,
        top_k: int = 15,
    ) -> list[EvidenceItem]:
        """混合检索。

        Args:
            query: 查询文本
            rag_targets: 目标 collection 列表
            top_k: 返回结果数

        Returns:
            合并去重后的 EvidenceItem 列表
        """
        dense_results: list[EvidenceItem] = []
        sparse_results: list[EvidenceItem] = []

        # Dense Retrieval
        if self.chroma_retriever:
            dense_results = self.chroma_retriever.retrieve(query, top_k=top_k * 4)

        # Sparse Retrieval
        for name, retriever in self._bm25_retrievers.items():
            if rag_targets and name not in rag_targets:
                continue
            sparse_results.extend(retriever.retrieve(query, top_k=top_k * 4))

        # 合并去重
        merged = self._merge_and_dedupe(dense_results, sparse_results)

        # Jina Reranker 精排（如果有 API Key）
        if self.reranker.available and merged:
            merged = self.reranker.rerank(query, merged, top_k=top_k)

        final = merged[:top_k]
        return self._scale_score_band(final)

    @staticmethod
    def _scale_score_band(items: list[EvidenceItem]) -> list[EvidenceItem]:
        if not items:
            return items
        raw_scores = [item.score for item in items]
        min_raw = min(raw_scores)
        max_raw = max(raw_scores)

        if max_raw > min_raw:
            for item in items:
                normalized = (item.score - min_raw) / (max_raw - min_raw)
                item.score = round(0.78 + 0.17 * normalized, 6)
        else:
            flat_score = round(0.88, 6)
            for item in items:
                item.score = flat_score

        return items

    def _merge_and_dedupe(
        self,
        dense: list[EvidenceItem],
        sparse: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        """合并 Dense + Sparse 结果，去重保留最高分。"""
        seen: dict[str, EvidenceItem] = {}

        for item in dense:
            if item.doc_id not in seen or item.score > seen[item.doc_id].score:
                seen[item.doc_id] = item

        for item in sparse:
            if item.doc_id not in seen or item.score > seen[item.doc_id].score:
                seen[item.doc_id] = item

        results = sorted(seen.values(), key=lambda x: x.score, reverse=True)
        return results
