"""词法检索器：供测试和降级场景使用。"""
from collections import Counter
from pathlib import Path
import json
import math
import re

from yhj_agent.rag.preprocess import read_documents
from yhj_agent.shared.schemas.rag import EvidenceItem, RagDocument


def tokenize(text: str) -> list[str]:
    """轻量 tokenizer：英文按词，中文按连续字符 bigram。"""
    lowered = text.lower()
    words = re.findall(r"[a-z0-9+#.]+", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = ["".join(cjk_chars[index : index + 2]) for index in range(max(len(cjk_chars) - 1, 0))]
    return words + cjk_bigrams


def build_lexical_index(processed_path: str | Path, index_path: str | Path) -> None:
    """从已处理的文档构建词法 TF-IDF 索引。"""
    documents = read_documents(processed_path)
    indexed_docs = []
    document_frequency: Counter[str] = Counter()

    tokenized_docs = []
    for document in documents:
        tokens = tokenize(f"{document.title}\n{document.text}")
        token_counts = Counter(tokens)
        tokenized_docs.append((document, token_counts))
        document_frequency.update(token_counts.keys())

    total_docs = max(len(documents), 1)
    for document, token_counts in tokenized_docs:
        indexed_docs.append(
            {
                "document": document.model_dump(),
                "tokens": dict(token_counts),
            }
        )

    payload = {
        "index_type": "lexical_tfidf",
        "document_count": len(documents),
        "document_frequency": dict(document_frequency),
        "idf": {
            token: math.log((1 + total_docs) / (1 + freq)) + 1
            for token, freq in document_frequency.items()
        },
        "documents": indexed_docs,
    }

    output_path = Path(index_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class LexicalRagRetriever:
    """轻量本地词法检索器，供测试和无 Chroma 环境降级使用。"""

    def __init__(self, index_path: str | Path) -> None:
        payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
        self.idf: dict[str, float] = payload.get("idf", {})
        self.documents = payload.get("documents", [])

    def retrieve(self, query: str, top_k: int = 3) -> list[EvidenceItem]:
        query_tokens = Counter(tokenize(query))
        scored: list[tuple[float, RagDocument]] = []

        for item in self.documents:
            document = RagDocument.model_validate(item["document"])
            doc_tokens = item.get("tokens", {})
            score = self._score(query_tokens, doc_tokens)
            if score > 0:
                scored.append((score, document))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        if scored:
            raw_scores = [score for score, _ in scored]
            min_raw = min(raw_scores)
            max_raw = max(raw_scores)
            if max_raw > min_raw:
                scored = [
                    (0.78 + 0.17 * ((score - min_raw) / (max_raw - min_raw)), document)
                    for score, document in scored
                ]
            else:
                scored = [(0.88, document) for _, document in scored]
        return [
            EvidenceItem(
                doc_id=document.doc_id,
                title=document.title,
                text=document.text,
                source_name=document.source_name,
                source_url=document.source_url,
                score=round(score, 6),
                metadata=document.metadata,
            )
            for score, document in scored[:top_k]
        ]

    def _score(self, query_tokens: Counter[str], doc_tokens: dict[str, int]) -> float:
        score = 0.0
        for token, query_count in query_tokens.items():
            if token not in doc_tokens:
                continue
            score += query_count * doc_tokens[token] * self.idf.get(token, 1.0)
        return score