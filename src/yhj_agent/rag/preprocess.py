from collections.abc import Iterable
from pathlib import Path
from typing import Any
import csv
import hashlib
import json

from yhj_agent.shared.schemas.rag import RagDocument


def load_raw_records(path: str | Path) -> list[dict[str, Any]]:
    """读取 csv / json / jsonl 原始数据。

    parquet 交给后续成熟库处理；当前模块不自造 parquet 解析器。
    """

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as file:
            return list(csv.DictReader(file))
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in file_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("data", [])
    raise ValueError(f"Unsupported raw data format: {suffix}")


def normalize_records(
    records: Iterable[dict[str, Any]],
    source_name: str,
    source_url: str,
    limit: int | None = None,
) -> list[RagDocument]:
    """把不同字段名的数据清洗成统一 RagDocument。"""

    documents: list[RagDocument] = []
    seen_hashes: set[str] = set()

    for index, record in enumerate(records):
        if limit is not None and len(documents) >= limit:
            break

        title, text = normalize_record_text(record)
        if not text:
            continue

        content_hash = hashlib.sha1(f"{title}\n{text}".encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)

        documents.append(
            RagDocument(
                doc_id=f"{source_name}:{content_hash[:12]}",
                title=title or f"document_{index}",
                text=text,
                source_name=source_name,
                source_url=source_url,
                metadata={"row_index": index},
            )
        )

    return documents


def normalize_record_text(record: dict[str, Any]) -> tuple[str, str]:
    """兼容 QA、SFT messages、岗位技能数据等常见公开数据格式。"""

    lowered = {str(key).strip().lower(): value for key, value in record.items()}

    question = _first_value(lowered, ("question", "questions", "query", "prompt", "instruction"))
    answer = _first_value(lowered, ("answer", "answers", "response", "completion", "output"))
    if question and answer:
        return str(question), f"Question: {question}\nAnswer: {answer}"

    messages = lowered.get("messages")
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            messages = None
    if isinstance(messages, list):
        user_parts = [item.get("content", "") for item in messages if item.get("role") == "user"]
        assistant_parts = [item.get("content", "") for item in messages if item.get("role") == "assistant"]
        title = user_parts[0] if user_parts else "career coach conversation"
        text = "\n".join(user_parts + assistant_parts)
        return title, text

    title = _first_value(lowered, ("title", "job_title", "role", "position", "name"))
    description = _first_value(lowered, ("description", "job_description", "summary", "content", "text"))
    skills = _first_value(lowered, ("skills", "required_skills", "skill"))
    parts = [part for part in (description, skills) if part]
    if title and parts:
        return str(title), "\n".join(str(part) for part in parts)

    text = _first_value(lowered, ("text", "content", "body"))
    if text:
        return str(title or "document"), str(text)

    return "", ""


def write_jsonl(documents: Iterable[RagDocument], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        for document in documents:
            file.write(json.dumps(document.model_dump(), ensure_ascii=False) + "\n")


def read_documents(path: str | Path) -> list[RagDocument]:
    return [
        RagDocument.model_validate(json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None

