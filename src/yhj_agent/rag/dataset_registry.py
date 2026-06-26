from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetSource(BaseModel):
    """RAG 数据源登记信息。

    这里只登记来源和下载策略，不直接下载或清洗数据。
    """

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    source_url: str
    download_url: str | None = None
    license: str | None = None
    format: Literal["csv", "json", "jsonl", "parquet", "kaggle", "unknown"]
    is_cleaned: bool
    language: str = "en"
    expected_schema: list[str] = Field(default_factory=list)
    use_case: str
    raw_path: str
    processed_path: str
    metadata_path: str
    known_limitations: list[str] = Field(default_factory=list)


def get_default_dataset_sources(project_root: str | Path = ".") -> list[DatasetSource]:
    """返回当前项目优先调研的数据源候选。

    数据源基于公开页面登记；真正采用前仍需复核许可证和字段。
    """

    root = Path(project_root)
    return [
        DatasetSource(
            dataset_name="Pradeep016/career-guidance-qa-dataset",
            source_url="https://huggingface.co/datasets/Pradeep016/career-guidance-qa-dataset",
            download_url=(
                "https://huggingface.co/datasets/Pradeep016/"
                "career-guidance-qa-dataset/resolve/main/Career%20QA%20Dataset.csv"
            ),
            license="cc-by-4.0",
            format="csv",
            is_cleaned=True,
            expected_schema=["question", "answer"],
            use_case="职业咨询 QA 检索与解释型问答证据",
            raw_path=str(root / "data/raw/career_guidance_qa.csv"),
            processed_path=str(root / "data/processed/career_guidance_qa.jsonl"),
            metadata_path=str(root / "data/metadata/career_guidance_qa.json"),
            known_limitations=["英文数据，需要中文 query 时做翻译或双语 query refine"],
        ),
        DatasetSource(
            dataset_name="aramasethu/career-coach-sft-dataset",
            source_url="https://huggingface.co/datasets/aramasethu/career-coach-sft-dataset",
            download_url=None,
            license=None,
            format="parquet",
            is_cleaned=True,
            expected_schema=["messages"],
            use_case="职业教练对话和面试准备类建议检索",
            raw_path=str(root / "data/raw/career_coach_sft.parquet"),
            processed_path=str(root / "data/processed/career_coach_sft.jsonl"),
            metadata_path=str(root / "data/metadata/career_coach_sft.json"),
            known_limitations=["parquet 需要额外依赖或 Hugging Face datasets 才能直接读取"],
        ),
        DatasetSource(
            dataset_name="Kaggle AI-Powered Job Recommendations",
            source_url="https://www.kaggle.com/datasets/samayashar/ai-powered-job-recommendations",
            download_url=None,
            license=None,
            format="kaggle",
            is_cleaned=True,
            expected_schema=["job_title", "skills", "salary", "description"],
            use_case="岗位推荐、技能匹配和现实职业约束检索",
            raw_path=str(root / "data/raw/ai_powered_job_recommendations.csv"),
            processed_path=str(root / "data/processed/job_recommendations.jsonl"),
            metadata_path=str(root / "data/metadata/job_recommendations.json"),
            known_limitations=["Kaggle 下载通常需要账号 token，不能在仓库提交凭证"],
        ),
    ]


def find_dataset_source(dataset_name: str, project_root: str | Path = ".") -> DatasetSource:
    for source in get_default_dataset_sources(project_root):
        if source.dataset_name == dataset_name:
            return source
    raise ValueError(f"Unknown dataset source: {dataset_name}")

