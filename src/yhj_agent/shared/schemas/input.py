"""输入归一化 schema。"""
from pydantic import BaseModel, ConfigDict


class NormalizedInput(BaseModel):
    """经字段校验和格式标准化后的结构化输入。"""

    model_config = ConfigDict(extra="forbid")

    birth_year: int
    birth_month: int
    birth_day: int
    birth_hour: str
    birth_location: str
    gender: str
    output_style: str
    calendar_type: str  # "solar" | "lunar"
    missing_fields: list[str]
    is_valid: bool
