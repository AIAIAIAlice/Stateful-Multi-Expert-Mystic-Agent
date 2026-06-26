"""Node15: ClarificationNode — 澄清提问。

模板字典匹配缺失字段 → 生成澄清问题 → interrupt/resume 机制。
兜底逻辑：clarification_count >= 2 时用默认值填充，强制继续。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


# 缺失字段 → 澄清问题模板
CLARIFICATION_TEMPLATES: dict[str, str] = {
    "birth_year": "请告诉我您的出生年份（如 1993 年）",
    "birth_month": "请告诉我您的出生月份（如 8 月）",
    "birth_day": "请告诉我您的出生日期（如 15 日）",
    "birth_hour": "请告诉我您的出生时间（如 上午10点 或 巳时）",
    "birth_location": "请告诉我您的出生地点（如 杭州）",
    "gender": "请告诉我您的性别（男/女）",
    "question": "请描述您想咨询的问题",
}

# 默认值（兜底时使用）
DEFAULT_VALUES: dict[str, Any] = {
    "birth_hour": "未知",
    "birth_location": "",
    "gender": "male",
}

MAX_CLARIFICATION_COUNT = 2


class ClarificationOutput(BaseModel):
    """ClarificationNode 输出。"""

    model_config = ConfigDict(extra="forbid")

    needs_interrupt: bool = False
    clarification_question: str = ""
    merged_input: dict[str, Any] = {}
    turn_type: str = "clarification_answer"
    degraded: bool = False


class ClarificationNode:
    """澄清提问节点。"""

    def process(
        self,
        missing_fields: list[str],
        normalized_input: dict[str, Any],
        clarification_answer: str = "",
        clarification_count: int = 0,
    ) -> ClarificationOutput:
        """处理澄清逻辑。

        Args:
            missing_fields: 缺失字段列表
            normalized_input: 当前标准化输入
            clarification_answer: 用户对澄清的回答（resume 时有值）
            clarification_count: 已澄清次数

        Returns:
            ClarificationOutput
        """
        # 有回答 → 合并并继续
        if clarification_answer:
            merged = self._merge_answer(normalized_input, clarification_answer, missing_fields)
            return ClarificationOutput(
                needs_interrupt=False,
                merged_input=merged,
                turn_type="clarification_resolved",
                degraded=False,
            )

        # 兜底：澄清次数耗尽
        if clarification_count >= MAX_CLARIFICATION_COUNT:
            merged = self._apply_defaults(normalized_input, missing_fields)
            return ClarificationOutput(
                needs_interrupt=False,
                merged_input=merged,
                turn_type="clarification_resolved",
                degraded=True,
            )

        # 生成澄清问题
        questions = []
        for field in missing_fields[:2]:  # 每轮最多 2 个
            template = CLARIFICATION_TEMPLATES.get(field, f"请补充 {field}")
            questions.append(template)

        question_text = "；".join(questions) + "。"

        return ClarificationOutput(
            needs_interrupt=True,
            clarification_question=question_text,
            merged_input=normalized_input,
            turn_type="clarification_answer",
            degraded=False,
        )

    def _merge_answer(
        self,
        normalized_input: dict[str, Any],
        answer: str,
        missing_fields: list[str],
    ) -> dict[str, Any]:
        """将用户回答合并到 normalized_input（解析完整回答）。"""
        import re
        merged = dict(normalized_input)

        # 尝试从回答中提取所有可解析字段
        date_match = re.search(r"(\d{4})\s*[年\-/\.]\s*(\d{1,2})\s*[月\-/\.]\s*(\d{1,2})", answer)
        if date_match:
            merged["birth_year"] = int(date_match.group(1))
            merged["birth_month"] = int(date_match.group(2))
            merged["birth_day"] = int(date_match.group(3))

        time_match = re.search(r"(凌晨|早上|上午|中午|下午|晚上|夜里)?\s*(\d{1,2})\s*[点时]", answer)
        if time_match:
            merged["birth_hour"] = f"{time_match.group(1) or ''}{time_match.group(2)}点"
        else:
            for kw in ("子时", "丑时", "寅时", "卯时", "辰时", "巳时", "午时", "未时", "申时", "酉时", "戌时", "亥时"):
                if kw in answer:
                    merged["birth_hour"] = kw
                    break

        for city in ("成都", "重庆", "北京", "上海", "广州", "深圳", "杭州", "南京", "武汉", "西安"):
            if city in answer:
                merged["birth_location"] = city
                break

        if "男" in answer:
            merged["gender"] = "male"
        elif "女" in answer:
            merged["gender"] = "female"

        # 对于无法自动解析的字段，按 missing_fields 顺序逐个填充
        for field in missing_fields:
            if not merged.get(field):
                if field == "birth_hour":
                    merged["birth_hour"] = answer
                elif field == "birth_location":
                    merged["birth_location"] = answer
                elif field == "gender":
                    merged["gender"] = "male" if "男" in answer else "female"
                else:
                    merged[field] = answer

        # 重新检查缺失字段
        all_required = ["birth_year", "birth_month", "birth_day", "birth_hour", "birth_location", "gender", "question"]
        still_missing = [f for f in all_required if not merged.get(f)]
        merged["missing_fields"] = still_missing
        merged["is_valid"] = len(still_missing) == 0

        return merged

    def _apply_defaults(
        self,
        normalized_input: dict[str, Any],
        missing_fields: list[str],
    ) -> dict[str, Any]:
        """用默认值填充缺失字段（兜底）。"""
        merged = dict(normalized_input)

        for field in missing_fields:
            if field in DEFAULT_VALUES:
                merged[field] = DEFAULT_VALUES[field]

        merged["missing_fields"] = []
        merged["is_valid"] = True
        return merged
