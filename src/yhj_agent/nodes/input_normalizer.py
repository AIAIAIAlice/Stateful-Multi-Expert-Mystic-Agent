"""Node1: InputNormalizer — 输入归一化。"""
from __future__ import annotations

import re
from typing import Any

from yhj_agent.shared.schemas.input import NormalizedInput


class InputNormalizer:
    """字段校验 + 格式标准化 + 历法预处理。"""

    REQUIRED_FIELDS = ("birth_year", "birth_month", "birth_day", "birth_hour", "birth_location", "gender")

    def normalize(self, raw_input: dict[str, Any]) -> NormalizedInput:
        # 1. 提取字段
        birth_year = self._extract_year(str(raw_input.get("birth_year", "")))
        birth_month = self._extract_number(str(raw_input.get("birth_month", "")))
        birth_day = self._extract_number(str(raw_input.get("birth_day", "")))
        birth_hour = self._normalize_hour(str(raw_input.get("birth_hour", "")))
        birth_location = str(raw_input.get("birth_location", "")).strip()
        gender = self._normalize_gender(str(raw_input.get("gender", "")))
        output_style = str(raw_input.get("output_style", "深刻且实际")).strip()

        # 2. 必填字段检查
        missing_fields: list[str] = []
        if not birth_year:
            missing_fields.append("birth_year")
        if not birth_month:
            missing_fields.append("birth_month")
        if not birth_day:
            missing_fields.append("birth_day")
        if not birth_hour:
            missing_fields.append("birth_hour")
        if not birth_location:
            missing_fields.append("birth_location")
        if not gender:
            missing_fields.append("gender")

        # 3. 日期有效性校验
        invalid_date = False
        if birth_year and birth_month and birth_day:
            invalid_date = not self._is_valid_date(birth_year, birth_month, birth_day)
            if invalid_date:
                missing_fields.append("invalid_date")

        # 4. 历法预处理
        calendar_type = "solar"
        if raw_input.get("calendar_type") == "lunar":
            calendar_type = "lunar"

        is_valid = len(missing_fields) == 0 and not invalid_date

        return NormalizedInput(
            birth_year=birth_year or 0,
            birth_month=birth_month or 0,
            birth_day=birth_day or 0,
            birth_hour=birth_hour or "",
            birth_location=birth_location,
            gender=gender,
            output_style=output_style,
            calendar_type=calendar_type,
            missing_fields=missing_fields,
            is_valid=is_valid,
        )

    def _extract_year(self, text: str) -> int:
        """提取年份数字。"""
        match = re.search(r"(\d{4})", text)
        return int(match.group(1)) if match else 0

    def _extract_number(self, text: str) -> int:
        """提取数字。"""
        match = re.search(r"(\d{1,2})", text)
        return int(match.group(1)) if match else 0

    def _normalize_hour(self, text: str) -> str:
        """标准化时辰。"""
        if not text:
            return ""

        # 已经是时辰格式
        if text in ("子时", "丑时", "寅时", "卯时", "辰时", "巳时", "午时", "未时", "申时", "酉时", "戌时", "亥时"):
            return text

        # 24 小时制 → 时辰
        hour_map = {
            23: "子时", 0: "子时",
            1: "丑时", 2: "丑时",
            3: "寅时", 4: "寅时",
            5: "卯时", 6: "卯时",
            7: "辰时", 8: "辰时",
            9: "巳时", 10: "巳时",
            11: "午时", 12: "午时",
            13: "未时", 14: "未时",
            15: "申时", 16: "申时",
            17: "酉时", 18: "酉时",
            19: "戌时", 20: "戌时",
            21: "亥时", 22: "亥时",
        }

        # 先检查中文时段关键词（优先级高于数字提取）
        if "凌晨" in text or "早" in text:
            return "寅时"
        if "上午" in text:
            return "巳时"
        if "中午" in text:
            return "午时"
        if "下午" in text:
            return "申时"
        if "晚上" in text or "夜" in text:
            return "戌时"

        # 尝试提取数字（24 小时制）
        match = re.search(r"(\d{1,2})", text)
        if match:
            hour = int(match.group(1))
            if 0 <= hour <= 23:
                return hour_map.get(hour, "")

        return text

    def _normalize_gender(self, text: str) -> str:
        """标准化性别。"""
        if not text:
            return ""
        text = text.strip().lower()
        if text in ("男", "male", "m"):
            return "male"
        if text in ("女", "female", "f"):
            return "female"
        return ""

    def _is_valid_date(self, year: int, month: int, day: int) -> bool:
        """校验日期有效性。"""
        if year < 1900 or year > 2100:
            return False
        if month < 1 or month > 12:
            return False
        if day < 1:
            return False
        # 每月天数
        days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        # 闰年
        if month == 2 and (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)):
            days_in_month[2] = 29
        return day <= days_in_month[month]
