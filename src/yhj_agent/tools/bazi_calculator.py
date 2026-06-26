"""真实八字排盘工具。

使用 lunar-python 库实现四柱八字排盘、五行分析、十神、大运等计算。
"""
from __future__ import annotations

import time
from typing import Any


# 天干
STEMS = "甲乙丙丁戊己庚辛壬癸"
# 地支
BRANCHES = "子丑寅卯辰巳午未申酉戌亥"
# 五行映射
STEM_ELEMENTS = {
    "甲": "木", "乙": "木",
    "丙": "火", "丁": "火",
    "戊": "土", "己": "土",
    "庚": "金", "辛": "金",
    "壬": "水", "癸": "水",
}
BRANCH_ELEMENTS = {
    "子": "水", "丑": "土", "寅": "木", "卯": "木",
    "辰": "土", "巳": "火", "午": "火", "未": "土",
    "申": "金", "酉": "金", "戌": "土", "亥": "水",
}
# 五行英文
ELEMENT_EN = {"木": "wood", "火": "fire", "土": "earth", "金": "metal", "水": "water"}
# 十神映射（日主天干 → 其他天干的关系）
TEN_GODS_MAP = {
    # 日主: {其他天干: 十神}
    "甲": {"甲": "比肩", "乙": "劫财", "丙": "食神", "丁": "伤官", "戊": "偏财", "己": "正财", "庚": "七杀", "辛": "正官", "壬": "偏印", "癸": "正印"},
    "乙": {"甲": "劫财", "乙": "比肩", "丙": "伤官", "丁": "食神", "戊": "正财", "己": "偏财", "庚": "正官", "辛": "七杀", "壬": "正印", "癸": "偏印"},
    "丙": {"甲": "偏印", "乙": "正印", "丙": "比肩", "丁": "劫财", "戊": "食神", "己": "伤官", "庚": "偏财", "辛": "正财", "壬": "七杀", "癸": "正官"},
    "丁": {"甲": "正印", "乙": "偏印", "丙": "劫财", "丁": "比肩", "戊": "伤官", "己": "食神", "庚": "正财", "辛": "偏财", "壬": "正官", "癸": "七杀"},
    "戊": {"甲": "七杀", "乙": "正官", "丙": "偏印", "丁": "正印", "戊": "比肩", "己": "劫财", "庚": "食神", "辛": "伤官", "壬": "偏财", "癸": "正财"},
    "己": {"甲": "正官", "乙": "七杀", "丙": "正印", "丁": "偏印", "戊": "劫财", "己": "比肩", "庚": "伤官", "辛": "食神", "壬": "正财", "癸": "偏财"},
    "庚": {"甲": "偏财", "乙": "正财", "丙": "七杀", "丁": "正官", "戊": "偏印", "己": "正印", "庚": "比肩", "辛": "劫财", "壬": "食神", "癸": "伤官"},
    "辛": {"甲": "正财", "乙": "偏财", "丙": "正官", "丁": "七杀", "戊": "正印", "己": "偏印", "庚": "劫财", "辛": "比肩", "壬": "伤官", "癸": "食神"},
    "壬": {"甲": "食神", "乙": "伤官", "丙": "偏财", "丁": "正财", "戊": "七杀", "己": "正官", "庚": "偏印", "辛": "正印", "壬": "比肩", "癸": "劫财"},
    "癸": {"甲": "伤官", "乙": "食神", "丙": "正财", "丁": "偏财", "戊": "正官", "己": "七杀", "庚": "正印", "辛": "偏印", "壬": "劫财", "癸": "比肩"},
}


class BaziCalculator:
    """真实八字排盘工具。"""

    name = "BaziCalculator"

    def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """执行八字排盘。

        Args:
            params: {"birth_date": "1993-08-15", "birth_time": "巳时" 或 "10:00", "gender": "male"}

        Returns:
            完整八字排盘结果
        """
        start_time = time.monotonic()

        birth_date = str(params.get("birth_date", ""))
        birth_time = str(params.get("birth_time", ""))
        gender = str(params.get("gender", "male"))

        # 解析出生日期
        year, month, day = self._parse_date(birth_date)

        # 解析时辰 → 小时
        hour = self._parse_hour(birth_time)

        # 使用 lunar-python 计算
        try:
            from lunar_python import Solar
            solar = Solar.fromYmdHms(year, month, day, hour, 0, 0)
            lunar = solar.getLunar()
            eight_char = lunar.getEightChar()

            # 四柱
            year_gz = eight_char.getYear()
            month_gz = eight_char.getMonth()
            day_gz = eight_char.getDay()
            time_gz = eight_char.getTime()

            # 五行分布
            five_elements = self._count_elements(year_gz, month_gz, day_gz, time_gz)

            # 日主
            day_master = day_gz[0]  # 日柱天干
            day_master_element = STEM_ELEMENTS.get(day_master, "")

            # 日主强弱（简化判断）
            day_master_strength = self._judge_strength(day_master_element, five_elements, month_gz)

            # 喜用神
            favorable, unfavorable = self._get_favorable(day_master_element, day_master_strength)

            # 十神
            ten_gods = {
                "year": TEN_GODS_MAP.get(day_master, {}).get(year_gz[0], ""),
                "month": TEN_GODS_MAP.get(day_master, {}).get(month_gz[0], ""),
                "day": "日主",
                "hour": TEN_GODS_MAP.get(day_master, {}).get(time_gz[0], ""),
            }

            # 大运
            gender_int = 1 if gender == "male" else 0
            yun = eight_char.getYun(gender_int)
            da_yun_list = yun.getDaYun()
            major_cycles = []
            current_cycle = {}
            current_age = 2026 - year  # 假设当前年份 2026

            for i, dy in enumerate(da_yun_list):
                if not dy.getGanZhi():
                    continue
                start_age = dy.getStartYear() - year + 1
                end_age = dy.getEndYear() - year + 1
                cycle = {
                    "start_age": start_age,
                    "end_age": end_age,
                    "heavenly_stem": dy.getGanZhi()[0] if dy.getGanZhi() else "",
                    "earthly_branch": dy.getGanZhi()[1:] if dy.getGanZhi() and len(dy.getGanZhi()) > 1 else "",
                }
                major_cycles.append(cycle)
                if start_age <= current_age <= end_age:
                    current_cycle = cycle

            # 流年（用当前真实日期计算）
            from datetime import date as _date
            _today = _date.today()
            _today_solar = Solar.fromYmdHms(_today.year, _today.month, _today.day, 12, 0, 0)
            _today_lunar = _today_solar.getLunar()
            current_year = {
                "heavenly_stem": _today_lunar.getYearGan(),
                "earthly_branch": _today_lunar.getYearZhi(),
                "year": _today.year,
            }

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            return {
                "subsystem": "bazi",
                "result": {
                    "four_pillars": {
                        "year": {"heavenly_stem": year_gz[0], "earthly_branch": year_gz[1]},
                        "month": {"heavenly_stem": month_gz[0], "earthly_branch": month_gz[1]},
                        "day": {"heavenly_stem": day_gz[0], "earthly_branch": day_gz[1]},
                        "hour": {"heavenly_stem": time_gz[0], "earthly_branch": time_gz[1]},
                    },
                    "five_elements": five_elements,
                    "day_master": day_master,
                    "day_master_strength": day_master_strength,
                    "favorable_elements": favorable,
                    "unfavorable_elements": unfavorable,
                    "ten_gods": ten_gods,
                    "major_cycles": major_cycles,
                    "current_cycle": current_cycle,
                    "current_year": current_year,
                    "confidence": "deterministic",
                },
                "confidence": "deterministic",
                "computation_time_ms": elapsed_ms,
            }

        except Exception as e:
            # 降级：返回简化结果
            return {
                "subsystem": "bazi",
                "result": {
                    "four_pillars": {},
                    "five_elements": {},
                    "day_master": "",
                    "day_master_strength": "unknown",
                    "favorable_elements": [],
                    "unfavorable_elements": [],
                    "ten_gods": {},
                    "major_cycles": [],
                    "current_cycle": {},
                    "current_year": {},
                    "confidence": "error",
                    "error": str(e),
                },
                "confidence": "error",
                "computation_time_ms": int((time.monotonic() - start_time) * 1000),
            }

    def _parse_date(self, birth_date: str) -> tuple[int, int, int]:
        """解析出生日期。"""
        import re
        match = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", birth_date)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        return 0, 0, 0

    def _parse_hour(self, birth_time: str) -> int:
        """解析时辰为小时数。"""
        import re

        # 时辰 → 小时映射
        shichen_map = {
            "子时": 0, "丑时": 2, "寅时": 4, "卯时": 6,
            "辰时": 8, "巳时": 10, "午时": 12, "未时": 14,
            "申时": 16, "酉时": 18, "戌时": 20, "亥时": 22,
        }

        # 直接匹配时辰
        for shichen, hour in shichen_map.items():
            if shichen in birth_time:
                return hour

        # 提取数字
        match = re.search(r"(\d{1,2})", birth_time)
        if match:
            return int(match.group(1))

        # 中文描述
        if "凌晨" in birth_time or "早" in birth_time:
            return 4
        if "上午" in birth_time:
            return 10
        if "中午" in birth_time:
            return 12
        if "下午" in birth_time:
            return 16
        if "晚上" in birth_time or "夜" in birth_time:
            return 20

        return 10  # 默认巳时

    def _count_elements(self, year_gz: str, month_gz: str, day_gz: str, time_gz: str) -> dict[str, int]:
        """统计五行分布。"""
        elements = {"wood": 0, "fire": 0, "earth": 0, "metal": 0, "water": 0}

        for gz in [year_gz, month_gz, day_gz, time_gz]:
            if gz and len(gz) >= 2:
                stem, branch = gz[0], gz[1]
                element = STEM_ELEMENTS.get(stem, "")
                if element:
                    elements[ELEMENT_EN[element]] += 1
                element = BRANCH_ELEMENTS.get(branch, "")
                if element:
                    elements[ELEMENT_EN[element]] += 1

        return elements

    def _judge_strength(self, day_master_element: str, five_elements: dict[str, int], month_gz: str) -> str:
        """判断日主强弱（简化版）。"""
        if not day_master_element:
            return "unknown"

        element_en = ELEMENT_EN.get(day_master_element, "")
        if not element_en:
            return "unknown"

        # 得令：月支是否生助日主
        month_branch_element = BRANCH_ELEMENTS.get(month_gz[1] if month_gz and len(month_gz) > 1 else "", "")
        month_branch_en = ELEMENT_EN.get(month_branch_element, "")

        # 生助关系（同我、生我）
        help_elements = {element_en}
        if element_en == "wood":
            help_elements.add("water")  # 水生木
        elif element_en == "fire":
            help_elements.add("wood")
        elif element_en == "earth":
            help_elements.add("fire")
        elif element_en == "metal":
            help_elements.add("earth")
        elif element_en == "water":
            help_elements.add("metal")

        strength_score = 0
        if month_branch_en in help_elements:
            strength_score += 2  # 得令

        strength_score += five_elements.get(element_en, 0)

        return "strong" if strength_score >= 3 else "weak"

    def _get_favorable(self, day_master_element: str, strength: str) -> tuple[list[str], list[str]]:
        """推断喜用神和忌神。"""
        if not day_master_element:
            return [], []

        # 五行生克关系
        generate_map = {"木": "火", "火": "土", "土": "金", "金": "水", "水": "木"}  # 我生
        control_map = {"木": "土", "火": "金", "土": "水", "金": "木", "水": "火"}  # 我克

        if strength == "strong":
            # 身强喜克泄耗
            favorable = [generate_map.get(day_master_element, ""), control_map.get(day_master_element, "")]
            # 克我的
            reverse_control = {v: k for k, v in control_map.items()}
            favorable.append(reverse_control.get(day_master_element, ""))
            favorable = [e for e in favorable if e]
            unfavorable = [day_master_element]  # 忌比劫
        else:
            # 身弱喜生助
            reverse_generate = {v: k for k, v in generate_map.items()}
            favorable = [day_master_element, reverse_generate.get(day_master_element, "")]
            favorable = [e for e in favorable if e]
            unfavorable = [generate_map.get(day_master_element, ""), control_map.get(day_master_element, "")]
            unfavorable = [e for e in unfavorable if e]

        return favorable, unfavorable
