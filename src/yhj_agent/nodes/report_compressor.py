"""Node13: ReportCompressor — 报告压缩（纯规则截断）。"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict


class ReportCompressorOutput(BaseModel):
    """ReportCompressor 输出。"""

    model_config = ConfigDict(extra="forbid")

    compressed_report: str
    original_word_count: int = 0
    compressed_word_count: int = 0
    compression_ratio: float = 0.0
    format_used: str = "short"


class ReportCompressor:
    """按 section 保留前 2-3 句，提取关键要点。"""

    def compress(
        self,
        report_text: str,
        format_type: str = "short",
        max_sentences_per_section: int = 3,
    ) -> ReportCompressorOutput:
        """压缩报告。

        Args:
            report_text: 原始报告文本
            format_type: "short" | "medium" | "bullet"
            max_sentences_per_section: 每个 section 保留的最大句子数

        Returns:
            ReportCompressorOutput
        """
        if not report_text:
            return ReportCompressorOutput(
                compressed_report="暂无报告内容。",
                format_used=format_type,
            )

        original_len = len(report_text)

        # 按段落分割
        sections = re.split(r"\n{2,}", report_text)
        compressed_sections: list[str] = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if format_type == "bullet":
                # 提取关键句（保留前 N 句）
                sentences = self._split_sentences(section)
                compressed = "\n".join(f"- {s}" for s in sentences[:max_sentences_per_section])
            elif format_type == "medium":
                # 保留前 N 句
                sentences = self._split_sentences(section)
                compressed = " ".join(sentences[:max_sentences_per_section])
            else:  # short
                # 保留前 2 句
                sentences = self._split_sentences(section)
                compressed = " ".join(sentences[:2])

            if compressed:
                compressed_sections.append(compressed)

        compressed_text = "\n\n".join(compressed_sections)
        compressed_len = len(compressed_text)

        return ReportCompressorOutput(
            compressed_report=compressed_text,
            original_word_count=original_len,
            compressed_word_count=compressed_len,
            compression_ratio=round(compressed_len / max(original_len, 1), 2),
            format_used=format_type,
        )

    def _split_sentences(self, text: str) -> list[str]:
        """按句子分割。"""
        # 中文句子分割
        sentences = re.split(r"[。！？\n]", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def restyle(
        self,
        report_text: str,
        target_style: str = "深刻且实际",
    ) -> ReportCompressorOutput:
        """LLM-driven style rewrite.

        Args:
            report_text: original report text
            target_style: desired style description

        Returns:
            ReportCompressorOutput with restyled text
        """
        if not report_text.strip():
            return ReportCompressorOutput(
                compressed_report="",
                format_used="style_only",
            )

        original_len = len(report_text)

        try:
            from yhj_agent.services.llm_client import MimoLLMClient
            from yhj_agent.shared.schemas.llm import LLMMessage, LLMRequest

            llm = MimoLLMClient()

            system_prompt = (
                "你是一位报告风格调整助手。根据用户要求的风格，改写报告文本。\n"
                "要求：\n"
                "1. 保持报告的核心信息和结论不变\n"
                "2. 仅调整表达方式、语气和文风\n"
                "3. 不添加新信息，不删除关键结论\n"
                "4. 直接输出改写后的报告文本，不要加任何前后说明"
            )
            user_prompt = f"目标风格：{target_style}\n\n原文：\n{report_text}\n\n请输出改写后的报告："

            request = LLMRequest(
                messages=[
                    LLMMessage(role="system", content=system_prompt),
                    LLMMessage(role="user", content=user_prompt),
                ],
                model=llm.config.light_model,
                temperature=0.5,
            )
            response = llm.chat(request)
            restyled_text = response.content

            return ReportCompressorOutput(
                compressed_report=restyled_text,
                original_word_count=original_len,
                compressed_word_count=len(restyled_text),
                compression_ratio=round(len(restyled_text) / max(original_len, 1), 2),
                format_used="style_only",
            )
        except Exception:
            # Fallback to rule-based compression on LLM failure
            return self.compress(report_text=report_text, format_type="short")

