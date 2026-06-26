from __future__ import annotations

from pathlib import Path

from yhj_agent.common.prompt_loader import PromptLoader


PROMPT_ROOT = Path("src/yhj_agent/prompts")
BROKEN_PROMPTS = [
    PROMPT_ROOT / "report_generator/user.j2",
    PROMPT_ROOT / "report_generator/shot.j2",
    PROMPT_ROOT / "synthesis/user.j2",
    PROMPT_ROOT / "synthesis/shot.j2",
]


def test_broken_prompts_do_not_contain_mojibake_tokens():
    for path in BROKEN_PROMPTS:
        text = path.read_text(encoding="utf-8")
        assert "????" not in text
        assert "浣犳槸" not in text


def test_intent_router_user_prompt_has_no_bom():
    text = (PROMPT_ROOT / "intent_router/user.j2").read_text(encoding="utf-8")

    assert not text.startswith("\ufeff")


def test_prompt_loader_renders_repaired_templates_with_chinese_keywords():
    loader = PromptLoader()

    synthesis_user = loader.render(
        "synthesis/user.j2",
        specialists={},
        debate_output={"debate_occurred": False},
    )
    report_user = loader.render(
        "report_generator/user.j2",
        raw_input="事业咨询",
        focus_question="未来三年事业方向",
        output_style="深刻且实际",
        presentation_mode="prose",
        specialists={},
        synthesis={"synthesis_text": "综合判断"},
        evaluation={"overall_score": 4.2, "safety_score": 4.5},
        user_profile={"knowledge_level": "beginner"},
        special_format_requests=[],
    )

    assert "专家" in synthesis_user or "综合" in synthesis_user
    assert "用户原始输入" in report_user
    assert "综合研判" in report_user
