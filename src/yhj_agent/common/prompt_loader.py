"""Jinja2 Prompt 模板加载器。"""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


class PromptLoader:
    """从 prompts/ 目录加载并渲染 Jinja2 模板。"""

    def __init__(self, template_dir: str | Path | None = None) -> None:
        self.env = Environment(
            loader=FileSystemLoader(
                str(template_dir or Path(__file__).parent.parent / "prompts")
            ),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, template_path: str, **kwargs) -> str:
        """渲染指定模板，返回字符串。"""
        return self.env.get_template(template_path).render(**kwargs)

    def list_templates(self) -> list[str]:
        """列出所有可用模板。"""
        return self.env.list_templates(extensions=["j2"])
