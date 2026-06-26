"""LangGraph Checkpointer 集成。

使用 SqliteSaver 持久化 graph state，支持 interrupt/resume。
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path


def get_checkpointer(db_path: str | None = None) -> object:
    """获取 LangGraph Checkpointer。

    优先级：
    1. 显式传入 db_path
    2. 环境变量 CHECKPOINTER_DB_PATH
    3. 默认 data/checkpoints.db
    """
    if db_path is None:
        db_path = os.environ.get("CHECKPOINTER_DB_PATH", "data/checkpoints.db")

    from langgraph.checkpoint.sqlite import SqliteSaver

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)
