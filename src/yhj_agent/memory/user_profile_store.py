"""SQLite 用户画像存储。"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class UserProfileStore:
    """SQLite user_profile 表读写器。"""

    def __init__(self, db_path: str | Path = "data/user_profiles.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id TEXT PRIMARY KEY,
                knowledge_level TEXT DEFAULT 'beginner',
                metaphysical_familiarity TEXT DEFAULT '',
                preferred_style TEXT DEFAULT 'balanced',
                personality_traits TEXT DEFAULT '[]',
                past_consultation_types TEXT DEFAULT '[]',
                sensitivity_flags TEXT DEFAULT '[]',
                consultation_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def get(self, user_id: str) -> dict[str, Any]:
        """获取用户画像，不存在则返回默认值。"""
        row = self.conn.execute(
            "SELECT * FROM user_profile WHERE user_id = ?", (user_id,)
        ).fetchone()

        if not row:
            return self._default_profile(user_id)

        return {
            "user_id": row["user_id"],
            "knowledge_level": row["knowledge_level"],
            "metaphysical_familiarity": row["metaphysical_familiarity"],
            "preferred_style": row["preferred_style"],
            "personality_traits": json.loads(row["personality_traits"]),
            "past_consultation_types": json.loads(row["past_consultation_types"]),
            "sensitivity_flags": json.loads(row["sensitivity_flags"]),
            "consultation_count": row["consultation_count"],
        }

    def upsert(self, user_id: str, profile: dict[str, Any]) -> None:
        """插入或更新用户画像。"""
        self.conn.execute("""
            INSERT INTO user_profile (user_id, knowledge_level, metaphysical_familiarity,
                preferred_style, personality_traits, past_consultation_types,
                sensitivity_flags, consultation_count, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                knowledge_level = excluded.knowledge_level,
                metaphysical_familiarity = excluded.metaphysical_familiarity,
                preferred_style = excluded.preferred_style,
                personality_traits = excluded.personality_traits,
                past_consultation_types = excluded.past_consultation_types,
                sensitivity_flags = excluded.sensitivity_flags,
                consultation_count = excluded.consultation_count,
                updated_at = CURRENT_TIMESTAMP
        """, (
            user_id,
            profile.get("knowledge_level", "beginner"),
            profile.get("metaphysical_familiarity", ""),
            profile.get("preferred_style", "balanced"),
            json.dumps(profile.get("personality_traits", []), ensure_ascii=False),
            json.dumps(profile.get("past_consultation_types", []), ensure_ascii=False),
            json.dumps(profile.get("sensitivity_flags", []), ensure_ascii=False),
            profile.get("consultation_count", 0),
        ))
        self.conn.commit()

    def record_consultation(self, user_id: str, consultation_type: str, output_style: str) -> None:
        """记录一次咨询，更新画像成长。"""
        profile = self.get(user_id)

        # 追加 consultation_type（去重）
        past_types = profile.get("past_consultation_types", [])
        if consultation_type and consultation_type not in past_types:
            past_types.append(consultation_type)

        # 累计咨询次数
        count = profile.get("consultation_count", 0) + 1

        # 推断 knowledge_level
        knowledge_level = profile.get("knowledge_level", "beginner")
        if count >= 10:
            knowledge_level = "advanced"
        elif count >= 5:
            knowledge_level = "intermediate"

        self.upsert(user_id, {
            **profile,
            "past_consultation_types": past_types,
            "consultation_count": count,
            "knowledge_level": knowledge_level,
        })

    def _default_profile(self, user_id: str) -> dict[str, Any]:
        return {
            "user_id": user_id,
            "knowledge_level": "beginner",
            "metaphysical_familiarity": "",
            "preferred_style": "balanced",
            "personality_traits": [],
            "past_consultation_types": [],
            "sensitivity_flags": [],
            "consultation_count": 0,
        }

    def close(self) -> None:
        self.conn.close()
