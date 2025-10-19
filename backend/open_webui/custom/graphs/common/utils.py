# app/graphs/common/utils.py
from __future__ import annotations
from typing import List, Dict, Any

def format_chat_history(
    history: List[Dict[str, Any]] | None,
    rounds: int = 3
) -> str:
    """
    仅保留 user/assistant/answer 角色的最近 N 轮对话，拼成字符串。
    兼容 None/空列表。
    """
    if not history:
        return ""

    filtered = [h for h in history if h.get("role") in ("user", "assistant", "answer")]

    selected_rounds: list[dict] = []
    user_count = 0
    current_round: list[dict] = []

    # 从末尾倒序扫描
    for i in range(len(filtered) - 1, -1, -1):
        entry = filtered[i]
        if entry.get("role") == "user":
            if current_round:
                selected_rounds[:0] = current_round  # 等价于 unshift 全部
                current_round = []
            user_count += 1
            if user_count > rounds:
                break
        current_round.insert(0, {"role": entry.get("role", ""), "content": entry.get("content", "")})

    if current_round:
        selected_rounds[:0] = current_round

    return "\n".join(f"{e['role']}: {e['content']}" for e in selected_rounds if "content" in e)
