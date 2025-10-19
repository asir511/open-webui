# app/graphs/bi/nodes/init_state.py
from __future__ import annotations
from typing import Dict, Any, Optional
from open_webui.custom.graphs.bi.state import BiGraphState
from open_webui.custom.graphs.common.utils import format_chat_history
from langchain_core.runnables import RunnableConfig

async def init_state(state: BiGraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    等价 TS:
      - 压入当前 user 输入到 chatHistory
      - 生成 chatHistoryString（仅取最近 3 轮）
      - 重置各控制字段
    """
    history = list(state.get("chatHistory", []) or [])
    if state.get("input"):
        history.append({"role": "user", "content": state["input"]})

    chat_history_str = format_chat_history(history, rounds=3)

    return {
        "chatHistory": history,
        "chatHistoryString": chat_history_str,
        "intent": None,              # 重置意图
        "sql_input": None,           # 重置数据查询相关问题
        "chart_input": None,         # 重置图像生成相关问题
        "needs_data_query": False,   # 重置为 false
        "next_after_query": "end",   # 重置为 end
    }
