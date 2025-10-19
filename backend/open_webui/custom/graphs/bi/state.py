# app/graphs/bi/state.py
from __future__ import annotations

import operator
from typing import TypedDict, List, Literal, Annotated, Optional
from open_webui.custom.graphs.common.state_types import ChatItem

Intent = Literal["query_data", "generate_chart", "normal_QA"]
NextAfterQuery = Literal["generateChart", "end"]

class BiGraphState(TypedDict, total=False):
    # 原始用户输入
    input: str
    # 意图
    intent: Intent

    # 聊天历史（等价 TS 的 reducer: concat）
    chatHistory: Annotated[List[ChatItem], operator.add]
    chatHistoryString: str

    # BI 相关
    sql_input: str              # 数据查询相关的问题
    chart_input: str            # 图像生成相关的问题
    needs_data_query: bool      # 是否需要查询数据
    next_after_query: NextAfterQuery  # 数据查询后的下一步

    sql: str                    # 当前 SQL 查询
    columnHeaders: str
