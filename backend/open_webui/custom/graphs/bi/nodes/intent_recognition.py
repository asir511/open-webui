# app/graphs/bi/nodes/intent_recognition.py
from __future__ import annotations
from typing import Dict, Any, Literal, Optional
from pydantic import BaseModel
from open_webui.custom.graphs.bi.state import BiGraphState
from open_webui.custom.graphs.common.model_config import chat_model
from langchain_core.runnables import RunnableConfig

# ====== 等价 zod 的结构化输出约束 ======
IntentLiteral = Literal["query_data", "generate_chart", "normal_QA"]

class IntentResponse(BaseModel):
    intent: IntentLiteral

def _write(config: Dict[str, Any] | None, msg: str) -> None:
    if not config:
        return
    writer = config.get("writer")
    if callable(writer):
        writer(msg)

async def intent_recognition(state: BiGraphState, config: Optional[RunnableConfig] = None) -> Dict[str, Any]:
    """
    等价 TS 的意图识别节点：
      - 结构化输出 { intent: 'query_data' | 'generate_chart' | 'normal_QA' }
      - 根据 intent 回填 sql_input / chart_input
      - 兼容 config.writer 日志输出
    """
    _write(config, '{"node":"意图识别","state":"START"}')
    _write(config, '{"node":"意图识别","sub_node":"意图判断","state":"START","content":"识别中..."}')

    chat_history_str = state.get("chatHistoryString", "")
    user_input = state.get("input", "")

    prompt = (
        "请判断以下用户查询的意图，并返回以下英文标记之一："
        "query_data, generate_chart, normal_QA。\n\n"
        f"历史对话：{chat_history_str}\n\n"
        f"用户问题：{user_input}\n"
        "注意: 查询、统计、分类、排序等为目的都算 query_data；"
        "若目的是画图/绘图，则归类 generate_chart。"
    )

    if chat_model is None:
        # 占位：没有模型就兜底 normal_QA，避免阻断流程
        intent = "normal_QA"
        resp = IntentResponse(intent=intent)  # 校验
    else:
        # LangChain Python：with_structured_output(PydanticModel)
        structured_llm = chat_model.with_structured_output(IntentResponse)
        # 支持字符串 prompt 或 HumanMessage；这里直接字符串即可
        resp: IntentResponse = structured_llm.invoke(prompt)

    intent = resp.intent
    _write(config, f'{{"node":"意图识别","sub_node":"意图判断","state":"END","content":"{intent}"}}')
    _write(config, '{"node":"意图识别","state":"END"}')

    if intent == "query_data":
        return {"intent": intent, "sql_input": user_input}
    if intent == "generate_chart":
        return {"intent": intent, "chart_input": user_input}
    # normal_QA
    return {"intent": intent}
