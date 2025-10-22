# app/graphs/bi/nodes/normal_qa.py
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from langgraph.types import StreamWriter  # 关键：使用 StreamWriter

from open_webui.custom.graphs.bi.state import BiGraphState
from open_webui.custom.graphs.common.model_config import chat_model

# ========= 统一的流输出工具（与 query_data 保持一致） =========

def emit(
    writer: StreamWriter,
    *,
    channel: str,
    payload: Dict[str, Any],
) -> None:
    """统一往 LangGraph 的 custom 流里写任意 JSON。
    前端在 astream(..., stream_mode=["custom"]) 的 on_chunk 里拿到 {"custom": <payload>}。
    """
    if writer is None:
        return
    base = {"channel": channel}
    base.update(payload)
    writer(base)

def emit_think(
    writer: StreamWriter,
    *,
    node: str,
    sub_node: Optional[str] = None,
    state: str,                  # "START" | "END" | "INFO" ...
    content: str = "",
) -> None:
    """给“思考链/可视化步骤流”用：结构化，便于前端画开始/结束、节点树等。"""
    payload = {
        "node": node,
        "sub_node": sub_node,
        "state": state,
        "content": content,
    }
    emit(writer, channel="think", payload=payload)

def emit_flow(
    writer: StreamWriter,
    *,
    text: str,
    step: Optional[int] = None,
    node: Optional[str] = None,
) -> None:
    """给“流程链/用户可见自然语言”用：一句话描述当前动作。"""
    payload = {"text": text}
    if step is not None:
        payload["step"] = step
    if node is not None:
        payload["node"] = node
    emit(writer, channel="flow", payload=payload)

# ========= 可选：结构化回答（保留接口，后续可扩展） =========

class AnswerSchema(BaseModel):
    answer: str = Field(description="面向用户的最终中文回答")

# ========= 节点实现 =========

async def normal_qa(state: BiGraphState, writer: StreamWriter) -> Dict[str, Any]:
    """
    普通问答节点：
    - 读取 state.input（以及可选的 chatHistoryString）
    - 通过 chat_model 生成中文回答
    - 通过 StreamWriter 输出结构化流程事件
    - 返回 chatHistory（role: assistant）
    """
    # 0) 开始
    emit_think(writer, node="常规问答", state="START", content="开始常规问答流程")
    emit_flow(writer, step=1, node="常规问答", text="启动常规问答。")

    # 1) 读取输入与上下文
    user_input: str = state.get("input", "") or ""
    chat_history_str: str = state.get("chatHistoryString", "") or ""
    emit_think(
        writer,
        node="常规问答",
        sub_node="读取上下文",
        state="INFO",
        content=f"获取到用户输入与历史上下文（长度：{len(chat_history_str)}）。",
    )
    emit_flow(writer, step=2, node="常规问答", text="载入用户输入与上下文。")

    # 2) 生成回答
    emit_think(writer, node="常规问答", sub_node="生成回复", state="START", content="开始调用大模型生成回答")
    emit_flow(writer, step=3, node="常规问答", text="正在生成回答…")

    # 构造提示词（尽量简洁，优先中文，允许参考上下文）
    prompt = (
        "你是一个中文助手，请用清晰、简洁、准确的中文回答用户问题。\n\n"
        "（如上下文与当前问题存在关联，请合理引用上下文信息；如无关，请仅针对当前问题作答。）\n\n"
        f"【上下文摘录】\n{chat_history_str}\n\n"
        f"【用户问题】\n{user_input}\n\n"
        "【回答要求】\n"
        "1) 直接给出结论与必要解释；\n"
        "2) 条理清晰，分点叙述（如适用）；\n"
        "3) 不确定时坦诚说明，并给出可能的方向或补充信息需求。\n"
    )

    answer = ""
    try:
        if chat_model is None:
            # 兜底：未配置模型
            answer = "（系统未配置对话模型，暂时无法生成内容。）"
        else:
            # 可选：若未来需要结构化输出，可切到 .with_structured_output(AnswerSchema)
            resp = chat_model.invoke(prompt)
            # 兼容不同返回类型
            if hasattr(resp, "content"):
                answer = resp.content
            else:
                answer = str(resp)
    except Exception as e:
        answer = f"抱歉，生成回答时出现异常：{e!s}"

    emit_think(writer, node="常规问答", sub_node="生成回复", state="END", content="生成成功")
    emit_flow(writer, step=4, node="常规问答", text="回答已生成。")

    # 3) 结束
    emit_think(writer, node="常规问答", state="END", content="常规问答流程完成")
    emit_flow(writer, step=5, node="常规问答", text="流程结束。")

    return {
        "chatHistory": [
            {"role": "assistant", "content": answer}
        ]
    }
