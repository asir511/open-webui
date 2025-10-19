# app/graphs/bi/nodes/generate_chart.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langgraph.types import StreamWriter  # 关键：使用 StreamWriter

from open_webui.custom.graphs.bi.state import BiGraphState
from open_webui.custom.graphs.common.model_config import chat_model
# 约定的服务接口
from open_webui.custom.services.recipe_store import list_recipes, load_recipe_detail


# ---------- 结构化输出 Schema ----------
class NeedsSQLSchema(BaseModel):
    needs_new_sql: bool = Field(description="是否需要生成新的 SQL 查询")


class SplitQuestionSchema(BaseModel):
    sql_input: str = Field(description="与数据查询相关的问题")


class SelectRecipeSchema(BaseModel):
    key: str = Field(description="从 availableKeys 中选出的图文档 key")


class ChartTitle(BaseModel):
    title: str = Field(description="图表中文简短标题（<=30字）")


# ---------- AG Grid 全局最小规则（固定拼接） ----------
AGGRID_GLOBAL_MIN = """
只输出一段 JavaScript 代码，且必须为：
gridApi.value.createRangeChart({ ... });
要求：
1) cellRange.columns 中的列名必须严格来自 columnHeaders（不要杜撳）。
2) chartType 必须与所选图类型文档的要求匹配。
3) 如用户提到“前N条/Top N”，在 cellRange 增加 rowStartIndex: 0, rowEndIndex: N。
4) 可设置 chartThemeOverrides.common.title.enabled=true 并用简短中文 text 命名标题；legend.position 可选 right/left/top/bottom。
5) 严禁输出解释/注释/额外文本，只能输出这一段 JS 代码。
""".strip()


# ---------- 统一的流输出工具 ----------
def emit(writer: StreamWriter, *, channel: str, payload: Dict[str, Any]) -> None:
    if writer is None:
        return
    data = {"channel": channel}
    data.update(payload)
    writer(data)


def emit_think(
    writer: StreamWriter,
    *,
    node: str,
    sub_node: Optional[str] = None,
    state: str,
    content: str,
) -> None:
    emit(
        writer,
        channel="think",
        payload={"node": node, "sub_node": sub_node, "state": state, "content": content},
    )


def emit_flow(
    writer: StreamWriter,
    *,
    text: str,
    step: Optional[int] = None,
    node: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {"text": text}
    if step is not None:
        payload["step"] = step
    if node is not None:
        payload["node"] = node
    emit(writer, channel="flow", payload=payload)


# ---------- 节点主逻辑 ----------
async def generate_chart(state: BiGraphState, writer: StreamWriter) -> Dict[str, Any]:
    """
    图像生成节点（AG Grid）：
    - 数据充分 → 选择合适的 recipe_key → 用 prompt_text + columnHeaders 生成唯一一段 JS 代码
    - 数据不足 → 拆分 sql_input，回跳 queryData
    返回:
      {
        "chatHistory": [{"role": "chartCode", "content": "<JS代码>", "title": "<标题>"}],
        "needs_data_query": False,
      }
      或回跳:
      {
        "sql_input": "<查询问句>",
        "needs_data_query": True,
        "next_after_query": "generateChart"
      }
    """
    # START
    emit_think(writer, node="图像生成", state="START", content="开始图像生成流程")
    emit_flow(writer, step=1, node="图像生成", text="启动图像生成流程。")

    chat_history = state.get("chatHistoryString", "") or ""
    col_headers = state.get("columnHeaders", "") or ""
    current_sql = state.get("sql", "") or ""
    chart_input = state.get("chart_input", "") or state.get("input", "")

    # Step 1: 是否需要新的 SQL
    emit_think(writer, node="图像生成", sub_node="数据完整性检验", state="START", content="检查当前列是否足以绘图")
    emit_flow(writer, step=2, node="数据完整性检验", text="检查字段是否满足绘图需要。")

    need_prompt = f"""
基于以下信息判断绘图是否需要新的数据列（从而需要新的SQL）：
- 历史：{chat_history}
- 当前列头（逗号分隔）：{col_headers}
- 当前SQL：{current_sql}
- 用户问题：{chart_input}
若当前列头足以支撑绘图，返回 needs_new_sql=false；否则 true。
""".strip()

    if chat_model is None:
        needs_new_sql = False
    else:
        needs_new_sql = chat_model.with_structured_output(NeedsSQLSchema).invoke(need_prompt).needs_new_sql

    if needs_new_sql:
        emit_think(writer, node="图像生成", sub_node="数据完整性检验", state="END", content="需要新SQL")
        emit_flow(writer, step=3, node="数据完整性检验", text="需要补充字段，准备提取查询问句。")

        # 拆分查询问句
        emit_think(writer, node="图像生成", sub_node="问题拆分", state="START", content="提取与数据查询相关的问句")
        emit_flow(writer, step=4, node="问题拆分", text="从上下文中提取数据查询问句。")

        split_prompt = f"从历史与问题中提取与数据查询相关的问句（若不明确返回空）：\n历史：{chat_history}\n问题：{chart_input}"
        if chat_model is None:
            sql_input = ""
        else:
            sql_input = chat_model.with_structured_output(SplitQuestionSchema).invoke(split_prompt).sql_input.strip()

        if not sql_input:
            emit_think(writer, node="图像生成", sub_node="问题拆分", state="END", content="无法提取查询问句")
            emit_think(writer, node="图像生成", state="END", content="数据不足且无法提炼查询问句")
            emit_flow(writer, step=5, node="问题拆分", text="未能提取到明确的查询问句。")
            emit_flow(writer, step=6, node="图像生成", text="流程结束。")

            return {
                "chatHistory": [{"role": "assistant", "content": "当前数据列不足以绘图，请补充更清晰的查询字段/维度需求"}],
                "needs_data_query": False,
            }

        emit_think(writer, node="图像生成", sub_node="问题拆分", state="END", content="已提取查询问句，回跳数据查询")
        emit_flow(writer, step=5, node="问题拆分", text="已提取查询问句，准备回到数据查询节点。")
        emit_think(writer, node="图像生成", state="END", content="等待数据查询后继续")
        emit_flow(writer, step=6, node="图像生成", text="流程暂停，等待新数据。")

        return {"sql_input": sql_input, "needs_data_query": True, "next_after_query": "generateChart"}

    emit_think(writer, node="图像生成", sub_node="数据完整性检验", state="END", content="数据充分")
    emit_flow(writer, step=3, node="数据完整性检验", text="字段充分，可直接绘图。")

    # Step 2: 让 LLM 选择 recipe_key
    recipes = list_recipes(recipe_type="aggrid_chart", lang="zh")  # [{recipe_key, display_name, llm_hint}]
    if not recipes:
        return {"error": "未找到任何图表文档（app_aggrid_recipe 为空或未启用）"}

    available_keys = [r["recipe_key"] for r in recipes]
    hints_text = "\n".join([f"- {r['recipe_key']}: {r.get('llm_hint', '')}" for r in recipes])

    emit_think(writer, node="图像生成", sub_node="方案选择", state="START", content="开始选择最合适的图表方案")
    emit_flow(writer, step=4, node="方案选择", text="匹配最合适的图表类型。")

    choose_prompt = f"""
基于下述 llm_hint，从 availableKeys 中选择最适合的图表 key：
- 历史：{chat_history}
- 问题：{chart_input}
可选：{available_keys}
hint：
{hints_text}
只返回 JSON: {{ "key": "<one of availableKeys>" }}
""".strip()

    if chat_model is None:
        chosen_key = available_keys[0]
    else:
        chosen = chat_model.with_structured_output(SelectRecipeSchema).invoke(choose_prompt).key
        chosen_key = chosen if chosen in available_keys else available_keys[0]

    emit_think(writer, node="图像生成", sub_node="方案选择", state="END", content=f"已选择：{chosen_key}")
    emit_flow(writer, step=5, node="方案选择", text=f"选择图表方案：{chosen_key}。")

    # Step 3: 读取该 recipe 的 prompt_text 与 example_code
    detail = load_recipe_detail(chosen_key) or {}
    prompt_text = detail.get("prompt_text", "")
    example_code = detail.get("example_code", "")

    # Step 4: 生成最终代码（只允许输出一段 gridApi.value.createRangeChart({...});）
    header_list = [h.strip() for h in col_headers.split(",") if h.strip()]
    code_prompt = f"""
{AGGRID_GLOBAL_MIN}

[文档提示词]
{prompt_text}

[示例代码风格参考（可选）]
{example_code}

[上下文]
- columnHeaders: {header_list}
- 历史：{chat_history}
- 用户问题：{chart_input}

**只输出一段 JS：gridApi.value.createRangeChart({{ ... }});**
**不要任何解释与多余字符。**
""".strip()

    emit_think(writer, node="图像生成", sub_node="生成代码", state="START", content="开始生成图表代码")
    emit_flow(writer, step=6, node="生成代码", text="根据文档提示与列头生成 AG Grid 代码。")

    if chat_model is None:
        code_str = 'gridApi.value.createRangeChart({ cellRange:{ columns:[] }, chartType:"groupedColumn" });'
    else:
        # ChatModel 直接返回文本：必须是一段 JS 代码
        code_str = chat_model.invoke(code_prompt).content

    emit_think(writer, node="图像生成", sub_node="生成代码", state="END", content="代码生成完成")
    emit_flow(writer, step=7, node="生成代码", text="图表代码生成完成。")

    # Step 5: 标题
    emit_think(writer, node="图像生成", sub_node="图标题生成", state="START", content="生成图表标题")
    emit_flow(writer, step=8, node="图标题生成", text="生成图表标题。")

    if chat_model is None:
        title = "数据统计图"
    else:
        title_prompt = f"为该图生成一个不超过30字的中文标题：\n历史：{chat_history}\n问题：{chart_input}"
        title = chat_model.with_structured_output(ChartTitle).invoke(title_prompt).title

    emit_think(writer, node="图像生成", sub_node="图标题生成", state="END", content=f"标题：{title}")
    emit_flow(writer, step=9, node="图标题生成", text=f"标题确定：{title}")

    # END
    emit_think(writer, node="图像生成", state="END", content="完成")
    emit_flow(writer, step=10, node="图像生成", text="流程结束。")

    return {
        "chatHistory": [{"role": "chartCode", "content": code_str, "title": title}],
        "needs_data_query": False,
    }
