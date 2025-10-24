# app/graphs/bi/nodes/query_data.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langgraph.types import StreamWriter  # 关键：使用 StreamWriter
from open_webui.custom.graphs.bi.state import BiGraphState
from open_webui.custom.graphs.common.model_config import chat_model, embed_query
from open_webui.custom.services.db_service import list_enabled_tables
from open_webui.custom.services.external_db import run_sql_on_external_db
from open_webui.custom.services.milvus_service import search_sql_examples

# ========= 结构化输出 Schema =========

class TableNamesSchema(BaseModel):
    tableNames: List[str] = Field(description="相关的表名列表，必须是数据库中存在的表名")

class SqlSchema(BaseModel):
    sql: str = Field(description="可直接运行的 SQL 语句")

class SqlTitle(BaseModel):
    title: str = Field(description="查询sql的中文解释")

# ========= 统一的流输出工具 =========
def emit(
    writer: StreamWriter,
    *,
    channel: str,
    payload: Dict[str, Any],
) -> None:
    """统一往 LangGraph 的 custom 流里写任意 JSON。
    前端在 astream(..., stream_mode=["custom"]) 的 on_chunk 里拿到 {"custom": <这里的payload>}。
    """
    if writer is None:
        return
    # 可以附带时间戳、版本等
    base = {"channel": channel}
    base.update(payload)
    writer(base)

def emit_think(
    writer: StreamWriter,
    *,
    node: str,
    sub_node: Optional[str] = None,
    state: str,                  # "START" | "END" | "INFO" ...
    content: str,
) -> None:
    """给“思考链/可视化步骤流”用：结构化，便于前端画开始/结束、节点树等。"""
    payload = {
        "node": node,
        "sub_node": sub_node,
        "state": state,
        "content": content,
        # 需要的话加上唯一 ID 或时间戳
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

# ========= 节点实现（签名重点：writer: StreamWriter） =========
async def query_data(state: BiGraphState, writer: StreamWriter) -> Dict[str, Any]:
    # 0) 开始
    emit_think(writer, node="数据查询", state="START", content="开始数据查询流程")
    emit_flow(writer, step=1, node="数据查询", text="启动数据查询流程。")

    # 1) 获取数据库表信息
    tables = list_enabled_tables(state="Y")
    emit_think(writer, node="数据查询", sub_node="数据分类选择", state="START", content="开始识别用户问题相关的数据库表")
    emit_flow(writer, step=2, node="数据分类选择", text="分析问题并匹配可能相关的数据表。")

    chat_history_str = state.get("chatHistoryString", "")
    sql_input = state.get("sql_input", "") or state.get("input", "")

    # 2) 识别相关表
    prompt = (
        f"请综合下面的聊天记录（在上下文存在关联的情况，请根据上下文生成答案）: {chat_history_str}\n\n"
        f"用户的问题：{sql_input}\n\n"
        "数据库中的表信息:\n"
    )
    for t in tables:
        prompt += f"- {t.get('tableName')}: {t.get('tableDescribe')}\n"
    prompt += "\n请根据用户的输入和表信息，判断相关的表名，并输出表名列表。输出的表名必须来自上述表信息中的 tableName。"

    if chat_model is None:
        related_table_names = [t.get("tableName") for t in tables][:3]
    else:
        structured_llm = chat_model.with_structured_output(TableNamesSchema)
        resp: TableNamesSchema = structured_llm.invoke(prompt)
        related_table_names = resp.tableNames

    related_tables = [t for t in tables if t.get("tableName") in related_table_names]
    emit_think(
        writer,
        node="数据查询",
        sub_node="数据分类选择",
        state="END",
        content=f"识别到相关数据表：{', '.join(related_table_names) if related_table_names else '无'}",
    )
    emit_flow(writer, step=3, node="数据分类选择", text=f"已选中相关表：{', '.join(related_table_names) if related_table_names else '（未匹配）'}。")

    # 3) 生成表信息提示词
    table_info_prompt = "数据库表信息:\n"
    for t in related_tables:
        table_info_prompt += f"- 表名: {t.get('tableName')}\n"
        table_info_prompt += f"  描述: {t.get('prompt')}\n"
        table_info_prompt += f"  数据库类型: {t.get('databaseType_dictText')}\n\n"

    # 4) 从 Milvus 检索 SQL 示例
    emit_think(writer, node="数据查询", sub_node="获取相关示例", state="START", content="开始从Milvus数据库检索SQL示例")
    emit_flow(writer, step=4, node="获取相关示例", text="检索相似问题的 SQL 示例以辅助生成。")
    sql_examples_prompt = "参考 SQL 示例:\n"
    try:
        query_vector = embed_query(sql_input)
        sql_examples = search_sql_examples(
            query_vector=query_vector,
            collection_name="llm_jeecgboot_glj",
            vector_field="vector",
            top_k=2,
            output_fields=["question", "content"],
            expr='training_data_type == "sql"',
        )
        for idx, ex in enumerate(sql_examples, start=1):
            sql_examples_prompt += f"示例 {idx}: 问题：{ex.get('question')}\n 答案：{ex.get('content')}\n "
    except Exception:
        sql_examples_prompt += "(无可用示例)\n"

    emit_think(writer, node="数据查询", sub_node="获取相关示例", state="END", content="成功检索到SQL示例")
    emit_flow(writer, step=5, node="获取相关示例", text="已获取到可参考的 SQL 示例。")

    # 5) 生成 SQL
    emit_think(writer, node="数据查询", sub_node="生成查询语句", state="START", content="开始生成SQL查询语句")
    emit_flow(writer, step=6, node="生成查询语句", text="根据表信息与示例自动生成可执行的 SQL。")

    system_prompt = f"""
你是一个 SQL 专家，负责根据用户的问题和数据库表信息生成可直接运行的 SQL 语句。

{table_info_prompt}

{sql_examples_prompt}

请综合下面的聊天记录（在上下文存在关联的情况，请根据上下文生成答案）: {chat_history_str}

用户的问题：{sql_input}

请根据上述信息，生成一个可直接运行的 SQL 语句来查询用户需要的数据。
SQL 语句必须基于提供的表名和字段，且符合数据库类型的要求,展示的字段名尽量 as 为中文,注意别名需要用引号包裹,
需要注意的是如果是聚合的数值类数据为空或者不存在的情况尽量赋值为 0 或者不查询该字段。
    """.strip()

    if chat_model is None:
        generated_sql = "SELECT 1 as `示例`;"
    else:
        sql_resp: SqlSchema = chat_model.with_structured_output(SqlSchema).invoke(system_prompt)
        generated_sql = sql_resp.sql.strip()

    emit_think(writer, node="数据查询", sub_node="生成查询语句", state="END", content="SQL查询语句生成成功")
    emit_flow(writer, step=7, node="生成查询语句", text="SQL 已生成。")

    # 6) 执行 SQL
    emit_think(writer, node="数据查询", sub_node="查询数据", state="START", content="开始执行SQL查询")
    emit_flow(writer, step=8, node="查询数据", text="正在执行 SQL 并获取结果。")
    try:
        query_results = run_sql_on_external_db(generated_sql)
    except Exception:
        query_results = []

    column_headers = list(query_results[0].keys()) if query_results else []
    column_headers_string = ",".join(column_headers)

    emit_think(writer, node="数据查询", sub_node="查询数据", state="END", content="SQL查询执行成功，数据已获取")
    emit_flow(writer, step=9, node="查询数据", text=f"已获取 {len(query_results)} 条记录。")

    # 7) 标题生成
    emit_think(writer, node="数据查询", sub_node="标题生成", state="START", content="开始生成数据查询结果的标题")
    emit_flow(writer, step=10, node="标题生成", text="生成结果标题。")
    import datetime
    title_prompt = (
        f"请综合下面的聊天记录（在上下文存在关联的情况）: {chat_history_str}\n\n"
        f"用户的问题：{sql_input}\n\n"
        f"生成的sql语句：{generated_sql}\n\n"
        f"当前时间：{datetime.datetime.now()}\n\n"
        "生成sql语句所查询的数据的简要描述，简洁描述内容即可不要超过30个字"
    )
    if chat_model is None:
        title = "查询结果"
    else:
        title_resp: SqlTitle = chat_model.with_structured_output(SqlTitle).invoke(title_prompt)
        title = title_resp.title

    emit_think(writer, node="数据查询", sub_node="标题生成", state="END", content=f"成功生成标题：{title}")
    emit_flow(writer, step=11, node="标题生成", text=f"标题：{title}")

    # 8) 结束
    emit_think(writer, node="数据查询", state="END", content="数据查询流程完成")
    emit_flow(writer, step=12, node="数据查询", text="流程结束。")

    return {
        "chatHistory": [
            {"role": "answer", "content": f"生成的 SQL: {generated_sql}"},
            {"role": "queryResults", "content": query_results, "title": title},
        ],
        "sql": generated_sql,
        "columnHeaders": column_headers_string,
    }
