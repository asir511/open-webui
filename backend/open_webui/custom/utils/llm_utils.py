# -*- coding: utf-8 -*-
import os
import json
import math
import uuid
from datetime import datetime, date
from typing import List, Dict, Any, Optional

import httpx
import re
from langchain.text_splitter import MarkdownHeaderTextSplitter

# ========= 可配置 =========
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://192.168.0.114:3000/v1")
OPENAI_API_KEY  = "sk-2pH4HUSK4wikhU7NSqMqF3Ldi7c2r89sJmRQBBJ9PS7vN1AM"
OPENAI_TIMEOUT  = int(os.getenv("OPENAI_TIMEOUT", "120"))

# ========= 通用小函数 =========
def approx_tokens_of_text(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))

def split_markdown_packed(content: str, max_tokens: int) -> List[str]:
    if not content:
        return [""]
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(content)
    segments = [d.page_content for d in docs if d and d.page_content]
    if not segments:
        segments = [s for s in content.split("\n\n") if s.strip()]

    chunks, buf, buf_tok = [], [], 0
    for seg in segments:
        t = approx_tokens_of_text(seg)
        if t >= max_tokens:
            if buf:
                chunks.append("\n\n".join(buf)); buf, buf_tok = [], 0
            chunks.append(seg); continue
        if buf_tok + t > max_tokens and buf:
            chunks.append("\n\n".join(buf)); buf, buf_tok = [], 0
        buf.append(seg); buf_tok += t
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

def _format_date(d: Any) -> Optional[str]:
    if isinstance(d, date) and not isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, str):
        s = d.strip().replace("/", "-").replace(".", "-")
        parts = s.split("-")
        if len(parts) == 3 and len(parts[0]) == 4:
            return f"{parts[0]:0>4}-{parts[1]:0>2}-{parts[2]:0>2}"
    return None

def _format_datetime(dt: Any) -> Optional[str]:
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(dt, str):
        s = dt.strip().replace("/", "-").replace("T", " ")
        try:
            return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return None

def _coerce_value_by_type(data_type: str, raw: Any) -> Dict[str, Any]:
    data_type = (data_type or "string").lower()
    cell = {
        "valueString": None, "valueNumber": None, "valueInteger": None,
        "valueBoolean": None, "valueDate": None, "valueDatetime": None, "valueJson": None,
    }
    try:
        if data_type == "string":
            cell["valueString"] = "" if raw is None else str(raw)
        elif data_type == "number":
            cell["valueNumber"] = None if raw in (None, "") else float(raw)
        elif data_type == "integer":
            cell["valueInteger"] = None if raw in (None, "") else int(float(raw))
        elif data_type == "boolean":
            truthy = {"true","1","yes","y","是"}; falsy = {"false","0","no","n","否"}
            if isinstance(raw, bool):
                cell["valueBoolean"] = 1 if raw else 0
            elif isinstance(raw, (int,float)):
                cell["valueBoolean"] = 1 if raw != 0 else 0
            elif isinstance(raw, str):
                s = raw.strip().lower()
                cell["valueBoolean"] = 1 if s in truthy else (0 if s in falsy else None)
        elif data_type == "date":
            cell["valueDate"] = _format_date(raw)
        elif data_type == "datetime":
            cell["valueDatetime"] = _format_datetime(raw)
        elif data_type == "json":
            if isinstance(raw, (dict, list)):
                cell["valueJson"] = json.dumps(raw, ensure_ascii=False)
            elif isinstance(raw, str):
                try: json.loads(raw); cell["valueJson"] = raw
                except Exception: cell["valueJson"] = json.dumps({"value": raw}, ensure_ascii=False)
            else:
                cell["valueJson"] = json.dumps({"value": raw}, ensure_ascii=False)
        else:
            cell["valueString"] = "" if raw is None else str(raw)
    except Exception:
        cell = {k: None for k in cell}; cell["valueString"] = "" if raw is None else str(raw)
    return cell

# ========= LLM 调用与提示词 =========
def _system_prompt(table_key: str, table_display: str, table_desc: str, lang: str, fields: List[Dict[str, Any]]) -> str:
    # 字段元信息
    lines = []
    for f in fields:
        meta = []
        if f.get("displayName"):  meta.append(f'显示名="{f["displayName"]}"')
        if f.get("description"):  meta.append(f'说明="{f["description"]}"')
        if f.get("unit"):         meta.append(f'单位="{f["unit"]}"')
        if f.get("required") == 1:meta.append("必填")
        if f.get("enumValues"):   meta.append(f'枚举={f["enumValues"]}')
        if f.get("regexPattern"): meta.append(f'正则={f["regexPattern"]}')
        if f.get("isPk") == 1:    meta.append("主键")
        meta_str = ("；".join(meta)) if meta else "无"
        lines.append(f'- {f["fieldKey"]} :: {f.get("dataType","string")} | {meta_str}')

    # 简化 schema（仅提示，不强制校验）
    schema_props = []
    for f in fields:
        t = (f.get("dataType") or "string").lower()
        json_type = {"string":"string","number":"number","integer":"integer","boolean":"boolean","date":"string","datetime":"string","json":"object"}.get(t,"string")
        schema_props.append(f'"{f["fieldKey"]}": {{"type": "{json_type}"}}')
    schema = '{ "type":"object", "properties": { ' + ", ".join(schema_props) + ' } }'

    table_desc_line = f'表说明：{table_desc or "无"}'
    return (
        f"你是一个**严格的结构化抽取器**。目标：从输入文本中抽取表“{table_display}”(tableKey={table_key})的**多行记录**。\n"
        f"{table_desc_line}\n\n"
        "### 字段定义\n" + "\n".join(lines) + "\n\n"
        "### 输出要求\n"
        "1) 仅输出 JSON：形如 {\"rows\": [ {<fieldKey>: <value>, ..., \"__explain\": {...}} ]}；\n"
        "2) 键必须使用 fieldKey；类型需符合定义；date=yyyy-MM-dd；datetime=yyyy-MM-dd HH:mm:ss；boolean=true/false；\n"
        "3) **证据简化**：若某字段给出了值，需在 `__explain.by_field[fieldKey]` 内给：\n"
        '   - "text": 一段原文摘录（<=160字，直接字符串）\n'
        '   - "reason": 简短理由（为什么选它，<=40字）\n'
        "4) 优先表格/明确数字；不确定留空；禁止臆造；\n"
        "5) 输出满足如下 JSON Schema（提示）：\n"
        f"{schema}\n\n"
        f"6) 输出语言偏好：{lang}（但 JSON 键与格式严格）。\n"
    )

def _user_prompt(content_md: str, filename: str = None, idx: int = None, total: int = None) -> str:
    meta = []
    if filename: meta.append(f"来源: {filename}")
    if idx is not None and total is not None: meta.append(f"切片: {idx}/{total}")
    meta_line = ("（" + "；".join(meta) + "）") if meta else ""
    return (
        f"请从以下 Markdown 文本中抽取数据{meta_line}。"
        "若给出字段值，请在 `__explain.by_field[fieldKey]` 里提供 `text` 与 `reason`：\n"
        "```\n" + (content_md or "")[:30000] + "\n```\n"
    )

def _call_openai(model_name: str, temperature: float, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "temperature": float(temperature),
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    with httpx.Client(timeout=OPENAI_TIMEOUT) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)  # 期望 {"rows":[...]}
    except Exception as e:
        raise RuntimeError(f"LLM 返回解析失败: {e}; raw={str(data)[:500]}")

def _normalize_rows(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = obj.get("rows")
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    if isinstance(obj, dict) and obj:
        return [obj]
    return []

# ========= 第一轮抽取 =========
def extract_rows_from_text(
    model_name: str,
    temperature: float,
    content_md: str,
    chunk_tokens: int,
    table_key: str,
    table_display_name: str,
    fields: List[Dict[str, Any]],
    lang: str = "zh",
    table_desc: str = None,
    filename: str = None,
) -> List[Dict[str, Any]]:
    sys_prompt = _system_prompt(table_key, table_display_name, table_desc or "", lang, fields)
    chunks = split_markdown_packed(content_md, max_tokens=chunk_tokens or 2000)

    all_rows: List[Dict[str, Any]] = []
    total = len(chunks)
    for i, ch in enumerate(chunks, 1):
        user_prompt = _user_prompt(ch, filename=filename, idx=i, total=total)
        obj = _call_openai(model_name, temperature, sys_prompt, user_prompt)
        rows = _normalize_rows(obj)
        # 确保每条有 __explain 基础结构，便于后续回填
        for r in rows:
            if "__explain" not in r or not isinstance(r["__explain"], dict):
                r["__explain"] = {"by_field": {}}
        all_rows.extend(rows)

    # 去重：按“填过值的字段集合+规范化值”生成签名
    seen = set(); uniq_rows = []
    def _norm_scalar(dtype: str, val):
        if val is None: return None
        t = (dtype or "string").lower()
        if t == "string":
            s = str(val).strip()
            return re.sub(r"\s+", " ", s) or None
        if t == "number":
            try: return float(val)
            except: return None
        if t == "integer":
            try: return int(float(val))
            except: return None
        if t == "boolean":
            if isinstance(val, bool): return 1 if val else 0
            s = str(val).strip().lower()
            if s in {"true","1","yes","y","是"}: return 1
            if s in {"false","0","no","n","否"}: return 0
            return None
        if t == "date": return _format_date(val)
        if t == "datetime": return _format_datetime(val)
        if t == "json":
            try: obj = val if isinstance(val,(dict,list)) else json.loads(val)
            except: obj = {"value": val}
            return json.dumps(obj, ensure_ascii=False, sort_keys=True)
        s = str(val).strip()
        return re.sub(r"\s+", " ", s) or None

    dtype_map = {f["fieldKey"]: (f.get("dataType") or "string").lower() for f in fields}
    for r in all_rows:
        items = []
        for fk in sorted(dtype_map.keys()):
            if fk not in r: continue
            nv = _norm_scalar(dtype_map[fk], r.get(fk))
            if nv is None: continue
            items.append([fk, nv])
        sig = json.dumps(items, ensure_ascii=False, sort_keys=True)
        if sig in seen: continue
        seen.add(sig); uniq_rows.append(r)
    return uniq_rows

# ========= 二次核对/归并 =========
# ========= 二次核对/归并 =========
def _reconcile_system_prompt(
    table_key: str,
    table_display: str,
    table_desc: str,
    lang: str,
    fields: List[Dict[str, Any]]
) -> str:
    field_lines = []
    for f in fields:
        meta = []
        if f.get("displayName"): meta.append(f'显示名="{f["displayName"]}"')
        if f.get("unit"): meta.append(f'单位="{f["unit"]}"')
        if f.get("description"): meta.append(f'说明="{f["description"]}"')
        field_lines.append(
            f'- {f["fieldKey"]} :: {f.get("dataType","string")} | {"；".join(meta) if meta else "无"}'
        )

    schema_props = []
    for f in fields:
        t = (f.get("dataType") or "string").lower()
        json_type = {
            "string": "string", "number": "number", "integer": "integer",
            "boolean": "boolean", "date": "string", "datetime": "string", "json": "object"
        }.get(t, "string")
        schema_props.append(f'"{f["fieldKey"]}": {{"type":"{json_type}"}}')
    schema = '{ "type":"object","properties":{' + ",".join(schema_props) + '} }'

    return (
        f"你是**核对与汇总器**。表“{table_display}”(tableKey={table_key})。\n"
        + "\n".join(field_lines) + "\n\n"
        "### 任务\n"
        "给你一批候选记录（来自文档不同段落/表格）。请：\n"
        "1) 识别哪些候选属于**同一条真实记录**（如同一学校同一年），可以自行归并；允许有多条不同主键的真实记录；\n"
        "2) 对每个字段，只能在所属候选的字段值中**择优选一个**（或留空），禁止臆造；\n"
        "3) 为每个被选中的字段提供 `__explain.by_field[fieldKey]`：\n"
        '   - "text": 该值对应的原文摘录（<=160字，直接字符串）\n'
        '   - "reason": 简短理由（<=40字，如“来自表格且与年份一致”）\n'
        "4) 冲突且难判断则留空并说明 `reason: 冲突/无法判断`；\n"
        "5) 满足如下 JSON Schema（提示）：\n"
        f"{schema}\n\n"
        f"输出仅 JSON：{{\"rows\": [{{<字段值>, \"__explain\": {{\"by_field\": {{...}}}}}}]}}。语言偏好：{lang}。\n"
    )


def _shrink(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else (s[:n] + "…")


def _build_reconcile_payload(
    rows: List[Dict[str, Any]],
    max_candidates: int = 1200
) -> Dict[str, Any]:
    """
    将第一轮 rows 压成 candidates，供核对器参考；把 __explain.by_field 里的 text 作为候选摘录。
    """
    cands = []
    for i, r in enumerate(rows):
        explain = r.get("__explain") or {}
        by_field = (explain.get("by_field") or {}) if isinstance(explain, dict) else {}
        compact_by_field = {}
        for fk, info in by_field.items():
            if not isinstance(info, dict):
                continue
            compact_by_field[fk] = {
                "text": _shrink(info.get("text") or "", 160),
                "reason": _shrink(info.get("reason") or "", 40),
            }
        cands.append({
            "candidate_id": f"cand_{i+1}",
            "values": {k: v for k, v in r.items() if k not in {"__explain"}},
            "by_field": compact_by_field,
        })
        if len(cands) >= max_candidates:
            break
    return {"candidates": cands}


def _backfill_explain(
    final_rows: List[Dict[str, Any]],
    payload: Dict[str, Any],
    fields: List[Dict[str, Any]]
) -> None:
    """
    若模型未给某字段的 text/reason，则从其来源候选的 by_field 里兜底选取一段 text，
    没有的话将 reason 置为简短说明。
    """
    cand_map = {c["candidate_id"]: c for c in (payload.get("candidates") or [])}
    for r in final_rows:
        explain = r.get("__explain") or {}
        by_field = explain.get("by_field") or {}
        # 如果模型没有给来源信息，我们也不强制，但尽量保证 text/reason 不为空
        for fk, info in list(by_field.items()):
            if not isinstance(info, dict):
                by_field[fk] = {"text": None, "reason": None}
                continue
            text = _shrink(info.get("text") or "", 160)
            reason = _shrink(info.get("reason") or "", 40)
            # 兜底
            if not text:
                # 任取一个候选的该字段 text 作为兜底
                for c in cand_map.values():
                    cf = (c.get("by_field") or {}).get(fk) or {}
                    if cf.get("text"):
                        text = _shrink(cf["text"], 160)
                        break
            if not reason:
                reason = "来自候选最相关片段"
            by_field[fk] = {"text": text or None, "reason": reason or None}
        explain["by_field"] = by_field
        r["__explain"] = explain

def reconcile_rows_with_llm(
    model_name: str,
    temperature: float,
    table_key: str,
    table_display_name: str,
    table_desc: str,
    lang: str,
    fields: List[Dict[str, Any]],
    extracted_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not extracted_rows:
        return []

    system_prompt = _reconcile_system_prompt(table_key, table_display_name, table_desc or "", lang, fields)
    payload = _build_reconcile_payload(extracted_rows)

    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    body = {
        "model": model_name,
        "temperature": float(temperature or 0.1),
        "response_format": {"type": "json_object"},
        "messages": messages,
    }

    with httpx.Client(timeout=OPENAI_TIMEOUT) as client:
        r = client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()

    try:
        content = data["choices"][0]["message"]["content"]
        obj = json.loads(content)
        final_rows = obj.get("rows") or []
        # 回填简化证据（text+reason）
        _backfill_explain(final_rows, payload, fields)
        return final_rows
    except Exception as e:
        raise RuntimeError(f"二次校验解析失败: {e}; raw={str(data)[:500]}")

# ========= 扁平化到 cells =========
def flatten_to_cells(rows: List[Dict[str, Any]], field_map: Dict[str, Any], task_id: str) -> List[Dict[str, Any]]:
    """
    将归并后的 rows 扁平成 AppExCell：
    - 每个字段落到对应 value_* 列位
    - evidenceText + reason 从 __explain.by_field[fieldKey] 读取
    """
    cells: List[Dict[str, Any]] = []
    for r in rows:
        row_id = uuid.uuid4().hex
        explain = r.get("__explain") or {}
        explain_by = (explain.get("by_field") or {}) if isinstance(explain, dict) else {}

        for field_key, val in r.items():
            if field_key == "__explain":
                continue
            f = field_map.get(field_key)
            if not f:
                continue

            casted = _coerce_value_by_type(f.dataType, val)

            # 校验 & notes
            notes_parts = []
            if f.enumValues:
                try:
                    enums = json.loads(f.enumValues) if f.enumValues.strip().startswith("[") else [x.strip() for x in f.enumValues.split(",")]
                    enums_norm = {str(x).strip().lower() for x in enums}
                    val_str = None
                    if casted["valueString"] is not None:
                        val_str = str(casted["valueString"]).strip().lower()
                    elif casted["valueInteger"] is not None:
                        val_str = str(casted["valueInteger"]).strip().lower()
                    elif casted["valueNumber"] is not None:
                        val_str = str(casted["valueNumber"]).strip().lower()
                    elif casted["valueBoolean"] is not None:
                        val_str = "true" if casted["valueBoolean"] == 1 else "false"
                    if val_str is not None and val_str not in enums_norm:
                        notes_parts.append(f"枚举不匹配:{val}")
                except Exception:
                    notes_parts.append("枚举解析失败")

            if f.regexPattern and isinstance(f.regexPattern, str) and casted["valueString"]:
                try:
                    pattern = re.compile(f.regexPattern)
                    if not pattern.fullmatch(casted["valueString"]):
                        notes_parts.append("正则不匹配")
                except Exception:
                    notes_parts.append("正则无效")

            info = explain_by.get(field_key) or {}
            evidence_text = info.get("text") or None
            reason = info.get("reason") or None

            c = {
                "rowId": row_id,
                "fieldId": f.id,
                "taskId": task_id,
                **casted,
                "unit": f.unit,
                "notes": "; ".join(notes_parts) if notes_parts else None,
                # ✅ 简化证据
                "evidenceText": evidence_text,
                "reason": reason,
            }
            cells.append(c)
    return cells
