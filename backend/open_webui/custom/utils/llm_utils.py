# -*- coding: utf-8 -*-
import os
import json
import math
import uuid
from datetime import datetime, date
from typing import List, Dict, Any, Iterable, Optional

import httpx

# ========= 可配置：你给的默认值 =========
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://192.168.0.114:3000/v1")
# OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY",  "sk-2pH4HUSK4wikhU7NSqMqF3Ldi7c2r89sJmRQBBJ9PS7vN1AM")
OPENAI_API_KEY  = "sk-2pH4HUSK4wikhU7NSqMqF3Ldi7c2r89sJmRQBBJ9PS7vN1AM"
OPENAI_TIMEOUT  = int(os.getenv("OPENAI_TIMEOUT", "120"))

# ========== 基础工具 ==========
def approx_tokens_of_text(s: str) -> int:
    # 粗略换算：4 字符 ≈ 1 token（中英文混排保守一些）
    return max(1, math.ceil(len(s) / 4))

def split_by_tokens(text: str, max_tokens: int) -> List[str]:
    if not text or max_tokens <= 0:
        return [text]
    # 简单分段（按段落或句号尽量切）
    chunks, buf, buf_tokens = [], [], 0
    seps = ["\n\n", "\n", "。", "！", "？", ".", "!", "?"]
    parts = [text]
    for sep in seps:
        tmp = []
        for p in parts:
            tmp.extend(p.split(sep))
        parts = tmp
    for piece in parts:
        if not piece:
            continue
        t = approx_tokens_of_text(piece)
        if buf_tokens + t > max_tokens and buf:
            chunks.append("".join(buf))
            buf, buf_tokens = [], 0
        buf.append(piece)
        buf_tokens += t
    if buf:
        chunks.append("".join(buf))
    if not chunks:
        chunks = [text]
    return chunks

def _system_prompt(table_key: str, table_display: str, table_desc: str, lang: str, fields: List[Dict[str, Any]]) -> str:
    """
    把表/字段元数据完整喂给 LLM，并明确输出 JSON 结构与校验要求。
    """
    # 字段行：把可用元数据都带上
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

    # JSON Schema（简化型）引导
    schema_props = []
    required_keys = []
    for f in fields:
        t = (f.get("dataType") or "string").lower()
        json_type = {
            "string": "string",
            "number": "number",
            "integer": "integer",
            "boolean": "boolean",
            "date": "string",
            "datetime": "string",
            "json": "object"
        }.get(t, "string")
        schema_props.append(f'"{f["fieldKey"]}": {{"type": "{json_type}"}}')
        if f.get("required") == 1:
            required_keys.append(f'"{f["fieldKey"]}"')

    schema = (
        '{'
        ' "type": "object",'
        ' "properties": {'
        + ", ".join(schema_props) +
        '}'
        + (', "required": [' + ", ".join(required_keys) + ']' if required_keys else '')
        + '}'
    )

    table_desc_line = f'表说明：{table_desc}' if table_desc else '表说明：无'

    return (
        f"你是一个**严格的结构化抽取器**。目标：从输入文本中抽取出表“{table_display}”(tableKey={table_key})的**多行记录**。\n"
        f"{table_desc_line}\n\n"
        "### 字段定义（fieldKey :: 数据类型 | 元数据）\n"
        + "\n".join(lines) + "\n\n"
        "### 输出要求\n"
        "1) 只能输出 JSON：形如 {\"rows\": [ {<fieldKey>: <value>, ...}, ... ]}。\n"
        "2) 所有键必须使用 **fieldKey**（不是显示名）。\n"
        "3) 类型遵循上面的数据类型；date 用 yyyy-MM-dd，datetime 用 yyyy-MM-dd HH:mm:ss；boolean 用 true/false。\n"
        "4) 如有枚举/正则，请优先满足；若不满足，请留空该字段或不输出该字段，**不要臆造**。\n"
        "5) 支持多条记录；若文本中没有对应数据，rows 可以为空数组。\n"
        "6) 输出应满足如下 JSON Schema（简化）：\n"
        f"{schema}\n"
        f"7) 理解语境优先使用 {lang}，如果要求文本和文本单位不同请尝试正确的转换单位，但结构输出必须是 JSON。\n"
    )


def _user_prompt(content_md: str, filename: str = None, idx: int = None, total: int = None) -> str:
    meta = []
    if filename:
        meta.append(f"来源文件: {filename}")
    if idx is not None and total is not None:
        meta.append(f"切片: {idx}/{total}")
    meta_line = ("（" + "；".join(meta) + "）") if meta else ""

    return (
        f"请从以下 Markdown 文本中抽取数据{meta_line}：\n"
        "```\n" + content_md[:30000] + "\n```\n"
    )


def _call_openai(model_name: str, temperature: float, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    """
    直接 HTTP 调用 OpenAI 兼容接口（更通用，无需 SDK）。
    期望模型支持 response_format=json_object（v1 兼容）。
    """
    url = f"{OPENAI_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
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
        return json.loads(content)  # 应该是 {"rows": [...]}
    except Exception as e:
        # 打印部分响应帮助排查
        raise RuntimeError(f"LLM 返回解析失败: {e}; raw={str(data)[:500]}")

def _normalize_rows(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = obj.get("rows")
    if isinstance(rows, list):
        # 只接受对象列表
        return [r for r in rows if isinstance(r, dict)]
    # 兼容单对象
    if isinstance(obj, dict) and obj:
        return [obj]
    return []


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

    # ✅ 用新的“语义切 + 贪心打包”切片器，保证每块尽量接近上限，但不超
    chunks = split_markdown_packed(content_md, max_tokens=chunk_tokens or 2000)

    all_rows: List[Dict[str, Any]] = []
    total = len(chunks)
    for i, ch in enumerate(chunks, 1):
        user_prompt = _user_prompt(ch, filename=filename, idx=i, total=total)
        obj = _call_openai(model_name, temperature, sys_prompt, user_prompt)
        rows = _normalize_rows(obj)
        all_rows.extend(rows)

    # ===== 去重（若你已按主键去重的版本，这里保持那版） =====
    seen = set()
    uniq_rows = []
    for r in all_rows:
        sig = _row_signature_by_full_values(r, fields)
        if sig in seen:
            continue
        seen.add(sig)
        uniq_rows.append(r)

    return uniq_rows

# -------- 将 rows 扁平化为 AppExCell 一组记录 --------

def _format_date(d: Any) -> str:
    """尽量把各种日期输入规范成 yyyy-MM-dd"""
    if isinstance(d, date) and not isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    if isinstance(d, str):
        s = d.strip().replace("/", "-").replace(".", "-")
        # 最常见三段式
        parts = s.split("-")
        if len(parts) == 3 and len(parts[0]) == 4:
            return f"{parts[0]:0>4}-{parts[1]:0>2}-{parts[2]:0>2}"
    return None

def _format_datetime(dt: Any) -> str:
    """尽量把各种日期时间输入规范成 yyyy-MM-dd HH:mm:ss"""
    if isinstance(dt, datetime):
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(dt, str):
        s = dt.strip().replace("/", "-").replace("T", " ").replace(".", "-")
        # 粗略处理
        try:
            # 优先尝试完整解析
            _ = datetime.fromisoformat(s)
            # 再格式化
            return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass
    return None

def _coerce_value_by_type(data_type: str, raw: Any) -> Dict[str, Any]:
    """
    落到 AppExCell 的 value_* 列位。
    只设置一个主列，其余列留空。
    """
    data_type = (data_type or "string").lower()
    cell = {
        "valueString": None,
        "valueNumber": None,
        "valueInteger": None,
        "valueBoolean": None,
        "valueDate": None,
        "valueDatetime": None,
        "valueJson": None,
    }

    try:
        if data_type == "string":
            cell["valueString"] = "" if raw is None else str(raw)
        elif data_type == "number":
            if raw is None or raw == "":
                val = None
            else:
                val = float(raw)
            cell["valueNumber"] = val
        elif data_type == "integer":
            if raw is None or raw == "":
                val = None
            else:
                val = int(raw)
            cell["valueInteger"] = val
        elif data_type == "boolean":
            # 统一转 1/0
            truthy = {"true", "1", "yes", "y", "是"}
            falsy = {"false", "0", "no", "n", "否"}
            if isinstance(raw, bool):
                cell["valueBoolean"] = 1 if raw else 0
            elif isinstance(raw, (int, float)):
                cell["valueBoolean"] = 1 if raw != 0 else 0
            elif isinstance(raw, str):
                s = raw.strip().lower()
                if s in truthy:
                    cell["valueBoolean"] = 1
                elif s in falsy:
                    cell["valueBoolean"] = 0
            else:
                cell["valueBoolean"] = None
        elif data_type == "date":
            cell["valueDate"] = _format_date(raw)
        elif data_type == "datetime":
            cell["valueDatetime"] = _format_datetime(raw)
        elif data_type == "json":
            # 保持 JSON 字符串
            if isinstance(raw, (dict, list)):
                cell["valueJson"] = json.dumps(raw, ensure_ascii=False)
            elif isinstance(raw, str):
                # 若是字符串，尽量验证为 json
                try:
                    json.loads(raw)
                    cell["valueJson"] = raw
                except Exception:
                    cell["valueJson"] = json.dumps({"value": raw}, ensure_ascii=False)
            else:
                cell["valueJson"] = json.dumps({"value": raw}, ensure_ascii=False)
        else:
            # 未知类型，当作字符串
            cell["valueString"] = "" if raw is None else str(raw)
    except Exception:
        # 强转失败也作为字符串落位，避免整行丢失
        cell = {k: None for k in cell}
        cell["valueString"] = "" if raw is None else str(raw)

    return cell

import re

def flatten_to_cells(rows, field_map, task_id: str) -> List[Dict[str, Any]]:
    cells: List[Dict[str, Any]] = []
    for r in rows:
        row_id = uuid.uuid4().hex
        for field_key, val in r.items():
            f = field_map.get(field_key)
            if not f:
                continue
            casted = _coerce_value_by_type(f.dataType, val)

            notes_parts = []
            # 枚举校验（字符串比较，大小写不敏感；json/number 略过）
            if f.enumValues:
                # enumValues 可支持 "A,B,C" 或 JSON 数组字符串
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

            # 正则校验（只对 string/date/datetime 做）
            if f.regexPattern and isinstance(f.regexPattern, str) and casted["valueString"]:
                try:
                    pattern = re.compile(f.regexPattern)
                    if not pattern.fullmatch(casted["valueString"]):
                        notes_parts.append("正则不匹配")
                except Exception:
                    notes_parts.append("正则无效")

            c = {
                "rowId": row_id,
                "fieldId": f.id,
                "taskId": task_id,
                **casted,
                "unit": f.unit,
                "notes": "; ".join(notes_parts) if notes_parts else None,
            }
            cells.append(c)
    return cells


from langchain.text_splitter import MarkdownHeaderTextSplitter

def split_markdown_packed(content: str, max_tokens: int) -> list[str]:
    """
    先用 MarkdownHeaderTextSplitter 按 #/##/### 拆成语义段，
    再用“贪心打包”把多段拼到接近 max_tokens；若再加一段就会超过，则开始新块。
    不做重叠；过细问题基本解决。
    """
    if not content:
        return [""]

    # 1) 语义分段（标题级别可自行加/减）
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(content)
    segments = [d.page_content for d in docs if d and d.page_content]

    # 如果没有识别出任何头，退化为按双换行分段
    if not segments:
        segments = [s for s in content.split("\n\n") if s.strip()]

    # 2) 贪心打包
    chunks, buf, buf_tok = [], [], 0
    for seg in segments:
        t = approx_tokens_of_text(seg)
        # 单段本身就超大：独立成块（避免死循环）
        if t >= max_tokens:
            # 先把已有 buffer 作为块提交
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_tok = [], 0
            chunks.append(seg)
            continue

        # 加上这个 seg 会超，先提交当前块再开新块
        if buf_tok + t > max_tokens and buf:
            chunks.append("\n\n".join(buf))
            buf, buf_tok = [], 0

        buf.append(seg)
        buf_tok += t

    if buf:
        chunks.append("\n\n".join(buf))

    return chunks

import re
import json
from typing import Tuple

def _normalize_scalar_for_signature(dtype: str, val):
    if val is None:
        return None
    t = (dtype or "string").lower()

    if t == "string":
        s = str(val).strip()
        # 折叠多空白，避免“多个空格/换行”造成的伪差异
        s = re.sub(r"\s+", " ", s)
        return s if s != "" else None

    if t == "number":
        try:
            return float(val)
        except Exception:
            # 回退到字符串规范化
            s = str(val).strip()
            return float(s) if re.fullmatch(r"[-+]?\d+(\.\d+)?", s) else s or None

    if t == "integer":
        try:
            return int(float(val))
        except Exception:
            s = str(val).strip()
            return int(float(s)) if re.fullmatch(r"[-+]?\d+(\.0+)?", s) else s or None

    if t == "boolean":
        if isinstance(val, bool):
            return 1 if val else 0
        s = str(val).strip().lower()
        if s in {"true", "1", "yes", "y", "是"}:
            return 1
        if s in {"false", "0", "no", "n", "否"}:
            return 0
        return None

    if t == "date":
        # 统一成 yyyy-MM-dd（你已有 _format_date，可复用）
        from datetime import datetime, date as _date
        if isinstance(val, _date) and not isinstance(val, datetime):
            return val.strftime("%Y-%m-%d")
        if isinstance(val, datetime):
            return val.strftime("%Y-%m-%d")
        if isinstance(val, str):
            s = val.strip().replace("/", "-").replace(".", "-")
            parts = s.split("-")
            if len(parts) == 3 and len(parts[0]) == 4:
                y = parts[0]; m = parts[1].zfill(2); d = parts[2].zfill(2)
                return f"{y}-{m}-{d}"
        return None

    if t == "datetime":
        from datetime import datetime
        if isinstance(val, datetime):
            return val.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(val, str):
            s = val.strip().replace("T", " ").replace("/", "-")
            try:
                # 尝试多种解析
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                    try:
                        return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M:%S")
                    except Exception:
                        pass
            except Exception:
                pass
        return None

    if t == "json":
        try:
            obj = val if isinstance(val, (dict, list)) else json.loads(val)
        except Exception:
            obj = {"value": val}
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)

    # 默认当 string
    s = str(val).strip()
    s = re.sub(r"\s+", " ", s)
    return s if s != "" else None

def _row_signature_by_full_values(row: dict, fields: list) -> str:
    """
    用“整行实际有值的字段”生成稳定签名：
    - 仅纳入有值字段（更贴合“完整相同的数据才算重复”）
    - 字段按 field_key 排序
    - 值做数据类型规范化
    返回一个稳定的 JSON 字符串作为签名 key
    """
    # field_key -> data_type
    dtype_map = {f["fieldKey"]: (f.get("dataType") or "string").lower() for f in fields}

    items = []
    for fk in sorted(dtype_map.keys()):
        if fk not in row:
            continue
        norm_val = _normalize_scalar_for_signature(dtype_map[fk], row.get(fk))
        if norm_val is None:
            continue  # ✅ 无值的字段不参与签名；如需参与，请改为 items.append([fk, None])
        items.append([fk, norm_val])

    return json.dumps(items, ensure_ascii=False, sort_keys=True)

