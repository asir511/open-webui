# -*- coding: utf-8 -*-
import os
import json
import math
import uuid
import asyncio
from datetime import datetime, date
from typing import List, Dict, Any, Optional

import httpx
import re
from langchain.text_splitter import MarkdownHeaderTextSplitter

# ========= 可配置 =========
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://192.168.0.114:3000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-2pH4HUSK4wikhU7NSqMqF3Ldi7c2r89sJmRQBBJ9PS7vN1AM")
OPENAI_TIMEOUT  = int(os.getenv("OPENAI_TIMEOUT", "300"))

# 并发控制（按你的要求：3~4 并发）
EXTRACT_CONCURRENCY   = int(os.getenv("EXTRACT_CONCURRENCY", "8"))
RECONCILE_CONCURRENCY = int(os.getenv("RECONCILE_CONCURRENCY", "8"))

# ========= 通用小函数 =========
def approx_tokens_of_text(s: str) -> int:
    return max(1, math.ceil(len(s) / 4))

def denoise_markdown(md: str) -> str:
    """
    文本去噪（不改变语义）：
    1) 移除开头的“目录/TOC”块（常见的编号/点线/锚链接列表）
    2) 去除所有图片/视频外链（Markdown/HTML img），以及纯图片/视频链接行
    """
    if not md:
        return md

    text = md

    # 1) 去除 Markdown 图片语法 & HTML <img>
    text = re.sub(r'!\[[^\]]*\]\([^)]+\)', '', text)               # ![alt](url)
    text = re.sub(r'(?is)<img\b[^>]*>', '', text)                  # <img ...>

    # 2) 行级资源外链（常见图片/视频后缀）
    exts = r'(?:jpg|jpeg|png|gif|bmp|webp|svg|mp4|mov|avi|mkv|flv|wmv)'
    text = re.sub(r'^\s*https?://\S+\.(?:' + exts + r')\s*$', '', text, flags=re.IGNORECASE|re.MULTILINE)
    text = re.sub(r'\((https?://[^)]+\.(?:' + exts + r'))\)', '()', text, flags=re.IGNORECASE)

    # 3) 移除开头的目录块（启发式）
    lines = text.splitlines()
    def is_toc_line(s: str) -> bool:
        s = s.strip()
        if not s:
            return False
        # 锚链接式目录: - [标题](#anchor)
        if re.match(r'^[-*+]\s*\[[^\]]+\]\(#.+\)$', s):
            return True
        # 数字/中文编号 + 标题 + 可能的点线/页码
        if re.match(r'^(\d+[\.\)]|[（(]?\d+[）)]|[一二三四五六七八九十]+[、.)]|[（(][一二三四五六七八九十]+[）)])\s*.+', s):
            return True
        # 带点线 leader 的目录行
        if re.match(r'^.+?[\.·‧･•⋯…]{2,}\s*\d+\s*$', s):
            return True
        return False

    # 若首 80 行里，有 >=5 行连续/高密度目录行，视为 TOC，剔除其连续块
    cut_idx = 0
    window = lines[:80] if len(lines) >= 80 else lines[:]
    toc_marks = [is_toc_line(x) or (x.strip() == "目录") for x in window]
    if sum(toc_marks) >= 5:
        i = 0
        # 跳过“目录”标题
        if i < len(lines) and lines[i].strip() in {"目录", "目 录", "# 目录", "## 目录"}:
            i += 1
        # 连续的 TOC 样式行
        while i < len(lines) and (is_toc_line(lines[i]) or not lines[i].strip()):
            i += 1
        # 若下一行是一级/二级标题，也可停止
        cut_idx = i

    if cut_idx > 0 and cut_idx < len(lines):
        text = "\n".join(lines[cut_idx:])

    # 清理重复空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text

try:
    import tiktoken
except Exception:
    tiktoken = None


def _get_encoder(encoding_name: str = "cl100k_base"):
    """
    返回与推理端一致的 tokenizer 编码器。
    - 默认 cl100k_base（OpenAI GPT-3.5/4 家族常用）。
    - 如本地服务兼容 o200k，可传 encoding_name="o200k_base"。
    """
    if tiktoken is None:
        raise RuntimeError(
            "tiktoken 未安装：请先 pip install tiktoken；"
            "严格的 token 上限需要与模型一致的 tokenizer 才能保证。"
        )
    try:
        # 优先按 encoding_name 获取
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        # 兜底：常见编码
        return tiktoken.get_encoding("cl100k_base")


def _chunk_text_by_tokens_force(text: str, max_tokens: int, overlap_tokens: int, enc) -> list[str]:
    """
    最后一层兜底：直接在 token 维度滑窗硬切，并将切片 decode 回文本。
    始终保证每块 token 数 <= max_tokens。
    """
    if not text:
        return [""]
    toks = enc.encode(text)
    n = len(toks)
    if n <= max_tokens:
        return [text]

    # 滑动步长，确保有 overlap（避免跨界丢上下文）
    stride = max(1, max_tokens - max(0, overlap_tokens))
    chunks = []
    i = 0
    while i < n:
        j = min(i + max_tokens, n)
        piece = enc.decode(toks[i:j])
        # 防御性清理：避免 decode 造成的多余空白
        piece = re.sub(r'\s+\n', '\n', piece)
        piece = re.sub(r'\n{3,}', '\n\n', piece).strip()
        chunks.append(piece if piece else enc.decode(toks[i:j]))  # 保底
        if j == n:
            break
        i += stride
    return chunks

def _repack_greedy(units: list[str], max_tokens: int, enc, *, joiner: str = "\n\n") -> list[str]:
    """
    将已保证单元 <= max_tokens 的 'units' 顺序贪心打包为更接近 max_tokens 的大块。
    - 不打乱顺序，不跨越原子边界；
    - 计算拼接分隔符 joiner 的 token 成本，避免超限；
    - 如遇单个 unit 本就接近 max_tokens，会单独成块。
    """
    if not units:
        return [""]

    def tlen(s: str) -> int:
        return len(enc.encode(s))

    joiner_tok = tlen(joiner)
    out, buf, buf_tok = [], [], 0

    for u in units:
        if not u.strip():
            continue
        utok = tlen(u)
        # 这里假定所有 unit 事先已经 <= max_tokens
        if not buf:
            # 新建缓冲
            buf, buf_tok = [u], utok
            continue

        # 预计加入 u 后的 token：已有 + 分隔符 + utok
        need = buf_tok + (joiner_tok if buf else 0) + utok
        if need <= max_tokens:
            buf.append(u)
            buf_tok = need
        else:
            # 先吐出当前缓冲，再以 u 开新块
            out.append(joiner.join(buf))
            buf, buf_tok = [u], utok

    if buf:
        out.append(joiner.join(buf))
    return out

def split_markdown_packed(
    content: str,
    max_tokens: int,
    *,
    overlap_tokens: int = 80,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """
    Token-aware 的 Markdown 切片：严格保证每个 chunk 的 token 数 <= max_tokens。
    - 先按标题层级切块（保留语义结构）；
    - 对每块做 token 计数：
        * 若 <= max_tokens：直接收录；
        * 若 >  max_tokens：依次按 段落/句子 细分；
        * 若仍有超限长句：退化为 token 维度滑窗硬切（并 decode）。
    - overlap_tokens：相邻块的 token 重叠量，用于跨界字段的取证/计算。
    - encoding_name：与推理端一致的分词编码（默认 cl100k_base）。
    """
    if not content:
        return [""]

    enc = _get_encoder(encoding_name)

    def tlen(s: str) -> int:
        return len(enc.encode(s))

    # 1) 语义级初切：按 MD 标题
    headers_to_split_on = [("#", "H1"), ("##", "H2"), ("###", "H3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(content)
    segments = [d.page_content for d in docs if d and d.page_content] or \
               [s for s in content.split("\n\n") if s.strip()]

    out_chunks: list[str] = []

    for seg in segments:
        if tlen(seg) <= max_tokens:
            out_chunks.append(seg)
            continue

        # 2) 二级：按段落细分
        paras = [p for p in re.split(r"\n{2,}", seg) if p.strip()]
        buf = []
        buf_tokens = 0

        def flush_buf():
            nonlocal buf, buf_tokens
            if not buf:
                return
            combined = "\n\n".join(buf)
            # 若组合后仍超限，则进一步细分（句子级/硬切）
            if tlen(combined) <= max_tokens:
                out_chunks.append(combined)
            else:
                # 3) 三级：按句号/分号/换行做更细切
                sentences = re.split(r'(?<=[。！？!?；;:])\s+|\n+', combined)
                sbuf, sbuf_tokens = [], 0
                for s in sentences:
                    if not s.strip():
                        continue
                    tl = tlen(s)
                    if tl > max_tokens:
                        # 4) 兜底：对超长句按 token 窗口硬切
                        hard = _chunk_text_by_tokens_force(s, max_tokens, overlap_tokens, enc)
                        out_chunks.extend(hard)
                        continue
                    if sbuf_tokens + tl > max_tokens and sbuf:
                        out_chunks.append("".join(sbuf))
                        sbuf, sbuf_tokens = [], 0
                    sbuf.append(s)
                    sbuf_tokens += tl
                if sbuf:
                    out_chunks.append("".join(sbuf))
            buf, buf_tokens = [], 0  # reset

        for p in paras:
            tl = tlen(p)
            if tl > max_tokens:
                # 该段过长，先把已有缓冲 flush，再对该段做句级/硬切
                flush_buf()
                sentences = re.split(r'(?<=[。！？!?；;:])\s+|\n+', p)
                sbuf, sbuf_tokens = [], 0
                for s in sentences:
                    if not s.strip():
                        continue
                    sl = tlen(s)
                    if sl > max_tokens:
                        hard = _chunk_text_by_tokens_force(s, max_tokens, overlap_tokens, enc)
                        out_chunks.extend(hard)
                        continue
                    if sbuf_tokens + sl > max_tokens and sbuf:
                        out_chunks.append("".join(sbuf))
                        sbuf, sbuf_tokens = [], 0
                    sbuf.append(s)
                    sbuf_tokens += sl
                if sbuf:
                    out_chunks.append("".join(sbuf))
                continue

            # 正常段落累积
            if buf_tokens + tl > max_tokens and buf:
                flush_buf()
            buf.append(p)
            buf_tokens += tl

        flush_buf()
    out_chunks = _repack_greedy(out_chunks, max_tokens, enc, joiner="\n\n")
    # 5) token 级 overlap：在最终块之间追加重叠（仅当需要）
    if overlap_tokens and overlap_tokens > 0 and out_chunks:
        with_overlap: list[str] = []
        prev_tail_tokens = []
        for idx, ch in enumerate(out_chunks):
            if idx == 0:
                with_overlap.append(ch)
                # 预计算尾部 token 以供下块添加
                prev_tail_tokens = enc.encode(ch)[-overlap_tokens:] if tlen(ch) > overlap_tokens else enc.encode(ch)
                continue
            cur_tokens = enc.encode(ch)
            # 把上块的尾部 token 解码并作为“前缀 overlap”拼接到当前块前方
            if prev_tail_tokens:
                prefix = enc.decode(prev_tail_tokens)
                merged = (prefix + ("\n" if prefix and not prefix.endswith("\n") else "")) + ch
                # 如因 overlap 导致超限，则回退到原 ch（不强制）
                if tlen(merged) <= max_tokens:
                    with_overlap.append(merged)
                else:
                    with_overlap.append(ch)
            else:
                with_overlap.append(ch)
            prev_tail_tokens = enc.encode(ch)[-overlap_tokens:] if tlen(ch) > overlap_tokens else enc.encode(ch)
        out_chunks = with_overlap

    # 6) 最终守卫：确保所有块都不超限（硬性保证）
    bad = [i for i, c in enumerate(out_chunks) if tlen(c) > max_tokens]
    if bad:
        # 理论上不会触发；以防万一对超限者做硬切替换
        fixed = []
        for i, c in enumerate(out_chunks):
            if i in bad:
                fixed.extend(_chunk_text_by_tokens_force(c, max_tokens, overlap_tokens, enc))
            else:
                fixed.append(c)
        out_chunks = fixed
        # 再校验一遍
        assert all(tlen(c) <= max_tokens for c in out_chunks), "内部错误：仍有 chunk 超过 max_tokens"

    return out_chunks

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
def _system_prompt(table_key: str, table_display: str, table_desc: str, lang: str, fields: list[dict]) -> str:
    lines = []
    for f in fields:
        meta = []
        if f.get("displayName"):   meta.append(f'显示名="{f["displayName"]}"')
        if f.get("description"):   meta.append(f'说明="{f["description"]}"')
        if f.get("unit"):          meta.append(f'单位="{f["unit"]}"')
        if f.get("required") == 1: meta.append("必填")
        if f.get("enumValues"):    meta.append(f'枚举={f["enumValues"]}')
        if f.get("regexPattern"):  meta.append(f'正则={f["regexPattern"]}')
        if f.get("isPk") == 1:     meta.append("主键")
        meta_str = "；".join(meta) if meta else "无"
        lines.append(f'- {f["fieldKey"]} :: {f.get("dataType","string")} | {meta_str}')

    schema_props = []
    for f in fields:
        t = (f.get("dataType") or "string").lower()
        json_type = {
            "string": "string","number": "number","integer": "integer",
            "boolean": "boolean","date": "string","datetime": "string","json": "object"
        }.get(t,"string")
        schema_props.append(f'"{f["fieldKey"]}": {{"type": "{json_type}"}}')
    schema = '{ "type": "object", "properties": { ' + ", ".join(schema_props) + ' } }'

    table_desc_line = f'表说明：{table_desc or "无"}'

    return (
        f"你是一个**严格的结构化抽取器**。目标：从输入文本中抽取表“{table_display}”(tableKey={table_key})的**多行记录**。\n"
        f"{table_desc_line}\n\n"
        "### 字段定义\n" + "\n".join(lines) + "\n\n"
        "### 抽取原则\n"
        "1) 所有字段值必须直接来源于原文，或通过原文中可验证的明确推导；**不得臆测**。\n"
        "2) 不可仅凭单一句话下结论，应结合上下文/上下段/表格整体内容综合判断。\n"
        "3) 在没有字段描述的情况下，根据字段的名称来准确的定义语义，对应找出的数据(如果有的话)必须完全符合字段语义，不能有任何臆测、幻想的数值匹配。\n"
        "4) 若缺乏充分证据，字段值留空。\n\n"
        "### 输出要求\n"
        "1) 仅输出 JSON：形如 {\"rows\": [ {<fieldKey>: <value>, ..., \"__explain\": {...}} ]}。\n"
        "2) 键必须使用 fieldKey；类型需符合定义；date=yyyy-MM-dd；datetime=yyyy-MM-dd HH:mm:ss；boolean=true/false。\n"
        "3) 对于每个非空字段，必须在 `__explain.by_field[fieldKey]` 给出：\n"
        '   - \"text\": 一段原文摘录，不需要无关部分，可以截断原文拼凑成有效摘录,表格等携带格式的数据不需要带有字符标签如td tr等（<=200字，保证能支撑该值）；\n'
        '   - \"reason\": 对该值的推理说明（为什么能从原文确定，需体现上下文思考链，<=140字）。\n'
        "4) 优先选表格/明确数字；遇到歧义或冲突，说明为什么选择该值；不确定则留空。\n"
        "5) 输出需满足如下 JSON Schema（提示）：\n"
        f"{schema}\n\n"
        f"6) 输出语言偏好：{lang}（仅影响 text 与 reason，自身 JSON 键名与格式保持严格）。\n"
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

async def _acall_openai(model_name: str, temperature: float, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
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
    async with httpx.AsyncClient(timeout=OPENAI_TIMEOUT) as client:
        r = await client.post(url, headers=headers, json=payload)
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

# ========= 第一轮抽取（去噪 + 并发）=========
async def extract_rows_from_text(
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
    # 文本去噪
    cleaned = denoise_markdown(content_md or "")
    sys_prompt = _system_prompt(table_key, table_display_name, table_desc or "", lang, fields)
    chunks = split_markdown_packed(
        cleaned,
        max_tokens=chunk_tokens or 4000,
        overlap_tokens=80,
        encoding_name="cl100k_base"
    )

    sem = asyncio.Semaphore(EXTRACT_CONCURRENCY)

    async def run_one(i: int, ch: str, total: int):
        user_prompt = _user_prompt(ch, filename=filename, idx=i, total=total)
        async with sem:
            obj = await _acall_openai(model_name, temperature, sys_prompt, user_prompt)
        rows = _normalize_rows(obj)
        # 确保每条有 __explain 基础结构
        for r in rows:
            if "__explain" not in r or not isinstance(r["__explain"], dict):
                r["__explain"] = {"by_field": {}}
        return rows

    tasks = [run_one(i, ch, len(chunks)) for i, ch in enumerate(chunks, 1)]
    results: List[List[Dict[str, Any]]] = await asyncio.gather(*tasks)
    all_rows: List[Dict[str, Any]] = [r for batch in results for r in batch]

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

# ========= 二次核对/归并（按主键分组 + 并发 + 正确性验证）=========
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
        "给你一组候选记录（已按主键分组）。请：\n"
        "1) 仅在候选值中**择优并合并**为1条真实记录；禁止创造新值；\n"
        "2) **证据精炼**：对每个被选字段，从给定的 by_field.text 中提取**核心片段**（去冗余/去无关，<=120字），写入 `text`；\n"
        "3) **格式校正**：若可依据证据改写成符合字段类型/单位/日期格式的“正确值”，请直接输出校正后的值；若证据不足或冲突，**该字段置空**；\n"
        "4) 冲突难判时留空，并在 reason 中简述原因（<=40字）；\n"
        "5) 输出仅 JSON：{\"rows\":[{<字段值>, \"__explain\":{\"by_field\":{fieldKey:{text,reason}}}}]}，满足下述 JSON Schema（提示）：\n"
        f"{schema}\n"
        f"输出语言偏好：{lang}。\n"
    )

def _shrink(s: str, n: int) -> str:
    if not s:
        return ""
    s = s.strip().replace("\n", " ")
    return s if len(s) <= n else (s[:n] + "…")

def _build_reconcile_payload(rows: List[Dict[str, Any]], max_candidates: int = 1200) -> Dict[str, Any]:
    cands = []
    for i, r in enumerate(rows):
        explain = r.get("__explain") or {}
        by_field = (explain.get("by_field") or {}) if isinstance(explain, dict) else {}
        compact_by_field = {}
        for fk, info in by_field.items():
            if not isinstance(info, dict):
                continue
            compact_by_field[fk] = {
                "text": _shrink(info.get("text") or "", 160),   # 提供原始摘录，由模型精炼核心片段
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

def _backfill_explain(final_rows: List[Dict[str, Any]], payload: Dict[str, Any]) -> None:
    cand_map = {c["candidate_id"]: c for c in (payload.get("candidates") or [])}
    # 这里不强制来源 id，但若模型未给 text/reason，则兜底取任意候选的 by_field
    for r in final_rows:
        explain = r.get("__explain") or {}
        by_field = explain.get("by_field") or {}
        for fk, info in list(by_field.items()):
            if not isinstance(info, dict):
                by_field[fk] = {"text": None, "reason": None}
                continue
            text = _shrink(info.get("text") or "", 120)
            reason = _shrink(info.get("reason") or "", 40)
            if not text:
                # 任取一个候选的该字段 text 作为兜底
                for c in cand_map.values():
                    cf = (c.get("by_field") or {}).get(fk) or {}
                    if cf.get("text"):
                        text = _shrink(cf["text"], 120)
                        break
            if not reason:
                reason = "来自候选最相关片段"
            by_field[fk] = {"text": text or None, "reason": reason or None}
        explain["by_field"] = by_field
        r["__explain"] = explain

def _validate_and_clean_row(row: Dict[str, Any], fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对二轮合并后的单行做类型/格式验证：
    - 无法按字段类型正确解析的值 → 置空，并在 reason 保留“格式不合法/证据不足”
    """
    by_field = ((row.get("__explain") or {}).get("by_field") or {})
    for f in fields:
        fk = f["fieldKey"]
        if fk not in row:
            continue
        casted = _coerce_value_by_type(f.get("dataType") or "string", row.get(fk))
        ok = any(v is not None for k,v in casted.items() if k.startswith("value"))
        if not ok:
            # 置空该字段，reason 补充
            row[fk] = None
            if fk not in by_field:
                by_field[fk] = {"text": None, "reason": "格式不合法或证据不足"}
            else:
                if not by_field[fk].get("reason"):
                    by_field[fk]["reason"] = "格式不合法或证据不足"
    if "__explain" not in row or not isinstance(row["__explain"], dict):
        row["__explain"] = {"by_field": by_field}
    else:
        row["__explain"]["by_field"] = by_field
    return row

async def reconcile_rows_with_llm(
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

    # 主键列表
    pk_fields = [f["fieldKey"] for f in fields if f.get("isPk") == 1]

    # 分组函数
    def pk_tuple(r: Dict[str, Any]):
        if not pk_fields:
            return ("__all__",)
        return tuple((str(r.get(pk) or "").strip()) for pk in pk_fields)

    # 分组
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in extracted_rows:
        groups.setdefault(pk_tuple(r), []).append(r)

    # 小组内并发归并
    sem = asyncio.Semaphore(RECONCILE_CONCURRENCY)
    sys_prompt = _reconcile_system_prompt(table_key, table_display_name, table_desc or "", lang, fields)

    async def merge_one_group(rows_in_group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # 组内只有 0/1 条，直接返回，无需 LLM
        if len(rows_in_group) <= 1:
            # 但仍做一次格式合法性校验
            return [_validate_and_clean_row(dict(r), fields) for r in rows_in_group]

        payload = _build_reconcile_payload(rows_in_group)
        user = json.dumps(payload, ensure_ascii=False)

        async with sem:
            obj = await _acall_openai(model_name, temperature or 0.1, sys_prompt, user)

        final_rows = obj.get("rows") or []
        # 回填证据（若缺失）并做最后格式验证
        _backfill_explain(final_rows, payload)
        final_rows = [_validate_and_clean_row(r, fields) for r in final_rows]
        return final_rows

    tasks = [merge_one_group(rows) for rows in groups.values()]
    results: List[List[Dict[str, Any]]] = await asyncio.gather(*tasks)

    # 汇总所有组的最终行
    merged: List[Dict[str, Any]] = [r for batch in results for r in batch]

    # 若无主键时，可能返回多条；保持与一轮一致（不再额外合并）
    return merged

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
                "evidenceText": evidence_text,
                "reason": reason,
            }
            cells.append(c)
    return cells
