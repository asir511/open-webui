import io
import logging
from open_webui.env import (
    SRC_LOG_LEVELS,
    DEVICE_TYPE,
    DOCKER,
    SENTENCE_TRANSFORMERS_BACKEND,
    SENTENCE_TRANSFORMERS_MODEL_KWARGS,
    SENTENCE_TRANSFORMERS_CROSS_ENCODER_BACKEND,
    SENTENCE_TRANSFORMERS_CROSS_ENCODER_MODEL_KWARGS,
)
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    Request,
    status,
    APIRouter,
)
from fastapi.responses import JSONResponse
from open_webui.utils.auth import (
    get_license_data,
    get_http_authorization_cred,
    decode_token,
    get_admin_user,
    get_verified_user,
)

import magic
import pymupdf4llm
import tempfile

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()

@router.get("/")
async def get_status(request: Request, user=Depends(get_verified_user)):
    return 'success'


def extract_pdf_to_md_bytesio(contents: bytes) -> str:
    """将 PDF 字节内容写入临时文件，并转换为 Markdown"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()
        return pymupdf4llm.to_markdown(tmp.name)


@router.post("/extract_text_to_markdown")
async def extract_text_to_markdown(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="仅支持 PDF 文件格式")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="文件为空或读取失败")

    try:
        mime = magic.from_buffer(contents, mime=True)
        if mime != "application/pdf":
            raise HTTPException(status_code=400, detail="上传内容不是有效 PDF 文件")
    except Exception:
        pass

    try:
        md_text = extract_pdf_to_md_bytesio(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Markdown 转换失败: {e}")

    return JSONResponse(content={"markdown": md_text})

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, ValidationError
from fastapi import Body

# LangChain (OpenAI 兼容)
try:
    from langchain_openai import ChatOpenAI
except Exception:
    # 兼容老版本包名
    from langchain.chat_models import ChatOpenAI  # type: ignore

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ========= 请求/响应 Pydantic =========

class IndicatorChildIn(BaseModel):
    name: str
    description: Optional[str] = None


class IndicatorIn(BaseModel):
    name: str
    description: Optional[str] = None
    children: List[IndicatorChildIn] = Field(default_factory=list)


class ExtractRequest(BaseModel):
    markdown: str = Field(..., description="待抽取的 Markdown 文本")
    indicators: List[IndicatorIn] = Field(..., description="两级指标树（顶级＋子项）")


# 每个顶级指标的抽取结果：rows 是“维度组合 + 单一度量列(列名=顶级指标名)”
class IndicatorRows(BaseModel):
    indicator: str = Field(..., description="顶级指标名")
    rows: List[Dict[str, Union[str, int, float, Dict[str, Any]]]] = Field(
        default_factory=list,
        description="每行包含 __dims__（子项=值）以及一个以顶级指标名为键的度量值",
    )


class ExtractResponse(BaseModel):
    items: List[IndicatorRows] = Field(default_factory=list)
    model_usage: Optional[Dict[str, int]] = None
    notes: Optional[str] = None


# ========= LangChain 组装 =========

OPENAI_BASE_URL = "http://192.168.0.114:3000/v1"
OPENAI_MODEL = "chat1"
OPENAI_API_KEY = "sk-2pH4HUSK4wikhU7NSqMqF3Ldi7c2r89sJmRQBBJ9PS7vN1AM"

SYSTEM_INSTRUCTIONS = """\
You are an information extraction model. You will receive:
1) A Markdown document.
2) A two-level indicator schema: top-level indicators and their child items (dimensions).

Your job:
- For EACH top-level indicator, produce a table of rows.
- Each row = a combination of child dimensions (__dims__) + ONE measure value.
- The measure key MUST be EXACTLY the top-level indicator's name (verbatim).
- Place dimension key-value pairs inside __dims__ (object).
- Use concise, normalized values for dimensions (strip whitespace, keep semantic meaning).
- If numeric values exist, output numbers (int/float). Otherwise leave as strings.
- If a certain indicator is not present in the doc, return an empty rows array for it (do NOT hallucinate).
- Prefer structured data present in tables, bullet lists, or obvious key-value fragments.
- If units exist, normalize to a single unit where possible and DO NOT include unit words in the numeric field. If unsure, keep raw text.

Output MUST conform to the provided schema.
"""

# 可选：更“强”的风格化提示，指导维度与度量映射
INDICATOR_GUIDE_TEMPLATE = """\
Two-level indicators definition:

{schema_text}

Rules:
- Only use dimension keys listed under each indicator's children.
- If a child is irrelevant for a specific row, you may omit it from __dims__.
- Each row MUST include exactly one measure field named exactly as the indicator (e.g., "资产数量": 35).
- Avoid duplications; aggregate when obviously the same dimension combination appears multiple times.
"""

def _schema_to_text(indicators: List[IndicatorIn]) -> str:
    lines = []
    for ind in indicators:
        lines.append(f"- Indicator: {ind.name}")
        if ind.description:
            lines.append(f"  Description: {ind.description}")
        if ind.children:
            lines.append(f"  Children (dimensions):")
            for c in ind.children:
                if c.description:
                    lines.append(f"    - {c.name}: {c.description}")
                else:
                    lines.append(f"    - {c.name}")
        else:
            lines.append("  Children (dimensions): (none)")
    return "\n".join(lines)


def _build_prompt(markdown: str, indicators: List[IndicatorIn]) -> ChatPromptTemplate:
    guide = INDICATOR_GUIDE_TEMPLATE.format(schema_text=_schema_to_text(indicators))
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_INSTRUCTIONS),
            ("system", guide),
            ("human", "Here is the Markdown document:\n\n```markdown\n{md}\n```"),
        ]
    ).partial(md=markdown)


def _validate_and_patch_items(
    indicators: List[IndicatorIn], raw_items: List[IndicatorRows]
) -> List[IndicatorRows]:
    """确保每个顶级 indicator 都有一项；并对 rows 进行基本校验/修补：
    - rows 中必须存在度量键 == indicator 名；
    - 维度装入 __dims__（object），仅允许子项里定义的维度名；
    """
    # 构建快速访问
    top_index = {i.name: i for i in indicators}

    # 先把模型输出转成 dict，便于我们修补
    out_map: Dict[str, IndicatorRows] = {}
    for it in raw_items:
        name = it.indicator
        if name not in top_index:
            # 未在定义里出现的指标，丢弃（防幻觉）
            continue
        fixed_rows: List[Dict[str, Any]] = []
        allowed_dims = {c.name for c in top_index[name].children}

        for row in it.rows:
            row = dict(row)  # copy
            dims = row.pop("__dims__", {})
            if not isinstance(dims, dict):
                dims = {}

            # 过滤/归位维度：仅保留定义过的子项名
            clean_dims = {}
            for k, v in dims.items():
                if k in allowed_dims:
                    clean_dims[k] = v

            # 查找度量列（必须和顶级指标名相同）
            measure_key = name
            measure_val = row.get(measure_key, None)

            # 如果模型把维度误塞到顶层，尽量搬回 __dims__
            for k in list(row.keys()):
                if k != measure_key and k in allowed_dims:
                    clean_dims[k] = row.pop(k)

            # 如果没有度量列，且顶层还剩下一个可解释的数值字段，兜底映射
            if measure_val is None:
                num_keys = [k for k, v in row.items() if isinstance(v, (int, float))]
                if len(num_keys) == 1:
                    measure_val = row.pop(num_keys[0])
                else:
                    # 无法修补，跳过此行
                    continue

            fixed_rows.append({"__dims__": clean_dims, measure_key: measure_val})

        out_map[name] = IndicatorRows(indicator=name, rows=fixed_rows)

    # 保证每个定义的顶级指标都存在
    for i in indicators:
        if i.name not in out_map:
            out_map[i.name] = IndicatorRows(indicator=i.name, rows=[])

    # 保持与输入指标顺序一致
    return [out_map[i.name] for i in indicators]


from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

MAX_CHARS_PER_CHUNK = 15_000  # 最大上下文字符数（可根据模型实际 context window 调整）

def split_markdown(md_text: str) -> list[str]:
    """
    将 Markdown 文本切片，保证每块不超过 MAX_CHARS_PER_CHUNK。
    优先按 Markdown 标题切，如果块仍过大则进一步按字符数递归切。
    """
    # 先按 Markdown 标题拆分
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_splits = splitter.split_text(md_text)

    # 把 header_splits 合并成纯文本块（MarkdownHeaderTextSplitter 会返回 Document 列表）
    md_chunks = [doc.page_content for doc in header_splits]

    # 对仍超过 MAX_CHARS_PER_CHUNK 的块再递归切分
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHARS_PER_CHUNK,
        chunk_overlap=200,   # 适当重叠以保持上下文
        separators=["\n\n", "\n", " ", ""],
    )
    final_chunks = []
    for chunk in md_chunks:
        if len(chunk) > MAX_CHARS_PER_CHUNK:
            final_chunks.extend(char_splitter.split_text(chunk))
        else:
            final_chunks.append(chunk)

    return final_chunks

@router.post("/extract_structured_from_markdown", response_model=ExtractResponse)
async def extract_structured_from_markdown(
    payload: ExtractRequest = Body(...),
    user=Depends(get_verified_user),
):
    if not payload.markdown or not payload.indicators:
        raise HTTPException(status_code=400, detail="markdown 与 indicators 不能为空")

    # 1) 切片 Markdown
    chunks = split_markdown(payload.markdown)

    # 初始化最终结果容器
    all_items: list[IndicatorRows] = []

    # 初始化模型
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.0,
        request_timeout=None
    )
    invoke = llm.invoke("你好")
    structured_llm = llm.with_structured_output(ExtractResponse)

    # 2) 对每个 chunk 单独抽取，再合并结果
    for idx, chunk in enumerate(chunks, start=1):
        prompt = _build_prompt(chunk, payload.indicators)
        try:
            part_result: ExtractResponse = await structured_llm.ainvoke(prompt.format_messages())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"第 {idx} 个切片抽取失败: {e}")

        # 合并到总结果
        for it in part_result.items:
            existing = next((x for x in all_items if x.indicator == it.indicator), None)
            if existing:
                existing.rows.extend(it.rows)
            else:
                all_items.append(it)

    # 3) 结果修补
    patched_items = _validate_and_patch_items(payload.indicators, all_items)
    return ExtractResponse(items=patched_items, model_usage=None)
