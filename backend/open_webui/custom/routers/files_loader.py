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
    Query
)
from fastapi.responses import JSONResponse
from open_webui.utils.auth import (
    get_license_data,
    get_http_authorization_cred,
    decode_token,
    get_admin_user,
    get_verified_user,
)

import pymupdf4llm
import tempfile

import os, sys
import subprocess
import shutil
import magic

import pypandoc
from pathlib import Path

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.table_utils import (
    extract_pdf_tables_with_camelot_bytes,
    extract_pdf_tables_to_dfs,
    extract_docx_tables_to_dfs,
    dfs_pack_as_render,
)

from utils.html_utils import (
    detect_source_type_strict,
    convert_file_to_html,
    enhance_html_tables_with_llm,
    compute_sha256_hex,
)

from utils.mineru_utils import (
    file_parse_smart,
    unwrap_md_from_mineru,
    sanitize_markdown
)

from utils.word_utils import (
    word_to_markdown
)

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()

@router.get("/")
async def get_status(request: Request, user=Depends(get_verified_user)):
    return 'success'

@router.post("/extract_html_structured")
async def extract_html_structured(file: UploadFile = File(...)):
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # 强识别：自动辨别 doc/docx/pdf_text/pdf_scanned/rtf/html/txt/md
    source_type = detect_source_type_strict(file.filename, file_bytes)

    base_html = convert_file_to_html(file_bytes, source_type, original_name=file.filename)
    final_html = enhance_html_tables_with_llm(base_html, filename=file.filename)

    html_sha256 = compute_sha256_hex(final_html)
    return JSONResponse(content={
        "source_type": source_type,  # 现在可能是 pdf_text 或 pdf_scanned
        "original_name": file.filename,
        "html_sha256": html_sha256,
        "content_html": final_html,
        "html_length": len(final_html),
        "lang": "zh",
        "title": file.filename.rsplit(".", 1)[0],
        "pages": None,
    })


def extract_pdf_to_md_bytesio(contents: bytes) -> str:
    """将 PDF 字节内容写入临时文件，并转换为 Markdown"""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()
        return pymupdf4llm.to_markdown(tmp.name)

# ================== 工具函数 ==================
def _guess_ext_and_mime(filename: str, contents: bytes):
    ext = (os.path.splitext(filename)[1] or "").lower()
    try:
        mime = magic.from_buffer(contents, mime=True)
    except Exception:
        mime = None
    return ext, mime

def _ensure_file(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)

def _pandoc_convert_file_to_md(in_path: str) -> str:
    """
    用 Pandoc 把文件转成 GFM Markdown（带管道表格），尽量避免换行破坏表格。
    """
    extra_args = [
        "--wrap=none",
        "--columns=999",
        # 需要时可加："--extract-media=media"   # 抽出内嵌图片到 media/ 并输出引用
        # 需要时可加："--reference-links"
    ]
    # 目标用 GFM 并启用 pipe_tables（管道表格），raw_html 允许保留一些行内 HTML（少量极端场景更稳）
    return pypandoc.convert_file(
        in_path,
        to="gfm+pipe_tables+raw_html",
        outputfile=None,
        extra_args=extra_args
    ).strip()

def _convert_docx_bytes_to_md_via_pandoc(contents: bytes) -> str:
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.docx")
        _ensure_file(in_path, contents)
        return _pandoc_convert_file_to_md(in_path)

def _convert_doc_bytes_to_md_via_lo_then_pandoc(contents: bytes) -> str:
    """
    .doc →（LibreOffice 无头）→ .docx → Pandoc → Markdown
    """
    with tempfile.TemporaryDirectory() as td:
        src_doc = os.path.join(td, "in.doc")
        _ensure_file(src_doc, contents)
        out_dir = td

        soffice = shutil.which("soffice") or "soffice"
        try:
            subprocess.run(
                [soffice, "--headless", "--convert-to", "docx", "--outdir", out_dir, src_doc],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="服务器未安装 LibreOffice，无法转换 .doc")
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f".doc 转换失败：{e.stderr.decode(errors='ignore')[:500]}")

        # 找到输出的 docx
        docx_path = None
        for name in os.listdir(td):
            if name.lower().endswith(".docx"):
                docx_path = os.path.join(td, name)
                break
        if not docx_path:
            raise HTTPException(status_code=500, detail=".doc 转 .docx 后未找到输出文件")

        return _pandoc_convert_file_to_md(docx_path)

# =============== （可选）PDF 表格增强：抽表格并拼回 =================
def _maybe_enhance_pdf_tables(contents: bytes) -> str:
    """
    如果你安装了 camelot 或 tabula，可在这里把 PDF 表格抽出来转成 Markdown 表格。
    默认返回空字符串（不开启），避免增加系统依赖。
    """
    # 示例：返回空，表示不增强。
    return ""

@router.post("/extract_text_to_markdown")
async def extract_text_to_markdown(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名缺失")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="文件为空或读取失败")

    ext, mime = _guess_ext_and_mime(file.filename, contents)

    is_pdf = (ext == ".pdf") or (mime == "application/pdf") or (file.content_type == "application/pdf")
    is_docx = (ext == ".docx") or (mime in {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"})
    is_doc  = (ext == ".doc")  or (mime in {"application/msword", "application/x-msword"})

    try:
        if is_pdf:
            # 1) 先用现有 PDF→MD（拿到文本）
            md_text = extract_pdf_to_md_bytesio(contents) or ""
            # 2) （可选）表格增强，抽出表格转为 Markdown 表格并拼接
            tables_md = _maybe_enhance_pdf_tables(contents)
            if tables_md:
                md_text = md_text.rstrip() + "\n\n" + tables_md.lstrip()

        elif is_docx:
            # 直接 Pandoc：DOCX→GFM（带 pipe_tables）
            md_text = _convert_docx_bytes_to_md_via_pandoc(contents)

        elif is_doc:
            # DOC→（LibreOffice）→DOCX→Pandoc
            md_text = _convert_doc_bytes_to_md_via_lo_then_pandoc(contents)

        else:
            # 大小写兜底
            if ext.lower() == ".docx":
                md_text = _convert_docx_bytes_to_md_via_pandoc(contents)
            elif ext.lower() == ".doc":
                md_text = _convert_doc_bytes_to_md_via_lo_then_pandoc(contents)
            else:
                raise HTTPException(status_code=400, detail=f"暂不支持的文件类型：{ext or mime or file.content_type}")

    except HTTPException:
        raise
    except OSError as e:
        # pandoc/soffice 不存在、权限问题等
        raise HTTPException(status_code=500, detail=f"服务器执行依赖失败：{str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Markdown 转换失败：{e}")

    return JSONResponse(content={"markdown": md_text})

APPEND_TABLES_TO_MD = True  # 是否把 GFM 表格附加到 md 文末

@router.post("/extract_text_to_markdown_structured")
async def extract_text_to_markdown_structured(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名缺失")

    original_name = file.filename
    ext = Path(original_name).suffix.lower()
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    allowed_exts = image_exts | {".pdf", ".doc", ".docx"}
    if ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")

    source_type = "image" if ext in image_exts else ext.lstrip(".")

    contents = await file.read()
    with tempfile.TemporaryDirectory(prefix="mineru_upload_") as tmpdir:
        tmp_path = Path(tmpdir) / original_name
        tmp_path.write_bytes(contents)

        # -------------------------------
        # ① Word：优先走本地转换
        # -------------------------------
        if ext in {".doc", ".docx"}:
            try:
                # 可选：通过环境变量给图片前缀（例如 CDN / 静态域名）
                image_base_url = os.getenv("WORD_MEDIA_BASE_URL")  # e.g. https://cdn.example.com/word-media
                wres = word_to_markdown(tmp_path, image_base_url=image_base_url or None)
                md_text = (wres.get("markdown") or "").strip()
                md_sha256 = hashlib.sha256(md_text.encode("utf-8")).hexdigest()
                return JSONResponse(
                    content={
                        "source_type": source_type,            # doc / docx
                        "original_name": original_name,
                        "md_sha256": md_sha256,
                        "content_md": md_text,
                        "md_length": len(md_text),
                        "lang": "zh",
                        "title": Path(original_name).stem,
                        "pages": None,
                        "media_dir": wres.get("media_dir"),    # 供你后续静态服务用
                        "images": wres.get("images", []),
                    }
                )
            except Exception as e:
                # 失败：回退到 MinerU（可改为直接 raise HTTPException）
                # print(f"Word to Markdown 失败，将回退到 MinerU。原因：{e}")
                pass

        # -------------------------------
        # ② 非 Word：图片先转 PDF，再交给 MinerU
        # -------------------------------
        path_to_process = tmp_path

        if source_type == "image":
            try:
                from PIL import Image
            except ImportError:
                raise HTTPException(
                    status_code=500,
                    detail="缺少 Pillow 以处理图片，请安装: pip install pillow",
                )
            img = Image.open(tmp_path)
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            pdf_path = Path(tmpdir) / (Path(original_name).stem + ".pdf")
            img.save(pdf_path, "PDF", resolution=300.0)
            path_to_process = pdf_path

        base_url = os.getenv("MINERU_API_BASE", "http://192.168.0.114:8181")
        try:
            result = file_parse_smart(
                inputs=[str(path_to_process)],
                base_url=base_url,
                backend="pipeline",
                lang="ch",
                parse_method="auto",
                formula_enable=True,
                table_enable=True,
                max_pages_per_request=25,
                convert_office=True,   # 仍保留对 doc/docx 的自动转 pdf（只有走到这里才会用到）
                timeout=900,
            )
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"调用 MinerU 失败: {e}")

    if not result:
        raise HTTPException(status_code=500, detail="MinerU 返回为空")

    first = next(iter(result.values()))
    raw_markdown_field = first.get("markdown") or ""

    # —— 抽取 + 反转义 + 清洗 ——
    md_text = unwrap_md_from_mineru(raw_markdown_field)
    image_base_url = os.getenv("MINERU_IMAGE_BASE_URL")  # 如果 MinerU 输出中有 images/ 相对路径
    md_text = sanitize_markdown(md_text, image_base_url=image_base_url or None).strip()

    total_pages = int(first.get("total_pages") or 0)
    md_sha256 = hashlib.sha256(md_text.encode("utf-8")).hexdigest()

    return JSONResponse(
        content={
            "source_type": source_type,  # pdf / image（已转pdf）
            "original_name": original_name,
            "md_sha256": md_sha256,
            "content_md": md_text,
            "md_length": len(md_text),
            "lang": "zh",
            "title": Path(original_name).stem,
            "pages": total_pages if total_pages > 0 else None,
        }
    )

from typing import Any, Dict, List, Optional, Union, Literal
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
    source_name: Optional[str] = Field(None, description="可选：来源文件名/标识")



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

# ========= 新增：服务端加工后的结构 =========
class Fact(BaseModel):
    id: str
    file: Optional[str] = None             # 可选，若前端传了来源文件名
    indicator: str                         # 顶级指标名（指标分类）
    value_number: Optional[float] = None
    value_text: Optional[str] = None
    unit: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

class FactDim(BaseModel):
    id: str
    fact_id: str
    dim_name: str
    dim_value: str

class LongTableResponse(BaseModel):
    facts: List[Fact] = Field(default_factory=list)
    fact_dims: List[FactDim] = Field(default_factory=list)

class FlatTableResponse(BaseModel):
    columns: List[str] = Field(default_factory=list)  # 列名清单
    rows: List[Dict[str, Any]] = Field(default_factory=list)  # 每行是同级字典


# ========= LangChain 组装 =========

OPENAI_BASE_URL = "http://192.168.0.114/v1"
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

MAX_CHARS_PER_CHUNK = 5_000  # 最大上下文字符数（可根据模型实际 context window 调整）

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

import hashlib, uuid

def _uuid5(ns: str, name: str) -> str:
    # 稳定 id（去重/可重算），避免重复插入；ns 建议用固定字符串防碰撞
    return str(uuid.uuid5(uuid.UUID(bytes=hashlib.sha1(ns.encode()).digest()[:16]), name))

def _to_long_tables(
    resp_items: List[IndicatorRows],
    source_name: Optional[str] = None
) -> LongTableResponse:
    facts: List[Fact] = []
    fact_dims: List[FactDim] = []

    for item in resp_items:
        indicator = item.indicator
        for row in item.rows:
            dims = row.get("__dims__", {}) or {}
            measure_val = row.get(indicator, None)

            # 生成稳定 fact_id：文件名 + 指标 + 维度 JSON（排序后）
            dims_key = "|".join(f"{k}={dims[k]}" for k in sorted(dims.keys()))
            fact_key = f"{source_name or ''}||{indicator}||{dims_key}||{measure_val}"
            fact_id = _uuid5("ns:extract:fact", fact_key)

            value_number = None
            value_text = None
            if isinstance(measure_val, (int, float)):
                value_number = float(measure_val)
            elif measure_val is not None:
                value_text = str(measure_val)

            # 额外键兜底
            extra_keys = [k for k in row.keys() if k not in ("__dims__", indicator)]
            extra = {k: row[k] for k in extra_keys} if extra_keys else None

            facts.append(Fact(
                id=fact_id,
                file=source_name,
                indicator=indicator,
                value_number=value_number,
                value_text=value_text,
                unit=None,
                extra=extra,
            ))

            for k, v in dims.items():
                dim_id = _uuid5("ns:extract:dim", f"{fact_id}||{k}||{v}")
                fact_dims.append(FactDim(
                    id=dim_id,
                    fact_id=fact_id,
                    dim_name=k,
                    dim_value="" if v is None else str(v),
                ))

    return LongTableResponse(facts=facts, fact_dims=fact_dims)


def _to_flat_table(long_resp: LongTableResponse) -> FlatTableResponse:
    # 1) 收集所有维度名
    dim_names = sorted({d.dim_name for d in long_resp.fact_dims})

    # 2) 构造 fact_id -> {dim_name: dim_value}
    dim_map: Dict[str, Dict[str, str]] = {}
    for d in long_resp.fact_dims:
        dim_map.setdefault(d.fact_id, {})[d.dim_name] = d.dim_value

    # 3) 生成平铺 rows
    rows: List[Dict[str, Any]] = []
    for f in long_resp.facts:
        dims_obj = dim_map.get(f.id, {})
        row: Dict[str, Any] = {
            "__file": f.file,
            "__indicator": f.indicator,
            "value": f.value_number if f.value_number is not None else f.value_text,
            "extra": f.extra,
        }
        for name in dim_names:
            row[f"dim_{name}"] = dims_obj.get(name)
        rows.append(row)

    columns = ["__file", "__indicator", *[f"dim_{n}" for n in dim_names], "value", "extra"]
    return FlatTableResponse(columns=columns, rows=rows)

@router.post("/extract_structured_from_markdown")
async def extract_structured_from_markdown(
    payload: ExtractRequest = Body(...),
    user=Depends(get_verified_user),
    output: Literal["raw", "long", "flat"] = Query("flat", description="返回形态：raw|long|flat")
):
    if not payload.markdown or not payload.indicators:
        raise HTTPException(status_code=400, detail="markdown 与 indicators 不能为空")

    # 1) 切片
    chunks = split_markdown(payload.markdown)

    # 2) LLM 初始化（你原来的保持不变）
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0.0,
        request_timeout=None,
    )
    structured_llm = llm.with_structured_output(ExtractResponse)

    # 3) 聚合 raw items
    all_items: List[IndicatorRows] = []
    for idx, chunk in enumerate(chunks, start=1):
        prompt = _build_prompt(chunk, payload.indicators)
        try:
            part: ExtractResponse = await structured_llm.ainvoke(prompt.format_messages())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"第 {idx} 个切片抽取失败: {e}")

        # 合并
        for it in part.items:
            ex = next((x for x in all_items if x.indicator == it.indicator), None)
            if ex:
                ex.rows.extend(it.rows)
            else:
                all_items.append(it)

    # 4) 修补
    patched_items = _validate_and_patch_items(payload.indicators, all_items)

    # 5) 根据 output 形态返回
    if output == "raw":
        return ExtractResponse(items=patched_items, model_usage=None)

    # 5.1 先转长表（统一父子结构）
    long_resp = _to_long_tables(patched_items, source_name=payload.source_name)

    if output == "long":
        return long_resp

    # 5.2 flat：返回同级宽表（含 columns）
    flat_resp = _to_flat_table(long_resp)
    return flat_resp