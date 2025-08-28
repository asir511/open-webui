# utils/table_utils.py
import io
import os
import re
import tempfile
import hashlib
import subprocess
from typing import Literal, Optional, Tuple

# ---- 第三方库依赖 ----
# pip install mammoth pdfminer.six lxml beautifulsoup4 openai
import mammoth
from bs4 import BeautifulSoup, NavigableString, Comment
from lxml import html as lxml_html  # 用于结构分析（可选）
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.converter import HTMLConverter
import PyPDF2
from pdf2image import convert_from_bytes
import magic
import pandas as pd
import camelot
import pdfplumber

# OpenAI 兼容客户端（vLLM/oneapi）
from openai import OpenAI

# --------- 配置（建议改用环境变量） ----------
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://47.243.192.2:15048/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-2pH4HUSK4wikhU7NSqMqF3Ldi7c2r89sJmRQBBJ9PS7vN1AM")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "chat1")

DocType = Literal[
    "doc", "docx",
    "pdf_text", "pdf_scanned",
    "rtf", "html", "txt", "md"
]

# 允许的结构性标签（可按需增减）
_ALLOWED_TAGS = {
    "html","head","meta","title","body",
    "h1","h2","h3","h4","h5","h6",
    "p","br","hr",
    "ul","ol","li",
    "table","thead","tbody","tr","th","td",
    "strong","em","b","i","u","code","pre","blockquote"
}
# 允许的属性（尽量少）；表格保留合并单元格
_ALLOWED_ATTRS = {"rowspan","colspan","lang"}

# --------- 类型识别 ----------
def detect_source_type_strict(filename: str, file_bytes: bytes) -> DocType:
    """
    完备的类型识别：
    1) libmagic MIME
    2) 魔数校验
    3) PDF 进一步判断是否扫描件
    4) 回退扩展名与尝试解析
    """
    mime = _sniff_mime(file_bytes)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    # --- 魔数快速路径 ---
    kind = _detect_by_magic_bytes(file_bytes)
    if kind:
        if kind == "pdf":
            return "pdf_scanned" if _looks_like_scanned_pdf(file_bytes) else "pdf_text"
        if kind in ("docx", "doc", "rtf"):
            return kind
        # OOXML 可能是 docx/xlsx/pptx，进一步看 [Content_Types].xml
        if kind == "ooxml_zip":
            ooxml_inner = _probe_ooxml_type(file_bytes)
            if ooxml_inner == "docx":
                return "docx"

    # --- MIME 路径 ---
    if mime:
        if mime in ("application/pdf",):
            return "pdf_scanned" if _looks_like_scanned_pdf(file_bytes) else "pdf_text"
        if mime in ("application/msword",):
            return "doc"
        if mime in ("application/vnd.openxmlformats-officedocument.wordprocessingml.document",):
            return "docx"
        if mime in ("text/rtf", "application/rtf"):
            return "rtf"
        if mime.startswith("text/html"):
            return "html"
        if mime.startswith("text/"):
            # 可能是 txt / md
            if ext in ("md", "markdown"):
                return "md"
            return "txt"

    # --- 扩展名回退 ---
    if ext == "pdf":
        return "pdf_scanned" if _looks_like_scanned_pdf(file_bytes) else "pdf_text"
    if ext == "doc":
        return "doc"
    if ext == "docx":
        return "docx"
    if ext in ("rtf",):
        return "rtf"
    if ext in ("html", "htm"):
        return "html"
    if ext in ("md", "markdown"):
        return "md"
    if ext in ("txt",):
        return "txt"

    # --- 解析探测（最后兜底） ---
    if _safe_try_pdf_head(file_bytes):
        return "pdf_scanned" if _looks_like_scanned_pdf(file_bytes) else "pdf_text"

    # 实在识别不了，按 txt 兜底
    return "txt"

def _sniff_mime(file_bytes: bytes) -> Optional[str]:
    try:
        return magic.Magic(mime=True).from_buffer(file_bytes[:4096])
    except Exception:
        return None

def _detect_by_magic_bytes(file_bytes: bytes) -> Optional[str]:
    head = file_bytes[:8]
    # PDF: %PDF-
    if head.startswith(b"%PDF-"):
        return "pdf"
    # OOXML zip: "PK\x03\x04"
    if head.startswith(b"PK\x03\x04"):
        # 进一步区分由 _probe_ooxml_type 完成
        return "ooxml_zip"
    # OLE2 (doc/xls/ppt 旧格式): D0 CF 11 E0 A1 B1 1A E1
    if head.startswith(b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"):
        return "doc"
    # RTF: "{\rtf"
    if file_bytes[:5] == b"{\\rtf":
        return "rtf"
    return None

def _probe_ooxml_type(file_bytes: bytes) -> Optional[str]:
    """
    读取 zip 中的 [Content_Types].xml / docProps 尝试识别是否 Word 文档
    """
    import zipfile, io
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = set(zf.namelist())
            # Word 文档一般含有 word/ 或 docProps/ 以及 [Content_Types].xml
            if "word/document.xml" in names:
                return "docx"
    except Exception:
        return None
    return None

def _safe_try_pdf_head(file_bytes: bytes) -> bool:
    return file_bytes[:5] == b"%PDF-"

def _looks_like_scanned_pdf(file_bytes: bytes, sample_pages: int = 3) -> bool:
    """
    判断 PDF 是否扫描件：
      - 用 PyPDF2 粗查：若前 N 页几乎无文本但含图片对象 => 扫描件
      - 或用 pdfminer 提取文本长度很短
    """
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        n = min(len(reader.pages), sample_pages)
        text_chars = 0
        image_xobjects = 0
        for i in range(n):
            page = reader.pages[i]
            try:
                text = page.extract_text() or ""
                text_chars += len(text.strip())
            except Exception:
                pass
            # 统计 XObject 图片
            try:
                resources = page.get("/Resources")
                if resources and "/XObject" in resources:
                    xobjs = resources["/XObject"]
                    if hasattr(xobjs, "get_object"):
                        xobjs = xobjs.get_object()
                    for k in xobjs:
                        x = xobjs[k]
                        obj = x.get_object() if hasattr(x, "get_object") else x
                        if obj.get("/Subtype") == "/Image":
                            image_xobjects += 1
            except Exception:
                pass
        # 经验阈值：文本极少而图片较多 => 扫描件
        if text_chars < 40 and image_xobjects >= 1:
            return True
    except Exception:
        # 读不出就按扫描件倾向（避免走文本管线报错）
        return True
    return False

# --------- 主转化入口 ----------
def convert_file_to_html(file_bytes: bytes, source_type: str, original_name: str = "") -> str:
    """
    兼容旧签名；如希望“强识别”，请在路由里先调用 detect_source_type_strict。
    """
    # 允许传入老的三值 doc/docx/pdf；也兼容新枚举
    st = source_type
    if source_type == "pdf":
        # 默认把 pdf 当“文本 PDF”处理，后续可切换成 strict 检测
        st = "pdf_text"

    if st == "docx":
        html_str = _convert_docx_to_html(file_bytes)
    elif st == "doc":
        html_str = _convert_doc_to_html_with_libreoffice(file_bytes, original_name=original_name)
    elif st == "pdf_text":
        html_str = _convert_pdf_to_html_text_based(file_bytes)
    elif st == "pdf_scanned":
        html_str = _convert_pdf_scanned_to_html(file_bytes)
    elif st == "rtf":
        html_str = _convert_rtf_to_html(file_bytes)
    elif st == "html":
        html_str = _wrap_html(file_bytes.decode("utf-8", errors="ignore"))
    elif st in ("txt", "md"):
        html_str = _convert_textlike_to_html(file_bytes, kind=st)
    else:
        raise ValueError(f"Unsupported source_type: {source_type}")

    html_str = _basic_postprocess_html(html_str, original_name=original_name)
    return html_str


# --------- DOCX -> HTML ----------
def _convert_docx_to_html(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as f:
        result = mammoth.convert_to_html(f, style_map=_mammoth_style_map())
        html = result.value  # HTML string
    return _wrap_html(html)

def _mammoth_style_map() -> str:
    # 这里可以根据需要做更详细的样式映射
    return """
p[style-name='Title'] => h1:fresh
p[style-name='Subtitle'] => h2:fresh
r[style-name='Strong'] => strong
table => table
"""

# --------- DOC (二进制旧格式) -> HTML ----------
def _convert_doc_to_html_with_libreoffice(file_bytes: bytes, original_name: str = "") -> str:
    """
    采用 LibreOffice headless 转换。需要系统已安装 libreoffice。
    Linux:   apt-get install -y libreoffice
    MacOS:   brew install --cask libreoffice
    Windows: 安装后将 soffice 加入 PATH
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 将 doc 落盘
        input_path = os.path.join(tmpdir, original_name or "input.doc")
        with open(input_path, "wb") as f:
            f.write(file_bytes)

        # 转换到 HTML
        # --convert-to html:"HTML (StarWriter)" 的过滤器名称在不同版本可能略有差异
        # 这里使用更通用的 --convert-to html
        cmd = [
            "soffice",
            "--headless",
            "--norestore",
            "--convert-to",
            "html",
            "--outdir",
            tmpdir,
            input_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"LibreOffice convert failed: {proc.stderr}")

        # 找到输出 html
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_html = os.path.join(tmpdir, f"{base_name}.html")
        if not os.path.exists(output_html):
            # 有些版本会输出 .htm
            output_html = os.path.join(tmpdir, f"{base_name}.htm")
            if not os.path.exists(output_html):
                raise RuntimeError("LibreOffice did not produce an HTML file.")

        with open(output_html, "r", encoding="utf-8", errors="ignore") as f:
            html_str = f.read()

    return _wrap_html(html_str)

# --------- PDF -> HTML（pdfminer.six） ----------
def _convert_pdf_to_html(file_bytes: bytes) -> str:
    """
    使用 pdfminer.six 的 HTMLConverter。保留基本块结构。
    对于复杂表格，后续交给 LLM 修复。
    """
    output = io.StringIO()
    laparams = LAParams(
        line_margin=0.2,
        word_margin=0.1,
        char_margin=1.0,
        boxes_flow=None,  # 让版面更自由，避免错误合并
        all_texts=True
    )
    rsrcmgr = PDFResourceManager()
    with io.BytesIO(file_bytes) as fp:
        with HTMLConverter(rsrcmgr, output, codec="utf-8", laparams=laparams) as conv:
            extract_text_to_fp(fp, outfp=conv, laparams=laparams, output_type="html", codec="utf-8")
    raw_html = output.getvalue()
    return _wrap_html(raw_html)

# ---- PDF(文本) -> HTML：用你之前修复后的 BytesIO 版本 ----
def _convert_pdf_to_html_text_based(file_bytes: bytes) -> str:
    """
    矢量/可检索PDF：
      1) Camelot 抽表格 -> 生成<table>
      2) pdfplumber 抽正文行 -> 过滤噪声 -> <p>
      3) 按页合并（先正文后表格，简单稳定；需要更精准可按bbox插入）
    """
    # 1) 落盘供 Camelot 读取（Camelot 需要路径）
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "input.pdf")
        with open(pdf_path, "wb") as f:
            f.write(file_bytes)

        # 2) 读取表格（先 lattice 再 stream）
        tables = []
        try:
            t_lattice = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
        except Exception:
            t_lattice = []
        try:
            t_stream = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        except Exception:
            t_stream = []

        for tset in (t_lattice or []):
            tables.append(tset)
        for tset in (t_stream or []):
            tables.append(tset)

        # 组装：page -> [<table_html>...]
        page_tables = {}
        for tbl in tables:
            df: pd.DataFrame = tbl.df  # Camelot统一输出字符串DataFrame
            # 简单策略：首行为表头（后续让 LLM 修复更复杂的多级表头/合并单元格）
            table_html = _df_to_min_table_html(df)
            page_tables.setdefault(tbl.page, []).append(table_html)

        # 3) 抽正文（pdfplumber 提取更干净）
        pages_html = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for p_idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
                lines = _clean_pdf_text_lines(text.splitlines())
                para_html = "\n".join(f"<p>{_escape_html(ln)}</p>" for ln in lines if ln.strip())
                tables_html = "\n".join(page_tables.get(p_idx, []))
                # 简单：每页“正文在前，表格在后”
                pages_html.append(f'<section data-page="{p_idx}">\n{para_html}\n{tables_html}\n</section>')

        html = "<div>\n" + "\n".join(pages_html) + "\n</div>"
        return _wrap_html(html)

def _df_to_min_table_html(df: "pd.DataFrame") -> str:
    # 以第一行作为表头；若第一行全是空则退化为无表头
    if df.shape[0] == 0:
        return "<table><tbody></tbody></table>"
    header = [str(x).strip() for x in list(df.iloc[0].values)]
    body = df.iloc[1:] if df.shape[0] > 1 else pd.DataFrame()

    thead = "<thead><tr>" + "".join(f"<th>{_escape_html(h)}</th>" for h in header) + "</tr></thead>"
    tbody_rows = []
    for _, row in body.iterrows():
        tds = "".join(f"<td>{_escape_html(str(v).strip())}</td>" for v in row.values)
        tbody_rows.append(f"<tr>{tds}</tr>")
    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"

_DOT_LEADER = re.compile(r"^[\s\.·•∙⋯]{3,}[\s\d]*$")  # 目录中的“..... 1”
_ONLY_NUM = re.compile(r"^\s*\d+\s*$")                 # 孤立页码
_REPEAT_CHAR = re.compile(r"(.)\1{2,}")                # 连续>=3的同字符（（（（、———、……）

def _clean_pdf_text_lines(lines):
    cleaned = []
    header_footer_cache = {}

    for ln in lines:
        raw = ln.strip()
        if not raw:
            continue
        if _DOT_LEADER.match(raw):
            continue
        if _ONLY_NUM.match(raw):
            continue
        # 压缩重复字符
        raw = _REPEAT_CHAR.sub(r"\1", raw)
        # 去掉看似目录项里末尾孤立数字（比如 “一、学校概述 ........ 1”）
        raw = re.sub(r"[\s\.·•∙⋯]{3,}\s*\d+$", "", raw).strip()
        if not raw:
            continue
        cleaned.append(raw)

    # （可选）进一步去页眉页脚：找频繁出现的首/尾行，这里简化忽略
    return cleaned

# ---- PDF(扫描) -> HTML：用 OCR (pytesseract + pdf2image) 产出 hOCR，再包成 HTML ----
def _convert_pdf_scanned_to_html(file_bytes: bytes) -> str:
    # 将前若干页转图（可全部页；考虑性能这里给个上限）
    images = convert_from_bytes(file_bytes, dpi=300)  # 需要 poppler
    html_parts = []
    for idx, img in enumerate(images):
        # 生成 hOCR（包含坐标，可后续做定位/高亮）
        hocr = pytesseract.image_to_pdf_or_hocr(img, extension='hocr')
        page_html = hocr.decode("utf-8", errors="ignore")
        # 包一层容器，并保留原图（可选）
        html_parts.append(f'<section class="page" data-page="{idx+1}">{page_html}</section>')
    return _wrap_html("\n".join(html_parts))

# ---- RTF -> HTML（用 LibreOffice 最稳；如无 soffice，可换 pyth -> html 简版） ----
def _convert_rtf_to_html(file_bytes: bytes) -> str:
    # 直接复用 LibreOffice 路径
    return _convert_doc_to_html_with_libreoffice(file_bytes, original_name="input.rtf")

# ---- 纯文本/Markdown -> HTML ----
def _convert_textlike_to_html(file_bytes: bytes, kind: str) -> str:
    text = file_bytes.decode("utf-8", errors="ignore")
    if kind == "md":
        try:
            import markdown  # pip install markdown
            html = markdown.markdown(text, extensions=["tables", "fenced_code"])
        except Exception:
            # 回退简单换行
            html = "<pre>" + _escape_html(text) + "</pre>"
    else:
        # txt
        html = "<pre>" + _escape_html(text) + "</pre>"
    return _wrap_html(html)

# --------- HTML 基础清洗 ----------
def _basic_postprocess_html(html_str: str, original_name: str = "") -> str:
    """
    第一阶段：轻清洗 + 基础表格兜底（不引入样式）。
    第二阶段（可选）：调用 _simplify_to_structure_only 去掉所有样式/非结构信息。
    """
    soup = BeautifulSoup(html_str, "lxml")

    # 去空标签
    for tag in soup.find_all():
        if tag.name in ["p","span","div"] and not tag.get_text(strip=True):
            tag.decompose()

    # 表格兜底
    for table in soup.find_all("table"):
        _ensure_table_sections(table)

    # —— 阶段2：极简化 —— #
    title_text = (original_name or "Document")
    minimal = _simplify_to_structure_only(str(soup), doc_title=title_text)
    return minimal

def _ensure_table_sections(table_tag):
    # 如果没有 thead/tbody，则尝试创建
    if not table_tag.find("thead") and table_tag.find("tr"):
        first_tr = table_tag.find("tr")
        # 简单启发：第一行当表头
        thead = table_tag.new_tag("thead")
        thead.append(first_tr.extract())
        table_tag.insert(0, thead)
    if not table_tag.find("tbody"):
        tbody = table_tag.new_tag("tbody")
        # 把剩余 tr 放到 tbody
        for tr in table_tag.find_all("tr"):
            if tr.parent.name != "thead":
                tbody.append(tr.extract())
        table_tag.append(tbody)

def _simplify_to_structure_only(html_str: str, doc_title: str = "Document") -> str:
    """
    将输入 HTML 简化为“无样式、仅结构+文本”的极简 HTML：
      - 删除 <style>, <script>, 注释
      - 剥离 <span>, <font>, <div> 等无语义/纯样式标签（保留其文本与子节点）
      - 删除所有非白名单标签与属性
      - 合理合并空白
      - 保留表格结构，并兜底 thead/tbody
    """
    soup = BeautifulSoup(html_str, "lxml")  # lxml 解析更稳

    # 1) 删除 style/script/注释
    for t in soup(["style", "script"]):
        t.decompose()
    for c in soup.find_all(string=lambda x: isinstance(x, Comment)):
        c.extract()

    # 2) 剥离纯表现标签：span/font/center 等（保留内容）
    for tag_name in ["span","font","center","u","b","i"]:
        for t in soup.find_all(tag_name):
            t.unwrap()

    # 3) 把“纯布局 div”尽量变为 p；保留 table 里的 div（不动）
    for t in soup.find_all("div"):
        if t.find_parent("table"):
            # 表格内部的 div 不做强制替换，以免破坏单元格内容
            t.name = "p" if t.name == "div" and not t.contents else t.name
        else:
            # 非表格内 div -> p（更语义化）
            t.name = "p"

    # 4) 非白名单标签：用 unwrap 只留下文本/子节点
    for t in soup.find_all():
        if t.name not in _ALLOWED_TAGS:
            t.unwrap()

    # 5) 删除所有不在白名单的属性
    for t in soup.find_all():
        if t.attrs:
            # 保留极少数安全属性
            t.attrs = {k: v for k, v in t.attrs.items() if k in _ALLOWED_ATTRS}

    # 6) 规范表格区块（确保 thead/tbody）
    for table in soup.find_all("table"):
        _ensure_table_sections(table)

    # 7) 删除空段/空行（保留换行 br）
    for t in list(soup.find_all(["p","li"])):
        if not t.get_text(strip=True):
            t.decompose()

    # 8) 文本空白归一化：把多空白压缩为一个空格；保留换行语义（p/li/tr）
    def _compress_text_nodes(node):
        for child in list(node.children):
            if isinstance(child, NavigableString):
                s = " ".join(str(child).split())
                child.replace_with(s)
            elif getattr(child, "children", None):
                _compress_text_nodes(child)
    body = soup.body or soup
    _compress_text_nodes(body)

    # 9) 重建最小外壳（无任何样式）
    title_text = (doc_title or "Document").rsplit(".", 1)[0]
    minimal_html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>{_escape_html(title_text)}</title>
</head>
<body>
{str(body)}
</body>
</html>"""
    return minimal_html

def _wrap_html(inner_html: str) -> str:
    """
    对已是完整 HTML 的场景不重复套壳，仅做轻度规范。
    """
    if "<html" in inner_html.lower() and "<body" in inner_html.lower():
        return inner_html
    return f"<div>{inner_html}</div>"

def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )

# --------- LLM 表格增强 ----------
def enhance_html_tables_with_llm(html_str: str, filename: str = "") -> str:
    """
    扫描 HTML，当存在“疑似复杂/跨页/嵌套异常”的表格时，调用 LLM 做结构修复与增强：
      - 合并跨页拆断的表格
      - 重建 thead（多级表头支持）
      - 识别合并单元格，填充合适的 row/colspan
      - 保留原有文本与顺序，不随意新增/删除内容
    若未检测到需要增强的情况，直接返回原文。
    """
    # 粗略启发式：单页过长表格、包含大量空单元格/孤立 span 的表格，尝试增强
    soup = BeautifulSoup(html_str, "html.parser")
    candidate_tables = []
    for table in soup.find_all("table"):
        text_len = len(table.get_text(strip=True))
        td_count = len(table.find_all(["td", "th"]))
        suspicious_spans = len(table.find_all("span"))
        if text_len > 3000 or td_count > 150 or suspicious_spans > 80:
            candidate_tables.append(table)

    if not candidate_tables:
        return html_str  # 不处理

    client = OpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

    # 将候选表格单独发给模型进行修复，避免输入超长
    for table in candidate_tables:
        table_html = str(table)
        prompt = _build_table_repair_prompt(filename=filename, table_html=table_html)

        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": "你是一个严谨的文档表格修复与标准化助手。只输出修改后的 <table>...</table> 片段，不要解释。"},
                    {"role": "user", "content": prompt},
                ],
            )
            fixed = resp.choices[0].message.content.strip()
            if "<table" in fixed.lower():
                # 用修复片段替换原表格
                new_table = BeautifulSoup(fixed, "html.parser")
                table.replace_with(new_table)
        except Exception:
            # LLM 失败时跳过，不影响主流程
            continue

    # 返回整体增强后的 HTML
    return str(soup) if "<html" not in html_str.lower() else soup.prettify()

def _build_table_repair_prompt(filename: str, table_html: str) -> str:
    return f"""
文件名：{filename}

下面是一段从文档中抽取的 HTML 表格片段，存在跨页/表头缺失/单元格合并信息丢失等问题。请进行**仅限结构层面的修复**并输出“替换用”的 <table>...</table> 片段（不要包含其它任何说明或标签）：

修复要求：
1) 推断并补齐 thead（支持多级表头），tbody 放数据行。
2) 尽量还原合并单元格，合理设置 rowSpan/colSpan。
3) 不要删除原始文本，除非是重复/明显噪声；不要新增凭空内容。
4) 保持单元格文本原样（允许轻度去噪/去重空白），不要意译。
5) 输出必须是可直接替换的 <table> 片段，且语法有效。

待修复表格：
{table_html}
""".strip()

# --------- 工具函数 ----------
def compute_sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
