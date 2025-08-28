# mineru_client.py
from __future__ import annotations
import os
import time
import json
import shutil
import mimetypes
import tempfile
import subprocess
from pathlib import Path
from typing import Iterable, List, Union, Optional, Tuple, Dict, Any

import requests
from pypdf import PdfReader

MineruPath = Union[str, Path]


class MinerUError(RuntimeError):
    pass


def _bool_str(v: bool) -> str:
    # FastAPI 表单里布尔用小写字符串更稳
    return "true" if v else "false"


def _detect_mime(p: Path) -> str:
    m, _ = mimetypes.guess_type(p.name)
    return m or "application/octet-stream"


def _has_soffice(soffice: str = "soffice") -> bool:
    return shutil.which(soffice) is not None


def _office_to_pdf(src: Path, out_dir: Path, soffice: str = "soffice") -> Path:
    if not _has_soffice(soffice):
        raise MinerUError(
            "LibreOffice (soffice) 未安装，无法将 Office 文档转换为 PDF。"
            " 在 Debian/Ubuntu 上执行：sudo apt-get update && sudo apt-get install -y libreoffice"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    # 明确使用 writer_pdf_Export，并开启一些有利于保真和可访问性的参数
    # 参考：https://wiki.openoffice.org/wiki/API/Tutorials/PDF_export
    pdf_filter = (
        "pdf:writer_pdf_Export"
        # 下面这些参数不是所有版本都识别，但多数新版本可用；不识别会忽略
        ":SelectPdfVersion=1"            # PDF 1.4；如需更高兼容可用 1
        ":UseTaggedPDF=true"             # Tagged PDF
        ":ExportBookmarks=true"
        ":ExportNotes=false"
        ":EmbedStandardFonts=true"       # 尽量嵌入基础字体
        ":ExportFormFields=false"
    )

    cmd = [
        soffice, "--headless",
        "--convert-to", pdf_filter,
        "--outdir", str(out_dir),
        str(src),
    ]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if cp.returncode != 0:
        raise MinerUError(f"LibreOffice 转换失败（{cp.returncode}）：{cp.stdout}")

    pdf = out_dir / (src.stem + ".pdf")
    if not pdf.exists():
        raise MinerUError(f"未找到转换后的 PDF：{pdf}")
    return pdf



def _pdf_num_pages(pdf_path: Path) -> int:
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        return len(reader.pages)


def _extract_markdown(resp: requests.Response) -> str:
    """
    尽力从响应里取 Markdown。
    - 优先 application/json，尝试常见 key：md / markdown / markdown_list / content / data 等
    - 否则直接返回文本
    """
    ctype = resp.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            obj = resp.json()
        except json.JSONDecodeError:
            return resp.text
        # 常见几种返回形态
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            for k in ("md", "markdown", "markdown_text", "result", "data", "content"):
                v = obj.get(k)
                if isinstance(v, str):
                    return v
                if isinstance(v, list):
                    # 若是段落/页的列表
                    return "\n\n".join([x for x in v if isinstance(x, str)])
            # 尝试更深一层
            for k in obj:
                v = obj[k]
                if isinstance(v, dict):
                    for kk in ("md", "markdown", "markdown_list"):
                        vv = v.get(kk)
                        if isinstance(vv, str):
                            return vv
                        if isinstance(vv, list):
                            return "\n\n".join([x for x in vv if isinstance(x, str)])
        if isinstance(obj, list):
            return "\n\n".join([x for x in obj if isinstance(x, str)])
        # 实在不行就回落到文本
        return resp.text
    # 非 JSON 直接当作文本
    return resp.text


def _post_one_range(
    endpoint: str,
    pdf_path: Path,
    *,
    backend: str,
    lang: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    start_page_id: int,
    end_page_id: int,
    output_dir: str,
    return_md: bool,
    return_middle_json: bool,
    return_model_output: bool,
    return_content_list: bool,
    return_images: bool,
    server_url: Optional[str],
    timeout: int,
) -> str:
    # 只传单文件（更符合分页语义）
    with open(pdf_path, "rb") as f:
        files_field = [("files", (pdf_path.name, f, _detect_mime(pdf_path)))]
        data: List[Tuple[str, str]] = [
            ("output_dir", output_dir),
            ("backend", backend),
            ("lang", lang),
            ("parse_method", parse_method),
            ("formula_enable", _bool_str(formula_enable)),
            ("table_enable", _bool_str(table_enable)),
            ("return_md", _bool_str(return_md)),
            ("return_middle_json", _bool_str(return_middle_json)),
            ("return_model_output", _bool_str(return_model_output)),
            ("return_content_list", _bool_str(return_content_list)),
            ("return_images", _bool_str(return_images)),
            ("start_page_id", str(start_page_id)),
            ("end_page_id", str(end_page_id)),
        ]
        if server_url:
            data.append(("server_url", server_url))

        resp = requests.post(endpoint, data=data, files=files_field, timeout=timeout)
        resp.raise_for_status()
        return _extract_markdown(resp)


def file_parse_smart(
    inputs: Iterable[MineruPath],
    *,
    base_url: str = "http://192.168.0.114:8181",
    backend: str = "pipeline",           # "pipeline" 或 "vlm"
    lang: Optional[str] = "ch",
    parse_method: str = "auto",
    formula_enable: bool = True,
    table_enable: bool = True,
    output_dir: str = "./output",
    max_pages_per_request: int = 25,     # 你要求的默认 25
    timeout: int = 900,
    retries: int = 2,
    retry_backoff: float = 1.6,
    convert_office: bool = True,         # 你要求：默认支持 doc/docx -> 自动转 PDF
    soffice_path: str = "soffice",
    server_url: Optional[str] = None,    # backend=vlm 时传形如 http://127.0.0.1:30000
) -> Dict[str, Dict[str, Any]]:
    """
    对每个输入（pdf/doc/docx）自动分页多次请求，最后拼接 markdown。
    返回：{ 原文件名: { "markdown": str, "chunks": [(start,end), ...], "total_pages": int, "converted_pdf": str|None } }
    """
    endpoint = f"{base_url.rstrip('/')}/file_parse"

    results: Dict[str, Dict[str, Any]] = {}
    temp_dir_obj: Optional[tempfile.TemporaryDirectory] = None
    try:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="mineru_chunks_")
        temp_dir = Path(temp_dir_obj.name)

        for item in inputs:
            src = Path(item)
            if not src.exists():
                raise FileNotFoundError(f"Input not found: {src}")

            # 若是 doc/docx 自动转 PDF
            pdf_path: Path
            if convert_office and src.suffix.lower() in {".doc", ".docx"}:
                pdf_path = _office_to_pdf(src, temp_dir, soffice=soffice_path)
                converted_pdf_str = str(pdf_path)
            else:
                if src.suffix.lower() not in {".pdf"}:
                    raise MinerUError(
                        f"不支持的文件类型：{src.suffix}. "
                        f"若要处理 Office，请设置 convert_office=True（默认已开启）并安装 LibreOffice。"
                    )
                pdf_path = src
                converted_pdf_str = None

            # 页面统计与分块（0-based，end 包含）
            try:
                total_pages = _pdf_num_pages(pdf_path)
            except Exception as e:
                # 页数统计失败也尝试给一大块（最多 25 页）
                total_pages = max_pages_per_request
            chunks: List[Tuple[int, int]] = []
            start = 0
            while start < total_pages:
                end = min(start + max_pages_per_request - 1, total_pages - 1)
                chunks.append((start, end))
                start = end + 1

            combined_md_parts: List[str] = []
            for (s, e) in chunks:
                last_exc = None
                for attempt in range(retries + 1):
                    try:
                        md = _post_one_range(
                            endpoint, pdf_path,
                            backend=backend,
                            lang=lang,
                            parse_method=parse_method,
                            formula_enable=formula_enable,
                            table_enable=table_enable,
                            start_page_id=s,
                            end_page_id=e,
                            output_dir=output_dir,
                            return_md=True,
                            return_middle_json=False,
                            return_model_output=False,
                            return_content_list=False,
                            return_images=False,
                            server_url=server_url,
                            timeout=timeout,
                        )
                        combined_md_parts.append(md.strip())
                        break
                    except (requests.RequestException, MinerUError) as ex:
                        last_exc = ex
                        if attempt < retries:
                            time.sleep(retry_backoff ** attempt)
                        else:
                            raise
                if last_exc:
                    raise last_exc  # 理论到不了

            results[src.name] = {
                "markdown": "\n\n".join([p for p in combined_md_parts if p]),
                "chunks": chunks,
                "total_pages": total_pages,
                "converted_pdf": converted_pdf_str,
            }
        return results
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()

import json
import re

def unwrap_md_from_mineru(markdown_field: str) -> str:
    """
    只提取 JSON 文本中的 md_content 原样内容：
    - 不 json.loads，不反转义，不 strip，不添加分隔符
    - 若存在多个 md_content，按出现顺序直接拼接返回
    - 若未匹配到，原样返回传入的字符串
    """
    s = markdown_field or ""
    if not s:
        return ""

    # 捕获 "md_content":"<这里是原样内容，可能包含转义字符>"
    pattern = r'"md_content"\s*:\s*"((?:\\.|[^"\\])*)"'
    matches = re.findall(pattern, s)
    if matches:
        return "".join(matches)

    return s


# --- 1) 依赖 ---
import json
import re
import html
import hashlib
from pathlib import Path

# --- 2) 更健壮的抽取 + 反转义 ---
def _json_string_unescape(s: str) -> str:
    """
    使用 JSON 语义做最安全的反转义：
    给 s 外面再套一层引号，交给 json.loads 还原 \n, \t, \", \\ 等。
    避免使用 unicode_escape 误伤 LaTeX 反斜杠。
    """
    try:
        return json.loads(f'"{s}"')
    except Exception:
        # 如果 s 本来就不需要反转义，原样返回
        return s

def unwrap_md_from_mineru(markdown_field: str) -> str:
    """
    尽可能从 MinerU 的响应里拿到“真实 Markdown 文本”（已反转义）：
    - 情况 A：顶层就是 JSON（dict/list），从常见 key(md/markdown/md_content/...)里提取并 join；
    - 情况 B：顶层是字符串，但里面包含多段 `"md_content":"..."`
              用正则抓取每段并用 json.loads 反转义后拼接；
    - 情况 C：既不是 JSON 也没匹配到 md_content：
              如果看起来包含转义序列（比如 '\n' 的字面量），尝试整体反转义一次。
    """
    s = markdown_field or ""
    if not s:
        return ""

    # 情况 A：尝试解析顶层 JSON
    def _collect_from_obj(obj) -> list[str]:
        out = []

        def dfs(x):
            if isinstance(x, str):
                out.append(x)
                return
            if isinstance(x, dict):
                # 常见 key 优先
                for k in ("md", "markdown", "markdown_text", "md_content", "result", "content", "data"):
                    if k in x and isinstance(x[k], (str, list)):
                        dfs(x[k])
                # 兜底：遍历所有字段（避免漏掉嵌套）
                for v in x.values():
                    dfs(v)
            elif isinstance(x, list):
                for it in x:
                    dfs(it)

        dfs(obj)
        return out

    try:
        obj = json.loads(s)
        parts = _collect_from_obj(obj)
        # 逐段确保没有转义残留
        parts = [_json_string_unescape(p) for p in parts]
        # 去空并用空行分隔
        return "\n\n".join([p for p in parts if p.strip()])
    except Exception:
        pass  # 不是顶层 JSON，继续走下去

    # 情况 B：顶层非 JSON，尝试抓取多段 "md_content":"..."
    pattern = r'"md_content"\s*:\s*"((?:\\.|[^"\\])*)"'
    matches = re.findall(pattern, s)
    if matches:
        parts = [_json_string_unescape(m) for m in matches]
        return "\n\n".join([p for p in parts if p.strip()])

    # 情况 C：兜底
    # 如果包含明显的转义序列，则整体尝试反转义
    if r"\n" in s or r"\"" in s or r"\\u" in s:
        s2 = _json_string_unescape(s)
        return s2

    return s

# --- 3) Markdown 清洗（可选）---
def sanitize_markdown(md: str, *, image_base_url: str | None = None) -> str:
    """
    - 统一换行为 '\n'
    - 去掉 BOM / NUL
    - HTML 实体反转义（&amp; -> &，避免 <table> 被实体化）
    - 如需要，把 'images/xxx.jpg' 这类相对路径补成完整 URL（方便前端加载）
    """
    if not md:
        return ""

    # 统一换行
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    # 去 BOM / NUL
    md = md.lstrip("\ufeff").replace("\x00", "")

    # HTML 实体 -> 字符（避免 &lt;table&gt; 这种被当字面量）
    md = html.unescape(md)

    # 可选：图片相对路径补全
    if image_base_url:
        # 只替换典型的 Markdown 图片语法中的相对路径
        md = re.sub(
            r'(!\[[^\]]*\]\()(?!(?:https?:)?//)(images/[^)\s]+)(\))',
            lambda m: f"{m.group(1)}{image_base_url.rstrip('/')}/{m.group(2)}{m.group(3)}",
            md,
        )
        # 也处理 <img src="images/...">
        md = re.sub(
            r'(<img[^>]+src=["\'])(?!(?:https?:)?//)(images/[^"\']+)(["\'])',
            lambda m: f'{m.group(1)}{image_base_url.rstrip("/")}/{m.group(2)}{m.group(3)}',
            md,
        )

    return md
