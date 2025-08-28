# word_utils.py
from __future__ import annotations
import os
import re
import html
import json
import shutil
import hashlib
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

class WordConvertError(RuntimeError):
    pass

# -----------------------
# 工具：依赖检查
# -----------------------
def has_pandoc() -> bool:
    return shutil.which("pandoc") is not None

def has_soffice(soffice: str = "soffice") -> bool:
    return shutil.which(soffice) is not None

# -----------------------
# 清洗 & 规范化
# -----------------------
def sanitize_markdown(md: str, *, image_base_url: str | None = None) -> str:
    """
    - 统一换行
    - 去 BOM/NUL
    - HTML 实体反转义（避免 &lt;table&gt; 之类）
    - 可选：为 Markdown/HTML 中的相对 images/ 路径补前缀
    """
    if not md:
        return ""
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = md.lstrip("\ufeff").replace("\x00", "")
    md = html.unescape(md)

    if image_base_url:
        base = image_base_url.rstrip("/")
        # ![alt](images/xxx)
        md = re.sub(
            r'(!\[[^\]]*\]\()(?!(?:https?:)?//)(images/[^)\s]+)(\))',
            lambda m: f"{m.group(1)}{base}/{m.group(2)}{m.group(3)}",
            md,
        )
        # <img src="images/xxx">
        md = re.sub(
            r'(<img[^>]+src=["\'])(?!(?:https?:)?//)(images/[^"\']+)(["\'])',
            lambda m: f'{m.group(1)}{base}/{m.group(2)}{m.group(3)}',
            md,
        )
    return md

# -----------------------
# 转换实现
# -----------------------
def _doc_to_docx_with_soffice(src: Path, out_dir: Path, soffice: str = "soffice") -> Path:
    """
    用 LibreOffice 把 .doc 转 .docx；需要已安装 soffice。
    """
    if not has_soffice(soffice):
        raise WordConvertError(
            "未检测到 LibreOffice(soffice)，无法将 .doc 转为 .docx。"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        soffice, "--headless",
        "--convert-to", "docx:MS Word 2007 XML",
        "--outdir", str(out_dir),
        str(src),
    ]
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if cp.returncode != 0:
        raise WordConvertError(f"soffice 转换 .doc→.docx 失败（{cp.returncode}）：{cp.stdout}")
    docx = out_dir / (src.stem + ".docx")
    if not docx.exists():
        raise WordConvertError(f"未找到转换后的 DOCX：{docx}")
    return docx

def _docx_to_markdown_with_pandoc(
    src: Path,
    media_out_dir: Path,
    *,
    wrap: str = "none",
) -> str:
    """
    Pandoc: .docx → GitHub Flavored Markdown
    - 抽出图片到 media_out_dir（相对路径 images/...）
    - 保留 LaTeX 数学（$...$ / $$...$$）
    - 允许 HTML 表格以保持复杂表格结构
    """
    if not has_pandoc():
        raise WordConvertError("未检测到 pandoc，请安装后再试。")

    media_out_dir.mkdir(parents=True, exist_ok=True)

    # 让 pandoc 把媒体导出到 media_out_dir/images 下，
    # 并让 md 中引用使用相对路径 images/xxx
    # 实现方式：切换工作目录到 media_out_dir，令 --extract-media=.
    # 然后把生成的 ./<doc-name>/ 里的文件移动到 ./images/
    # 最终 md 里是 images/xxx 这种相对引用。
    with tempfile.TemporaryDirectory(prefix="pandoc_md_") as tdir:
        tmp_md = Path(tdir) / "out.md"
        extract_root = media_out_dir  # 就用目标目录作为根

        args = [
            "pandoc",
            "-f", "docx",
            "-t", "gfm+tex_math_dollars+pipe_tables+raw_html",
            "--wrap", wrap,
            "--extract-media", str(extract_root),
            "-o", str(tmp_md),
            str(src),
        ]

        cp = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if cp.returncode != 0:
            raise WordConvertError(f"Pandoc 转 Markdown 失败：{cp.stdout}")

        md_text = tmp_md.read_text(encoding="utf-8")

    # 将 pandoc 导出的媒体规范化到 images/ 前缀（默认情况下，pandoc 会在 extract_root 下建一个以文档名命名的子目录）
    # 我们把所有文件搬到 media_out_dir/images 下，然后把 md 中的路径替换为 images/...
    images_dir = media_out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # 找出现有的子目录
    for p in media_out_dir.iterdir():
        if p.is_dir() and p.name != "images":
            # 把里面的文件平铺搬到 images/
            for f in p.rglob("*"):
                if f.is_file():
                    target = images_dir / f.name
                    # 同名文件简单重命名避免覆盖
                    if target.exists():
                        target = images_dir / f"{f.stem}_{hashlib.md5(str(f).encode()).hexdigest()[:8]}{f.suffix}"
                    shutil.move(str(f), str(target))
            shutil.rmtree(p, ignore_errors=True)

    # 将 md 中的相对路径统一替换为 images/xxx
    md_text = re.sub(
        r'(!\[[^\]]*\]\()(?:(?:\./)?[^)/\\]+/)+([^)\s]+)(\))',
        r'\1images/\2\3',
        md_text,
    )
    md_text = re.sub(
        r'(<img[^>]+src=["\'])(?:(?:\./)?[^/"\']+/)+([^"\']+)(["\'])',
        r'\1images/\2\3',
        md_text,
    )

    return md_text

# -----------------------
# 对外主函数
# -----------------------
def word_to_markdown(
    src: Path,
    *,
    soffice: str = "soffice",
    image_base_url: str | None = None,
) -> Dict[str, Any]:
    """
    把 .doc / .docx 转为 Markdown（含图片抽取）。
    返回：
    {
      "source_type": "doc"|"docx",
      "markdown": "<md string>",
      "media_dir": "<目录路径字符串>",  # 抽出的图片所在目录（包含 images/）
      "images": ["images/xxx1.png", ...], # md 内相对路径
    }
    """
    if not src.exists():
        raise WordConvertError(f"文件不存在：{src}")
    ext = src.suffix.lower()
    if ext not in {".doc", ".docx"}:
        raise WordConvertError(f"不支持的扩展名：{ext}")

    with tempfile.TemporaryDirectory(prefix="word_media_") as tmpdir:
        media_root = Path(tmpdir)

        # .doc 先转 .docx
        work_src = src
        src_type = "docx" if ext == ".docx" else "doc"
        if ext == ".doc":
            work_src = _doc_to_docx_with_soffice(src, media_root, soffice=soffice)
            src_type = "doc"

        # .docx → Markdown
        md = _docx_to_markdown_with_pandoc(work_src, media_root)

        # 清洗，且可补全图片 URL 前缀
        md = sanitize_markdown(md, image_base_url=image_base_url)

        # 收集图片相对路径（images/ 开头）
        images_rel: List[str] = []
        for p in (media_root / "images").glob("*"):
            if p.is_file():
                images_rel.append(f"images/{p.name}")

        # 把媒体目录复制到一个可持久位置（可按你项目需求调整，这里默认把目录搬到 /tmp 的新位置，并返回路径）
        persist_dir = Path(tempfile.mkdtemp(prefix="word_media_persist_"))
        # 复制 media_root 全部内容
        for child in media_root.iterdir():
            target = persist_dir / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)

        return {
            "source_type": src_type,
            "markdown": md,
            "media_dir": str(persist_dir),  # 里面含 images/
            "images": images_rel,
        }
