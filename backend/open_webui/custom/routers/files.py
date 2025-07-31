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
