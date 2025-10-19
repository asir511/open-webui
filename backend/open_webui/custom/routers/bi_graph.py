import io
import logging
from open_webui.env import (
    SRC_LOG_LEVELS,
)
from fastapi import (
    HTTPException,
    APIRouter,
)
import os, sys
from open_webui.custom.graphs.bi import run_workflow

from pydantic import BaseModel

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()

class WorkflowRequest(BaseModel):
    input_text: str
    thread_id: str

@router.post("/run-bigraph-workflow")
async def run_bigraph_workflow(request: WorkflowRequest):
    """处理请求并启动工作流"""
    input_text = request.input_text
    thread_id = request.thread_id

    result_chunks = []

    # 定义一个临时函数来处理返回的 chunk
    def on_chunk(chunk):
        result_chunks.append(chunk)

    try:
        # 调用 `run_workflow`，并通过 `on_chunk` 收集结果
        await run_workflow(input_text, thread_id, on_chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {str(e)}")

    return {"result": result_chunks}
