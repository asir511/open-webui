# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from utils.llm_utils import extract_rows_from_text, flatten_to_cells, reconcile_rows_with_llm

router = APIRouter()

# ---- Pydantic 入参模型 ----
class AbsFieldIn(BaseModel):
    id: str
    tableKey: str
    fieldKey: str
    displayName: Optional[str] = None
    dataType: Literal[
        "string", "number", "integer", "boolean", "date", "datetime", "json"
    ] = Field(..., description="和你的 data_type 字典一致")
    unit: Optional[str] = None
    description: Optional[str] = None
    required: Optional[int] = 0
    enumValues: Optional[str] = None
    regexPattern: Optional[str] = None
    sortOrder: Optional[int] = None
    isPk: Optional[int] = 0

class DocumentIn(BaseModel):
    id: int
    contentMd: str
    lang: Optional[str] = None
    title: Optional[str] = None
    originalName: Optional[str] = None

class ExtractReq(BaseModel):
    taskId: str
    modelName: str
    temperature: Optional[float] = 0.1
    chunkTokens: Optional[int] = 5000
    tableKey: str
    tableDisplayName: Optional[str] = None
    tableDescription: Optional[str] = None
    fields: List[AbsFieldIn]
    document: DocumentIn

# ---- 出参（按 AppExCell 需要的字段给 Java） ----
class CellOut(BaseModel):
    rowId: str
    fieldId: str
    taskId: str
    valueString: Optional[str] = None
    valueNumber: Optional[float] = None
    valueInteger: Optional[int] = None
    valueBoolean: Optional[int] = None  # 1/0
    valueDate: Optional[str] = None     # "yyyy-MM-dd"
    valueDatetime: Optional[str] = None # "yyyy-MM-dd HH:mm:ss"
    valueJson: Optional[str] = None
    unit: Optional[str] = None
    notes: Optional[str] = None
    # ✅ 每个字段只保留一段原文摘录 + 简短理由
    evidenceText: Optional[str] = None
    reason: Optional[str] = None

class ExtractResp(BaseModel):
    taskId: str
    cells: List[CellOut]
    rows_count: int

@router.post("/extract", response_model=ExtractResp)
async def extract(req: ExtractReq):
    """
    单子任务抽取（异步并发版）：
    - 文本去噪（目录/图片视频外链）→ 切片
    - 第一轮对每个切片并发抽取
    - 第一轮结果按主键分组，只有组内>1条才并发进行二轮合并研判与格式验证
    - 扁平为 cells
    """
    if not req.fields:
        raise HTTPException(status_code=400, detail="字段定义不能为空")
    if not req.document or not req.document.contentMd:
        raise HTTPException(status_code=400, detail="文档内容 contentMd 不能为空")

    try:
        rows_stage1 = await extract_rows_from_text(
            model_name=req.modelName,
            temperature=req.temperature or 0.1,
            content_md=req.document.contentMd,
            chunk_tokens=req.chunkTokens or 4000,
            table_key=req.tableKey,
            table_display_name=req.tableDisplayName or req.tableKey,
            fields=[f.model_dump() for f in req.fields],
            lang=req.document.lang or "zh",
            table_desc=req.tableDescription or "",
            filename=(req.document.originalName or req.document.title)
        )

        # —— 二次核对与归并（按主键分组；组内>1条才调用 LLM；并发）——
        rows_final = await reconcile_rows_with_llm(
            model_name=req.modelName,
            temperature=(req.temperature or 0.1),
            table_key=req.tableKey,
            table_display_name=req.tableDisplayName or req.tableKey,
            table_desc=req.tableDescription or "",
            lang=req.document.lang or "zh",
            fields=[f.model_dump() for f in req.fields],
            extracted_rows=rows_stage1
        )

        field_map = {f.fieldKey: f for f in req.fields}
        cells = flatten_to_cells(
            rows=rows_final,
            field_map=field_map,
            task_id=req.taskId
        )

        return ExtractResp(
            taskId=req.taskId,
            cells=[CellOut(**c) for c in cells],
            rows_count=len({c["rowId"] for c in cells})
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"抽取失败：{e}")
