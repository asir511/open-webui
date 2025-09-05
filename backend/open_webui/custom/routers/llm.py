# -*- coding: utf-8 -*-
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from utils.llm_utils import extract_rows_from_text, flatten_to_cells

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


# ---- 出参（直接按 AppExCell 结构需要的字段给到 Java） ----
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

class ExtractResp(BaseModel):
    taskId: str
    cells: List[CellOut]
    rows_count: int

@router.post("/extract", response_model=ExtractResp)
def extract(req: ExtractReq):
    """
    单子任务抽取：
    - 按 chunkTokens 对 contentMd 切片
    - 对每个切片进行结构化抽取
    - 汇总为 rows，再扁平为 cells（按字段类型落位）
    """
    if not req.fields:
        raise HTTPException(status_code=400, detail="字段定义不能为空")
    if not req.document or not req.document.contentMd:
        raise HTTPException(status_code=400, detail="文档内容 contentMd 不能为空")

    try:
        # 1) 调 LLM，返回 rows: List[Dict[fieldKey, Any]]
        rows = extract_rows_from_text(
            model_name=req.modelName,
            temperature=req.temperature or 0.2,
            content_md=req.document.contentMd,
            chunk_tokens=req.chunkTokens or 2000,
            table_key=req.tableKey,
            table_display_name=req.tableDisplayName or req.tableKey,
            fields=[f.model_dump() for f in req.fields],
            lang=req.document.lang or "zh",
            table_desc=req.tableDescription or "",
            filename=(req.document.originalName or req.document.title)  # ✅ 带上来源文件名
        )

        # 2) 扁平化为 cells（按照 AppExCell 的 value_* 列位）
        field_map = {f.fieldKey: f for f in req.fields}
        cells = flatten_to_cells(
            rows=rows,
            field_map=field_map,
            task_id=req.taskId
        )

        return ExtractResp(taskId=req.taskId, cells=[CellOut(**c) for c in cells], rows_count=len(set([c["rowId"] for c in cells])))

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"抽取失败：{e}")
