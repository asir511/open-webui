# app/services/milvus_service.py
from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from pymilvus import connections, Collection

MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT", "http://192.168.0.114:19530")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")

def _connect_once():
    # idempotent：多次调用也只保留一个连接
    if "default" not in connections.list_connections():
        connections.connect(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)

def search_sql_examples(
    query_vector: List[float],
    collection_name: str = "llm_jeecgboot_springboot3",
    partition_name: Optional[str] = "sql_prompt",
    vector_field: str = "vector",
    top_k: int = 2,
    output_fields: Optional[List[str]] = None,
    expr: Optional[str] = 'training_data_type == "sql"',
    metric_type: str = "IP",
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    返回形如 [{"question": "...", "content": "..."}, ...]
    """
    _connect_once()
    coll = Collection(collection_name)
    if partition_name:
        coll.load(partition_names=[partition_name])
    else:
        coll.load()

    search_params = {"metric_type": metric_type}
    if params:
        search_params.update(params)

    res = coll.search(
        data=[query_vector],
        anns_field=vector_field,
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=output_fields or ["question", "content"],
        partition_names=[partition_name] if partition_name else None,
    )

    # res 是一个 list[Hits]；我们只取第 1 条查询的 hits
    hits = res[0]
    out: List[Dict[str, Any]] = []
    for h in hits:
        # h.entity.get("field")
        item = {}
        for f in (output_fields or ["question", "content"]):
            item[f] = h.entity.get(f)
        out.append(item)
    return out
