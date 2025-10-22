from __future__ import annotations
import os
from typing import List, Dict, Any, Optional
from pymilvus import connections, db, utility, Collection

MILVUS_ADDR = os.getenv("MILVUS_ADDR", "192.168.0.114:19530")  # 不要 http://
MILVUS_USER = os.getenv("MILVUS_USER", "root")
MILVUS_PASS = os.getenv("MILVUS_PASS", "Milvus")
MILVUS_DB   = os.getenv("MILVUS_DB", "sql_prompt")

def connect_milvus():
    connections.connect(
        alias="default",
        address=MILVUS_ADDR,
        user=MILVUS_USER,
        password=MILVUS_PASS,
        secure=False,
    )
    db.using_database(MILVUS_DB)
    # 连通性 & 版本
    print("Milvus version:", utility.get_server_version())

def search_sql_examples(
    query_vector: List[float],
    collection_name: str = "llm_jeecgboot_springboot3",
    vector_field: str = "vector",
    top_k: int = 2,
    output_fields: Optional[List[str]] = None,
    expr: Optional[str] = 'training_data_type == "sql"',
    metric_type: str = "IP",
    params: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    connect_milvus()

    if not utility.has_collection(collection_name):
        raise RuntimeError(f"集合不存在: {collection_name} (db={MILVUS_DB})")

    coll = Collection(collection_name)

    # 分区检查
    load_kwargs = {}

    coll.load(**load_kwargs)

    search_params = {
        "metric_type": "COSINE",
        "params": {"ef": 64}
    }

    res = coll.search(
        data=[query_vector],
        anns_field=vector_field,
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=output_fields or ["question", "content"],
    )

    hits = res[0]
    fields = output_fields or ["question", "content"]
    out: List[Dict[str, Any]] = []
    for h in hits:
        item = {f: h.entity.get(f) for f in fields}
        out.append(item)
    return out
