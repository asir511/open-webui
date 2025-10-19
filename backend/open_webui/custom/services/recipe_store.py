# app/services/recipe_store.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from sqlalchemy import text
from open_webui.custom.services.db_service import get_internal_engine

def list_recipes(recipe_type: str = "aggrid_chart", lang: str = "zh") -> List[Dict[str, Any]]:
    """
    返回 [{recipe_key, display_name, llm_hint}]，供 LLM 做选择。
    """
    sql = text("""
      SELECT recipe_key, display_name, llm_hint
      FROM app_aggrid_recipe
      WHERE recipe_type=:rt AND enabled=1 AND lang=:lang
      ORDER BY update_time DESC
    """)
    with get_internal_engine().connect() as conn:
        return [dict(r._mapping) for r in conn.execute(sql, {"rt": recipe_type, "lang": lang})]

def load_recipe_detail(recipe_key: str) -> Optional[Dict[str, Any]]:
    """
    返回 {"prompt_text": str, "example_code": str}
    """
    sql = text("""
      SELECT prompt_text, example_code
      FROM app_aggrid_recipe
      WHERE recipe_key=:k AND enabled=1
      LIMIT 1
    """)
    with get_internal_engine().connect() as conn:
        row = conn.execute(sql, {"k": recipe_key}).mappings().first()
        return dict(row) if row else None
