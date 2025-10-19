# app/graphs/common/state_types.py
from __future__ import annotations
from typing import TypedDict, Literal, Optional

class ChatItem(TypedDict, total=False):
    role: Literal["user", "assistant", "system"]
    content: str
    nodeName: Optional[str]
