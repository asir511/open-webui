# app/graphs/bi/__init__.py
from __future__ import annotations
from typing import Any, Callable
from langgraph.checkpoint.memory import InMemorySaver
from open_webui.custom.graphs.bi.graph import build_graph
from langchain_core.runnables import RunnableConfig

_GLOBAL_CHECKPOINTER = InMemorySaver()

_GRAPH = build_graph().compile(checkpointer=_GLOBAL_CHECKPOINTER)


def compile_graph():
    """返回已编译的全局图"""
    return _GRAPH


async def run_workflow(
    input_text: str,
    thread_id: str,
    on_chunk: Callable[[Any], None] | None = None,
) -> None:
    graph = compile_graph()
    config = RunnableConfig(configurable={"thread_id": thread_id})

    async for chunk in graph.astream(
        {"input": input_text},
        config=config,
        stream_mode=["values", "custom"],
    ):
        if on_chunk:
            on_chunk(chunk)
