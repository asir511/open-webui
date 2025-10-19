# app/graphs/bi/graph.py
from __future__ import annotations

from langgraph.graph import StateGraph, START, END
from typing import Any, Dict
from open_webui.custom.graphs.bi.state import BiGraphState
from open_webui.custom.graphs.bi.nodes.init_state import init_state
from open_webui.custom.graphs.bi.nodes.intent_recognition import intent_recognition
from open_webui.custom.graphs.bi.nodes.query_data import query_data
from open_webui.custom.graphs.bi.nodes.generate_chart import generate_chart
from open_webui.custom.graphs.bi.nodes.normal_qa import normal_qa

def build_graph() -> StateGraph[BiGraphState]:
    g = (
        StateGraph(BiGraphState)
        .add_node("initState", init_state)
        .add_node("intentRecognition", intent_recognition)
        .add_node("queryData", query_data)
        .add_node("generateChart", generate_chart)
        .add_node("normalQA", normal_qa)
        .add_edge(START, "initState")
        .add_edge("initState", "intentRecognition")
        .add_conditional_edges(
            "intentRecognition",
            lambda s: s.get("intent", "default"),
            {
                "query_data": "queryData",
                "generate_chart": "generateChart",
                "normal_QA": "normalQA",
                "default": "normalQA",
            },
        )
        .add_conditional_edges(
            "generateChart",
            lambda s: "queryData" if s.get("needs_data_query") else END,
        )
        .add_conditional_edges(
            "queryData",
            lambda s: s.get("next_after_query", "end"),
            {
                "generateChart": "generateChart",
                "end": END,
            },
        )
        .add_edge("normalQA", END)
    )
    return g
