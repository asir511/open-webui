# app/graphs/common/model_config.py
from __future__ import annotations

from typing import List

from openai import base_url

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    chat_model = ChatOpenAI(model="chat1", temperature=0, base_url = "http://192.168.0.114:3000/v1", api_key="sk-2pH4HUSK4wikhU7NSqMqF3Ldi7c2r89sJmRQBBJ9PS7vN1AM")
    from langchain_community.embeddings import OllamaEmbeddings
    _embedder = OllamaEmbeddings(model="bge-m3:latest",base_url="http://192.168.0.114:11434")
    async_mode = False
except Exception:
    chat_model = None
    _embedder = None
    async_mode = False

def embed_query(text: str) -> List[float]:
    """
    等价 TS 的 ollamaEmbedding.embedQuery。你若用 Ollama，可换成
    langchain_community.embeddings.OllamaEmbeddings。
    """
    if _embedder is None:
        return [0.0] * 1024
    return _embedder.embed_query(text)
