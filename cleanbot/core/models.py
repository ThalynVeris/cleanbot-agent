from __future__ import annotations

import time
from functools import lru_cache
from typing import Any

import dashscope
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from cleanbot.core.config import get_settings


class DashScopeEmbeddingModel(Embeddings):
    """Small LangChain adapter around the maintained DashScope SDK."""

    def __init__(self, model: str, api_key: str, max_retries: int = 2) -> None:
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.batch_size = 10 if model in {"text-embedding-v3", "text-embedding-v4"} else 25

    @staticmethod
    def _items(response: Any) -> list[dict[str, Any]]:
        output = getattr(response, "output", None)
        if output is None and isinstance(response, dict):
            output = response.get("output")
        if isinstance(output, dict):
            items = output.get("embeddings", [])
        else:
            items = getattr(output, "embeddings", [])
        if not isinstance(items, list) or not items:
            raise RuntimeError("Embedding provider returned no vectors")
        return items

    def _call(self, input_data: str | list[str], text_type: str) -> list[list[float]]:
        for attempt in range(self.max_retries + 1):
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=input_data,
                text_type=text_type,
                api_key=self.api_key,
            )
            status_code = int(getattr(response, "status_code", 200))
            if status_code == 200:
                return [list(map(float, item["embedding"])) for item in self._items(response)]
            message = getattr(response, "message", "unknown provider error")
            if status_code in {400, 401}:
                raise ValueError(f"Embedding provider returned {status_code}: {message}")
            if attempt == self.max_retries:
                raise RuntimeError(f"Embedding provider returned {status_code}: {message}")
            time.sleep(0.5 * (2**attempt))
        raise RuntimeError("Embedding provider retry loop ended unexpectedly")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            vectors.extend(self._call(texts[offset : offset + self.batch_size], "document"))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._call(text, "query")[0]


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for the chat model")
    return ChatOpenAI(
        model=settings.chat_model_name,
        api_key=settings.dashscope_api_key,
        base_url=settings.dashscope_base_url,
        temperature=0,
        streaming=True,
        stream_usage=True,
        timeout=settings.model_timeout_seconds,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    settings = get_settings()
    if not settings.dashscope_api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is required for the embedding model")
    return DashScopeEmbeddingModel(
        model=settings.embedding_model_name,
        api_key=settings.dashscope_api_key,
    )
