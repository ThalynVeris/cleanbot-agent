from __future__ import annotations

from types import SimpleNamespace

import dashscope

from cleanbot.core.models import DashScopeEmbeddingModel


def test_dashscope_embedding_adapter_batches_documents_and_marks_query(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_call(**kwargs):
        values = kwargs["input"] if isinstance(kwargs["input"], list) else [kwargs["input"]]
        calls.append((kwargs["text_type"], len(values)))
        return SimpleNamespace(
            status_code=200,
            output={"embeddings": [{"embedding": [float(index), 1.0]} for index, _ in enumerate(values)]},
        )

    monkeypatch.setattr(dashscope.TextEmbedding, "call", fake_call)
    model = DashScopeEmbeddingModel("text-embedding-v4", "test-key")

    assert len(model.embed_documents([f"document-{index}" for index in range(11)])) == 11
    assert model.embed_query("query") == [0.0, 1.0]
    assert calls == [("document", 10), ("document", 1), ("query", 1)]

