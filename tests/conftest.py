from __future__ import annotations

from pathlib import Path

import pytest

from cleanbot.core.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    runtime = tmp_path / "runtime"
    upload = runtime / "uploads"
    vector = runtime / "chroma"
    runtime.mkdir()
    upload.mkdir()
    vector.mkdir()
    return Settings(
        project_root=tmp_path,
        data_dir=Path(__file__).resolve().parents[1] / "data",
        runtime_dir=runtime,
        upload_dir=upload,
        vector_dir=vector,
        database_url=f"sqlite:///{runtime / 'test.db'}",
        collection_name="test_collection",
        chat_model_name="fake-chat",
        embedding_model_name="fake-embedding",
        rerank_model_name="fake-rerank",
        dashscope_api_key=None,
        dashscope_base_url="https://example.invalid/v1",
        enable_rerank=False,
        admin_token="test-admin-token",
        app_env="test",
        log_level="WARNING",
        dense_top_k=10,
        sparse_top_k=10,
        rerank_top_n=5,
        answer_top_n=4,
        min_retrieval_score=0.05,
        max_history_messages=12,
        model_timeout_seconds=5,
        weather_timeout_seconds=1,
        max_upload_bytes=1024 * 1024,
    )
