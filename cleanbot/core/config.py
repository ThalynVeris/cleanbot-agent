from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

RerankPolicy = Literal["always", "disagreement"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_rerank_policy(value: str | None) -> RerankPolicy:
    policy = (value or "always").strip().lower()

    if policy == "always":
        return "always"

    if policy == "disagreement":
        return "disagreement"

    raise ValueError("RERANK_POLICY must be 'always' or 'disagreement'")


def _as_path(value: str | None, default: Path) -> Path:
    path = Path(value).expanduser() if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    project_root: Path
    data_dir: Path
    runtime_dir: Path
    upload_dir: Path
    vector_dir: Path
    database_url: str
    collection_name: str
    chat_model_name: str
    embedding_model_name: str
    rerank_model_name: str
    dashscope_api_key: str | None
    dashscope_base_url: str
    enable_rerank: bool
    rerank_policy: RerankPolicy
    admin_token: str
    app_env: str
    log_level: str
    dense_top_k: int
    sparse_top_k: int
    rerank_top_n: int
    answer_top_n: int
    min_retrieval_score: float
    max_history_messages: int
    model_timeout_seconds: float
    weather_timeout_seconds: float
    max_upload_bytes: int
    device_checkpoint_path: Path
    device_mcp_url: str | None
    device_mcp_timeout_seconds: float

    def ensure_runtime_dirs(self) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.vector_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    runtime_dir = _as_path(os.getenv("RUNTIME_DIR"), PROJECT_ROOT / ".runtime")
    vector_dir = _as_path(os.getenv("VECTOR_DIR"), runtime_dir / "chroma")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        database_url = f"sqlite:///{runtime_dir / 'cleanbot.db'}"

    settings = Settings(
        project_root=PROJECT_ROOT,
        data_dir=_as_path(os.getenv("DATA_DIR"), PROJECT_ROOT / "data"),
        runtime_dir=runtime_dir,
        upload_dir=_as_path(os.getenv("UPLOAD_DIR"), runtime_dir / "uploads"),
        vector_dir=vector_dir,
        database_url=database_url,
        collection_name=os.getenv("COLLECTION_NAME", "cleanbot_knowledge_v2"),
        chat_model_name=os.getenv("CHAT_MODEL_NAME", "qwen3-max"),
        embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4"),
        rerank_model_name=os.getenv("RERANK_MODEL_NAME", "qwen3-rerank"),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        dashscope_base_url=os.getenv(
            "DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        enable_rerank=_as_bool(os.getenv("ENABLE_RERANK"), True),
        rerank_policy=_as_rerank_policy(os.getenv("RERANK_POLICY")),
        admin_token=os.getenv("ADMIN_TOKEN", "change-me-before-deployment"),
        app_env=os.getenv("APP_ENV", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        dense_top_k=int(os.getenv("DENSE_TOP_K", "20")),
        sparse_top_k=int(os.getenv("SPARSE_TOP_K", "20")),
        rerank_top_n=int(os.getenv("RERANK_TOP_N", "5")),
        answer_top_n=int(os.getenv("ANSWER_TOP_N", "4")),
        min_retrieval_score=float(os.getenv("MIN_RETRIEVAL_SCORE", "0.10")),
        max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "12")),
        model_timeout_seconds=float(os.getenv("MODEL_TIMEOUT_SECONDS", "45")),
        weather_timeout_seconds=float(os.getenv("WEATHER_TIMEOUT_SECONDS", "6")),
        max_upload_bytes=int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024))),
        device_checkpoint_path=_as_path(
            os.getenv("DEVICE_CHECKPOINT_PATH"),
            runtime_dir / "device-checkpoints.sqlite",
        ),
        device_mcp_url=(os.getenv("DEVICE_MCP_URL") or None),
        device_mcp_timeout_seconds=float(
            os.getenv(
                "DEVICE_MCP_TIMEOUT_SECONDS",
                "5",
            )
        ),
    )

    settings.ensure_runtime_dirs()
    return settings
