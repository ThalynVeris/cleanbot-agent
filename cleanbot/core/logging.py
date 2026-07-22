from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from cleanbot.core.config import get_settings

_SENSITIVE_KEYS = {"api_key", "authorization", "token", "admin_token", "password"}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "***" if key.lower() in _SENSITIVE_KEYS else _redact(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if context:
            payload["context"] = _redact(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    settings = get_settings()
    root = logging.getLogger()
    if getattr(root, "_cleanbot_configured", False):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(settings.log_level)
    root._cleanbot_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
