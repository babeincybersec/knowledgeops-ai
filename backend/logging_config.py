"""Structured JSON logging setup for the backend."""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone


# Per-request context — set in middleware, read by the formatter
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """Emit each log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }

        # Attach any extra fields the caller passed via `extra=`
        for key in ("event", "latency_ms", "tokens_in", "tokens_out",
                    "status", "source_count", "found", "path", "method"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Set up root logger to emit JSON to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet down noisy third-party loggers
    for name in ("uvicorn.access", "httpx", "urllib3", "chromadb"):
        logging.getLogger(name).setLevel(logging.WARNING)


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]