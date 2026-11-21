"""Structured logging + a tiny in-process metrics registry for run summaries."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from threading import Lock


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "ctx", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class PrettyFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{time.strftime('%H:%M:%S')} {record.levelname:<7} {record.name:<22} {record.getMessage()}"
        extra = getattr(record, "ctx", None)
        if extra:
            base += "  " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def setup_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = os.getenv("LOG_FORMAT", "pretty").lower()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else PrettyFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    for noisy in ("asyncio", "urllib3", "trafilatura", "htmldate", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log(logger: logging.Logger, level: int, msg: str, **ctx) -> None:
    logger.log(level, msg, extra={"ctx": ctx} if ctx else None)


class Metrics:
    """Thread/async-safe counters, dumped into output/run_summary.json."""

    def __init__(self) -> None:
        self._c: Counter[str] = Counter()
        self._lock = Lock()
        self.started = time.time()

    def inc(self, key: str, n: int = 1) -> None:
        with self._lock:
            self._c[key] += n

    def get(self, key: str) -> int:
        return self._c.get(key, 0)

    def snapshot(self) -> dict:
        with self._lock:
            d = dict(sorted(self._c.items()))
        d["elapsed_seconds"] = round(time.time() - self.started, 1)
        return d


METRICS = Metrics()
