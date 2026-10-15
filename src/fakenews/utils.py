# MIT License
#
# Copyright (c) 2024 fake-news-jev contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
"""Pure-function helpers: stable hashing, deterministic seeding,
filesystem helpers, structured logging setup.

Everything here is stdlib-only and side-effect free except `set_seed`
(seeds numpy/random if available) and `get_logger` (configures a logger).
The simulated Jev backend relies on `content_hash` for deterministic tie
breaking so that two runs over the same article text always produce the
same TypedAnswer[].
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional

# Module-level constants
_HASH_TRUNC = 16  # 64-bit-friendly short hash; collisions are fine for caching


def content_hash(text: str) -> str:
    """Return a stable, truncated SHA-256 hex digest of `text`.

    Used as the LRU-cache key for `JevClient.ask` and as a deterministic
    tie-breaker in the simulated backend. Truncation to 16 hex chars (64 bits)
    is more than enough for an in-memory cache; collisions only ever produce
    a stale-but-equivalent hit because the cache value depends solely on the
    question list, which is itself stable per `config/questions.yaml` version.

    Args:
        text: arbitrary string (article state, prompt concatenation, ...).

    Returns:
        16-char lowercase hex string.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_HASH_TRUNC]


def set_seed(seed: int) -> None:
    """Seed the Python `random` module.

    Optional numpy seeding is guarded — numpy is not a hard dependency of
    the Jev core; only the downstream meta-classifier (lightgbm) pulls it in.

    Args:
        seed: integer seed.
    """
    random.seed(seed)
    try:  # pragma: no cover - optional dep
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:  # pragma: no cover - optional dep
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create `path` (and parents) if missing; return it as a Path.

    Idempotent and safe to call on existing directories.

    Args:
        path: directory path.

    Returns:
        pathlib.Path for `path`.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def truncate(text: str, n: int = 80) -> str:
    """Return `text` truncated to at most `n` chars, with an ellipsis.

    Args:
        text: input.
        n:    maximum length (including the ellipsis when truncated).

    Returns:
        Possibly-truncated string.
    """
    if text is None:
        return ""
    if len(text) <= n:
        return text
    if n <= 1:
        return "…"
    return text[: n - 1].rstrip() + "…"


def get_logger(
    name: str,
    level: str = "INFO",
    fmt: str = "json",
) -> logging.Logger:
    """Return a configured logger with JSON or plain-text output to stdout.

    Re-fetching the same `name` returns the same logger; handlers are not
    duplicated across calls.

    JSON format emits one log record per line as
    `{"ts": "...", "level": "INFO", "logger": "...", "msg": "..."}` —
    easy to ship to Loki/CloudWatch/Elastic.

    Args:
        name:  logger name (usually `__name__`).
        level: "DEBUG"|"INFO"|"WARNING"|"ERROR"|"CRITICAL".
        fmt:   "json" or "text".

    Returns:
        logging.Logger configured with a single StreamHandler to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False  # avoid duplicate lines via root logger

    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logger.level)

    if fmt == "json":
        formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter — no third-party dep."""

    def format(self, record: logging.LogRecord) -> str:
        import json
        import time

        payload = {
            "ts": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


__all__ = [
    "content_hash",
    "set_seed",
    "ensure_dir",
    "truncate",
    "get_logger",
]
