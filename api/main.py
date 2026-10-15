# Copyright (c) 2025 fake-news-jev contributors
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
"""Fake News Detector — FastAPI serving layer.

Exposes the 3-stage :class:`fakenews.pipeline.FakeNewsPipeline` over HTTP:

    POST /predict          single-article verdict
    POST /predict/batch    concurrent batch
    GET  /health           liveness + model-loaded + jev provider
    GET  /metrics          live counters (requests, latency, score buckets,
                           % routed to review)

Configured with CORS, structured JSON logging via
:func:`fakenews.config.get_logger`, and a lazy-loaded singleton pipeline +
monitor so the module imports cleanly even when the heavy ML stack has not
been initialised yet (e.g. during container warm-up).

Run with::

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

or directly::

    python -m api.main
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, List, Optional

# Make ``fakenews`` importable when the API is launched from the repo root
# (``uvicorn api.main:app``). ``src/`` is added to ``sys.path`` so the
# namespace package resolves without requiring ``pip install -e .``.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_PATH = os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# ---------------------------------------------------------------------------
# Contract imports — guarded where the spec requires it.
# ---------------------------------------------------------------------------
try:
    from fakenews.config import get_logger, load_config  # type: ignore
except ImportError:  # pragma: no cover - parallel-task fallback
    import logging

    def get_logger(name: str, level: str = "INFO", fmt: str = "json") -> logging.Logger:
        return logging.getLogger(name)

    def load_config() -> dict:
        return {}

from fakenews.pipeline import FakeNewsPipeline, build_default_pipeline  # type: ignore
from fakenews.monitoring import Monitor  # type: ignore

# FastAPI / pydantic / uvicorn — guarded per spec so the module imports
# cleanly even if the deps were not installed (they are in this sandbox).
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover - spec requires the guard
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = Exception  # type: ignore[assignment,misc]
    CORSMiddleware = None  # type: ignore[assignment,misc]
    JSONResponse = None  # type: ignore[assignment,misc]
    Request = None  # type: ignore[assignment,misc]

    # Minimal BaseModel shim so Pydantic-style class bodies don't crash.
    class BaseModel:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    def Field(*args: Any, **kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None

try:
    import uvicorn  # type: ignore
    _HAS_UVICORN = True
except ImportError:  # pragma: no cover
    _HAS_UVICORN = False


_logger = get_logger("fakenews.api")


# ---------------------------------------------------------------------------
# Request / response Pydantic models
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel if _FASTAPI_AVAILABLE else object):  # type: ignore[misc]
    """Single-article POST body for /predict."""

    headline: str = ""
    body: str = Field(..., min_length=1, description="Article body text (required)")
    source: Optional[str] = None
    url: Optional[str] = None
    date: Optional[str] = None


class BatchRequest(BaseModel if _FASTAPI_AVAILABLE else object):  # type: ignore[misc]
    """Batch POST body for /predict/batch."""

    articles: List[PredictRequest]


class SignalOut(BaseModel if _FASTAPI_AVAILABLE else object):  # type: ignore[misc]
    """Serialised Signal — name/value/probability/type/contribution."""

    name: str
    value: Any
    probability: float = 0.0
    type: str = ""
    contribution: float = 0.0


class EvidenceOut(BaseModel if _FASTAPI_AVAILABLE else object):  # type: ignore[misc]
    """Serialised Evidence — claims/findings/source."""

    claims: List[str] = []
    findings: List[Any] = []
    source: str = ""
    adjusted_probability: Optional[float] = None


class VerdictOut(BaseModel if _FASTAPI_AVAILABLE else object):  # type: ignore[misc]
    """Serialised Verdict — the JSON shape returned by /predict."""

    label: str
    probability: float
    signals: List[Any] = []
    explanation: str = ""
    evidence: Optional[Any] = None
    latency_ms: float = 0.0
    model_version: str = ""


# ---------------------------------------------------------------------------
# Singleton pipeline + monitor (lazy-loaded)
# ---------------------------------------------------------------------------
_pipeline: Optional[FakeNewsPipeline] = None
_monitor: Optional[Monitor] = None
_pipeline_lock = __import__("threading").Lock()


def get_pipeline() -> FakeNewsPipeline:
    """Return the singleton :class:`FakeNewsPipeline`, building it on first call."""
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                t0 = time.perf_counter()
                cfg = load_config()
                _pipeline = build_default_pipeline(cfg)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                _logger.info(
                    "api.pipeline.loaded",
                    extra={
                        "latency_ms": round(dt_ms, 2),
                        "model_version": _pipeline.model_version,
                        "jev_provider": _pipeline.jev_provider,
                        "using_fallback": _pipeline.using_fallback,
                    },
                )
    return _pipeline


def get_monitor() -> Monitor:
    """Return the singleton :class:`Monitor`."""
    global _monitor
    if _monitor is None:
        with _pipeline_lock:
            if _monitor is None:
                _monitor = Monitor()
    return _monitor


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------
def _signal_to_dict(signal: Any) -> dict[str, Any]:
    """Serialise a Signal-like object to a JSON-safe dict."""
    if signal is None:
        return {}
    if isinstance(signal, dict):
        return {
            "name": str(signal.get("name", "")),
            "value": _coerce_json(signal.get("value")),
            "probability": float(signal.get("probability", 0.0) or 0.0),
            "type": _type_str(signal.get("type")),
            "contribution": float(signal.get("contribution", 0.0) or 0.0),
        }
    # tuple of (name, value) — what fakenews.explain returns directly.
    if isinstance(signal, (tuple, list)) and len(signal) == 2 and isinstance(signal[0], str):
        name, value = signal
        try:
            contrib = float(value)
        except (TypeError, ValueError):
            contrib = 0.0
        return {
            "name": str(name),
            "value": 0.0,
            "probability": 0.0,
            "type": "noul",
            "contribution": contrib,
        }
    # dataclass Signal.
    return {
        "name": str(getattr(signal, "name", "") or ""),
        "value": _coerce_json(getattr(signal, "value", 0.0)),
        "probability": float(getattr(signal, "probability", 0.0) or 0.0),
        "type": _type_str(getattr(signal, "type", None)),
        "contribution": float(getattr(signal, "contribution", 0.0) or 0.0),
    }


def _evidence_to_dict(evidence: Any) -> Optional[dict[str, Any]]:
    """Serialise an Evidence-like object to a JSON-safe dict."""
    if evidence is None:
        return None
    if isinstance(evidence, dict):
        return {
            "claims": [str(c) for c in evidence.get("claims", []) or []],
            "findings": evidence.get("findings", []) or [],
            "source": str(evidence.get("source", "") or ""),
            "adjusted_probability": _safe_float(evidence.get("adjusted_probability")),
        }
    return {
        "claims": [str(c) for c in (getattr(evidence, "claims", None) or [])],
        "findings": list(getattr(evidence, "findings", None) or []),
        "source": str(getattr(evidence, "source", "") or ""),
        "adjusted_probability": _safe_float(getattr(evidence, "adjusted_probability", None)),
    }


def _verdict_to_dict(verdict: Any) -> dict[str, Any]:
    """Serialise a Verdict to the JSON shape returned by /predict."""
    signals_raw = getattr(verdict, "signals", None) or []
    signals_out: list[dict[str, Any]] = []
    if isinstance(signals_raw, (list, tuple)):
        for s in signals_raw:
            signals_out.append(_signal_to_dict(s))
    return {
        "label": str(getattr(verdict, "label", "NEEDS_REVIEW") or "NEEDS_REVIEW"),
        "probability": float(getattr(verdict, "probability", 0.0) or 0.0),
        "signals": signals_out,
        "explanation": str(getattr(verdict, "explanation", "") or ""),
        "evidence": _evidence_to_dict(getattr(verdict, "evidence", None)),
        "latency_ms": float(getattr(verdict, "latency_ms", 0.0) or 0.0),
        "model_version": str(getattr(verdict, "model_version", "") or ""),
    }


def _coerce_json(value: Any) -> Any:
    """Coerce a value to something json-serialisable."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    # Enums (e.g. AnswerType) — use .value.
    val = getattr(value, "value", None)
    if val is not None and isinstance(val, (str, int, float, bool)):
        return val
    return str(value)


def _type_str(t: Any) -> str:
    """Return the lowercase string value of an AnswerType (or empty)."""
    if t is None:
        return ""
    val = getattr(t, "value", t)
    return str(val).lower()


def _safe_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _request_to_article(req: PredictRequest) -> Any:
    """Build an Article from a PredictRequest (works for any Article class)."""
    try:
        from fakenews.types import Article  # type: ignore
        return Article(
            headline=req.headline or "",
            body=req.body or "",
            source=req.source or "",
            url=req.url or "",
            date=req.date or "",
        )
    except ImportError:  # pragma: no cover - parallel-task fallback
        from fakenews.pipeline import Article as _Article  # type: ignore
        return _Article(
            headline=req.headline or "",
            body=req.body or "",
            source=req.source or "",
            url=req.url or "",
            date=req.date or "",
        )


# ---------------------------------------------------------------------------
# FastAPI app — only built when the framework is available.
# ---------------------------------------------------------------------------
app: Optional[Any] = None
if _FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Fake News Detector API",
        description=(
            "Jev-typed-decision fake-news detection. 3-stage pipeline: "
            "signal extraction -> meta-classifier -> optional evidence check."
        ),
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    @app.post(
        "/predict",
        response_model=None,  # we return a plain JSONResponse for control
        tags=["predict"],
        summary="Classify a single article",
    )
    async def predict(req: PredictRequest) -> Any:  # type: ignore[override]
        """Run the 3-stage pipeline on one article and return a Verdict."""
        if not req.body or not req.body.strip():
            raise HTTPException(status_code=400, detail="`body` is required")
        pipe = get_pipeline()
        mon = get_monitor()
        article = _request_to_article(req)
        try:
            verdict = await pipe.predict(article)
        except Exception as exc:
            mon.record_error()
            _logger.error("api.predict.error", extra={"error": str(exc)})
            raise HTTPException(status_code=500, detail=f"pipeline error: {exc}") from exc
        mon.record(verdict)
        return JSONResponse(_verdict_to_dict(verdict))

    @app.post(
        "/predict/batch",
        response_model=None,
        tags=["predict"],
        summary="Classify a batch of articles concurrently",
    )
    async def predict_batch(req: BatchRequest) -> Any:  # type: ignore[override]
        """Run /predict on every article in ``req.articles`` concurrently."""
        if not req.articles:
            return JSONResponse({"verdicts": []})
        pipe = get_pipeline()
        mon = get_monitor()
        articles = [_request_to_article(r) for r in req.articles]

        async def _one(article: Any) -> dict[str, Any]:
            try:
                v = await pipe.predict(article)
                mon.record(v)
                return _verdict_to_dict(v)
            except Exception as exc:
                mon.record_error()
                _logger.error("api.batch.item_error", extra={"error": str(exc)})
                return {
                    "label": "ERROR",
                    "probability": 0.5,
                    "signals": [],
                    "explanation": f"item failed: {exc}",
                    "evidence": None,
                    "latency_ms": 0.0,
                    "model_version": "",
                }

        # Bounded concurrency to keep the Jev backend from being
        # thundering-herded; the rate-limiter inside AsyncJevClient also
        # guards this, but we cap fan-out locally too.
        import asyncio
        sem = asyncio.Semaphore(max(1, int(os.environ.get("FAKENEWS_BATCH_CONCURRENCY", "8"))))

        async def _guarded(a: Any) -> dict[str, Any]:
            async with sem:
                return await _one(a)

        verdicts = await asyncio.gather(*[_guarded(a) for a in articles])
        return JSONResponse({"verdicts": list(verdicts)})

    @app.get(
        "/health",
        tags=["ops"],
        summary="Liveness probe + model status",
    )
    async def health() -> Any:  # type: ignore[override]
        """Return ``{status, model_loaded, jev_provider, model_version}``."""
        pipe = get_pipeline()
        return {
            "status": "ok",
            "model_loaded": not pipe.using_fallback,
            "using_fallback": pipe.using_fallback,
            "jev_provider": pipe.jev_provider,
            "model_version": pipe.model_version,
        }

    @app.get(
        "/metrics",
        tags=["ops"],
        summary="Live pipeline counters",
    )
    async def metrics() -> Any:  # type: ignore[override]
        """Return live request counters, latency percentiles, score buckets."""
        return JSONResponse(get_monitor().export_metrics())

    @app.get(
        "/",
        tags=["ops"],
        summary="Service info",
    )
    async def root() -> Any:  # type: ignore[override]
        """Tiny landing page so curl http://localhost:8000/ shows something."""
        return {
            "service": "fake-news-detector",
            "version": "0.1.0",
            "endpoints": [
                "POST /predict",
                "POST /predict/batch",
                "GET  /health",
                "GET  /metrics",
            ],
            "docs": "/docs",
        }


# ---------------------------------------------------------------------------
# Direct entry point — guarded so importing the module never starts uvicorn.
# ---------------------------------------------------------------------------
def _run() -> None:
    """Launch uvicorn (only when invoked as ``python -m api.main``)."""
    if not _HAS_UVICORN:
        print("uvicorn is not installed. Install with:  pip install uvicorn", file=sys.stderr)
        sys.exit(1)
    if not _FASTAPI_AVAILABLE:
        print("fastapi is not installed. Install with:  pip install fastapi", file=sys.stderr)
        sys.exit(1)
    host = os.environ.get("FAKENEWS_API_HOST", "0.0.0.0")
    port = int(os.environ.get("FAKENEWS_API_PORT", "8000"))
    _logger.info("api.launch", extra={"host": host, "port": port})
    uvicorn.run(  # type: ignore[no-redef]
        "api.main:app",
        host=host,
        port=port,
        reload=bool(os.environ.get("FAKENEWS_API_RELOAD", "0") == "1"),
        log_level=os.environ.get("FAKENEWS_API_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    _run()
