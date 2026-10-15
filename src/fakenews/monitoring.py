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
"""Fake News Detection — monitoring, drift detection, and shadow mode.

This module ships two small, stdlib-only observability primitives:

``Monitor``
    Live counters + rolling-window latency / score histograms consumed by
    the FastAPI ``/metrics`` endpoint. Also writes human-feedback records
    to ``data/feedback.jsonl`` so retrainer scripts can consume them.

``ShadowMode``
    A/B-comparison helper for running the new pipeline alongside an
    existing (legacy) system. Tracks label-agreement and probability
    divergence over a window so a safe rollout decision can be made.

Both classes are thread-safe (a single ``threading.Lock`` guards all
mutable state) and importable even when ``numpy`` is unavailable — the
drift / latency-percentile helpers fall back to pure-Python paths.

No third-party dependencies. No secrets.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional config-logger import; fall back to stdlib logging if unavailable.
# ---------------------------------------------------------------------------
try:
    from fakenews.config import get_logger  # type: ignore
except ImportError:  # pragma: no cover
    import logging

    def get_logger(name: str, level: str = "INFO", fmt: str = "json") -> logging.Logger:
        return logging.getLogger(name)


_logger = get_logger("fakenews.monitoring")

# Default rolling-window size for latency / score buffers.
_DEFAULT_WINDOW = 1000

# Default feedback file path relative to the repo root (or CWD if the
# repo-root resolution fails).
_DEFAULT_FEEDBACK_PATH = os.environ.get(
    "FAKENEWS_FEEDBACK_PATH", "data/feedback.jsonl"
)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an object (attr or dict key), else ``default``."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Pure-Python percentile (no numpy) for ``p`` in ``[0, 100]``."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (p / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = rank - lo
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac)


class Monitor:
    """Live pipeline observability counters + drift hooks.

    The monitor is intentionally simple — every record is appended to a
    bounded rolling window (default 1000 samples) so memory stays flat
    even under sustained traffic. All public methods are thread-safe.

    Exposed via :meth:`export_metrics` for the ``/metrics`` HTTP endpoint
    and via :meth:`drift_check` for scheduled-drift alerting.

    Parameters
    ----------
    window_size : int
        Maximum number of latency / score samples retained.
    feedback_path : str, optional
        Path to the JSONL file written by :meth:`feedback_loop`. Defaults
        to ``data/feedback.jsonl`` (or the ``FAKENEWS_FEEDBACK_PATH`` env
        var if set).
    """

    LABEL_REAL = "REAL"
    LABEL_FAKE = "FAKE"
    LABEL_REVIEW = "NEEDS_REVIEW"

    def __init__(
        self,
        window_size: int = _DEFAULT_WINDOW,
        feedback_path: Optional[str] = None,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.feedback_path = feedback_path or _DEFAULT_FEEDBACK_PATH
        self._lock = threading.Lock()

        # Counters (cumulative since process start).
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._total_evidence_routed: int = 0
        self._label_counts: Counter[str] = Counter()

        # Rolling windows.
        self._latencies: deque[float] = deque(maxlen=self.window_size)
        self._scores: deque[float] = deque(maxlen=self.window_size)

        # Start time for uptime reporting.
        self._started_at = time.time()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    def record(self, verdict: Any) -> None:
        """Ingest one ``Verdict`` produced by the pipeline.

        Reads ``label``, ``probability``, ``latency_ms`` from the verdict
        (works with both the real :class:`fakenews.types.Verdict` dataclass
        and dict-shaped JSON the API may decode back into).
        """
        label = str(_get_attr(verdict, "label", "UNKNOWN") or "UNKNOWN")
        prob = float(_get_attr(verdict, "probability", 0.0) or 0.0)
        lat = float(_get_attr(verdict, "latency_ms", 0.0) or 0.0)

        with self._lock:
            self._total_requests += 1
            self._label_counts[label] += 1
            if label == self.LABEL_REVIEW:
                self._total_evidence_routed += 1
            self._latencies.append(lat)
            self._scores.append(max(0.0, min(1.0, prob)))

    def record_error(self) -> None:
        """Increment the error counter (call from exception handlers)."""
        with self._lock:
            self._total_errors += 1
            self._total_requests += 1

    # ------------------------------------------------------------------
    # Histograms + percentile
    # ------------------------------------------------------------------
    def score_distribution(self, n_bins: int = 10) -> dict[str, int]:
        """Return a histogram of calibrated ``p(fake)`` over ``[0, 1]``.

        Keys are ``"0.0-0.1"``, ``"0.1-0.2"``, …; values are counts.
        """
        n_bins = max(1, int(n_bins))
        with self._lock:
            scores = list(self._scores)
        bins = [0] * n_bins
        for p in scores:
            idx = min(int(p * n_bins), n_bins - 1)
            if idx < 0:
                idx = 0
            bins[idx] += 1
        return {
            f"{i / n_bins:.1f}-{(i + 1) / n_bins:.1f}": bins[i]
            for i in range(n_bins)
        }

    def latency_percentiles(self) -> dict[str, float]:
        """Return ``p50`` / ``p95`` / ``p99`` / ``avg`` latency in ms."""
        with self._lock:
            lats = sorted(self._latencies)
            n = len(lats)
        if n == 0:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0, "n": 0}
        return {
            "p50": _percentile(lats, 50.0),
            "p95": _percentile(lats, 95.0),
            "p99": _percentile(lats, 99.0),
            "avg": sum(lats) / n,
            "n": n,
        }

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------
    def drift_check(
        self,
        recent: dict[str, float],
        baseline: dict[str, float],
        threshold: float = 0.10,
    ) -> dict[str, Any]:
        """KL-divergence drift check between two score distributions.

        Both inputs are probability distributions summing to ~1.0 (i.e.
        normalise :meth:`score_distribution` counts before passing in).
        Returns a dict with the KL value, a boolean ``drifted`` flag, and
        a human-readable message.

        Parameters
        ----------
        recent : dict[str, float]
            Recent-window distribution (e.g. last hour).
        baseline : dict[str, float]
            Baseline distribution (e.g. the first 24h after deploy).
        threshold : float
            KL-divergence above which the system is considered drifted.
            Default 0.10 nats — a conservative starting point.

        Notes
        -----
        KL(P || Q) = sum_i P_i * log(P_i / Q_i) with Laplace smoothing
        applied to both distributions to avoid ``log(0)`` and division by
        zero.
        """
        keys = sorted(set(recent) | set(baseline))
        if not keys:
            return {"kl": 0.0, "drifted": False, "message": "no data", "n_bins": 0}
        # Laplace-smooth both distributions over the union of keys.
        eps = 1e-6
        p_sum = sum(max(recent.get(k, 0.0), eps) for k in keys)
        q_sum = sum(max(baseline.get(k, 0.0), eps) for k in keys)
        kl = 0.0
        for k in keys:
            p_i = max(recent.get(k, 0.0), eps) / p_sum
            q_i = max(baseline.get(k, 0.0), eps) / q_sum
            kl += p_i * math.log(p_i / q_i)
        kl = max(0.0, kl)
        drifted = kl > threshold
        return {
            "kl": round(kl, 6),
            "threshold": float(threshold),
            "drifted": bool(drifted),
            "message": (
                f"drift detected (KL={kl:.4f} > {threshold:.4f})"
                if drifted
                else f"no drift (KL={kl:.4f} <= {threshold:.4f})"
            ),
            "n_bins": len(keys),
        }

    # ------------------------------------------------------------------
    # Feedback loop (human-in-the-loop retraining)
    # ------------------------------------------------------------------
    def feedback_loop(
        self,
        article: Any,
        verdict: Any,
        human_label: str,
        *,
        path: Optional[str] = None,
    ) -> str:
        """Append a human-labelled example to the feedback JSONL store.

        Each line is a self-contained JSON record with the article text,
        the pipeline's verdict (label + probability), and the human label
        (``"real"`` / ``"fake"`` / ``"uncertain"``). Retrainer scripts can
        stream-read this file to grow the labelled training set.

        Parameters
        ----------
        article : Article
            The original article.
        verdict : Verdict
            The pipeline's verdict for the same article.
        human_label : str
            Reviewer's ground-truth label.
        path : str, optional
            Override the file path (defaults to ``self.feedback_path``).

        Returns
        -------
        str
            The path actually written to (so callers can log it).
        """
        target = path or self.feedback_path
        # Make the directory so the open() call doesn't fail.
        try:
            os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
        except Exception:
            pass

        record = {
            "ts": _utc_now_iso(),
            "article": {
                "headline": _get_attr(article, "headline", ""),
                "body": _get_attr(article, "body", ""),
                "source": _get_attr(article, "source", ""),
                "url": _get_attr(article, "url", ""),
                "date": _get_attr(article, "date", ""),
            },
            "verdict": {
                "label": _get_attr(verdict, "label", None),
                "probability": _get_attr(verdict, "probability", None),
                "model_version": _get_attr(verdict, "model_version", None),
            },
            "human_label": str(human_label).lower(),
        }
        line = json.dumps(record, ensure_ascii=False)
        try:
            # Thread-safe append (per-process). Multiple processes would need
            # a file lock — out of scope for this module.
            with self._lock:
                with open(target, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
        except Exception as exc:  # pragma: no cover - filesystem failure
            _logger.warning("monitor.feedback.write_failed", extra={"error": str(exc), "path": target})
            return target
        return target

    # ------------------------------------------------------------------
    # Snapshot for /metrics
    # ------------------------------------------------------------------
    def export_metrics(self) -> dict[str, Any]:
        """Return a snapshot of all live counters for the ``/metrics`` route."""
        with self._lock:
            total = self._total_requests
            label_counts = dict(self._label_counts)
            errors = self._total_errors
            evidence_routed = self._total_evidence_routed
            scores = list(self._scores)
            lats = list(self._latencies)
        # Latency percentiles.
        sorted_lats = sorted(lats)
        pcts = {
            "p50_ms": round(_percentile(sorted_lats, 50.0), 3),
            "p95_ms": round(_percentile(sorted_lats, 95.0), 3),
            "p99_ms": round(_percentile(sorted_lats, 99.0), 3),
            "avg_ms": round(sum(lats) / len(lats), 3) if lats else 0.0,
            "n": len(lats),
        }
        # Label distribution (% of total).
        label_pct = {
            lbl: round(cnt / total * 100.0, 2) if total else 0.0
            for lbl, cnt in label_counts.items()
        }
        # % routed to review.
        pct_review = (evidence_routed / total * 100.0) if total else 0.0
        return {
            "status": "ok",
            "uptime_s": round(time.time() - self._started_at, 1),
            "requests": total,
            "errors": errors,
            "error_rate_pct": round(errors / total * 100.0, 3) if total else 0.0,
            "label_distribution": label_counts,
            "label_distribution_pct": label_pct,
            "score_distribution": self.score_distribution(),
            "latency_ms": pcts,
            "pct_routed_to_review": round(pct_review, 3),
            "window_size": self.window_size,
        }


# ---------------------------------------------------------------------------
# Shadow mode
# ---------------------------------------------------------------------------
class ShadowMode:
    """A/B-comparison helper for running next to an existing system.

    Records ``(new_verdict, old_verdict)`` pairs and exposes agreement /
    probability-divergence metrics so a safe rollout decision can be made.

    Typical usage::

        shadow = ShadowMode()
        new_v = await new_pipeline.predict(article)
        old_v = await legacy_system.predict(article)
        shadow.compare(new_v, old_v)
        if shadow.summary()["agreement_rate"] < 0.90:
            alert("shadow disagreement too high")
    """

    def __init__(
        self,
        monitor: Optional[Monitor] = None,
        window_size: int = _DEFAULT_WINDOW,
    ) -> None:
        self.monitor = monitor if monitor is not None else Monitor(window_size=window_size)
        self.window_size = max(1, int(window_size))
        self._lock = threading.Lock()
        self._pairs: deque[tuple[Any, Any]] = deque(maxlen=self.window_size)
        self._agreements: int = 0
        self._total: int = 0
        self._prob_divs: deque[float] = deque(maxlen=self.window_size)

    # ------------------------------------------------------------------
    def compare(self, new_verdict: Any, old_verdict: Any) -> dict[str, Any]:
        """Record a single (new, old) verdict pair and return a per-row summary."""
        new_label = str(_get_attr(new_verdict, "label", "?") or "?")
        old_label = str(_get_attr(old_verdict, "label", "?") or "?")
        new_prob = float(_get_attr(new_verdict, "probability", 0.0) or 0.0)
        old_prob = float(_get_attr(old_verdict, "probability", 0.0) or 0.0)
        agreement = new_label == old_label
        divergence = abs(new_prob - old_prob)

        with self._lock:
            self._pairs.append((new_verdict, old_verdict))
            self._total += 1
            if agreement:
                self._agreements += 1
            self._prob_divs.append(divergence)

        # Also forward the new verdict to the live monitor so /metrics
        # includes the shadow traffic.
        try:
            self.monitor.record(new_verdict)
        except Exception:  # pragma: no cover
            pass

        return {
            "agreement": bool(agreement),
            "new_label": new_label,
            "old_label": old_label,
            "new_prob": round(new_prob, 4),
            "old_prob": round(old_prob, 4),
            "prob_divergence": round(divergence, 4),
        }

    # ------------------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        """Aggregate shadow-mode statistics over the rolling window."""
        with self._lock:
            total = self._total
            agreements = self._agreements
            divs = list(self._prob_divs)
        if total == 0:
            return {
                "n": 0,
                "agreement_rate": 0.0,
                "avg_prob_divergence": 0.0,
                "max_prob_divergence": 0.0,
            }
        rate = agreements / total if total else 0.0
        return {
            "n": total,
            "agreements": agreements,
            "agreement_rate": round(rate, 4),
            "disagreements": total - agreements,
            "avg_prob_divergence": round(sum(divs) / len(divs), 4) if divs else 0.0,
            "max_prob_divergence": round(max(divs), 4) if divs else 0.0,
        }


__all__ = ["Monitor", "ShadowMode"]
