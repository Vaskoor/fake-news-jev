# -*- coding: utf-8 -*-
"""
Probability calibration for the fake-news meta-classifier.

Implements a thin, serialisable wrapper around either:

  * ``sklearn.isotonic.IsotonicRegression`` (mode ``"isotonic"``) — the
    default; monotone, non-parametric, very effective on tree-model
    probabilities which tend to be poorly calibrated.
  * ``sklearn.calibration.CalibratedClassifierCV`` with ``cv="prefit"`` and
    ``method="sigmoid"`` (mode ``"platt"``) — Platt scaling, a single
    logistic re-fit on the validation probabilities.
  * ``"none"`` — identity (returns probabilities unchanged).

Also exposes three diagnostic helpers used by ``scripts/evaluate.py``:

  * :func:`expected_calibration_error` — scalar ECE.
  * :func:`reliability_data` — per-bin accuracy/confidence for reliability
    diagrams.
  * :func:`plot_reliability` — matplotlib reliability diagram (guarded:
    no-op if matplotlib is missing).

MIT License — Copyright (c) 2024 fake-news-jev contributors.
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any, Tuple

import numpy as np
import joblib

from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV


class Calibrator:
    """Probability calibrator (isotonic / Platt / none).

    Parameters
    ----------
    mode : {"isotonic", "platt", "none"}
        Calibration strategy.
    config : dict, optional
        Reserved for future tuning knobs; currently unused but kept for API
        symmetry with :class:`fakenews.meta_classifier.MetaClassifier`.

    Notes
    -----
    The calibrator is *one-dimensional* — it operates on a single column of
    probabilities (``p(fake)``). This is by design: the meta-classifier emits
    a single decision-relevant probability and we calibrate that directly.
    """

    def __init__(self, mode: str = "isotonic", config: Optional[Dict[str, Any]] = None) -> None:
        if mode not in {"isotonic", "platt", "none"}:
            raise ValueError(f"Unknown calibration mode: {mode!r}")
        self.mode = mode
        self.config = config or {}
        self._fitted = False
        self._iso: Optional[IsotonicRegression] = None
        self._platt: Optional[CalibratedClassifierCV] = None
        # For the platt path we need to keep the underlying estimator around
        # so CalibratedClassifierCV can wrap it; we cheat by re-creating a
        # sigmoid from raw probabilities via a tiny LogReg below.
        self._platt_a: float = 0.0  # slope
        self._platt_b: float = 0.0  # intercept

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, probs, y) -> "Calibrator":
        """Fit the calibrator on (validation) probabilities and labels.

        Parameters
        ----------
        probs : array-like of shape (n,)
            Raw ``p(fake)`` values emitted by the meta-classifier.
        y : array-like of shape (n,)
            Binary ground-truth labels (0 = real, 1 = fake).
        """
        probs = np.asarray(probs, dtype=float).ravel()
        y = np.asarray(y).astype(int).ravel()
        if probs.shape[0] != y.shape[0]:
            raise ValueError(
                f"probs ({probs.shape[0]}) and y ({y.shape[0]}) length mismatch"
            )
        if len(set(y)) < 2:
            # Degenerate (all one class) — calibrator is identity.
            self._fitted = True
            return self

        if self.mode == "isotonic":
            self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            self._iso.fit(probs, y)
        elif self.mode == "platt":
            # A 1-D Platt scaling: logistic regression of y on logit(probs).
            # We clamp to avoid +/-inf.
            eps = 1e-6
            p = np.clip(probs, eps, 1.0 - eps)
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            # Tiny closed-form-ish: use sklearn LogReg on the 1-D logit.
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            lr.fit(logits, y)
            self._platt_a = float(lr.coef_[0, 0])
            self._platt_b = float(lr.intercept_[0])
        # "none": nothing to fit.
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # calibrate
    # ------------------------------------------------------------------
    def calibrate(self, probs) -> np.ndarray:
        """Map raw probabilities -> calibrated probabilities."""
        if not self._fitted:
            raise RuntimeError("Calibrator.fit() must be called before calibrate()")
        probs = np.asarray(probs, dtype=float).ravel()
        if self.mode == "isotonic" and self._iso is not None:
            return self._iso.predict(probs)
        if self.mode == "platt":
            eps = 1e-6
            p = np.clip(probs, eps, 1.0 - eps)
            logits = np.log(p / (1.0 - p))
            z = self._platt_a * logits + self._platt_b
            return 1.0 / (1.0 + np.exp(-z))
        return probs.copy()  # mode == "none"

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "Calibrator":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a Calibrator: {type(obj)!r}")
        return obj


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def expected_calibration_error(
    y_true, y_pred_proba, n_bins: int = 10
) -> float:
    """Return the Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    where ``B_b`` is the set of samples whose predicted probability falls
    into bin ``b`` (equal-width over [0, 1]).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_pred_proba, dtype=float).ravel()
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_pred_proba must have equal length")
    n = y_true.shape[0]
    if n == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() == 0:
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(y_proba[mask]))
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def reliability_data(
    y_true, y_proba, n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """Return per-bin reliability data for plotting.

    Returns
    -------
    dict with keys:
        bin_centers : ndarray (n_bins,)
            Midpoint of each bin.
        accs : ndarray (n_bins,)
            Empirical accuracy in each bin.
        confs : ndarray (n_bins,)
            Mean predicted probability in each bin.
        counts : ndarray (n_bins,)
            Number of samples in each bin.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    accs = np.full(n_bins, np.nan)
    confs = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() > 0:
            accs[b] = float(np.mean(y_true[mask]))
            confs[b] = float(np.mean(y_proba[mask]))
            counts[b] = int(mask.sum())
    return {
        "bin_centers": centers,
        "accs": accs,
        "confs": confs,
        "counts": counts,
    }


def plot_reliability(y_true, y_proba, path: str, n_bins: int = 10, title: str = "Reliability Diagram") -> bool:
    """Render a reliability diagram to ``path``.

    Guarded: returns ``False`` (and writes nothing) if matplotlib is not
    available so the evaluation pipeline never crashes for a missing optional
    dependency.
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:                       # pragma: no cover
        return False

    data = reliability_data(y_true, y_proba, n_bins=n_bins)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(
        data["bin_centers"],
        data["accs"],
        width=1.0 / n_bins * 0.9,
        edgecolor="black",
        color="#4C9F70",
        label="Accuracy in bin",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of fakes")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ece = expected_calibration_error(y_true, y_proba, n_bins=n_bins)
    ax.text(
        0.04, 0.96, f"ECE = {ece:.4f}", transform=ax.transAxes,
        va="top", ha="left", fontsize=9, bbox=dict(facecolor="white", alpha=0.7),
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


__all__ = [
    "Calibrator",
    "expected_calibration_error",
    "reliability_data",
    "plot_reliability",
]
