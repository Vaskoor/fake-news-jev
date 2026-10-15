# -*- coding: utf-8 -*-
"""
Decision policy + threshold tuning for the fake-news pipeline.

Three-bucket policy on the calibrated ``p(fake)``:

  * ``p < real_max``           -> ``("REAL",      "below fake threshold")``
  * ``real_max <= p < fake_min`` -> ``("NEEDS_REVIEW", "uncertain — routed to evidence check")``
  * ``p >= fake_min``          -> ``("FAKE",      "above fake threshold")``

The ``review_min`` knob is honoured as a sanity floor: if it differs from
``real_max`` the gap region is still treated as NEEDS_REVIEW (the safest
choice for an uncertain prediction).

``tune_thresholds`` sweeps ``real_max`` and ``fake_min`` over the
validation probabilities to hit a target precision or recall on the FAKE
class — never by guessing.

MIT License — Copyright (c) 2024 fake-news-jev contributors.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


DEFAULT_THRESHOLDS: Dict[str, float] = {
    "real_max": 0.30,
    "review_min": 0.30,
    "fake_min": 0.70,
}


def decide(p_fake: float, thresholds: Optional[Dict[str, float]] = None) -> Tuple[str, str]:
    """Map a calibrated ``p(fake)`` to ``(label, reason)``.

    Parameters
    ----------
    p_fake : float
        Calibrated probability of the article being fake, in ``[0, 1]``.
    thresholds : dict, optional
        Must contain ``real_max`` and ``fake_min`` (``review_min`` optional,
        defaults to ``real_max``). If omitted, :data:`DEFAULT_THRESHOLDS`
        is used.

    Returns
    -------
    (label, reason) : tuple[str, str]
        ``label`` ∈ {"REAL", "NEEDS_REVIEW", "FAKE"}.
    """
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update({k: float(v) for k, v in thresholds.items() if k in t})
    # review_min defaults to real_max if not provided / equal.
    t["review_min"] = t.get("review_min", t["real_max"])

    p = float(p_fake)
    if p < t["real_max"]:
        return "REAL", "below fake threshold"
    if p >= t["fake_min"]:
        return "FAKE", "above fake threshold"
    # Everything between real_max and fake_min is the review zone.
    return "NEEDS_REVIEW", "uncertain — routed to evidence check"


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------
def _precision_recall_at(p, y, fake_min: float, real_max: float):
    """Precision / recall / review-rate for the FAKE class at given thresholds.

    FAKE predictions: p >= fake_min. NEEDS_REVIEW: real_max <= p < fake_min.
    REAL: p < real_max.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    if n == 0:
        result = {"precision": float("nan"), "recall": float("nan"),
                "review_rate": float("nan"), "accuracy": float("nan")}

    fake_mask = p >= fake_min
    real_mask = p < real_max
    review_mask = ~fake_mask & ~real_mask

    # FAKE-class precision / recall (NEEDS_REVIEW is excluded from both
    # precision and recall denominators to keep the metrics policy-meaningful).
    tp = int((fake_mask & (y == 1)).sum())
    fp = int((fake_mask & (y == 0)).sum())
    fn_fake = int(((~fake_mask) & (y == 1)).sum())  # includes review-zone fakes
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn_fake) if (tp + fn_fake) > 0 else float("nan")
    review_rate = float(review_mask.sum()) / n
    # "Decision accuracy": review-zone excluded, correct = REAL true-neg + FAKE true-pos.
    correct = int((real_mask & (y == 0)).sum()) + int((fake_mask & (y == 1)).sum())
    decided = int((real_mask | fake_mask).sum())
    accuracy = correct / decided if decided > 0 else float("nan")
    return {
        "precision": precision,
        "recall": recall,
        "review_rate": review_rate,
        "accuracy": accuracy,
    }


def tune_thresholds(
    y_true,
    y_proba,
    target_precision: Optional[float] = None,
    target_recall: Optional[float] = None,
    default: Optional[Dict[str, float]] = None,
    grid_size: int = 101,
) -> Dict[str, float]:
    """Sweep ``real_max`` / ``fake_min`` on validation to hit a target.

    Strategy
    --------
    * If ``target_precision`` is set: find the lowest ``fake_min`` such that
      the FAKE-class precision is >= target_precision (lower ``fake_min`` =
      higher recall but lower precision, so we pick the threshold meeting
      the precision floor that maximises recall). Falls back to default if
      no threshold satisfies the target.
    * If ``target_recall`` is set: find the highest ``fake_min`` such that
      FAKE-class recall >= target_recall (higher ``fake_min`` = higher
      precision). Falls back to default if no threshold satisfies the target.
    * If both are set: pick the smallest ``fake_min`` satisfying the
      precision floor, then the highest ``real_max`` that maintains the
      recall floor.
    * If neither is set: return the (possibly customised) defaults.

    ``real_max`` is set equal to ``fake_min * 0.5`` by default when not
    otherwise constrained, i.e. a symmetric review band centred on 0.5;
    when ``target_recall`` is set we sweep it independently.

    Parameters
    ----------
    y_true, y_proba : array-like
        Validation labels and calibrated ``p(fake)``.
    target_precision, target_recall : float, optional
        Policy targets in ``[0, 1]``.
    default : dict, optional
        Starting thresholds (defaults to :data:`DEFAULT_THRESHOLDS`).
    grid_size : int
        Number of grid points for the sweep.

    Returns
    -------
    dict
        ``{"real_max": float, "review_min": float, "fake_min": float}``.
    """
    if default is None:
        default = dict(DEFAULT_THRESHOLDS)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_proba, dtype=float).ravel()

    # If no target is given, return the (possibly customised) defaults
    # untouched — we do NOT guess thresholds without a policy goal.
    if target_precision is None and target_recall is None:
        return {
            "real_max": float(default["real_max"]),
            "review_min": float(default.get("review_min", default["real_max"])),
            "fake_min": float(default["fake_min"]),
            "lower": float(default["real_max"]),
            "upper": float(default["fake_min"]),
            "low": float(default["real_max"]),
            "high": float(default["fake_min"]),
        }

    grid = np.linspace(0.01, 0.99, grid_size)
    fake_min = float(default["fake_min"])
    real_max = float(default["real_max"])

    # --- fake_min sweep (precision and/or recall) ---------------------------
    if target_precision is not None or target_recall is not None:
        best_fake_min = None
        # Iterate from high to low threshold: precision is non-increasing,
        # recall is non-decreasing as fake_min decreases.
        candidates = []
        for fm in grid:
            m = _precision_recall_at(p, y, fake_min=fm, real_max=fm * 0.5)
            candidates.append((fm, m))
        # Precision-target: lowest fm with precision >= target.
        if target_precision is not None:
            ok = [c for c in candidates
                  if not np.isnan(c[1]["precision"]) and c[1]["precision"] >= target_precision]
            if ok:
                # Lowest fm maximises recall among precision-satisfying thresholds.
                best_fake_min = min(c[0] for c in ok)
        # Recall-target: highest fm with recall >= target.
        if target_recall is not None:
            ok = [c for c in candidates
                  if not np.isnan(c[1]["recall"]) and c[1]["recall"] >= target_recall]
            if ok:
                # Highest fm maximises precision among recall-satisfying thresholds.
                cand_recall = max(c[0] for c in ok)
                best_fake_min = (
                    max(best_fake_min, cand_recall)
                    if best_fake_min is not None
                    else cand_recall
                )
        if best_fake_min is not None:
            fake_min = float(best_fake_min)

    # --- real_max sweep ----------------------------------------------------
    # Sweep real_max to hit a recall floor on the REAL class (true-negatives
    # correctly let through) without pushing too many real articles into
    # review. When target_recall is set on the FAKE class, we keep the
    # default band. Otherwise, choose real_max that minimises the review
    # rate subject to keeping decision-accuracy high.
    best_real_max = real_max
    best_score = -np.inf
    for rm in grid:
        if rm >= fake_min:
            continue
        m = _precision_recall_at(p, y, fake_min=fake_min, real_max=rm)
        # Score: decision accuracy minus a small penalty for review rate.
        score = (m["accuracy"] if not np.isnan(m["accuracy"]) else 0.0) \
                - 0.5 * (m["review_rate"] if not np.isnan(m["review_rate"]) else 0.0)
        if score > best_score:
            best_score = score
            best_real_max = float(rm)
    if best_score > -np.inf:
        real_max = best_real_max

    return {
        "real_max": float(real_max),
        "review_min": float(real_max),
        "fake_min": float(fake_min),
        "lower": float(real_max),
        "upper": float(fake_min),
        "low": float(real_max),
        "high": float(fake_min),
    }


__all__ = ["decide", "tune_thresholds", "DEFAULT_THRESHOLDS"]
