# MIT License
#
# Copyright (c) 2024 Bhaskar Kaphle <kaphlebhaskar012345@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# Tests for `fakenews.decision` — threshold-based decisioning and tuning.

"""Tests for fakenews.decision — thresholded decisions + threshold tuning.

Decision contract: ``decide(p, thresholds) -> (label, reason)``.
A ``DEFAULT_THRESHOLDS`` mapping (or similar) is exported alongside
``tune_thresholds(probs, labels, target_precision=...)``.

These tests are deterministic and offline. ``decide`` is a pure function of
the probability and threshold table; ``tune_thresholds`` is verified on a
perfectly separable synthetic validation set.
"""

import os
import sys

# Make the src-layout package importable without an installed wheel.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fakenews.decision import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    decide,
    tune_thresholds,
)


# ---------------------------------------------------------------------------
# decide — boundary behaviour at the three named probability bands
# ---------------------------------------------------------------------------

def _label_str(label) -> str:
    """Best-effort stringification of a Label enum / str / Literal."""
    return str(label).upper().replace("LABEL.", "").replace(".", "")


def test_decide_low_probability_returns_real():
    label, reason = decide(0.1, DEFAULT_THRESHOLDS)
    s = _label_str(label)
    assert "REAL" in s or "TRUE" in s, f"expected REAL for p=0.1, got {label!r}"


def test_decide_mid_probability_returns_needs_review():
    label, reason = decide(0.5, DEFAULT_THRESHOLDS)
    s = _label_str(label)
    assert "REVIEW" in s or "NEEDS" in s, (
        f"expected NEEDS_REVIEW for p=0.5, got {label!r}"
    )


def test_decide_high_probability_returns_fake():
    label, reason = decide(0.9, DEFAULT_THRESHOLDS)
    s = _label_str(label)
    assert "FAKE" in s, f"expected FAKE for p=0.9, got {label!r}"


def test_decide_returns_nonempty_reason():
    label, reason = decide(0.5, DEFAULT_THRESHOLDS)
    assert isinstance(reason, str)
    assert len(reason) > 0


def test_decide_monotone_in_probability():
    """As probability rises, the verdict must not regress from FAKE to REAL."""
    verdicts = [decide(p, DEFAULT_THRESHOLDS)[0] for p in (0.05, 0.2, 0.4, 0.6, 0.8, 0.95)]
    s = [_label_str(v) for v in verdicts]
    # The first must be REAL, the last must be FAKE.
    assert "REAL" in s[0]
    assert "FAKE" in s[-1]
    # No REAL verdict should appear after the first FAKE verdict.
    seen_fake = False
    for v in s:
        if "FAKE" in v:
            seen_fake = True
        if seen_fake and "REAL" in v:
            raise AssertionError(
                f"Non-monotone decision: REAL appears after FAKE in {s}"
            )


# ---------------------------------------------------------------------------
# tune_thresholds — respects target precision on a separable synthetic set
# ---------------------------------------------------------------------------

def _extract_thresholds(tuned):
    """Accept dict, namedtuple, or tuple of (lower, upper) thresholds."""
    if isinstance(tuned, dict):
        upper = (
            tuned.get("upper_threshold")
            or tuned.get("upper")
            or tuned.get("fake_threshold")
            or tuned.get("high")
        )
        lower = (
            tuned.get("lower_threshold")
            or tuned.get("lower")
            or tuned.get("real_threshold")
            or tuned.get("low")
        )
        return lower, upper
    if hasattr(tuned, "upper_threshold"):
        return tuned.lower_threshold, tuned.upper_threshold
    if hasattr(tuned, "_fields"):
        fields = tuned._fields
        idx_low = fields.index("lower_threshold") if "lower_threshold" in fields else 0
        idx_high = fields.index("upper_threshold") if "upper_threshold" in fields else 1
        return tuned[idx_low], tuned[idx_high]
    # Tuple / list fallback: (lower, upper)
    if isinstance(tuned, (tuple, list)) and len(tuned) >= 2:
        return tuned[0], tuned[1]
    raise AssertionError(f"Cannot extract thresholds from {tuned!r}")


def _precision_at_upper(probs, labels, upper):
    """Precision of the FAKE class when thresholding at `upper`."""
    pred_fake = [1 if p >= upper else 0 for p in probs]
    tp = sum(1 for p, l in zip(pred_fake, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(pred_fake, labels) if p == 1 and l == 0)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def test_tune_thresholds_returns_threshold_object():
    probs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.70, 0.80, 0.85, 0.90, 0.95]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    tuned = tune_thresholds(probs, labels, target_precision=0.9)
    assert tuned is not None
    lower, upper = _extract_thresholds(tuned)
    assert 0.0 <= lower <= 1.0
    assert 0.0 <= upper <= 1.0
    assert lower <= upper


def test_tune_thresholds_respects_target_precision_on_separable_set():
    """On a perfectly separable synthetic set, tuned upper threshold must
    achieve precision >= target_precision (1.0 here, target 0.9 leaves
    headroom) while still flagging at least one FAKE.
    """
    probs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.70, 0.80, 0.85, 0.90, 0.95]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    tuned = tune_thresholds(probs, labels, target_precision=0.9)
    lower, upper = _extract_thresholds(tuned)

    # Upper threshold must lie strictly between the two classes so that
    # the FAKE band actually contains any positives.
    assert max(p for p, l in zip(probs, labels) if l == 0) < upper, (
        f"upper={upper} must exceed the highest REAL prob"
    )
    assert upper <= min(p for p, l in zip(probs, labels) if l == 1), (
        f"upper={upper} must not exceed the lowest FAKE prob (else zero recall)"
    )
    precision = _precision_at_upper(probs, labels, upper)
    assert precision >= 0.9, (
        f"tuned precision {precision:.3f} below target 0.9 (upper={upper})"
    )


def test_tune_thresholds_low_target_precision_allows_more_positives():
    """A lower precision target should yield at least as many flagged FAKES
    as a higher target (threshold non-increasing in target)."""
    probs = [0.05, 0.10, 0.30, 0.45, 0.50, 0.55, 0.65, 0.85, 0.90, 0.95]
    labels = [0, 0, 0, 0, 1, 0, 1, 1, 1, 1]  # overlap zone in the middle
    high_target = tune_thresholds(probs, labels, target_precision=0.95)
    low_target = tune_thresholds(probs, labels, target_precision=0.50)
    _, upper_high = _extract_thresholds(high_target)
    _, upper_low = _extract_thresholds(low_target)
    # Lower precision target → upper threshold cannot be stricter
    # (i.e. upper_low <= upper_high). Equality is acceptable.
    assert upper_low <= upper_high + 1e-9, (
        f"upper_low={upper_low} should be <= upper_high={upper_high}"
    )
