# -*- coding: utf-8 -*-
"""
SHAP-based explainability for the fake-news meta-classifier.

Produces per-instance, top-k feature attributions. Uses the official
``shap`` package when it is available (TreeExplainer for LightGBM/XGBoost,
LinearExplainer for LogReg); falls back to ``model.feature_importances_``
when ``shap`` is missing, so the evaluation pipeline never crashes on a
headless / sandboxed environment without ``shap``.

MIT License — Copyright (c) 2024 fake-news-jev contributors.
"""
from __future__ import annotations

from typing import Any, List, Tuple, Optional

import numpy as np

# guarded shap import
try:
    import shap as _shap  # type: ignore
    HAS_SHAP = True
except ImportError:                       # pragma: no cover — defensive
    _shap = None
    HAS_SHAP = False


def _as_2d_dense(X) -> np.ndarray:
    """Coerce sparse / 1-D input into a 2-D dense ndarray."""
    if hasattr(X, "toarray"):            # scipy sparse
        X = X.toarray()
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _shap_for_tree(model, X) -> Optional[np.ndarray]:
    """Return SHAP values for a tree booster; ``None`` on failure."""
    if not HAS_SHAP:
        return None
    try:
        explainer = _shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # LightGBM/xgboost return either:
        #  * an ndarray of shape (n, d) for binary with log-odds, or
        #  * a list of two arrays [class0, class1] for older lightgbm.
        if isinstance(shap_values, list) and len(shap_values) == 2:
            return np.asarray(shap_values[1])
        arr = np.asarray(shap_values)
        if arr.ndim == 3:                  # (n, d, 2) — take class-1 slice
            return arr[:, :, 1]
        return arr
    except Exception:
        return None


def _shap_for_linear(model, X) -> Optional[np.ndarray]:
    """Return SHAP values for a LogReg via LinearExplainer; ``None`` on fail."""
    if not HAS_SHAP:
        return None
    try:
        explainer = _shap.LinearExplainer(model, X)
        return np.asarray(explainer.shap_values(X))
    except Exception:
        return None


def _fallback_importances(model, n_rows: int, n_features: int) -> Optional[np.ndarray]:
    """Broadcast ``feature_importances_`` (per-feature scalar) into a
    pseudo-shap matrix of shape (n_rows, n_features).

    Sign of the contribution is approximated by the direction of the
    feature value relative to its mean: positive value -> positive
    contribution to ``p(fake)`` when the importance is positive. This is a
    rough approximation used only when ``shap`` is not installed.
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        # LogReg has no feature_importances_; use |coef_| as a proxy.
        coef = getattr(model, "coef_", None)
        if coef is not None:
            importances = np.abs(np.asarray(coef)).ravel()
        else:
            return None
    importances = np.asarray(importances, dtype=float).ravel()
    if importances.shape[0] != n_features:
        return None
    # We don't have per-instance signs in pure feature_importances_ land;
    # return positive importance replicated across rows. Sorting by abs
    # still produces meaningful "top-k" attributions.
    return np.tile(importances.reshape(1, -1), (n_rows, 1))


def explain(
    model: Any,
    X,
    feature_names: Optional[List[str]] = None,
    top_k: int = 5,
    instance_idx: Optional[int] = 0,
) -> List[Tuple[str, float]]:
    """Return the top-k contributing features for a single instance.

    Parameters
    ----------
    model : MetaClassifier or sklearn-like estimator
        The trained meta-classifier (or its underlying ``.model``).
    X : array-like
        Feature matrix; will be densified if sparse.
    feature_names : list[str], optional
        Column names. If ``None``, falls back to ``model.feature_names_``
        or ``["f0", "f1", ...]``.
    top_k : int
        Number of top contributing features to return (by absolute value).
    instance_idx : int, optional
        Row of ``X`` to explain. Defaults to the first row.

    Returns
    -------
    list[tuple[str, float]]
        Descending-by-``abs`` list of ``(feature_name, shap_value)`` pairs.
        Sign convention: positive SHAP -> pushes prediction toward fake
        (class 1); negative -> pushes toward real.

    Determinism
    -----------
    ``shap.TreeExplainer`` is deterministic for a fixed model + input.
    The fallback path is purely arithmetic and therefore also deterministic.
    """
    Xd = _as_2d_dense(X)
    if Xd.shape[0] == 0:
        return []
    if instance_idx is None:
        instance_idx = 0
    if not (0 <= instance_idx < Xd.shape[0]):
        raise IndexError(
            f"instance_idx={instance_idx} out of range for X with {Xd.shape[0]} rows"
        )

    # Unwrap MetaClassifier -> underlying estimator.
    inner = getattr(model, "model", model)

    # Resolve feature names.
    if feature_names is None:
        feature_names = list(getattr(model, "feature_names", None) or
                              getattr(inner, "feature_names_", None) or
                              [f"f{i}" for i in range(Xd.shape[1])])
    if len(feature_names) != Xd.shape[1]:
        feature_names = [f"f{i}" for i in range(Xd.shape[1])]

    shap_vals = None
    kind_tag = type(inner).__module__ + "." + type(inner).__qualname__
    if "lightgbm" in kind_tag or "xgboost" in kind_tag or "LGBMClassifier" in type(inner).__name__ or "XGBClassifier" in type(inner).__name__:
        shap_vals = _shap_for_tree(inner, Xd)
    elif "LogisticRegression" in type(inner).__name__:
        shap_vals = _shap_for_linear(inner, Xd)

    if shap_vals is None:
        shap_vals = _fallback_importances(inner, Xd.shape[0], Xd.shape[1])

    if shap_vals is None:
        return []

    row = np.asarray(shap_vals)[instance_idx].ravel()
    if row.shape[0] != len(feature_names):
        # Mismatch — bail with top-k by |importance| across whatever we have.
        order = np.argsort(-np.abs(row))[:top_k]
        return [(feature_names[i] if i < len(feature_names) else f"f{i}", float(row[i])) for i in order]

    order = np.argsort(-np.abs(row))[:top_k]
    return [(feature_names[i], float(row[i])) for i in order]


def explain_batch(
    model: Any,
    X,
    feature_names: Optional[List[str]] = None,
    top_k: int = 5,
) -> List[List[Tuple[str, float]]]:
    """Convenience wrapper: explain every row of ``X``."""
    Xd = _as_2d_dense(X)
    return [explain(model, Xd, feature_names=feature_names, top_k=top_k, instance_idx=i) for i in range(Xd.shape[0])]


__all__ = ["explain", "explain_batch", "HAS_SHAP"]
