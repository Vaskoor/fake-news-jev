# -*- coding: utf-8 -*-
"""
Meta-classifier for the fake-news-detection system (Stage 2).

This module wraps three backend learners behind a single, serialisable
``MetaClassifier`` interface:

  * ``lightgbm.LGBMClassifier``  (default, fast tree boosting)
  * ``xgboost.XGBClassifier``     (alternative boosting backend)
  * ``sklearn.linear_model.LogisticRegression`` (linear baseline)

The classifier consumes the hybrid feature matrix produced by
``fakenews.features.build_feature_matrix`` (Jev signals + TF-IDF + metadata)
and predicts ``p(fake)``.

A ``TFIDFOnlyBaseline`` (LogReg on the TF-IDF block only) is provided for
ablation comparisons, and ``train_meta`` is the convenience entry-point used
by ``scripts/train.py``.

MIT License — Copyright (c) 2024 fake-news-jev contributors.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import joblib

# --- guarded backend imports -------------------------------------------------
try:                                   # LightGBM is installed per spec.
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:                    # pragma: no cover — defensive
    HAS_LIGHTGBM = False

try:                                   # XGBoost is installed per spec.
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:                    # pragma: no cover — defensive
    HAS_XGBOOST = False

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


SEED = 42

# --- default hyperparameters -------------------------------------------------
DEFAULT_LGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.10,
    "reg_lambda": 0.10,
    "random_state": SEED,
    "n_jobs": -1,
    "verbosity": -1,
}

DEFAULT_XGB_PARAMS: Dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.10,
    "reg_lambda": 0.10,
    "random_state": SEED,
    "n_jobs": -1,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "use_label_encoder": False,
}

DEFAULT_LOGREG_PARAMS: Dict[str, Any] = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": SEED,
    "n_jobs": -1,
}


class MetaClassifier:
    """Uniform wrapper over LightGBM / XGBoost / LogReg meta-classifiers.

    Parameters
    ----------
    kind : {"lightgbm", "xgboost", "logreg"}
        Backend learner.
    config : dict, optional
        May carry hyper-parameter overrides under ``config["model"]["params"]``
        or directly under ``config["params"]``. All keys are merged on top of
        the ``DEFAULT_*_PARAMS`` table for the selected backend.

    Attributes
    ----------
    model : object
        The underlying estimator instance.
    kind : str
        Backend name.
    feature_names : Optional[list[str]]
        Feature names (populated by ``fit`` if X is a pandas frame).
    val_metrics : dict
        Validation metrics, populated by ``train_meta`` or by an explicit
        ``fit(..., eval_set=...)`` call.
    """

    def __init__(self, kind: str = "lightgbm", config: Optional[Dict[str, Any]] = None) -> None:
        if kind not in {"lightgbm", "xgboost", "logreg"}:
            raise ValueError(f"Unknown meta-classifier kind: {kind!r}")
        self.kind = kind
        self.config = config or {}
        self.feature_names: Optional[list] = None
        self.val_metrics: Dict[str, float] = {}
        self._build()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _resolve_params(self) -> Dict[str, Any]:
        """Pull hyper-parameter overrides from the config dict."""
        if "model" in self.config and isinstance(self.config["model"], dict):
            params = self.config["model"].get("params", {})
        else:
            params = self.config.get("params", {})
        return params

    def _build(self) -> None:
        params = self._resolve_params()
        if self.kind == "lightgbm":
            if not HAS_LIGHTGBM:
                raise ImportError(
                    "lightgbm is not available; install it or use --model logreg"
                )
            self.model = lgb.LGBMClassifier(**{**DEFAULT_LGB_PARAMS, **params})
        elif self.kind == "xgboost":
            if not HAS_XGBOOST:
                raise ImportError(
                    "xgboost is not available; install it or use --model logreg"
                )
            self.model = xgb.XGBClassifier(**{**DEFAULT_XGB_PARAMS, **params})
        else:  # logreg
            self.model = LogisticRegression(**{**DEFAULT_LOGREG_PARAMS, **params})

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------
    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        feature_names: Optional[list] = None,
    ) -> "MetaClassifier":
        """Fit the meta-classifier.

        If ``X_val`` / ``y_val`` are provided they are used as the early-stopping
        / eval set (where supported) and validation metrics are stored on
        ``self.val_metrics``.
        """
        y = np.asarray(y).astype(int)
        self.feature_names = feature_names

        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            y_val_arr = np.asarray(y_val).astype(int)
            if self.kind == "lightgbm":
                fit_kwargs["eval_set"] = [(X_val, y_val_arr)]
                fit_kwargs["callbacks"] = [lgb.early_stopping(20, verbose=False)]
            elif self.kind == "xgboost":
                fit_kwargs["eval_set"] = [(X_val, y_val_arr)]
                fit_kwargs["early_stopping_rounds"] = 20
                fit_kwargs["verbose"] = False
            # sklearn LogReg has no eval_set concept.

        self.model.fit(X, y, **fit_kwargs)
        self._classes = np.asarray(getattr(self.model, "classes_", np.array([0, 1])))

        if X_val is not None and y_val is not None:
            self.val_metrics = self._compute_metrics(X_val, np.asarray(y_val).astype(int))
        return self

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def predict_proba(self, X) -> np.ndarray:
        """Return probability matrix; ``[:, 1]`` is ``p(fake)``."""
        proba = self.model.predict_proba(X)
        # Guarantee two-column shape even for degenerate single-class fits.
        if proba.ndim == 1 or proba.shape[1] == 1:
            p1 = proba.ravel() if proba.ndim == 1 else proba[:, 0]
            proba = np.column_stack([1.0 - p1, p1])
        return proba

    def predict(self, X) -> np.ndarray:
        """Return argmax class labels (0 = real, 1 = fake)."""
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # metrics helper
    # ------------------------------------------------------------------
    def _compute_metrics(self, X, y) -> Dict[str, float]:
        y = np.asarray(y).astype(int)
        proba = self.predict_proba(X)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y, proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y, proba))
        except Exception:
            metrics["pr_auc"] = float("nan")
        return metrics

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist the wrapper via joblib. ``path`` should end with ``.joblib``."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "MetaClassifier":
        """Load a wrapper previously saved with :meth:`save`."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a MetaClassifier: {type(obj)!r}")
        return obj


class TFIDFOnlyBaseline:
    """Logistic-regression baseline trained on the TF-IDF block only.

    Used by ``scripts/evaluate.py`` for ablation: hybrid (Jev+TF-IDF+meta)
    vs. TF-IDF-only. Mirrors the :class:`MetaClassifier` surface so the
    evaluation harness can treat both uniformly.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        params = self.config.get("params", DEFAULT_LOGREG_PARAMS)
        self.model = LogisticRegression(**{**DEFAULT_LOGREG_PARAMS, **params})
        self.val_metrics: Dict[str, float] = {}

    def fit(self, X_tfidf, y, X_val=None, y_val=None) -> "TFIDFOnlyBaseline":
        y = np.asarray(y).astype(int)
        self.model.fit(X_tfidf, y)
        self._classes = np.asarray(getattr(self.model, "classes_", np.array([0, 1])))
        if X_val is not None and y_val is not None:
            self.val_metrics = self._compute_metrics(X_val, np.asarray(y_val).astype(int))
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model.predict_proba(X)
        if proba.ndim == 1 or proba.shape[1] == 1:
            p1 = proba.ravel() if proba.ndim == 1 else proba[:, 0]
            proba = np.column_stack([1.0 - p1, p1])
        return proba

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def _compute_metrics(self, X, y) -> Dict[str, float]:
        y = np.asarray(y).astype(int)
        proba = self.predict_proba(X)[:, 1]
        y_pred = (proba >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, proba)) if len(set(y)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y, proba)) if len(set(y)) > 1 else float("nan"),
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "TFIDFOnlyBaseline":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a TFIDFOnlyBaseline: {type(obj)!r}")
        return obj


def train_meta(
    X_train,
    y_train,
    X_val,
    y_val,
    config: Optional[Dict[str, Any]] = None,
    feature_names: Optional[list] = None,
) -> MetaClassifier:
    """Build and fit a :class:`MetaClassifier` from a config dict.

    Parameters
    ----------
    X_train, y_train : training data and binary labels (0 = real, 1 = fake).
    X_val, y_val : held-out validation data, used for early stopping & metrics.
    config : dict. ``config["model"]["meta_classifier"]`` selects the backend
        (``lightgbm`` | ``xgboost`` | ``logreg``); defaults to ``lightgbm``.
    feature_names : optional list of column names.

    Returns
    -------
    MetaClassifier
        The fitted wrapper; validation metrics are stashed on
        ``mc.val_metrics``.
    """
    config = config or {}
    model_cfg = config.get("model", {}) if isinstance(config.get("model", {}), dict) else {}
    kind = model_cfg.get("meta_classifier", "lightgbm")
    mc = MetaClassifier(kind=kind, config=config)
    mc.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_names)
    return mc


__all__ = [
    "MetaClassifier",
    "TFIDFOnlyBaseline",
    "train_meta",
    "DEFAULT_LGB_PARAMS",
    "DEFAULT_XGB_PARAMS",
    "DEFAULT_LOGREG_PARAMS",
    "SEED",
]
