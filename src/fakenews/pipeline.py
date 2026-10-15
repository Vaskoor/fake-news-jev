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
"""Fake News Detection — 3-stage pipeline orchestrator.

This module wires together the three stages of fake-news classification:

    Stage 1  signal extraction    one Jev call (``SignalExtractor``).
    Stage 2  feature build + classification + calibration + decision.
    Stage 3  evidence check (only for uncertain verdicts — adds latency).

The fast path (Stages 1 + 2, no evidence check) returns well under one
second with the simulated Jev backend. Stage 3 is invoked only when the
calibrated probability falls inside the configured "review band"
(``decision.evidence_check.low`` .. ``decision.evidence_check.high``) so
the steady-state cost stays low.

Self-bootstrapping: if no trained ``MetaClassifier`` artefact is on disk
yet (i.e. fn-C has not produced a model artefact), the orchestrator falls
back to a deterministic ``RuleBasedFallback`` scorer that combines the
Jev NOUL signals with config-supplied weights. The same is true if the
contract modules shipped by the parallel fn-A / fn-B / fn-C tasks are
not yet importable: local stand-in dataclasses and a heuristic
``FallbackSignalExtractor`` keep the system runnable end-to-end.

Public surface:

    FakeNewsPipeline        the orchestrator
    RuleBasedFallback       weighted-sum scorer used pre-training
    FallbackSignalExtractor heuristic signal extractor (pre-fn-A)
    build_default_pipeline  convenience factory
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Contract imports (fn-A / fn-B / fn-C). All guarded so this module loads
# cleanly even when the parallel tasks have not yet shipped their exports.
# ---------------------------------------------------------------------------
try:  # fn-A: types
    from fakenews.types import (  # type: ignore
        AnswerType,
        Article,
        Evidence,
        Signal,
        Verdict,
    )
    _HAS_TYPES = True
except ImportError:  # pragma: no cover - parallel-task fallback
    _HAS_TYPES = False
    from enum import Enum

    class AnswerType(str, Enum):  # type: ignore[no-redef]
        NOUL = "noul"
        SCORE = "score"
        CHOICE = "choice"

    @dataclass
    class Article:  # type: ignore[no-redef]
        headline: str = ""
        body: str = ""
        source: str = ""
        url: str = ""
        date: str = ""

    @dataclass
    class Signal:  # type: ignore[no-redef]
        name: str = ""
        value: Any = 0.0
        probability: float = 0.5
        type: "AnswerType" = AnswerType.NOUL  # type: ignore[assignment]
        contribution: float = 0.0

    @dataclass
    class Evidence:  # type: ignore[no-redef]
        claims: list = field(default_factory=list)
        findings: list = field(default_factory=list)
        adjusted_probability: Optional[float] = None
        source: str = ""

    @dataclass
    class Verdict:  # type: ignore[no-redef]
        label: str = "NEEDS_REVIEW"
        probability: float = 0.5
        signals: list = field(default_factory=list)
        explanation: str = ""
        evidence: Any = None
        latency_ms: float = 0.0
        model_version: str = ""

try:  # fn-A: signal extractor
    from fakenews.signal_extractor import SignalExtractor  # type: ignore
    _HAS_SIGNAL_EXTRACTOR = True
except ImportError:  # pragma: no cover
    SignalExtractor = None  # type: ignore
    _HAS_SIGNAL_EXTRACTOR = False

# fn-A: Jev client factory. The exact module name is not fixed by the spec;
# try a few common candidates before giving up.
make_jev_client: Optional[Callable[[dict], Any]] = None
for _mod_name in ("fakenews.jev_client", "fakenews.client", "fakenews.jev"):
    try:
        _mod = __import__(_mod_name, fromlist=["make_jev_client"])
        make_jev_client = getattr(_mod, "make_jev_client", None)  # type: ignore[assignment]
        if make_jev_client is not None:
            break
    except ImportError:
        continue

try:  # fn-B: feature engineering (optional; not all deployments ship it)
    from fakenews.features import build_feature_matrix, extract_jev_vector  # type: ignore
    _HAS_FEATURES = True
except ImportError:  # pragma: no cover
    build_feature_matrix = None  # type: ignore
    extract_jev_vector = None  # type: ignore
    _HAS_FEATURES = False

# fn-C: classifier / calibrator / explainer / decider / evidence checker.
MetaClassifier: Any = None
Calibrator: Any = None
explain: Optional[Callable[..., Any]] = None
decide: Optional[Callable[[float, dict], Tuple[str, str]]] = None
EvidenceChecker: Any = None
_HAS_CLASSIFIER = False
for _mod_name in ("fakenews.meta_classifier", "fakenews.classifier", "fakenews.model"):
    try:
        _mod = __import__(_mod_name, fromlist=["MetaClassifier"])
        MetaClassifier = getattr(_mod, "MetaClassifier", MetaClassifier)
        _HAS_CLASSIFIER = True
        break
    except ImportError:
        continue
for _mod_name in ("fakenews.calibrate", "fakenews.calibration"):
    try:
        _mod = __import__(_mod_name, fromlist=["Calibrator"])
        Calibrator = getattr(_mod, "Calibrator", Calibrator)
        break
    except ImportError:
        continue
for _mod_name in ("fakenews.decision", "fakenews.policy"):
    try:
        _mod = __import__(_mod_name, fromlist=["decide"])
        decide = getattr(_mod, "decide", decide)
        break
    except ImportError:
        continue
for _mod_name in ("fakenews.explain", "fakenews.explainer"):
    try:
        _mod = __import__(_mod_name, fromlist=["explain"])
        explain = getattr(_mod, "explain", explain)
        break
    except ImportError:
        continue
for _mod_name in ("fakenews.evidence_check", "fakenews.evidence"):
    try:
        _mod = __import__(_mod_name, fromlist=["EvidenceChecker"])
        EvidenceChecker = getattr(_mod, "EvidenceChecker", EvidenceChecker)
        break
    except ImportError:
        continue

try:  # shared config + logging
    from fakenews.config import get_logger, load_config, load_questions  # type: ignore
except ImportError:  # pragma: no cover
    import logging

    _LOGGER = logging.getLogger("fakenews")

    def get_logger(name: str, level: str = "INFO", fmt: str = "json") -> logging.Logger:
        return _LOGGER

    def load_config() -> dict:
        return {}

    def load_questions() -> list:  # type: ignore[return-value]
        return []

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Rule-based fallback classifier
# ---------------------------------------------------------------------------
class RuleBasedFallback:
    """Deterministic scorer used when no trained ``MetaClassifier`` is on disk.

    Computes ``p_fake`` as a weighted sum of the Jev NOUL signals (the spec's
    "weighted sum of Jev noul signals"), squashed through a logistic so the
    result lands in ``[0, 1]``. Weights are pulled from
    ``config['fallback']['weights']`` (keyed by question name); signals
    without an explicit weight use ``config['fallback']['default_weight']``
    (default 0.15). A configurable bias shifts the decision boundary.

    The class intentionally mirrors the public method
    ``MetaClassifier.predict_proba`` so the pipeline can treat both
    interchangeably. The fallback also exposes ``feature_importances_`` so
    the SHAP-style explain path in ``fakenews.explain`` can fall back to
    per-feature importances without crashing.
    """

    def __init__(self, config: dict) -> None:
        self.config = config or {}
        fb_cfg = self.config.get("fallback", {}) or {}
        self.weights: dict[str, float] = {
            str(k): float(v) for k, v in (fb_cfg.get("weights", {}) or {}).items()
        }
        self.default_weight: float = float(fb_cfg.get("default_weight", 0.15))
        self.bias: float = float(fb_cfg.get("bias", -0.10))
        # Used by explain.py's fallback path (broadcast importances).
        self.feature_importances_: Optional[list[float]] = None

    # -- sklearn-like surface --------------------------------------------
    def fit(self, *_args: Any, **_kwargs: Any) -> "RuleBasedFallback":
        """No-op fit so the fallback can be swapped into a training script."""
        return self

    def predict_proba(self, X: Any) -> Any:
        """Return ``p_fake`` for a single article.

        Accepts either:

        * a list of :class:`Signal` objects (preferred — uses raw signals), or
        * a 2-row / 1-row numeric matrix (used when the pipeline hands the
          fallback a feature matrix built by ``build_feature_matrix``).

        Returns a 2-column ``[[p_real, p_fake]]`` shape so sklearn-style
        ``proba[0, 1]`` indexing keeps working downstream.
        """
        if isinstance(X, dict) and "signals" in X:
            p_fake = self._score_signals(X["signals"])
        elif (
            isinstance(X, (list, tuple))
            and X
            and hasattr(X[0], "name")
            and hasattr(X[0], "value")
        ):
            p_fake = self._score_signals(X)
        else:
            # Numeric matrix path: sum features weighted by index, then squash.
            try:
                row = X[0] if (hasattr(X, "__getitem__") and len(X) > 0) else X
                values = list(row)
                z = self.bias
                for i, v in enumerate(values):
                    w = self.weights.get(f"f{i}", self.default_weight)
                    z += w * float(v)
                p_fake = self._sigmoid(z)
            except Exception:
                p_fake = 0.5
        # Mimic sklearn's 2-column shape: [p_real, p_fake].
        return [[1.0 - p_fake, p_fake]]

    # -- internals --------------------------------------------------------
    # NOUL signals where a HIGH value means LESS fake (anti-fake indicators).
    _ANTI_FAKE_NOULS = {"cites_verifiable_sources"}

    def _score_signals(self, signals: Sequence[Any]) -> float:
        z = self.bias
        importances: list[float] = []
        for s in signals:
            stype = getattr(s, "type", None)
            # Only NOUL signals (yes-probabilities in [0,1]) contribute to the
            # rule-based score; SCORE / CHOICE signals are skipped (the spec
            # literally says "weighted sum of Jev noul signals").
            if _answer_type_value(stype) != "noul":
                importances.append(0.0)
                continue
            try:
                val = float(getattr(s, "value", 0.0) or 0.0)
            except (TypeError, ValueError):
                val = 0.0
            name = getattr(s, "name", "") or ""
            w = float(self.weights.get(name, self.default_weight))
            # Anti-fake signals (high value = real) SUBTRACT from the fake score.
            if name in self._ANTI_FAKE_NOULS:
                z -= w * val
            else:
                z += w * val
            # Per-signal "importance" for the explain fallback: |w * val|.
            importances.append(abs(w * val))
        self.feature_importances_ = importances
        return self._sigmoid(z)

    @staticmethod
    def _sigmoid(z: float) -> float:
        try:
            if z >= 0:
                ez = math.exp(-z)
                return 1.0 / (1.0 + ez)
            ez = math.exp(z)
            return ez / (1.0 + ez)
        except OverflowError:
            return 1.0 if z > 0 else 0.0


# ---------------------------------------------------------------------------
# Fallback signal extractor (used when fn-A's SignalExtractor is unavailable)
# ---------------------------------------------------------------------------
class FallbackSignalExtractor:
    """Heuristic signal extractor for pre-fn-A / no-Jev-key operation.

    Produces a small set of lexical + stylistic NOUL signals straight from
    the article text so the pipeline can be exercised end-to-end without
    the Jev backend. Real Jev signals (semantic embedding, stance,
    credibility) are added once ``fakenews.signal_extractor`` is importable
    and ``make_jev_client`` produces a real client.
    """

    SENSATIONAL_TERMS: tuple[str, ...] = (
        "shocking", "exposed", "secret", "they don't want you to know",
        "breaking", "you won't believe", "doctors hate", "one weird trick",
        "mainstream media won't", "wake up",
    )

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    async def extract(self, article: Any) -> list[Any]:
        body = (getattr(article, "body", "") or "").lower()
        headline = (getattr(article, "headline", "") or "").lower()
        text = f"{headline} {body}"
        signals: list[Any] = []

        # Sensational-lexicon count.
        n_sens = sum(1 for term in self.SENSATIONAL_TERMS if term in text)
        signals.append(self._mk("sensational_count", float(n_sens), 0.7))

        # Headline ALL-CAPS ratio.
        caps = sum(1 for c in headline if c.isupper())
        letters = max(sum(1 for c in headline if c.isalpha()), 1)
        signals.append(self._mk("headline_caps_ratio", round(caps / letters, 4), 0.6))

        # Exclamation density.
        n_excl = body.count("!")
        words = max(len(body.split()), 1)
        signals.append(self._mk("exclamation_density", round(n_excl / words, 6), 0.5))

        # Body length signal.
        signals.append(self._mk("body_length", float(len(body)), 0.5))

        # Hedging / uncertainty.
        hedge_terms = ("allegedly", "reportedly", "rumour", " rumor", "some say", " sources say")
        n_hedge = sum(1 for term in hedge_terms if term in text)
        signals.append(self._mk("hedging_count", float(n_hedge), 0.6))

        # Source presence.
        source = getattr(article, "source", "") or ""
        signals.append(self._mk("has_source", 1.0 if source.strip() else 0.0, 0.4))

        return signals

    @staticmethod
    def _mk(name: str, value: float, prob: float) -> Any:
        if _HAS_TYPES:
            return Signal(
                name=name,
                value=value,
                probability=prob,
                type=AnswerType.NOUL,
                contribution=0.0,
            )
        # Fallback Signal dataclass.
        return Signal(
            name=name,
            value=value,
            probability=prob,
            type=AnswerType.NOUL,
            contribution=0.0,
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class FakeNewsPipeline:
    """3-stage fake-news classifier.

    Stages
    ------
    1. **Signal extraction** — ``await self.extractor.extract(article)``.
       With the simulated Jev backend this completes well under one second.
    2. **Classification + calibration + decision** — build the feature
       vector (Jev vector + metadata; TF-IDF only if the trained model used
       it), score with ``MetaClassifier.predict_proba`` (or the rule-based
       fallback), calibrate, then decide the label.
    3. **Evidence check (uncertain only)** — when the calibrated probability
       sits in the review band, an optional ``EvidenceChecker`` runs against
       the article and nudges the probability; verdicts that remain
       ambiguous are reported as ``NEEDS_REVIEW``.

    Parameters
    ----------
    config : dict
        Pipeline configuration. Reads ``config['decision']`` (thresholds +
        evidence_check) and ``config['jev']`` (provider). Falls back to
        sane defaults if absent.
    model : optional
        Pre-instantiated ``MetaClassifier`` (or compatible). If ``None`` the
        constructor tries to load one from ``config['model']['path']`` and
        falls back to :class:`RuleBasedFallback` if no artefact is found.
    calibrator : optional
        Pre-instantiated ``Calibrator``. If ``None`` the constructor tries
        to load one from ``config['calibrator']['path']``.
    jev_client : optional
        Pre-built Jev client. If ``None`` the constructor builds one via
        ``make_jev_client(config)`` (when available).
    evidence_checker : optional
        Pre-instantiated ``EvidenceChecker``. If ``None`` and
        ``config['decision']['evidence_check']['enabled']`` is truthy, one
        is built.
    questions : optional
        Pre-loaded canonical question list (``list[Question]``). If
        ``None`` the constructor loads them via ``load_questions()``.
    """

    DEFAULT_THRESHOLDS: dict[str, float] = {
        "real_max": 0.30,
        "review_min": 0.30,
        "fake_min": 0.70,
    }
    DEFAULT_REVIEW_RANGE: tuple[float, float] = (0.30, 0.70)

    def __init__(
        self,
        config: dict,
        model: Optional[Any] = None,
        calibrator: Optional[Any] = None,
        jev_client: Optional[Any] = None,
        evidence_checker: Optional[Any] = None,
        questions: Optional[list] = None,
    ) -> None:
        self.config: dict = config or {}
        self.logger = get_logger("fakenews.pipeline.FakeNewsPipeline")

        # -- Thresholds + decision config --------------------------------
        decision_cfg = self.config.get("decision", {}) or {}
        self.thresholds: dict[str, float] = {
            "real_max": float(decision_cfg.get("real_max", self.DEFAULT_THRESHOLDS["real_max"])),
            "review_min": float(decision_cfg.get("review_min", self.DEFAULT_THRESHOLDS["review_min"])),
            "fake_min": float(decision_cfg.get("fake_min", self.DEFAULT_THRESHOLDS["fake_min"])),
        }
        ec_cfg = decision_cfg.get("evidence_check", {}) or {}
        self.evidence_check_cfg: dict = ec_cfg
        self.review_low: float = float(ec_cfg.get("low", self.thresholds["real_max"]))
        self.review_high: float = float(ec_cfg.get("high", self.thresholds["fake_min"]))

        # -- Fallback weights (used by explain path too) -----------------
        fb_cfg = self.config.get("fallback", {}) or {}
        self.fallback_weights: dict[str, float] = {
            str(k): float(v) for k, v in (fb_cfg.get("weights", {}) or {}).items()
        }
        self.fallback_default_weight: float = float(fb_cfg.get("default_weight", 0.15))

        # -- Model --------------------------------------------------------
        self.model: Any = model
        if self.model is None:
            self.model = self._load_model(self.config)
        self.using_fallback: bool = isinstance(self.model, RuleBasedFallback)
        if self.using_fallback:
            self.logger.info("pipeline.model.fallback", extra={
                "reason": "no trained MetaClassifier available; using RuleBasedFallback",
            })

        # -- Calibrator --------------------------------------------------
        self.calibrator: Optional[Any] = calibrator
        if self.calibrator is None:
            self.calibrator = self._load_calibrator(self.config)

        # -- Jev client --------------------------------------------------
        self.jev_client: Optional[Any] = jev_client
        if self.jev_client is None and make_jev_client is not None:
            try:
                self.jev_client = make_jev_client(self.config)
            except Exception as exc:  # pragma: no cover - depends on fn-A
                self.logger.warning("pipeline.jev.client_failed", extra={"error": str(exc)})
                self.jev_client = None
        self.jev_provider: str = self._resolve_jev_provider()

        # -- Questions + signal extractor -------------------------------
        self.questions: list = list(questions) if questions else self._load_questions()
        self.extractor: Any = self._build_extractor()

        # -- Evidence checker -------------------------------------------
        self.evidence_checker: Optional[Any] = evidence_checker
        if (
            self.evidence_checker is None
            and ec_cfg.get("enabled", True)
            and EvidenceChecker is not None
        ):
            try:
                self.evidence_checker = EvidenceChecker(self.config)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("pipeline.evidence.disabled", extra={"error": str(exc)})
                self.evidence_checker = None

        # -- Model version tag -----------------------------------------
        self.model_version: str = self._compute_model_version()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _load_model(self, config: dict) -> Any:
        """Try to load a trained ``MetaClassifier`` from ``config['model']['path']``."""
        model_cfg = config.get("model", {}) or {}
        path = model_cfg.get("path")
        if not path or not isinstance(path, str) or not os.path.exists(path):
            return RuleBasedFallback(config)
        # Prefer fn-C's MetaClassifier.load(path) classmethod when available.
        if MetaClassifier is not None and hasattr(MetaClassifier, "load"):
            try:
                obj = MetaClassifier.load(path)  # type: ignore[attr-defined]
                if hasattr(obj, "predict_proba"):
                    return obj
            except Exception as exc:
                self.logger.warning("pipeline.model.load_failed", extra={
                    "path": path, "error": str(exc),
                })
                return RuleBasedFallback(config)
        # Generic joblib/pickle fallback.
        loader = None
        try:
            import joblib  # type: ignore
            loader = joblib.load
        except ImportError:
            import pickle
            loader = pickle.load  # type: ignore[assignment]
        try:
            obj = loader(path)
        except Exception as exc:
            self.logger.warning("pipeline.model.load_failed", extra={
                "path": path, "error": str(exc),
            })
            return RuleBasedFallback(config)
        if hasattr(obj, "predict_proba"):
            return obj
        self.logger.warning("pipeline.model.shape_unknown", extra={"path": path})
        return RuleBasedFallback(config)

    def _load_calibrator(self, config: dict) -> Optional[Any]:
        cal_cfg = config.get("calibrator", {}) or {}
        path = cal_cfg.get("path")
        if not path or not isinstance(path, str) or not os.path.exists(path):
            return None
        if Calibrator is not None and hasattr(Calibrator, "load"):
            try:
                obj = Calibrator.load(path)  # type: ignore[attr-defined]
                return obj
            except Exception as exc:
                self.logger.warning("pipeline.calibrator.load_failed", extra={
                    "path": path, "error": str(exc),
                })
                return None
        loader = None
        try:
            import joblib  # type: ignore
            loader = joblib.load
        except ImportError:
            import pickle
            loader = pickle.load  # type: ignore[assignment]
        try:
            obj = loader(path)
        except Exception as exc:
            self.logger.warning("pipeline.calibrator.load_failed", extra={
                "path": path, "error": str(exc),
            })
            return None
        if hasattr(obj, "calibrate") or hasattr(obj, "predict"):
            return obj
        return None

    def _load_questions(self) -> list:
        try:
            qs = load_questions()
            if qs:
                return list(qs)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("pipeline.questions.load_failed", extra={"error": str(exc)})
        return []

    def _resolve_jev_provider(self) -> str:
        jev_cfg = self.config.get("jev", {}) or {}
        provider = jev_cfg.get("provider") or jev_cfg.get("backend") or "unknown"
        if self.jev_client is not None:
            provider = getattr(self.jev_client, "provider", None) or provider
            cls = type(self.jev_client).__name__
            if cls == "JevSimulated":
                provider = "simulated"
        if not _HAS_SIGNAL_EXTRACTOR:
            provider = f"{provider}+fallback-extractor"
        return str(provider)

    def _build_extractor(self) -> Any:
        """Instantiate a real ``SignalExtractor`` when fn-A is available."""
        if (
            _HAS_SIGNAL_EXTRACTOR
            and SignalExtractor is not None
            and self.jev_client is not None
            and self.questions
        ):
            for ctor in (
                lambda: SignalExtractor(self.jev_client, self.questions),  # type: ignore[misc]
                lambda: SignalExtractor(client=self.jev_client, questions=self.questions),  # type: ignore[misc]
            ):
                try:
                    return ctor()
                except TypeError:
                    continue
        # Fall back to the local heuristic extractor.
        self.logger.info("pipeline.extractor.fallback", extra={
            "reason": "SignalExtractor unavailable or no Jev client; using FallbackSignalExtractor",
        })
        return FallbackSignalExtractor(self.config)

    def _compute_model_version(self) -> str:
        jev_tag = self.jev_provider.split("+", 1)[0]
        if self.using_fallback:
            return f"jev-{jev_tag}+rule-fallback-v1"
        kind = getattr(self.model, "kind", None) or type(self.model).__name__
        return f"jev-{jev_tag}+{kind}-v1"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def predict(self, article: Any) -> Any:
        """Run the full 3-stage pipeline for a single article.

        Parameters
        ----------
        article : Article
            Article to classify. Any object with ``headline`` / ``body`` /
            ``source`` attributes works; the fallback Article dataclass is
            fine.

        Returns
        -------
        Verdict
            ``label`` (REAL | FAKE | NEEDS_REVIEW), ``probability`` (final,
            post-evidence), ``signals`` (top-k for explainability),
            ``explanation``, ``evidence`` (None on the fast path),
            ``latency_ms``, ``model_version``.
        """
        if isinstance(article, str):
            from fakenews.types import Article
            article = Article(headline=article[:120], body=article)
        t0 = time.perf_counter()

        # Stage 1: signal extraction (one Jev call, <1s simulated) --------
        signals = await self._stage1_extract(article)

        # Stage 2: feature build + classification + calibration + decision -
        p_fake, feat_names = self._stage2_classify(article, signals)
        p_cal = self._calibrate(p_fake)
        label, reason = self._decide(p_cal)

        # Stage 3: evidence check (uncertain only) ------------------------
        evidence: Optional[Any] = None
        p_final = p_cal
        if self.evidence_checker is not None and self._should_check_evidence(p_cal):
            try:
                evidence = await self._run_evidence_check(article)
                if evidence is not None:
                    p_final = self._adjust_probability(p_cal, evidence)
                    if self._is_uncertain(p_final):
                        label = "NEEDS_REVIEW"
                        reason = (
                            "uncertain after evidence check "
                            f"(p_final={p_final:.3f} inside review band)"
                        )
            except Exception as exc:  # pragma: no cover - depends on fn-C
                self.logger.warning("pipeline.evidence.error", extra={"error": str(exc)})
                evidence = None

        # Explainability --------------------------------------------------
        top_signals = self._explain(article, signals, feat_names, p_final)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return self._build_verdict(
            label=label,
            probability=p_final,
            top_signals=top_signals,
            reason=reason,
            evidence=evidence,
            latency_ms=latency_ms,
        )

    async def predict_batch(self, articles: Sequence[Any]) -> list[Any]:
        """Classify a batch of articles concurrently.

        Uses :func:`asyncio.gather` so the Jev backend can overlap I/O when
        the underlying client supports it. Failures on a single article do
        not abort the batch.
        """
        async def _safe(a: Any) -> Any:
            try:
                return await self.predict(a)
            except Exception as exc:  # pragma: no cover
                self.logger.warning("pipeline.batch.item_failed", extra={"error": str(exc)})
                return self._error_verdict(str(exc))
        return list(await asyncio.gather(*[_safe(a) for a in articles]))

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------
    async def _stage1_extract(self, article: Any) -> list[Any]:
        """Stage 1: extract Jev signals for the article."""
        try:
            signals = await self.extractor.extract(article)
        except Exception as exc:  # pragma: no cover - depends on fn-A
            self.logger.warning("pipeline.stage1.extract_failed", extra={"error": str(exc)})
            signals = []
        return list(signals or [])

    def _stage2_classify(
        self, article: Any, signals: Sequence[Any]
    ) -> Tuple[float, list[str]]:
        """Stage 2: build features and run the classifier.

        Returns the raw (pre-calibration) ``p_fake`` plus the feature names
        (used later by the explainer).
        """
        # Build feature matrix if fn-B is available.
        X: Any = None
        feat_names: list[str] = []
        if (
            not self.using_fallback
            and _HAS_FEATURES
            and build_feature_matrix is not None
        ):
            try:
                X, feat_names = build_feature_matrix([article], signals_list=[signals])
            except Exception as exc:  # pragma: no cover - depends on fn-B
                self.logger.warning("pipeline.stage2.feature_build_failed", extra={"error": str(exc)})
                X = None

        if X is None:
            # Fallback feature vector = stacked signal values (numeric only).
            X, feat_names = self._fallback_features(signals)

        # Score.
        if self.using_fallback:
            # RuleBasedFallback works best with raw signals.
            p_fake = float(self.model.predict_proba(list(signals))[0][1])
        else:
            p_fake = self._model_predict_proba(X)
        return float(p_fake), list(feat_names)

    def _model_predict_proba(self, X: Any) -> float:
        """Call the trained model's ``predict_proba`` and extract ``p_fake``.

        Handles 2-row (sklearn ``[n_samples, n_classes]``) and 1-row return
        shapes transparently.
        """
        try:
            proba = self.model.predict_proba(X)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("pipeline.stage2.model_failed", extra={"error": str(exc)})
            return 0.5
        # 2D numpy / list-of-lists shape: take [0, 1].
        try:
            shape = getattr(proba, "shape", None)
            if shape is not None and len(tuple(shape)) == 2:
                return float(proba[0][1])
            if shape is not None and len(tuple(shape)) == 1:
                return float(proba[1]) if shape[0] > 1 else float(proba[0])
        except Exception:
            pass
        # list / tuple.
        if isinstance(proba, (list, tuple)):
            if len(proba) and isinstance(proba[0], (list, tuple)):
                return float(proba[0][1])
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        # scalar fallback.
        try:
            return float(proba)
        except (TypeError, ValueError):
            return 0.5

    def _calibrate(self, p_fake: float) -> float:
        """Apply the calibrator (if loaded) to a raw probability."""
        if self.calibrator is None:
            return float(p_fake)
        try:
            # Calibrator.calibrate takes an array; .predict is the sklearn alias.
            if hasattr(self.calibrator, "calibrate"):
                out = self.calibrator.calibrate([p_fake])
            else:
                out = self.calibrator.predict([p_fake])
            # out is typically a 1-element numpy array / list.
            try:
                p_cal = float(out[0])  # type: ignore[index]
            except (TypeError, IndexError, ValueError):
                p_cal = float(out)
            return max(0.0, min(1.0, p_cal))
        except Exception as exc:  # pragma: no cover
            self.logger.warning("pipeline.stage2.calibrate_failed", extra={"error": str(exc)})
            return float(p_fake)

    def _decide(self, p: float) -> Tuple[str, str]:
        """Map a calibrated probability to a (label, reason) pair."""
        if decide is not None:
            try:
                label, reason = decide(p, self.thresholds)
                return str(label), str(reason)
            except Exception:  # pragma: no cover
                pass
        if p < self.thresholds["real_max"]:
            return "REAL", f"p={p:.3f} < real_max {self.thresholds['real_max']:.3f}"
        if p >= self.thresholds["fake_min"]:
            return "FAKE", f"p={p:.3f} >= fake_min {self.thresholds['fake_min']:.3f}"
        return "NEEDS_REVIEW", (
            f"p={p:.3f} inside review band "
            f"[{self.review_low:.3f}, {self.review_high:.3f})"
        )

    def _should_check_evidence(self, p: float) -> bool:
        """Return ``True`` when Stage 3 should run for this probability."""
        if not self.evidence_check_cfg.get("enabled", True):
            return False
        # Prefer EvidenceChecker.should_check (knows about low/high) when available.
        if self.evidence_checker is not None and hasattr(self.evidence_checker, "should_check"):
            try:
                return bool(self.evidence_checker.should_check(p))
            except Exception:  # pragma: no cover
                pass
        return self.review_low <= p < self.review_high

    def _is_uncertain(self, p: float) -> bool:
        return self.review_low <= p < self.review_high

    async def _run_evidence_check(self, article: Any) -> Optional[Any]:
        """Stage 3 orchestration: claims → fact-check → Evidence."""
        ec = self.evidence_checker
        if ec is None:
            return None
        try:
            claims = ec.extract_claims(article)
        except Exception as exc:  # pragma: no cover
            self.logger.warning("pipeline.evidence.claims_failed", extra={"error": str(exc)})
            claims = []
        if not claims:
            # No claims to check — emit an empty Evidence so the verdict's
            # ``evidence`` field is still populated for transparency.
            return self._empty_evidence()
        # check_claims does sync HTTP — run in a thread so the event loop
        # is not blocked.
        try:
            evidence = await asyncio.to_thread(ec.check_claims, claims)
        except AttributeError:
            # Some EvidenceChecker variants expose a direct async check(article).
            check = getattr(ec, "check", None)
            if check is None:
                return self._empty_evidence()
            try:
                ev = check(article)
                if hasattr(ev, "__await__"):
                    ev = await ev  # type: ignore[func-returns-value]
                evidence = ev
            except Exception as exc:  # pragma: no cover
                self.logger.warning("pipeline.evidence.check_failed", extra={"error": str(exc)})
                return self._empty_evidence()
        except Exception as exc:  # pragma: no cover
            self.logger.warning("pipeline.evidence.check_failed", extra={"error": str(exc)})
            return self._empty_evidence()
        return evidence

    def _adjust_probability(self, p: float, evidence: Any) -> float:
        """Apply evidence to a probability (uses fn-C's method if available)."""
        if self.evidence_checker is not None and hasattr(self.evidence_checker, "adjust_probability"):
            try:
                p_new = float(self.evidence_checker.adjust_probability(p, evidence))
                return max(0.0, min(1.0, p_new))
            except Exception:  # pragma: no cover
                pass
        # Local fallback: bump toward 0 or 1 based on findings.
        findings = _get_attr(evidence, "findings", None) or []
        if not findings:
            return p
        delta = 0.0
        false_markers = ("false", "pants on fire", "misleading", "fabricated", "incorrect", "untrue", "fake")
        true_markers = ("true", "correct", "accurate", "verified")
        for f in findings:
            rating = str((f.get("rating") if isinstance(f, dict) else _get_attr(f, "rating", "")) or "").lower()
            if not rating:
                continue
            if any(m in rating for m in false_markers):
                delta += 0.10
            elif any(m in rating for m in true_markers):
                delta -= 0.10
        delta = max(-0.30, min(0.30, delta))
        return max(0.0, min(1.0, p + delta))

    def _explain(
        self,
        article: Any,
        signals: Sequence[Any],
        feat_names: Sequence[str],
        p_final: float,
    ) -> list[Any]:
        """Return the top-5 signals for explainability."""
        top_k = 5

        # Use fn-C's SHAP explainer when a trained model + feature matrix exist.
        if (
            explain is not None
            and not self.using_fallback
            and _HAS_FEATURES
            and build_feature_matrix is not None
        ):
            try:
                X, names = build_feature_matrix([article], signals_list=[signals])
                ranked: list = explain(self.model, X, list(names), top_k=top_k)
                if ranked:
                    return self._ranked_to_signals(ranked, signals)
            except Exception as exc:  # pragma: no cover
                self.logger.debug("pipeline.explain.explainer_failed", extra={"error": str(exc)})

        # Fallback: rank by abs(weight × value) over the Jev NOUL signals.
        scored: list[tuple[Any, float, float]] = []  # (signal, abs_contrib, signed_contrib)
        for s in signals:
            stype = _answer_type_value(getattr(s, "type", None))
            try:
                val = float(getattr(s, "value", 0.0) or 0.0)
            except (TypeError, ValueError):
                val = 0.0
            if stype != "noul":
                # Non-NOUL signals don't feed the rule-based score; use the
                # signal's pre-existing contribution (or 0) as a tie-breaker.
                contrib = float(getattr(s, "contribution", 0.0) or 0.0)
            else:
                name = getattr(s, "name", "") or ""
                w = float(self.fallback_weights.get(name, self.fallback_default_weight))
                contrib = w * val
            scored.append((s, abs(contrib), contrib))
        scored.sort(key=lambda t: t[1], reverse=True)

        # Build fresh Signal copies with the computed contribution so the UI
        # can render contribution bars.
        result: list[Any] = []
        for s, _, contrib in scored[:top_k]:
            new_sig = self._with_contribution(s, contrib)
            result.append(new_sig)
        return result

    @staticmethod
    def _with_contribution(signal: Any, contrib: float) -> Any:
        """Return a copy of ``signal`` with ``contribution`` set to ``contrib``."""
        try:
            # Mutable dataclass — copy fields, override contribution.
            if _HAS_TYPES and Signal is not None:
                return Signal(
                    name=getattr(signal, "name", ""),
                    value=getattr(signal, "value", 0.0),
                    probability=getattr(signal, "probability", 0.0),
                    type=getattr(signal, "type", AnswerType.NOUL),
                    contribution=float(contrib),
                )
        except Exception:  # pragma: no cover
            pass
        # Last resort: mutate the original signal in place (dataclass is mutable).
        try:
            signal.contribution = float(contrib)  # type: ignore[union-attr]
        except Exception:  # pragma: no cover
            pass
        return signal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ranked_to_signals(
        self, ranked: list, original_signals: Sequence[Any]
    ) -> list[Any]:
        """Convert ``explain``'s ``[(name, shap_value), ...]`` to ``Signal[]``.

        Looks up the corresponding Signal in ``original_signals`` by name and
        updates its ``contribution`` to the SHAP value. Falls back to a
        pseudo-Signal when the feature name is not a Jev signal name (e.g.
        a TF-IDF column).
        """
        if not _HAS_TYPES:
            # No real Signal class — return the tuples as-is, the API/UI
            # downstream will serialize them defensively.
            return list(ranked)  # type: ignore[return-value]

        by_name: dict[str, Any] = {}
        for s in original_signals:
            name = getattr(s, "name", "")
            if name:
                by_name[name] = s

        out: list[Any] = []
        for item in ranked:
            if isinstance(item, tuple) and len(item) == 2:
                name, contrib = item
            elif isinstance(item, dict):
                name = item.get("name", "")
                contrib = item.get("contribution", 0.0)
            else:
                # Already a Signal-like object.
                out.append(item)
                continue
            name = str(name)
            try:
                contrib_f = float(contrib)
            except (TypeError, ValueError):
                contrib_f = 0.0
            existing = by_name.get(name)
            if existing is not None:
                # Rebuild a Signal with the SHAP contribution attached.
                try:
                    new_sig = Signal(
                        name=getattr(existing, "name", name),
                        value=getattr(existing, "value", 0.0),
                        probability=getattr(existing, "probability", 0.0),
                        type=getattr(existing, "type", AnswerType.NOUL),
                        contribution=contrib_f,
                    )
                    out.append(new_sig)
                    continue
                except Exception:  # pragma: no cover
                    pass
            # Pseudo-Signal for a TF-IDF / metadata feature name.
            out.append(Signal(
                name=name,
                value=0.0,
                probability=0.0,
                type=AnswerType.NOUL,
                contribution=contrib_f,
            ))
        return out

    def _fallback_features(
        self, signals: Sequence[Any]
    ) -> Tuple[list[list[float]], list[str]]:
        names = [
            str(getattr(s, "name", f"sig_{i}")) or f"sig_{i}"
            for i, s in enumerate(signals)
        ]
        row: list[float] = []
        for s in signals:
            val = getattr(s, "value", 0.0)
            try:
                row.append(float(val))
            except (TypeError, ValueError):
                row.append(0.0)
        return [row], names

    def _empty_evidence(self) -> Any:
        if _HAS_TYPES and Evidence is not None:
            try:
                return Evidence(claims=[], findings=[], source="no_claims")
            except Exception:  # pragma: no cover
                pass
        return Evidence(claims=[], findings=[], source="no_claims")

    def _error_verdict(self, err: str) -> Any:
        if _HAS_TYPES and Verdict is not None:
            try:
                return Verdict(
                    label="NEEDS_REVIEW",
                    probability=0.5,
                    signals=[],
                    explanation=f"pipeline error: {err}",
                    evidence=None,
                    latency_ms=0.0,
                    model_version=self.model_version,
                )
            except Exception:  # pragma: no cover
                pass
        return Verdict(
            label="NEEDS_REVIEW",
            probability=0.5,
            signals=[],
            explanation=f"pipeline error: {err}",
            evidence=None,
            latency_ms=0.0,
            model_version=self.model_version,
        )

    def _build_verdict(
        self,
        *,
        label: str,
        probability: float,
        top_signals: list[Any],
        reason: str,
        evidence: Optional[Any],
        latency_ms: float,
    ) -> Any:
        explanation = self._build_explanation(reason, evidence)
        if _HAS_TYPES and Verdict is not None:
            try:
                return Verdict(
                    label=label,
                    probability=float(probability),
                    signals=list(top_signals),
                    explanation=explanation,
                    evidence=evidence,
                    latency_ms=float(latency_ms),
                    model_version=self.model_version,
                )
            except Exception:  # pragma: no cover
                pass
        return Verdict(
            label=label,
            probability=float(probability),
            signals=list(top_signals),
            explanation=explanation,
            evidence=evidence,
            latency_ms=float(latency_ms),
            model_version=self.model_version,
        )

    @staticmethod
    def _build_explanation(reason: str, evidence: Optional[Any]) -> str:
        parts: list[str] = [reason]
        if evidence is None:
            return " | ".join(parts)
        # Prefer the evidence's adjusted_probability if set.
        adj = _get_attr(evidence, "adjusted_probability", None)
        if adj is not None:
            parts.append(f"evidence adjusted p={float(adj):.3f}")
        source = _get_attr(evidence, "source", None)
        if source:
            parts.append(f"evidence source={source}")
        findings = _get_attr(evidence, "findings", None) or []
        if findings:
            parts.append(f"{len(findings)} fact-check finding(s)")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------------
def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from an object (attr or dict key), else ``default``."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _answer_type_value(t: Any) -> str:
    """Return the lowercase string value of an AnswerType (or empty)."""
    if t is None:
        return ""
    # Enum member.
    val = getattr(t, "value", t)
    return str(val).lower()


def build_default_pipeline(config: Optional[dict] = None) -> "FakeNewsPipeline":
    """Convenience factory used by the FastAPI service and the Streamlit UI.

    Loads config via ``fakenews.config.load_config`` when not provided.
    """
    cfg = config
    if cfg is None:
        try:
            cfg = load_config()
        except Exception:  # pragma: no cover
            cfg = {}
    return FakeNewsPipeline(cfg)


__all__ = [
    "FakeNewsPipeline",
    "RuleBasedFallback",
    "FallbackSignalExtractor",
    "build_default_pipeline",
]
