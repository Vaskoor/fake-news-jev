# -*- coding: utf-8 -*-
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
"""Feature extraction for the fake-news meta-classifier (Stage 2).

Three feature families, all consumed by the downstream
``fakenews.meta_classifier``:

1. **JEV signal vector** — 19 floats, one per question in
   ``config/questions.yaml`` (11 NOUL + 5 SCORE + 3 CHOICE). The order is
   the canonical file order; ``extract_jev_vector`` produces a fixed-length
   float vector from a list of `Signal` objects. CHOICE signals are encoded
   as the integer index of the value within the question's ``options``
   list (tree models like LightGBM/XGBoost handle this natively; SHAP
   scores the resulting column as a categorical).

2. **TF-IDF** — wrapped `sklearn.feature_extraction.text.TfidfVectorizer`
   with the project config's ``max_features`` and ``ngram_range``
   (defaults 5000 and (1,2)). Sublinear TF scaling and English stop words
   are on by default. The `TfidfFeaturizer` exposes a sklearn-style
   ``fit/transform/fit_transform`` API.

3. **Metadata features** — 7 handcrafted, low-cardinality features that
   capture stylometric signals not in the TF-IDF n-gram space:
   ``text_length`` (log), ``caps_ratio``, ``punct_ratio``,
   ``exclamation_count`` (log), ``question_count`` (log),
   ``headline_body_similarity`` (Jaccard on tokens > 2 chars), and
   ``source_reputation`` (lookup in `SOURCE_REPUTATION`, default 0.5).

`build_feature_matrix` concatenates the three blocks into a single design
matrix and returns it wrapped in a `FeatureBundle` with SHAP-friendly
column names. The function supports dense (default) or sparse output.
"""
from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

try:                                   # optional — features config
    from .config import load_config
    _HAS_CONFIG = True
except Exception:                      # pragma: no cover — defensive
    _HAS_CONFIG = False

from .types import AnswerType, Article, Signal

__all__ = [
    "SOURCE_REPUTATION",
    "DEFAULT_SOURCE_REPUTATION",
    "MetadataFeatures",
    "TfidfFeaturizer",
    "JEV_QUESTION_ORDER",
    "JEV_QUESTION_SPECS",
    "extract_jev_vector",
    "build_feature_matrix",
    "FeatureBundle",
]


# ---------------------------------------------------------------------------
# Source reputation — built-in lookup
# ---------------------------------------------------------------------------

SOURCE_REPUTATION: dict[str, float] = {
    # Tier 1 — international wire services and broadsheet of record
    "reuters": 0.95,
    "ap": 0.93,
    "associated press": 0.93,
    "bbc": 0.92,
    "bbc news": 0.92,
    "the new york times": 0.91,
    "new york times": 0.91,
    "nytimes": 0.91,
    "the washington post": 0.90,
    "washington post": 0.90,
    "npr": 0.89,
    "wall street journal": 0.88,
    "wsj": 0.88,
    "the guardian": 0.87,
    "guardian": 0.87,
    "financial times": 0.87,
    "ft": 0.87,
    "reuters news": 0.95,
    # Tier 2 — major US broadcast
    "cnn": 0.78,
    "nbc news": 0.78,
    "nbc": 0.78,
    "cbs news": 0.78,
    "cbs": 0.78,
    "abc news": 0.77,
    "abc": 0.77,
    "politico": 0.76,
    "the hill": 0.72,
    "usa today": 0.72,
    "time": 0.74,
    "time magazine": 0.74,
    "newsweek": 0.70,
    "the atlantic": 0.78,
    "the economist": 0.85,
    # Tier 3 — partisan-leaning mainstream
    "fox news": 0.70,
    "fox": 0.70,
    "msnbc": 0.68,
    "the new york post": 0.55,
    "new york post": 0.55,
    "nypost": 0.55,
    "daily mail": 0.55,
    "dailymail": 0.55,
    "daily mail online": 0.55,
    "the telegraph": 0.72,
    "buzzfeed news": 0.60,
    "buzzfeed": 0.50,
    "huffpost": 0.55,
    "huffington post": 0.55,
    "vox": 0.62,
    "slate": 0.60,
    # Tier 4 — explicitly partisan / alternative
    "newsmax": 0.45,
    "newsmax.com": 0.45,
    "the blaze": 0.35,
    "breitbart": 0.30,
    "breitbart news": 0.30,
    "the federalist": 0.40,
    "national review": 0.55,
    "the weekly standard": 0.55,
    "the daily wire": 0.40,
    "daily wire": 0.40,
    # Tier 5 — well-known misinformation vectors
    "oann": 0.25,
    "oan": 0.25,
    "one america news": 0.25,
    "rt": 0.18,
    "rt.com": 0.18,
    "rt america": 0.18,
    "sputnik": 0.15,
    "sputnik news": 0.15,
    "the gateway pundit": 0.18,
    "gateway pundit": 0.18,
    "zero hedge": 0.12,
    "zerohedge": 0.12,
    "infowars": 0.10,
    "natural news": 0.08,
    "naturalnews": 0.08,
    "news punch": 0.10,
    "your news wire": 0.08,
    "before it's news": 0.08,
    "global research": 0.15,
    "veterans today": 0.10,
    # Special tags used by our synthetic generator / WELFake anonymizer
    "anonymous-blog": 0.15,
    "fake-kaggle": 0.10,
    "welfake": 0.50,  # provenance-obscured
    "unknown": 0.50,
    "": 0.50,
}

DEFAULT_SOURCE_REPUTATION: float = 0.50


def _attr_str(obj: Any, name: str, default: str = "") -> str:
    """Duck-typed attribute access: returns ``str(getattr(obj, name, default))``.

    Coerces None, NaN, and pandas-N/A sentinels to the empty string. Never
    raises — safe to call on Article, SimpleNamespace, DataFrame row, or
    even a plain str (returns `default` in that case).
    """
    if isinstance(obj, str):
        return default  # a bare str has no headline/body attribute
    v = getattr(obj, name, None)
    if v is None:
        return default
    # NaN check: NaN is the only float that compares unequal to itself.
    if isinstance(v, float) and v != v:
        return default
    s = str(v).strip()
    if s.lower() in ("nan", "none", "null", "na", "<na>"):
        return default
    return s


def _resolve_reputation(source: str) -> float:
    """Return a float reputation score in [0, 1] for `source`.

    Lookup order:
      1. Exact lowercase match in SOURCE_REPUTATION.
      2. Substring match (prefer the longest key that is a substring of
         `source`). This catches "Reuters / News" or "BBC London desk".
      3. Default.

    Args:
        source: Raw source string (may be empty).

    Returns:
        Reputation score in [0, 1].
    """
    if not source:
        return DEFAULT_SOURCE_REPUTATION
    s = source.strip().lower()
    if not s:
        return DEFAULT_SOURCE_REPUTATION
    if s in SOURCE_REPUTATION:
        return SOURCE_REPUTATION[s]
    # Longest-substring match — prefer specificity.
    best_key: Optional[str] = None
    best_len = 0
    for k in SOURCE_REPUTATION:
        if k and len(k) > best_len and k in s:
            best_key = k
            best_len = len(k)
    if best_key is not None:
        return SOURCE_REPUTATION[best_key]
    return DEFAULT_SOURCE_REPUTATION


# ---------------------------------------------------------------------------
# JEV question order — canonical, mirrors config/questions.yaml
# ---------------------------------------------------------------------------

# IMPORTANT: this list is the source of truth for the JEV feature vector
# order in the meta-classifier. It MUST mirror the order questions appear
# in config/questions.yaml (11 NOUL, 5 SCORE, 3 CHOICE). If the YAML is
# edited to add/remove/reorder a question, the meta-classifier must be
# retrained from scratch (set version bump), and this list must be updated.
JEV_QUESTION_ORDER: list[str] = [
    # 11 NOUL
    "is_fake_overall",
    "is_clickbait",
    "has_unverified_claims",
    "uses_anonymous_sources",
    "contains_conspiracy_framing",
    "cites_verifiable_sources",
    "is_satire",
    "is_opinion_not_news",
    "has_statistical_claims",
    "impersonates_authority",
    "headline_contradicts_body",
    # 5 SCORE
    "emotional_intensity",
    "sensationalism",
    "factual_specificity",
    "political_bias",
    "writing_professionalism",
    # 3 CHOICE
    "topic",
    "claim_type",
    "manipulation_tactic",
]

# Per-question specs (type + bounds/options) — also mirrors questions.yaml.
# Used by `extract_jev_vector` to encode CHOICE values as integer indices
# (the only way to represent a categorical as a single float in a fixed-
# length float vector). For NOUL and SCORE, `value` is already a float.
JEV_QUESTION_SPECS: dict[str, dict[str, Any]] = {
    "is_fake_overall":             {"type": AnswerType.NOUL},
    "is_clickbait":                {"type": AnswerType.NOUL},
    "has_unverified_claims":       {"type": AnswerType.NOUL},
    "uses_anonymous_sources":      {"type": AnswerType.NOUL},
    "contains_conspiracy_framing": {"type": AnswerType.NOUL},
    "cites_verifiable_sources":    {"type": AnswerType.NOUL},
    "is_satire":                   {"type": AnswerType.NOUL},
    "is_opinion_not_news":         {"type": AnswerType.NOUL},
    "has_statistical_claims":      {"type": AnswerType.NOUL},
    "impersonates_authority":      {"type": AnswerType.NOUL},
    "headline_contradicts_body":   {"type": AnswerType.NOUL},
    "emotional_intensity":         {"type": AnswerType.SCORE, "min": 0.0, "max": 10.0},
    "sensationalism":              {"type": AnswerType.SCORE, "min": 0.0, "max": 10.0},
    "factual_specificity":         {"type": AnswerType.SCORE, "min": 0.0, "max": 10.0},
    "political_bias":              {"type": AnswerType.SCORE, "min": -5.0, "max": 5.0},
    "writing_professionalism":     {"type": AnswerType.SCORE, "min": 0.0, "max": 10.0},
    "topic":                       {"type": AnswerType.CHOICE,
                                     "options": ["politics", "health", "finance",
                                                 "science", "sports", "other"]},
    "claim_type":                   {"type": AnswerType.CHOICE,
                                     "options": ["fact", "opinion",
                                                 "prediction", "rumor"]},
    "manipulation_tactic":          {"type": AnswerType.CHOICE,
                                     "options": ["none", "fear", "outrage",
                                                 "false_authority",
                                                 "cherry_picking"]},
}

assert len(JEV_QUESTION_ORDER) == 19, (
    f"JEV_QUESTION_ORDER must have exactly 19 entries (per questions.yaml), "
    f"got {len(JEV_QUESTION_ORDER)}"
)
assert set(JEV_QUESTION_ORDER) == set(JEV_QUESTION_SPECS), (
    "JEV_QUESTION_ORDER and JEV_QUESTION_SPECS keys must match"
)


def extract_jev_vector(signals: Sequence[Signal]) -> np.ndarray:
    """Return the 19-float JEV feature vector for one article's signals.

    Output order is `JEV_QUESTION_ORDER` (the canonical file order in
    questions.yaml). For each question:

    * NOUL  -> ``value`` (float in [0, 1]; if missing, falls back to
               ``probability`` and then 0.0).
    * SCORE -> ``value`` (float in [min, max]; min/max normalization is
               left to the downstream model — LightGBM handles raw bounds
               fine).
    * CHOICE -> integer index of ``value`` within the question's
                ``options`` list. Unknown / missing value -> -1.0 so the
                model can learn "unknown" as its own bin.

    Args:
        signals: List of Signal objects produced by the Jev client. May
                 contain extra or out-of-order signals; only the 19
                 canonical names are read.

    Returns:
        ``np.ndarray`` of shape (19,) and dtype float32. Missing signals
        default to 0.0 (NOUL/SCORE) or -1.0 (CHOICE), so partial signal
        sets don't crash the meta-classifier.
    """
    n = len(JEV_QUESTION_ORDER)
    out = np.zeros(n, dtype=np.float32)
    # Pre-fill CHOICE slots with -1 (sentinel for "unknown categorical").
    for i, name in enumerate(JEV_QUESTION_ORDER):
        if JEV_QUESTION_SPECS[name]["type"] is AnswerType.CHOICE:
            out[i] = -1.0
    index_by_name = {name: i for i, name in enumerate(JEV_QUESTION_ORDER)}
    # Allow missing signals to fall back to "answer was 0".
    for sig in signals:
        name = getattr(sig, "name", None) or getattr(sig, "question_id", None)
        if name is None or name not in index_by_name:
            continue
        i = index_by_name[name]
        spec = JEV_QUESTION_SPECS[name]
        value = getattr(sig, "value", None)
        if spec["type"] is AnswerType.CHOICE:
            opts = spec["options"]
            if value is None:
                out[i] = -1.0
            else:
                sval = str(value).strip().lower()
                try:
                    out[i] = float(opts.index(sval))
                except ValueError:
                    out[i] = -1.0
        else:  # NOUL or SCORE
            if value is None:
                # Fall back to probability if value was dropped.
                value = getattr(sig, "probability", 0.0)
            try:
                out[i] = float(value)
            except (TypeError, ValueError):
                out[i] = 0.0
    return out


# ---------------------------------------------------------------------------
# Metadata featurizer — 7 handcrafted stylometric features
# ---------------------------------------------------------------------------

_TOKEN_PATTERN = re.compile(r"\b\w+\b")


@dataclass
class MetadataFeatures:
    """Hand-crafted, low-cardinality stylometric features (7 floats).

    Feature order is fixed; ``feature_names`` exposes the canonical
    order for downstream SHAP / inspection. The transform is stateless
    (no fit required to compute, but `fit` is provided so callers can
    include it in an sklearn Pipeline).

    Features:
      1. text_length              — log1p(len(headline + " " + body)).
      2. caps_ratio               — uppercase letters / total letters.
      3. punct_ratio              — punctuation chars / total chars.
      4. exclamation_count        — log1p(count('!')).
      5. question_count           — log1p(count('?')).
      6. headline_body_similarity — Jaccard(headline tokens, body tokens)
                                    with tokens > 2 chars (filters stopwords).
      7. source_reputation        — looked up in `SOURCE_REPUTATION`,
                                    default 0.5.

    Count features are log1p-scaled so they don't dominate the tree
    splits; ratios are already in [0, 1]; reputation is in [0, 1].
    """
    feature_names: tuple[str, ...] = (
        "text_length",
        "caps_ratio",
        "punct_ratio",
        "exclamation_count",
        "question_count",
        "headline_body_similarity",
        "source_reputation",
    )

    @property
    def n_features(self) -> int:
        """Number of features emitted per article (always 7)."""
        return len(self.feature_names)

    # sklearn-style API (so this can sit in a Pipeline if desired).
    def fit(self, articles: Iterable[Any]) -> "MetadataFeatures":
        """Stateless — return self. Provided for sklearn Pipeline compat."""
        return self

    def transform(self, article: Any) -> np.ndarray:
        """Compute the 7-feature vector for a single article-like object.

        Args:
            article: Article (fn-A), SimpleNamespace (fn-C), or any object
                     with ``headline`` / ``body`` / ``source`` attributes.
                     Already cleaned by `preprocessing.clean_article` is
                     best, but raw articles work too — we re-tokenize
                     defensively.

        Returns:
            np.ndarray shape (7,), dtype float32.
        """
        headline = _attr_str(article, "headline")
        body = _attr_str(article, "body")
        source = _attr_str(article, "source")
        text = headline + " " + body
        text_len = len(text)
        letters = sum(1 for c in text if c.isalpha())
        caps = sum(1 for c in text if c.isupper())
        caps_ratio = (caps / letters) if letters else 0.0
        punct = sum(1 for c in text if c in string.punctuation)
        punct_ratio = (punct / text_len) if text_len else 0.0
        excl = text.count("!")
        quest = text.count("?")
        h_tokens = {t for t in _TOKEN_PATTERN.findall(headline.lower())
                    if len(t) > 2}
        b_tokens = {t for t in _TOKEN_PATTERN.findall(body.lower())
                    if len(t) > 2}
        union = h_tokens | b_tokens
        jacc = (len(h_tokens & b_tokens) / len(union)) if union else 0.0
        reputation = _resolve_reputation(source)
        return np.array(
            [
                np.log1p(text_len),
                caps_ratio,
                punct_ratio,
                np.log1p(excl),
                np.log1p(quest),
                jacc,
                reputation,
            ],
            dtype=np.float32,
        )

    def transform_many(self, articles: Sequence[Any]) -> np.ndarray:
        """Vectorize `transform` over a list of article-like objects.

        Args:
            articles: List of Article / SimpleNamespace / etc.

        Returns:
            np.ndarray shape (n_articles, 7), dtype float32.
        """
        if not articles:
            return np.zeros((0, self.n_features), dtype=np.float32)
        out = np.empty((len(articles), self.n_features), dtype=np.float32)
        for i, art in enumerate(articles):
            out[i] = self.transform(art)
        return out

    def get_feature_names_out(self) -> np.ndarray:
        """Return the feature names as a numpy array (sklearn convention)."""
        return np.array(self.feature_names, dtype=object)


# ---------------------------------------------------------------------------
# TF-IDF featurizer — wraps sklearn TfidfVectorizer
# ---------------------------------------------------------------------------

@dataclass
class TfidfFeaturizer:
    """Thin wrapper around `sklearn.feature_extraction.text.TfidfVectorizer`.

    Configuration is pulled from `config/config.yaml` (``model.tfidf_max_features``
    and ``model.ngram_range``) by default; explicit constructor args override.

    The vectorizer is built lazily inside `fit` so the constructor never
    touches the filesystem (important for `__init__` in pickled pipelines).
    """
    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True
    lowercase: bool = True
    stop_words: Optional[str] = "english"
    norm: str = "l2"
    strip_accents: str = "unicode"
    token_pattern: str = r"(?u)\b\w\w+\b"
    _vectorizer: Optional[TfidfVectorizer] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Pull defaults from config.yaml if not explicitly overridden."""
        if _HAS_CONFIG:
            try:
                cfg = load_config()
                m = cfg.get("model", {})
                if m.get("tfidf_max_features") is not None:
                    self.max_features = int(m["tfidf_max_features"])
                if m.get("ngram_range") is not None:
                    ng = m["ngram_range"]
                    if isinstance(ng, (list, tuple)) and len(ng) == 2:
                        self.ngram_range = (int(ng[0]), int(ng[1]))
            except Exception:                       # pragma: no cover
                pass

    def _build(self) -> TfidfVectorizer:
        """Construct the underlying sklearn TfidfVectorizer instance."""
        return TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=self.lowercase,
            stop_words=self.stop_words,
            norm=self.norm,
            strip_accents=self.strip_accents,
            token_pattern=self.token_pattern,
        )

    def fit(self, corpus: Iterable[str]) -> "TfidfFeaturizer":
        """Fit the underlying vectorizer on a corpus of strings.

        Args:
            corpus: Iterable of (already-cleaned) article texts. We
                    pass ``headline + " " + body`` if the caller gives
                    us Articles — they should concatenate first. This
                    keeps the featurizer text-only and reusable.

        Returns:
            self, for chaining.
        """
        self._vectorizer = self._build()
        self._vectorizer.fit(list(corpus))
        return self

    def transform(self, texts) -> sparse.csr_matrix:
        """Transform `texts` into a sparse TF-IDF matrix.

        Args:
            texts: str | Iterable[str] | Article | Iterable[Article].
                   If Article(s) are passed, headline + " " + body is
                   used as the document text.

        Returns:
            scipy.sparse.csr_matrix shape (n_docs, vocab_size).

        Raises:
            RuntimeError: featurizer has not been fit yet.
        """
        if self._vectorizer is None:
            raise RuntimeError("TfidfFeaturizer must be fit() before transform()")
        strs = _coerce_to_strings(texts)
        return self._vectorizer.transform(strs).tocsr()

    def fit_transform(self, corpus) -> sparse.csr_matrix:
        """Convenience: fit then transform in one call."""
        strs = _coerce_to_strings(corpus)
        self._vectorizer = self._build()
        return self._vectorizer.fit_transform(strs).tocsr()

    @property
    def vocabulary_(self) -> dict[str, int]:
        """Return the fitted vocabulary mapping (empty if not fit)."""
        return self._vectorizer.vocabulary_ if self._vectorizer is not None else {}

    def get_feature_names_out(self) -> np.ndarray:
        """Return the fitted TF-IDF vocabulary as a numpy str array."""
        if self._vectorizer is None:
            return np.array([], dtype=object)
        return self._vectorizer.get_feature_names_out()


def _article_text(obj: Any) -> str:
    """Duck-typed access to an article-like object's text.

    Accepts Article (fn-A's dataclass), a SimpleNamespace with
    ``headline``/``body`` attributes (used by fn-C's synthetic generator),
    a DataFrame row with a ``text`` attribute, or a plain str. Always
    returns a non-None str.
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    headline = _attr_str(obj, "headline")
    body = _attr_str(obj, "body")
    if headline or body:
        return f"{headline} {body}".strip()
    text = _attr_str(obj, "text")
    if text:
        return text
    # Last-ditch fallback.
    return str(obj)


def _coerce_to_strings(texts: Any) -> list[str]:
    """Coerce `texts` (str / iterable of str / Article-like / iterable of
    Article-like) to list[str]."""
    if isinstance(texts, str):
        return [texts]
    if hasattr(texts, "headline") and hasattr(texts, "body"):
        return [_article_text(texts)]
    out: list[str] = []
    for t in texts:
        out.append(_article_text(t))
    return out


# ---------------------------------------------------------------------------
# FeatureBundle — assembled design matrix + column metadata for SHAP
# ---------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    """Design-matrix + column-name bundle returned by `build_feature_matrix`.

    Primary fields:
        X:               the design matrix (dense np.ndarray OR scipy.sparse
                         csr_matrix, depending on `dense` flag at build).
        column_names:    list of str, length == X.shape[1]. One per column
                         in the same order as `X`'s columns. Suitable for
                         passing directly as `feature_names` to SHAP
                         `Explainer(model, ..., feature_names=...)`.

    Slice fields (two equivalent forms are exposed):
        jev_slice:       (start, end) column indices of the JEV block.
        tfidf_slice:     (start, end) column indices of the TF-IDF block.
        meta_slice:      (start, end) column indices of the metadata block.
        jev_indices:     np.arange(jev_slice[0], jev_slice[1])  — for callers
                         that prefer a column-index array (fn-C shape).
        tfidf_indices:   np.arange(...) for the TF-IDF block.
        metadata_indices:np.arange(...) for the metadata block.

    Auxiliary fields:
        feature_names:   alias for `column_names` (sklearn / fn-C convention).
        metadata_keys:   metadata feature names without the "meta_" prefix
                         (i.e. as returned by `MetadataFeatures.feature_names`).
        tfidf_vocabulary: TF-IDF vocab strings in column order (no prefix).
        tfidf_vectorizer: optional reference to the underlying fitted
                         TfidfVectorizer (or TfidfFeaturizer). Downstream
                         callers can attach one here if `build_feature_matrix`
                         was called with an already-transformed sparse matrix.
        n_tfidf:         number of TF-IDF columns (= len(tfidf_vocabulary)).
        n_jev:           number of JEV columns (= 19, but exposed for safety).
        n_meta:          number of metadata columns (= 7).
    """
    X: Any
    column_names: list[str]
    jev_slice: tuple[int, int]
    tfidf_slice: tuple[int, int]
    meta_slice: tuple[int, int]
    tfidf_vocabulary: list[str] = field(default_factory=list)
    # fn-C compatibility aliases:
    feature_names: list[str] = field(default_factory=list)
    jev_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    tfidf_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    metadata_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    metadata_keys: list[str] = field(default_factory=list)
    tfidf_vectorizer: Optional[Any] = None
    n_tfidf: int = 0
    n_jev: int = 0
    n_meta: int = 0

    def __post_init__(self) -> None:
        """Populate fn-C compatibility aliases if not already set."""
        if not self.feature_names:
            self.feature_names = list(self.column_names)
        if len(self.jev_indices) == 0:
            self.jev_indices = np.arange(self.jev_slice[0], self.jev_slice[1],
                                        dtype=np.int64)
        if len(self.tfidf_indices) == 0:
            self.tfidf_indices = np.arange(self.tfidf_slice[0], self.tfidf_slice[1],
                                           dtype=np.int64)
        if len(self.metadata_indices) == 0:
            self.metadata_indices = np.arange(self.meta_slice[0], self.meta_slice[1],
                                              dtype=np.int64)
        if self.n_jev == 0:
            self.n_jev = int(self.jev_slice[1] - self.jev_slice[0])
        if self.n_tfidf == 0:
            self.n_tfidf = int(self.tfidf_slice[1] - self.tfidf_slice[0])
        if self.n_meta == 0:
            self.n_meta = int(self.meta_slice[1] - self.meta_slice[0])

    @property
    def n_features(self) -> int:
        """Total feature dimensionality (= X.shape[1])."""
        return int(self.X.shape[1])

    def column_name(self, idx: int) -> str:
        """Return the column name at index `idx` (bounds-checked)."""
        return self.column_names[idx]


# ---------------------------------------------------------------------------
# build_feature_matrix — assemble all three blocks
# ---------------------------------------------------------------------------

def _coerce_jev_block(
    jev_signals: Any,
    article_ids: list[str],
    n: int,
) -> tuple[np.ndarray, list[str]]:
    """Adapt `jev_signals` of any supported shape to a (n, 19) float32 matrix.

    Accepted shapes:
      1. ``Sequence[Sequence[Signal]]`` — the canonical fn-B shape; each
         inner list is run through ``extract_jev_vector``.
      2. ``dict[str, float]`` — single "jev_score" per article id (fn-C's
         simulated backend). The score is placed at index 0 (the
         ``is_fake_overall`` slot) and the remaining 18 slots are filled
         with 0.0 (NOUL) or -1.0 (CHOICE) so the model still sees a
         well-shaped 19-d vector.
      3. ``dict[str, list[Signal]]`` — per-article-id list of Signals.
      4. ``np.ndarray`` shape ``(n,)`` or ``(n, 19)`` — used as-is (the
         (n,) variant is broadcast to column 0 like case 2).
      5. ``None`` — all zeros.

    Args:
        jev_signals: see above.
        article_ids: list of article id strings, used to look up dict-keyed
                     inputs. Length must equal n.
        n:           number of articles.

    Returns:
        (jev_mat, jev_names) — float32 (n, 19) matrix and list of 19 column
        names ``["jev_<q>", ...]``.
    """
    jev_dim = len(JEV_QUESTION_ORDER)
    jev_mat = np.zeros((n, jev_dim), dtype=np.float32)
    # Pre-fill CHOICE slots with -1 (sentinel for "unknown categorical").
    for i, name in enumerate(JEV_QUESTION_ORDER):
        if JEV_QUESTION_SPECS[name]["type"] is AnswerType.CHOICE:
            jev_mat[:, i] = -1.0

    if jev_signals is None:
        pass  # leave zeros (with CHOICE = -1)
    elif isinstance(jev_signals, np.ndarray):
        arr = jev_signals
        if arr.ndim == 1:
            # (n,) — single score per article, place at column 0.
            jev_mat[:, 0] = arr.astype(np.float32)
        elif arr.ndim == 2:
            rows = min(arr.shape[0], n)
            cols = min(arr.shape[1], jev_dim)
            jev_mat[:rows, :cols] = arr[:rows, :cols].astype(np.float32)
    elif isinstance(jev_signals, dict):
        # dict[str, float] or dict[str, list[Signal]]
        for i, aid in enumerate(article_ids):
            v = jev_signals.get(aid)  # type: ignore[union-attr]
            if v is None:
                continue
            if isinstance(v, (int, float)):
                jev_mat[i, 0] = float(v)  # is_fake_overall slot
            elif isinstance(v, np.ndarray):
                row = v.ravel().astype(np.float32)
                cols = min(row.shape[0], jev_dim)
                jev_mat[i, :cols] = row[:cols]
            elif isinstance(v, (list, tuple)) and v and _looks_like_signal(v[0]):
                jev_mat[i] = extract_jev_vector(v)
            elif isinstance(v, (list, tuple)):
                # Bare numeric list — treat as raw vector.
                row = np.asarray(v, dtype=np.float32).ravel()
                cols = min(row.shape[0], jev_dim)
                jev_mat[i, :cols] = row[:cols]
            # else: leave zeros
    else:
        # Treat as iterable of per-article Signal-lists (or numbers).
        for i, sigs in enumerate(jev_signals):
            if i >= n:
                break
            if sigs is None:
                continue
            if isinstance(sigs, (int, float)):
                jev_mat[i, 0] = float(sigs)
            elif isinstance(sigs, np.ndarray):
                row = sigs.ravel().astype(np.float32)
                cols = min(row.shape[0], jev_dim)
                jev_mat[i, :cols] = row[:cols]
            elif isinstance(sigs, (list, tuple)) and sigs and _looks_like_signal(sigs[0]):
                jev_mat[i] = extract_jev_vector(sigs)
            elif isinstance(sigs, (list, tuple)):
                row = np.asarray(sigs, dtype=np.float32).ravel()
                cols = min(row.shape[0], jev_dim)
                jev_mat[i, :cols] = row[:cols]

    jev_names = [f"jev_{q}" for q in JEV_QUESTION_ORDER]
    return jev_mat, jev_names


def _looks_like_signal(obj: Any) -> bool:
    """Heuristic: True if `obj` looks like a Signal instance.

    A Signal has a `.name` and a `.value` (or `.probability`). Bare floats
    or strings don't qualify.
    """
    return hasattr(obj, "name") and hasattr(obj, "value")


def _coerce_tfidf_block(
    tfidf: Any,
    articles: Sequence[Any],
) -> tuple[sparse.csr_matrix, list[str], Optional[Any]]:
    """Adapt `tfidf` of any supported shape to a (n, k) sparse CSR matrix.

    Accepted shapes:
      1. ``TfidfFeaturizer`` — call ``transform()`` on the article texts.
      2. ``(vectorizer, X_sparse)`` tuple — extract feature names from the
         vectorizer, return the matrix directly.
      3. ``scipy.sparse`` matrix or 2D ndarray — used as-is. No vocab
         strings are recoverable; column names become ``tfidf_<idx>``.
      4. ``sklearn TfidfVectorizer`` instance (already fit) — transform the
         article texts directly.

    Args:
        tfidf:     see above.
        articles:  list of article-like objects (for shape 1/4 only).

    Returns:
        (tfidf_mat, tfidf_names, tfidf_vectorizer) — CSR matrix, list of
        vocab strings, and a reference to the underlying vectorizer (or
        None if none is available).
    """
    texts = [_article_text(a) for a in articles]
    # Case 2: (vectorizer, matrix) tuple
    if isinstance(tfidf, tuple) and len(tfidf) == 2:
        vec, X = tfidf
        X_csr = sparse.csr_matrix(X) if not sparse.issparse(X) else X.tocsr()
        try:
            names = list(vec.get_feature_names_out())  # type: ignore[union-attr]
        except Exception:
            names = [f"tfidf_{i}" for i in range(X_csr.shape[1])]
        return X_csr, names, vec
    # Case 1: TfidfFeaturizer instance
    if isinstance(tfidf, TfidfFeaturizer):
        X = tfidf.transform(texts).tocsr()
        names = list(tfidf.get_feature_names_out())
        return X, names, tfidf._vectorizer
    # Case 4: raw sklearn TfidfVectorizer
    if hasattr(tfidf, "transform") and hasattr(tfidf, "vocabulary_") and \
            not sparse.issparse(tfidf) and not hasattr(tfidf, "shape"):
        X = tfidf.transform(texts).tocsr()
        names = list(tfidf.get_feature_names_out())
        return X, names, tfidf
    # Case 3: raw sparse matrix / ndarray
    if sparse.issparse(tfidf) or isinstance(tfidf, np.ndarray):
        X_csr = sparse.csr_matrix(tfidf) if not sparse.issparse(tfidf) else tfidf.tocsr()
        names = [f"tfidf_{i}" for i in range(X_csr.shape[1])]
        return X_csr, names, None
    raise TypeError(
        f"build_feature_matrix: unsupported tfidf argument type {type(tfidf)!r}"
    )


def _coerce_metadata_block(
    metadata: Any,
    articles: Sequence[Any],
    article_ids: list[str],
    n: int,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Adapt `metadata` of any supported shape to a (n, 7) float32 matrix.

    Accepted shapes:
      1. ``MetadataFeatures`` instance — call ``transform_many(articles)``.
      2. ``dict[str, dict[str, float]]`` — per-article-id feature dict
         (fn-C's `build_metadata` output). All keys across all articles
         become the column order (alphabetically sorted for determinism).
      3. ``list[dict[str, float]]`` — per-article feature dict (positional).
      4. ``np.ndarray`` shape ``(n, k)`` — used as-is; column names are
         ``meta_0``, ``meta_1``, ... ``meta_{k-1}``.
      5. ``None`` — all zeros (no metadata columns).

    Args:
        metadata:   see above.
        articles:   list of article-like objects (for shape 1 only).
        article_ids: list of article id strings, used for dict-keyed inputs.
        n:          number of articles.

    Returns:
        (meta_mat, meta_block_names, meta_keys) — float32 (n, k) matrix,
        list of column names (with `meta_` prefix for use in the full design
        matrix), and the raw metadata key names (no prefix — useful for
        ablation reports).
    """
    if isinstance(metadata, MetadataFeatures):
        arr = metadata.transform_many(articles).astype(np.float32)
        keys = list(metadata.feature_names)
        names = list(keys)  # already human-readable; no prefix needed
        return arr, names, keys
    if metadata is None:
        return np.zeros((n, 0), dtype=np.float32), [], []
    if isinstance(metadata, np.ndarray):
        arr = metadata.astype(np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(n, -1)
        keys = [f"meta_{i}" for i in range(arr.shape[1])]
        return arr, keys, keys
    if isinstance(metadata, dict):
        # dict[str, dict[str, float]]
        all_keys = sorted({k for d in metadata.values() for k in (d or {}).keys()})
        arr = np.zeros((n, len(all_keys)), dtype=np.float32)
        for i, aid in enumerate(article_ids):
            d = metadata.get(aid, {}) or {}
            for j, k in enumerate(all_keys):
                try:
                    arr[i, j] = float(d.get(k, 0.0))
                except (TypeError, ValueError):
                    arr[i, j] = 0.0
        return arr, all_keys, all_keys
    if isinstance(metadata, (list, tuple)):
        # list[dict[str, float]] — positional, no ids needed
        all_keys = sorted({k for d in metadata for k in (d or {}).keys()}) if metadata else []
        arr = np.zeros((n, len(all_keys)), dtype=np.float32)
        for i in range(min(len(metadata), n)):
            d = metadata[i] or {}
            for j, k in enumerate(all_keys):
                try:
                    arr[i, j] = float(d.get(k, 0.0))
                except (TypeError, ValueError):
                    arr[i, j] = 0.0
        return arr, all_keys, all_keys
    raise TypeError(
        f"build_feature_matrix: unsupported metadata argument type {type(metadata)!r}"
    )


def build_feature_matrix(
    articles: Sequence[Any],
    jev_signals: Any,
    tfidf: Any,
    metadata: Any,
    top_k_tfidf: Optional[int] = None,
    dense: bool = True,
) -> tuple[Any, "FeatureBundle"]:
    """Concatenate JEV + TF-IDF + metadata into a single design matrix.

    Returns a ``(X, FeatureBundle)`` tuple so callers can both train on `X`
    and use `bundle.column_names` (alias `bundle.feature_names`) as SHAP
    feature names. The spec shorthand ``-> np.ndarray`` refers to the
    primary returned matrix ``X``; the bundle carries it (``bundle.X``) plus
    the SHAP column names and per-block slice indices.

    Input shapes are duck-typed for cross-task compatibility:

      * ``articles``:    Sequence of Article-like objects (any object with
        ``headline``/``body`` attributes; ``Article``, ``SimpleNamespace``,
        DataFrame row). The article's ``id`` attribute (or its index as a
        fallback) is used to key into dict-shaped inputs.
      * ``jev_signals``:  One of:
          - ``list[list[Signal]]`` (the canonical fn-B shape — 19 cols).
          - ``dict[str, float]`` keyed by article id (fn-C simulated Jev).
          - ``dict[str, list[Signal]]`` keyed by article id.
          - ``np.ndarray`` of shape (n,) or (n, 19).
          - ``None`` for an all-zero JEV block (still 19 cols).
      * ``tfidf``:        One of:
          - ``TfidfFeaturizer`` (fit).
          - ``(vectorizer, sparse_matrix)`` tuple.
          - raw sparse matrix / 2D ndarray.
          - already-fit ``sklearn.feature_extraction.text.TfidfVectorizer``.
      * ``metadata``:     One of:
          - ``MetadataFeatures`` instance (stateless).
          - ``dict[str, dict[str, float]]`` keyed by article id.
          - ``list[dict[str, float]]`` (positional).
          - ``np.ndarray`` shape (n, k).
          - ``None``.

    Args:
        articles:    list of article-like objects.
        jev_signals: see above.
        tfidf:       see above.
        metadata:    see above.
        top_k_tfidf: optional cap on TF-IDF columns (keep top-k by sum).
        dense:       True (default) -> densify the result into np.ndarray.
                     False -> return scipy.sparse.csr_matrix (useful when
                     TF-IDF vocab is large).

    Returns:
        ``(X, bundle)`` — X is np.ndarray (or csr_matrix if dense=False),
        bundle is a FeatureBundle with ``column_names``, ``feature_names``
        (alias), ``jev_slice``/``tfidf_slice``/``meta_slice``, and the
        fn-C-style ``{jev,tfidf,metadata}_indices`` np.ndarrays.

    Raises:
        ValueError: empty article list.
        TypeError:  unsupported input shape for jev_signals / tfidf / metadata.
    """
    n = len(articles)
    if n == 0:
        raise ValueError("build_feature_matrix: empty article list")

    # Resolve article ids (used for dict-keyed inputs). Fall back to index.
    article_ids: list[str] = []
    for i, a in enumerate(articles):
        aid = getattr(a, "id", None)
        if aid is None or aid == "":
            aid = str(i)
        article_ids.append(str(aid))

    # --- JEV block (always dense float32, 19 cols) ------------------------
    jev_mat, jev_names = _coerce_jev_block(jev_signals, article_ids, n)

    # --- TF-IDF block (sparse CSR; optionally densified) ------------------
    tfidf_mat, tfidf_names_all, tfidf_vectorizer = _coerce_tfidf_block(tfidf, articles)
    tfidf_dim = tfidf_mat.shape[1]
    if top_k_tfidf is not None and tfidf_dim > top_k_tfidf:
        col_sums = np.asarray(tfidf_mat.sum(axis=0)).ravel()
        # top-k by descending sum, then re-sort ascending for stability.
        top_idx = np.argsort(col_sums)[::-1][:top_k_tfidf]
        top_idx = np.sort(top_idx)
        tfidf_mat = tfidf_mat[:, top_idx]
        tfidf_names = [tfidf_names_all[i] for i in top_idx]
    else:
        tfidf_names = tfidf_names_all
    tfidf_block_names = [f"tfidf_{t}" for t in tfidf_names]

    # --- Metadata block (always dense float32, variable k cols) -----------
    meta_mat, meta_names, meta_keys = _coerce_metadata_block(
        metadata, articles, article_ids, n
    )

    # --- Concatenate ------------------------------------------------------
    if dense:
        if sparse.issparse(tfidf_mat):
            tfidf_dense = tfidf_mat.toarray().astype(np.float32)
        else:
            tfidf_dense = np.asarray(tfidf_mat, dtype=np.float32)
        if jev_mat.shape[1] == 0 and meta_mat.shape[1] == 0:
            X = tfidf_dense
        elif jev_mat.shape[1] == 0:
            X = np.hstack([tfidf_dense, meta_mat])
        elif meta_mat.shape[1] == 0:
            X = np.hstack([jev_mat, tfidf_dense])
        else:
            X = np.hstack([jev_mat, tfidf_dense, meta_mat])
    else:
        blocks = []
        if jev_mat.shape[1] > 0:
            blocks.append(sparse.csr_matrix(jev_mat))
        if tfidf_mat.shape[1] > 0:
            blocks.append(tfidf_mat if sparse.issparse(tfidf_mat)
                          else sparse.csr_matrix(tfidf_mat))
        if meta_mat.shape[1] > 0:
            blocks.append(sparse.csr_matrix(meta_mat))
        if not blocks:
            X = sparse.csr_matrix((n, 0), dtype=np.float32)
        else:
            X = sparse.hstack(blocks, format="csr")

    # Slices
    n_jev = len(jev_names)
    n_tfidf = len(tfidf_block_names)
    n_meta = len(meta_names)
    column_names = jev_names + tfidf_block_names + meta_names
    bundle = FeatureBundle(
        X=X,
        column_names=column_names,
        feature_names=list(column_names),
        jev_slice=(0, n_jev),
        tfidf_slice=(n_jev, n_jev + n_tfidf),
        meta_slice=(n_jev + n_tfidf, n_jev + n_tfidf + n_meta),
        tfidf_vocabulary=list(tfidf_names),
        metadata_keys=list(meta_keys),
        tfidf_vectorizer=tfidf_vectorizer,
        n_tfidf=n_tfidf,
        n_jev=n_jev,
        n_meta=n_meta,
    )
    return X, bundle
