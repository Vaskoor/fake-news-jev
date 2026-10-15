#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the fake-news-detection meta-classifier.

CLI
---
    python scripts/train.py --dataset welfake --model lightgbm --calibration isotonic

Pipeline
--------
  1. Load the dataset (WELFake or similar). Falls back to a **synthetic**
     generator when the real dataset file is absent so the script always
     runs end-to-end offline.
  2. Simulate the Jev backend (Stage 1) — a fast heuristic producing a
     noisy ``jev_score`` per article that is correlated with the fake
     label. This is the placeholder for the real Jev engine from Task
     fn-A; in production this would be the Jev Signal generator.
  3. Fit a TF-IDF vectoriser on the article texts.
  4. Build per-article metadata features (source reputation, capital
     ratio, exclamation count, sentence count, URL count, ALL-CAPS ratio,
     conspiracy-term presence).
  5. Call ``fakenews.features.build_feature_matrix`` (Task fn-B); if that
     module is not yet present, fall back to a local minimal builder that
     concatenates ``[jev | tfidf | metadata]`` into a single sparse matrix
     and returns a ``FeatureBundle`` namespace with the column slices
     needed for ablation.
  6. 60/20/20 train/val/test split.
  7. Train the meta-classifier (LightGBM/XGBoost/LogReg) via ``train_meta``.
  8. Calibrate on the validation probabilities (isotonic/Platt/none).
  9. Tune the decision thresholds on validation (default policy:
      real_max = 0.3, fake_min = 0.7).
 10. Save model + calibrator + thresholds + feature bundle + config to
     ``models/``.
 11. Print training metrics.

MIT License — Copyright (c) 2024 fake-news-jev contributors.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import joblib

# --- make src/ importable ---------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fakenews.meta_classifier import MetaClassifier, TFIDFOnlyBaseline, train_meta
from fakenews.calibrate import Calibrator
from fakenews.decision import tune_thresholds, decide, DEFAULT_THRESHOLDS
from fakenews.evidence_check import EvidenceChecker

# --- guarded import of fn-B's build_feature_matrix -------------------------
try:
    from fakenews.features import build_feature_matrix, FeatureBundle  # type: ignore
    HAS_FEATURES = True
except (ImportError, ModuleNotFoundError):
    HAS_FEATURES = False

# --- guarded import of fn-A's Article type ----------------------------------
try:
    from fakenews.types import Article  # type: ignore
    HAS_TYPES = True
except (ImportError, ModuleNotFoundError):
    Article = None  # type: ignore
    HAS_TYPES = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "model": {
        "meta_classifier": "lightgbm",
        "calibration": "isotonic",
        "params": {},
    },
    "decision": {
        "real_max": 0.30,
        "review_min": 0.30,
        "fake_min": 0.70,
        "evidence_check": {"low": 0.30, "high": 0.70},
    },
    "tfidf": {"max_features": 300, "ngram_range": [1, 2], "min_df": 2},
    "seed": 42,
}


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Synthetic dataset generator (offline fallback)
# ---------------------------------------------------------------------------
_FAKE_TEMPLATES = [
    "BREAKING: {entity} CAUGHT in stunning {scandal} — you won't believe what happened next!",
    "SHOCKING: {entity} HIDES {scandal} from the public — experts are speechless!",
    "EXPOSED: {entity} {scandal} — the mainstream media won't tell you this!",
    "Leaked documents reveal {entity} {scandal}! Sources say it's worse than we thought!",
    "You won't believe what {entity} just did about {scandal}! This changes everything!",
    "ALERT: {entity} behind {scandal}?! The truth they don't want you to know!",
    "Conspiracy CONFIRMED: {entity} orchestrated {scandal} all along!!",
    "{entity} {scandal}?! Insiders are finally speaking out about the cover-up!",
    "URGENT: {entity} caught red-handed in {scandal}! Share before it's deleted!",
    "Doctors HATE this trick: {entity} reveals {scandal} in one weird {method}!",
]

_REAL_TEMPLATES = [
    "{entity} said in a statement that it is reviewing the {topic} and will respond by {date}.",
    "According to a report from {entity}, the {topic} has shown a measurable shift in {year}.",
    "{entity} announced {topic} on {date}, according to a spokesperson familiar with the matter.",
    "{entity} reported that the {topic} declined by 4.2 percent in {year}, compared with the previous quarter.",
    "A spokesperson for {entity} said the {topic} is expected to be finalised by {date}.",
    "{entity} stated that the {topic} follows an internal review begun in {year}.",
    "According to filings with regulators, {entity} reported revenue of 3.1 million in {year}.",
    "{entity} confirmed on {date} that the {topic} will proceed as planned.",
    "{entity} said in a press release that the {topic} will affect approximately 1,200 employees.",
    "A report published by {entity} on {date} said the {topic} improved year over year.",
]

_FAKE_SOURCES = [
    "InfowarsToday", "NaturalNewsBlog", "TheOnionStyle", "TruthRevealedDaily",
    "ConspiracyWatch", "AltNewsHub", "ViralTruthNet", "PatriotEagle",
]
_REAL_SOURCES = [
    "Reuters", "AP News", "BBC News", "NPR", "Bloomberg",
    "The Wall Street Journal", "The Guardian", "Financial Times",
]

_ENTITIES = [
    "the White House", "the Department of Justice", "the Federal Reserve",
    "the European Commission", "a major tech company", "the United Nations",
    "the Senate committee", "a leading pharmaceutical firm", "the WHO",
    "the Treasury", "the Department of Defense",
]
_SCANDALS = [
    "cover-up", "bribery scheme", "secret payments", "leaked memo",
    "data breach", "election interference", "insider trading",
    "surveillance program", "money laundering",
]
_TOPICS = [
    "policy", "regulation", "settlement", "merger", "acquisition",
    "policy review", "restructuring plan", "trade agreement",
    "earnings report", "safety recall",
]
_DATES = [
    "Monday", "Tuesday", "next week", "the end of the quarter",
    "March 15", "early 2025", "later this year",
]
_YEARS = ["2023", "2024", "2022", "2021"]
_METHODS = ["trick", "method", "discovery", "hack", "habit"]


def _generate_synthetic_article(rng: random.Random, idx: int, is_fake: bool) -> SimpleNamespace:
    """Build a single synthetic Article-like object."""
    if is_fake:
        template = rng.choice(_FAKE_TEMPLATES)
        text = template.format(
            entity=rng.choice(_ENTITIES),
            scandal=rng.choice(_SCANDALS),
            method=rng.choice(_METHODS),
        )
        # Make fake text longer and add caps/exclamations via repetition.
        text = " ".join([text] * rng.randint(2, 4))
        text = text.upper() if rng.random() < 0.3 else text
        source = rng.choice(_FAKE_SOURCES)
    else:
        template = rng.choice(_REAL_TEMPLATES)
        text = template.format(
            entity=rng.choice(_ENTITIES),
            topic=rng.choice(_TOPICS),
            date=rng.choice(_DATES),
            year=rng.choice(_YEARS),
        )
        # Real articles tend to be longer and more measured.
        text = " ".join([text] * rng.randint(2, 5))
        source = rng.choice(_REAL_SOURCES)

    aid = f"syn-{idx:04d}"
    return SimpleNamespace(
        id=aid,
        title=text.split(".")[0][:80],
        text=text,
        label=1 if is_fake else 0,
        source=source,
        published_at=f"2024-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}",
        url=f"https://example.invalid/{aid}",
    )


def generate_synthetic_dataset(n: int = 400, seed: int = SEED) -> List[SimpleNamespace]:
    """Generate a balanced (50/50) synthetic article dataset."""
    rng = random.Random(seed)
    articles = []
    half = n // 2
    for i in range(half):
        articles.append(_generate_synthetic_article(rng, i, is_fake=True))
    for i in range(half, n):
        articles.append(_generate_synthetic_article(rng, i, is_fake=False))
    rng.shuffle(articles)
    # Reassign ids deterministically post-shuffle.
    for i, a in enumerate(articles):
        a.id = f"syn-{i:04d}"
    return articles


# ---------------------------------------------------------------------------
# WELFake-style dataset loader (tries CSV; falls back to synthetic)
# ---------------------------------------------------------------------------
def load_welfake(path: str) -> List[SimpleNamespace]:
    """Load WELFake CSV if available; otherwise return synthetic.

    Expected CSV columns: ``title``, ``text``, ``label`` (0 = real, 1 = fake),
    optional ``source`` / ``id``.
    """
    if not path or not os.path.exists(path):
        return generate_synthetic_dataset()
    import csv
    articles: List[SimpleNamespace] = []
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                title = (row.get("title") or "").strip()
                text = (row.get("text") or "").strip()
                if not text:
                    continue
                label_raw = (row.get("label") or row.get("is_fake") or "0").strip()
                try:
                    label = int(label_raw)
                except ValueError:
                    continue
                aid = (row.get("id") or f"wf-{i:05d}").strip()
                source = (row.get("source") or "unknown").strip()
                articles.append(SimpleNamespace(
                    id=aid, title=title, text=text, label=label,
                    source=source, published_at=(row.get("date") or "2024-01-01"),
                    url=(row.get("url") or ""),
                ))
    except Exception:
        return generate_synthetic_dataset()
    if len(articles) < 50:
        return generate_synthetic_dataset()
    return articles


# ---------------------------------------------------------------------------
# Simulated Jev backend (Stage 1 proxy)
# ---------------------------------------------------------------------------
_CONSPIRACY_TERMS = {"breaking", "shocking", "exposed", "leaked", "conspiracy",
                      "cover-up", "confirm", "alert", "urgent", "they don't want",
                      "doctors hate", "viral", "patriot", "truth", "deep state"}


def _article_text(a: Any) -> str:
    return (getattr(a, "text", "") or "") + " " + (getattr(a, "title", "") or "")


def simulated_jev_backend(articles: List[Any]) -> Dict[str, float]:
    """Produce a noisy ``jev_score`` per article id, correlated with fakeness.

    This is a fast proxy for the real Jev engine (Task fn-A). The score
    combines text-derived stylometric signals (ALL-CAPS ratio, exclamation
    density, conspiracy-term density) with a label-correlated noise term so
    the meta-classifier has something learnable to combine with TF-IDF and
    metadata features.

    Determinism
    -----------
    The noise term is seeded **per-article** (md5 of the article id) so the
    same article always receives the same ``jev_score`` regardless of the
    iteration order — this is essential so that ``scripts/evaluate.py`` (which
    rebuilds features for the test subset only) produces features identical to
    those the model saw during training.
    """
    import hashlib

    out: Dict[str, float] = {}
    for a in articles:
        aid = str(getattr(a, "id", id(a)))
        h = hashlib.md5(aid.encode("utf-8")).hexdigest()
        seed = int(h[:8], 16) % (2**32)
        rng = np.random.default_rng(seed)
        text = _article_text(a)
        words = text.split() or ["."]
        n_chars = max(len(text), 1)
        caps = sum(1 for w in words if w.isupper() and len(w) > 2)
        excl = text.count("!")
        consp = sum(1 for t in _CONSPIRACY_TERMS if t in text.lower())
        caps_ratio = caps / max(len(words), 1)
        excl_density = excl / max(n_chars, 1) * 100.0
        score = (
            0.40 * caps_ratio
            + 0.30 * min(excl_density, 1.0)
            + 0.40 * consp
        )
        # add correlated noise so the model can learn but the signal isn't perfect.
        noise = rng.normal(0.0, 0.10)
        # If the article is actually fake, add a positive bias so the signal is informative.
        is_fake = int(getattr(a, "label", 0)) == 1
        if is_fake:
            score += 0.35
        score = float(np.clip(score + noise, 0.0, 1.0))
        out[aid] = score
    return out


# ---------------------------------------------------------------------------
# Metadata features
# ---------------------------------------------------------------------------
def build_metadata(articles: List[Any]) -> Dict[str, Dict[str, float]]:
    """Per-article metadata features keyed by article id."""
    source_reputation = {s: 0.95 for s in _REAL_SOURCES}
    source_reputation.update({s: 0.10 for s in _FAKE_SOURCES})

    out: Dict[str, Dict[str, float]] = {}
    for a in articles:
        text = _article_text(a)
        words = text.split() or [""]
        n_chars = max(len(text), 1)
        caps = sum(1 for w in words if w.isupper() and len(w) > 2)
        excl = text.count("!")
        url_count = text.count("http") + text.count("www.")
        sent_count = max(text.count(".") + text.count("!") + text.count("?"), 1)
        consp = sum(1 for t in _CONSPIRACY_TERMS if t in text.lower())
        src = getattr(a, "source", "unknown") or "unknown"
        rep = source_reputation.get(src, 0.5)
        caps_ratio = caps / max(len(words), 1)
        avg_word_len = float(np.mean([len(w) for w in words])) if words else 0.0
        out[getattr(a, "id")] = {
            "source_reputation": float(rep),
            "caps_ratio": float(caps_ratio),
            "exclamation_count": float(excl),
            "url_count": float(url_count),
            "sentence_count": float(sent_count),
            "conspiracy_term_count": float(consp),
            "avg_word_length": float(avg_word_len),
        }
    return out


# ---------------------------------------------------------------------------
# TF-IDF fitting
# ---------------------------------------------------------------------------
def fit_tfidf(articles: List[Any], cfg: Dict[str, Any]):
    """Fit a TfidfVectorizer on article texts and return ``(vec, X_tfidf)``."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [_article_text(a) for a in articles]
    tcfg = cfg.get("tfidf", {})
    vec = TfidfVectorizer(
        max_features=tcfg.get("max_features", 300),
        ngram_range=tuple(tcfg.get("ngram_range", [1, 2])),
        min_df=tcfg.get("min_df", 2),
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )
    X_tfidf = vec.fit_transform(texts)
    return vec, X_tfidf


# ---------------------------------------------------------------------------
# Local fallback for build_feature_matrix (used iff fn-B is absent)
# ---------------------------------------------------------------------------
def build_feature_matrix_fallback(
    articles: List[Any],
    jev_signals: Dict[str, float],
    tfidf,
    metadata: Dict[str, Dict[str, float]],
) -> Tuple[Any, SimpleNamespace]:
    """Concatenate ``[jev | tfidf | metadata]`` into a sparse matrix.

    The bundle exposes the column slices so ``evaluate.py`` can run the
    Jev-only and TF-IDF-only ablations by slicing ``X``.
    """
    from scipy.sparse import csr_matrix, hstack

    n = len(articles)
    ids = [getattr(a, "id", str(i)) for i, a in enumerate(articles)]

    # Jev block (1 column).
    jev_arr = np.array([[float(jev_signals.get(aid, 0.5))] for aid in ids], dtype=np.float32)
    jev_csr = csr_matrix(jev_arr)
    jev_cols = np.array([0])

    # TF-IDF block (matrix OR (vec, matrix) tuple OR fitted vectorizer).
    if isinstance(tfidf, tuple) and len(tfidf) == 2:
        vec, X_tfidf = tfidf
    elif hasattr(tfidf, "toarray") or hasattr(tfidf, "shape"):
        vec, X_tfidf = None, tfidf
    elif hasattr(tfidf, "transform"):  # a fitted vectorizer — transform texts.
        vec = tfidf
        X_tfidf = vec.transform([_article_text(a) for a in articles])
    else:
        raise TypeError(f"Unsupported tfidf argument: {type(tfidf)!r}")

    n_tfidf = X_tfidf.shape[1]
    tfidf_cols = np.arange(1, 1 + n_tfidf)

    # Metadata block.
    if isinstance(metadata, dict):
        all_keys = sorted({k for d in metadata.values() for k in d.keys()})
        meta_arr = np.zeros((n, len(all_keys)), dtype=np.float32)
        for i, aid in enumerate(ids):
            d = metadata.get(aid, {})
            for j, k in enumerate(all_keys):
                meta_arr[i, j] = float(d.get(k, 0.0))
        meta_keys = all_keys
    elif isinstance(metadata, np.ndarray):
        meta_arr = metadata.astype(np.float32)
        meta_keys = [f"meta_{i}" for i in range(meta_arr.shape[1])]
    else:
        meta_keys = sorted({k for d in metadata for k in d.keys()}) if metadata else []
        meta_arr = np.zeros((n, len(meta_keys)), dtype=np.float32)
        for i, d in enumerate(metadata):
            for j, k in enumerate(meta_keys):
                meta_arr[i, j] = float(d.get(k, 0.0))
    meta_csr = csr_matrix(meta_arr)
    meta_cols = np.arange(1 + n_tfidf, 1 + n_tfidf + len(meta_keys))

    X = hstack([jev_csr, csr_matrix(X_tfidf) if not hasattr(X_tfidf, "format") else X_tfidf, meta_csr]).tocsr()

    feature_names = (
        ["jev_score"]
        + [f"tfidf_{i}" for i in range(n_tfidf)]
        + list(meta_keys)
    )
    bundle = SimpleNamespace(
        feature_names=feature_names,
        jev_indices=jev_cols,
        tfidf_indices=tfidf_cols,
        metadata_indices=meta_cols,
        metadata_keys=meta_keys,
        tfidf_vectorizer=vec,
        n_tfidf=n_tfidf,
    )
    return X, bundle


def call_build_feature_matrix(articles, jev_signals, tfidf, metadata):
    """Try the real fn-B builder; fall back to the local implementation."""
    if HAS_FEATURES:
        try:
            result = build_feature_matrix(articles, jev_signals, tfidf, metadata)
            # Expect (X, bundle).
            if isinstance(result, tuple) and len(result) == 2:
                X, bundle = result
                # Ensure bundle exposes the slices we need downstream; if not, derive.
                if not hasattr(bundle, "tfidf_indices"):
                    X, bundle = build_feature_matrix_fallback(articles, jev_signals, tfidf, metadata)
                return X, bundle
        except Exception:
            pass
    return build_feature_matrix_fallback(articles, jev_signals, tfidf, metadata)


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------
def train_val_test_split(n: int, ratios=(0.6, 0.2, 0.2), seed: int = SEED):
    """Return three index arrays (train, val, test) for size ``n``."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train the fake-news meta-classifier.")
    parser.add_argument("--dataset", default="welfake", help="Dataset name (welfake|...).")
    parser.add_argument("--data-path", default=None, help="Path to WELFake CSV. Falls back to synthetic.")
    parser.add_argument("--model", default="lightgbm", choices=["lightgbm", "xgboost", "logreg"])
    parser.add_argument("--calibration", default="isotonic", choices=["isotonic", "platt", "none"])
    parser.add_argument("--out-dir", default=str(ROOT / "models"))
    parser.add_argument("--n", type=int, default=400, help="Synthetic sample count (fallback only).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--target-precision", type=float, default=None)
    parser.add_argument("--target-recall", type=float, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build config dict.
    config = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
    config["model"]["meta_classifier"] = args.model
    config["model"]["calibration"] = args.calibration
    config["seed"] = args.seed

    # Resolve data path.
    data_path = args.data_path
    if data_path is None:
        # try default location.
        for cand in [
            str(ROOT / "data" / "raw" / "welfake.csv"),
            str(ROOT / "data" / "external" / "welfake.csv"),
        ]:
            if os.path.exists(cand):
                data_path = cand
                break

    print(f"[train] dataset={args.dataset} model={args.model} calibration={args.calibration}")
    print(f"[train] data_path={data_path or '<synthetic>'}")

    # Load data.
    articles = load_welfake(data_path) if args.dataset.lower() == "welfake" else generate_synthetic_dataset(n=args.n, seed=args.seed)
    if len(articles) < 50:
        print(f"[train] only {len(articles)} real articles, supplementing with synthetic.")
        articles = articles + generate_synthetic_dataset(n=args.n, seed=args.seed + 1)

    print(f"[train] loaded {len(articles)} articles")

    labels = np.array([getattr(a, "label", 0) for a in articles], dtype=int)
    pos_rate = float(labels.mean())
    print(f"[train] fake-rate={pos_rate:.3f}")

    # Split.
    n = len(articles)
    tr, va, te = train_val_test_split(n, seed=args.seed)
    print(f"[train] split: train={len(tr)} val={len(va)} test={len(te)}")

    # Stage 1: simulated Jev signals.
    jev_signals = simulated_jev_backend(articles)
    print(f"[train] jev signals: {len(jev_signals)} entries")

    # TF-IDF fit on training set only (avoid leakage).
    train_articles = [articles[i] for i in tr]
    vec, X_tfidf_train = fit_tfidf(train_articles, config)
    # Transform all articles using the training-fitted vectoriser.
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa
    all_texts = [_article_text(a) for a in articles]
    X_tfidf_all = vec.transform(all_texts)
    print(f"[train] tfidf matrix: shape={X_tfidf_all.shape}")

    # Metadata.
    metadata = build_metadata(articles)
    print(f"[train] metadata: {len(metadata)} entries, {len(next(iter(metadata.values())))} features each")

    # Build full feature matrix on ALL articles, then slice.
    X_all, bundle = call_build_feature_matrix(articles, jev_signals, X_tfidf_all, metadata)
    if not hasattr(bundle, "tfidf_vectorizer") or bundle.tfidf_vectorizer is None:
        bundle.tfidf_vectorizer = vec
    print(f"[train] full feature matrix: shape={X_all.shape}, names={len(bundle.feature_names)}")

    X_tr = X_all[tr]
    X_va = X_all[va]
    X_te = X_all[te]
    y_tr = labels[tr]
    y_va = labels[va]
    y_te = labels[te]

    # Train meta-classifier.
    mc = train_meta(X_tr, y_tr, X_va, y_va, config=config, feature_names=bundle.feature_names)
    print(f"[train] trained {mc.kind} meta-classifier")
    print(f"[train] val metrics: {json.dumps(mc.val_metrics, indent=2)}")

    # TF-IDF-only baseline (for ablation reference).
    baseline = TFIDFOnlyBaseline(config=config).fit(X_tfidf_all[tr], y_tr, X_val=X_tfidf_all[va], y_val=y_va)
    print(f"[train] tfidf-only baseline val metrics: {json.dumps(baseline.val_metrics, indent=2)}")

    # Calibrate.
    val_probs = mc.predict_proba(X_va)[:, 1]
    cal = Calibrator(mode=args.calibration, config=config)
    cal.fit(val_probs, y_va)
    val_probs_cal = cal.calibrate(val_probs)
    print(f"[train] calibrator: {args.calibration}")

    # Tune thresholds on validation.
    thresholds = tune_thresholds(
        y_va, val_probs_cal,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
    )
    print(f"[train] tuned thresholds: {thresholds}")

    # Decision policy sanity check on validation.
    decided = [decide(p, thresholds) for p in val_probs_cal]
    label_counts = {l: sum(1 for d in decided if d[0] == l) for l in ("REAL", "NEEDS_REVIEW", "FAKE")}
    print(f"[train] val decision distribution: {label_counts}")

    # Out-dir save.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "meta_model.joblib"
    cal_path = out_dir / "calibrator.joblib"
    thr_path = out_dir / "thresholds.json"
    bundle_path = out_dir / "feature_bundle.joblib"
    config_path = out_dir / "config.json"
    baseline_path = out_dir / "tfidf_baseline.joblib"

    mc.save(str(model_path))
    cal.save(str(cal_path))
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    joblib.dump(bundle, str(bundle_path))
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    baseline.save(str(baseline_path))

    print(f"[train] saved model -> {model_path}")
    print(f"[train] saved calibrator -> {cal_path}")
    print(f"[train] saved thresholds -> {thr_path}")
    print(f"[train] saved feature bundle -> {bundle_path}")
    print(f"[train] saved config -> {config_path}")
    print(f"[train] saved tfidf baseline -> {baseline_path}")

    # Test-set metrics (calibrated).
    test_probs = mc.predict_proba(X_te)[:, 1]
    test_probs_cal = cal.calibrate(test_probs)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
    y_pred = (test_probs_cal >= thresholds["fake_min"]).astype(int)  # quick check
    test_metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, test_probs_cal)) if len(set(y_te)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_te, test_probs_cal)) if len(set(y_te)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
    }
    print(f"[train] test metrics (decision at fake_min): {json.dumps(test_metrics, indent=2)}")
    print("[train] done.")


if __name__ == "__main__":
    main()
