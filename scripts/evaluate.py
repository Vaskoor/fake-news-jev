#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate the trained fake-news meta-classifier.

CLI
---
    python scripts/evaluate.py --model-path models/ --dataset welfake

Reports
-------
* Core metrics on the held-out **test** split:
    accuracy, precision, recall, F1 (decisive predictions only),
    ROC-AUC, PR-AUC, ECE, confusion matrix, % routed to review.
* Ablations: Jev-only vs TF-IDF-only vs Hybrid (the full model).
* Stage 3 evidence: with vs without the fact-check adjustment path.
* Robustness: paraphrased (sentence-shuffled + synonym-swapped
  deterministically), satirical, and mixed true/false articles.

Artifacts
---------
* ``results/eval.json``      — full metrics dict.
* ``results/reliability.png``— reliability diagram (matplotlib; guarded).

The script regenerates the **same** synthetic dataset / split that
``scripts/train.py`` used (deterministic seed), so the test indices match
the held-out partition the model was *not* trained on. The TF-IDF
vectoriser and metadata schema are loaded from the saved feature bundle.

MIT License — Copyright (c) 2024 fake-news-jev contributors.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import joblib

# --- make src/ and scripts/ importable -------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fakenews.meta_classifier import MetaClassifier, TFIDFOnlyBaseline
from fakenews.calibrate import (
    Calibrator, expected_calibration_error, reliability_data, plot_reliability,
)
from fakenews.decision import decide, tune_thresholds
from fakenews.explain import explain, HAS_SHAP
from fakenews.evidence_check import EvidenceChecker

# Import the data helpers from the sibling train.py (one of the 7 files).
from train import (  # type: ignore
    generate_synthetic_dataset,
    load_welfake,
    simulated_jev_backend,
    build_metadata,
    call_build_feature_matrix,
    train_val_test_split,
    _article_text,
    _FAKE_SOURCES,
    _REAL_SOURCES,
    DEFAULT_CONFIG,
    SEED,
)


# ===========================================================================
# Metrics helpers
# ===========================================================================
def _decisive_metrics(y_true, y_proba, thresholds):
    """Compute binary metrics on the decisive (non-review) predictions.

    NEEDS_REVIEW samples are *excluded* from accuracy/precision/recall/F1
    denominators; their share is reported separately as ``review_rate``.
    ROC-AUC / PR-AUC / ECE use the raw probabilities over all samples.
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, confusion_matrix,
    )
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float).ravel()

    labels = np.array([decide(p, thresholds)[0] for p in y_proba])
    # Map to binary y_pred on decisive samples.
    y_pred = np.full_like(y_true, -1)
    y_pred[labels == "REAL"] = 0
    y_pred[labels == "FAKE"] = 1
    # review stays -1

    decisive = y_pred != -1
    review_rate = float((~decisive).sum()) / len(y_true) if len(y_true) else float("nan")
    if decisive.sum() == 0:
        return {
            "accuracy": float("nan"), "precision": float("nan"),
            "recall": float("nan"), "f1": float("nan"),
            "roc_auc": float("nan"), "pr_auc": float("nan"),
            "ece": float("nan"), "review_rate": review_rate,
            "confusion_matrix": [[0, 0], [0, 0]],
            "decisive_count": 0,
        }

    yt = y_true[decisive]
    yp = y_pred[decisive]
    metrics = {
        "accuracy": float(accuracy_score(yt, yp)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall": float(recall_score(yt, yp, zero_division=0)),
        "f1": float(f1_score(yt, yp, zero_division=0)),
        "review_rate": review_rate,
        "decisive_count": int(decisive.sum()),
    }
    # Probabilistic metrics over ALL samples.
    if len(set(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    metrics["ece"] = float(expected_calibration_error(y_true, y_proba, n_bins=10))
    metrics["confusion_matrix"] = confusion_matrix(yt, yp, labels=[0, 1]).tolist()
    return metrics


def _proba_metrics(y_true, y_proba):
    """Pure probability metrics (no decision policy)."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(set(y_true)) > 1 else float("nan"),
        "ece": float(expected_calibration_error(y_true, y_proba, n_bins=10)),
    }


# ===========================================================================
# Ablation training
# ===========================================================================
def _train_jev_only(X_train, y_train, X_test, y_test, jev_indices):
    """Train a LogReg on the Jev block only and return test probabilities."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    jev_idx = np.asarray(jev_indices).ravel()
    Xtr = np.asarray(X_train.tocsr()[:, jev_idx].todense() if hasattr(X_train, "tocsr") else X_train[:, jev_idx], dtype=float)
    Xte = np.asarray(X_test.tocsr()[:, jev_idx].todense() if hasattr(X_test, "tocsr") else X_test[:, jev_idx], dtype=float)
    scaler = StandardScaler(with_mean=True, with_std=True).fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    Xte_s = scaler.transform(Xte)
    lr = LogisticRegression(C=1.0, max_iter=500, random_state=SEED)
    lr.fit(Xtr_s, y_train)
    return lr.predict_proba(Xte_s)[:, 1]


def _eval_tfidf_only(baseline: TFIDFOnlyBaseline, X_tfidf_test, y_test):
    """Use the saved TFIDFOnlyBaseline; return test probabilities."""
    return baseline.predict_proba(X_tfidf_test)[:, 1]


# ===========================================================================
# Robustness — deterministic perturbations
# ===========================================================================
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|[\n\r]+")
_SYNONYMS = {
    "stunning": ["remarkable", "striking"],
    "shocking": ["surprising", "alarming"],
    "caught": ["found", "discovered"],
    "revealed": ["disclosed", "uncovered"],
    "report": ["account", "account"],
    "announced": ["stated", "declared"],
    "percent": ["per cent", "pct"],
    "spokesperson": ["representative", "spokesperson"],
    "statement": ["declaration", "remark"],
    "finalised": ["completed", "concluded"],
}


def _paraphrase(text: str, seed: int = SEED) -> str:
    """Deterministic paraphrase: shuffle sentences + swap synonyms."""
    rng = random.Random(seed)
    sents = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if len(sents) > 1:
        rng.shuffle(sents)
    out_sents = []
    for s in sents:
        for src, tgt in _SYNONYMS.items():
            if src in s:
                s = s.replace(src, rng.choice(tgt))
        out_sents.append(s)
    return " ".join(out_sents)


def _paraphrase_articles(articles: List[Any], seed: int = SEED) -> List[SimpleNamespace]:
    """Return a list of paraphrased copies."""
    out = []
    for i, a in enumerate(articles):
        text = _article_text(a)
        ptext = _paraphrase(text, seed=seed + i)
        out.append(SimpleNamespace(
            id=getattr(a, "id") + "-para",
            title=getattr(a, "title", "") + " (paraphrased)",
            text=ptext,
            label=getattr(a, "label"),
            source=getattr(a, "source", "unknown"),
            published_at=getattr(a, "published_at", "2024-01-01"),
            url=getattr(a, "url", ""),
        ))
    return out


# Satirical articles: factual-looking tone, absurd claims (label = 0, real,
# but easy to misclassify as fake).
_SATIRE_TEMPLATES = [
    "Local cat announces candidacy for mayor, promises universal nap subsidies and mandatory sunbeam breaks.",
    "Senate passes resolution designating Tuesday as 'National Burrito Day'; bipartisan queso compromise reportedly near.",
    "Researchers confirm that pizza, when cut into triangles, is mathematically superior to all other polygons.",
    "City council debates whether clouds should be regulated under zoning ordinances; protests expected by Tuesday.",
    "Office coffee machine issues formal apology for Monday, citing overwhelming workload and inadequate creamer supply.",
]

# Mixed true/false articles: real lede + fabricated follow-up.
_MIXED_TEMPLATES = [
    "Reuters reported that the Federal Reserve raised rates by 0.25 percent on Monday. SECRET SOURCES say the real reason is a cover-up of an alien treaty.",
    "AP News confirmed that the Treasury published Q3 revenue of 3.1 million. The 3.1 million was actually embezzled to fund a deep-state weather machine!",
    "BBC News stated that the WHO confirmed an outbreak in 2024. The outbreak was actually engineered to control the population!",
]


def _build_satire_articles(n: int = 25, seed: int = SEED) -> List[SimpleNamespace]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = rng.choice(_SATIRE_TEMPLATES)
        out.append(SimpleNamespace(
            id=f"satire-{i:03d}",
            title=t.split(".")[0][:80],
            text=" ".join([t] * rng.randint(2, 4)),
            label=0,  # satire is not fake news
            source="TheOnionStyle",
            published_at="2024-01-01",
            url="",
        ))
    return out


def _build_mixed_articles(n: int = 25, seed: int = SEED) -> List[SimpleNamespace]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = rng.choice(_MIXED_TEMPLATES)
        out.append(SimpleNamespace(
            id=f"mixed-{i:03d}",
            title=t.split(".")[0][:80],
            text=" ".join([t] * rng.randint(2, 3)),
            label=1,  # mixed = fake (the fabricated claim dominates)
            source=rng.choice(_FAKE_SOURCES),
            published_at="2024-01-01",
            url="",
        ))
    return out


# ===========================================================================
# Main evaluation
# ===========================================================================
def _build_test_features(
    test_articles: List[Any],
    bundle: Any,
    seed: int = SEED,
):
    """Build the test feature matrix using the saved TF-IDF vectoriser.

    Returns the sparse X_test aligned with ``test_articles``.
    """
    jev_signals = simulated_jev_backend(test_articles)
    metadata = build_metadata(test_articles)
    vec = getattr(bundle, "tfidf_vectorizer", None)
    if vec is None:
        raise RuntimeError("Feature bundle has no tfidf_vectorizer; cannot transform test articles.")
    X_tfidf_test = vec.transform([_article_text(a) for a in test_articles])
    X_test, _ = call_build_feature_matrix(test_articles, jev_signals, X_tfidf_test, metadata)
    return X_test, X_tfidf_test, jev_signals, metadata


def _print_table(title: str, rows: List[Tuple[str, Dict[str, float]]], keys: List[str]):
    """Pretty-print a metrics table to stdout."""
    print()
    print("=" * 78)
    print(title)
    print("-" * 78)
    header = f"{'variant':<24}" + "".join(f"{k:>12}" for k in keys)
    print(header)
    print("-" * 78)
    for name, m in rows:
        line = f"{name:<24}"
        for k in keys:
            v = m.get(k, float("nan"))
            line += f"{v:>12.4f}" if isinstance(v, (int, float)) else f"{str(v):>12}"
        print(line)
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the fake-news meta-classifier.")
    parser.add_argument("--model-path", default=str(ROOT / "models"), help="Directory containing model artifacts.")
    parser.add_argument("--dataset", default="welfake")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--n", type=int, default=400, help="Synthetic sample count (must match train.py for offline test reproducibility).")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_dir = Path(args.model_path)
    print(f"[eval] loading artifacts from {model_dir}")
    mc = MetaClassifier.load(str(model_dir / "meta_model.joblib"))
    cal = Calibrator.load(str(model_dir / "calibrator.joblib"))
    with open(model_dir / "thresholds.json", "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    bundle = joblib.load(str(model_dir / "feature_bundle.joblib"))
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    baseline = TFIDFOnlyBaseline.load(str(model_dir / "tfidf_baseline.joblib"))

    print(f"[eval] meta-classifier kind: {mc.kind}")
    print(f"[eval] calibration mode: {cal.mode}")
    print(f"[eval] thresholds: {thresholds}")
    print(f"[eval] feature bundle: feature_names={len(bundle.feature_names)}")

    # --- regenerate the same articles + split used in train.py -----------
    data_path = args.data_path
    if data_path is None:
        for cand in [
            str(ROOT / "data" / "raw" / "welfake.csv"),
            str(ROOT / "data" / "external" / "welfake.csv"),
        ]:
            if os.path.exists(cand):
                data_path = cand
                break
    articles = load_welfake(data_path) if args.dataset.lower() == "welfake" else generate_synthetic_dataset(n=args.n, seed=args.seed)
    if len(articles) < 50:
        articles = articles + generate_synthetic_dataset(n=args.n, seed=args.seed + 1)
    labels = np.array([getattr(a, "label", 0) for a in articles], dtype=int)
    tr, va, te = train_val_test_split(len(articles), seed=args.seed)
    test_articles = [articles[i] for i in te]
    y_test = labels[te]
    print(f"[eval] test set: {len(test_articles)} articles, fake-rate={y_test.mean():.3f}")

    # --- build test features using the saved vectoriser -----------------
    X_test, X_tfidf_test, _, _ = _build_test_features(test_articles, bundle, seed=args.seed)
    print(f"[eval] test feature matrix: shape={X_test.shape}")

    # --- predict + calibrate ---------------------------------------------
    raw_probs = mc.predict_proba(X_test)[:, 1]
    cal_probs = cal.calibrate(raw_probs)

    # --- core metrics ----------------------------------------------------
    core = _decisive_metrics(y_test, cal_probs, thresholds)
    print(f"[eval] core metrics:")
    print(json.dumps(core, indent=2))

    # --- ablations: hybrid vs jev-only vs tfidf-only --------------------
    # Hybrid (already done).
    hybrid_metrics = _decisive_metrics(y_test, cal_probs, thresholds)
    # Jev-only: train a fresh LogReg on the train set's jev column, predict on test.
    train_articles = [articles[i] for i in tr]
    X_train_full, _, _, _ = _build_test_features(train_articles, bundle, seed=args.seed)
    y_train = labels[tr]
    jev_indices = np.asarray(getattr(bundle, "jev_indices", [0])).ravel()
    jev_test_proba = _train_jev_only(X_train_full, y_train, X_test, y_test, jev_indices)
    jev_metrics = _proba_metrics(y_test, jev_test_proba)
    # TF-IDF-only: use saved baseline.
    tfidf_test_proba = _eval_tfidf_only(baseline, X_tfidf_test, y_test)
    tfidf_metrics = _proba_metrics(y_test, tfidf_test_proba)

    # Also: hybrid *uncalibrated* to show calibration impact.
    hybrid_uncal = _proba_metrics(y_test, raw_probs)

    _print_table(
        "ABLATION: feature-set comparison (test set)",
        [
            ("Jev-only", jev_metrics),
            ("TF-IDF-only", tfidf_metrics),
            ("Hybrid (uncal.)", hybrid_uncal),
            ("Hybrid (calibr.)", hybrid_metrics),
        ],
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "ece"],
    )

    # --- Stage 3 evidence: with vs without ------------------------------
    checker = EvidenceChecker(config=config)
    print(f"[eval] evidence_check band: low={checker.low}, high={checker.high}")
    # Without stage 3: decisions as-is.
    decisions_without = [decide(p, thresholds) for p in cal_probs]
    # With stage 3: for NEEDS_REVIEW, run evidence + adjust + re-decide.
    decisions_with = []
    for i, p in enumerate(cal_probs):
        label, reason = decide(p, thresholds)
        if label == "NEEDS_REVIEW":
            claims = checker.extract_claims(test_articles[i])
            ev = checker.check_claims(claims)
            p_adj = checker.adjust_probability(p, ev)
            label, reason = decide(p_adj, thresholds)
            if label == "NEEDS_REVIEW":
                reason = f"{reason} (evidence: {ev.source})"
        decisions_with.append((label, reason))
    y_pred_with = np.array([1 if l == "FAKE" else 0 for l, _ in decisions_with])
    y_pred_without = np.array([1 if l == "FAKE" else 0 for l, _ in decisions_without])
    review_with = sum(1 for l, _ in decisions_with if l == "NEEDS_REVIEW")
    review_without = sum(1 for l, _ in decisions_without if l == "NEEDS_REVIEW")

    from sklearn.metrics import accuracy_score, f1_score
    stage3_metrics = {
        "without_evidence_accuracy": float(accuracy_score(y_test, y_pred_without)),
        "without_evidence_f1": float(f1_score(y_test, y_pred_without, zero_division=0)),
        "without_evidence_review_rate": float(review_without) / len(y_test),
        "with_evidence_accuracy": float(accuracy_score(y_test, y_pred_with)),
        "with_evidence_f1": float(f1_score(y_test, y_pred_with, zero_division=0)),
        "with_evidence_review_rate": float(review_with) / len(y_test),
        "note": "Offline: evidence API key absent → findings=[] → adjust_probability returns p unchanged; "
                "metrics should match unless evidence_check low/high band differs from thresholds.",
    }
    print(f"[eval] stage-3 evidence metrics: {json.dumps(stage3_metrics, indent=2)}")

    # --- SHAP explainability sample (top-k for first 3 test articles) --
    top_features_sample = []
    try:
        for i in range(min(3, len(test_articles))):
            topk = explain(mc, X_test[i:i + 1], feature_names=bundle.feature_names, top_k=5, instance_idx=0)
            top_features_sample.append({
                "article_id": getattr(test_articles[i], "id"),
                "p_fake_calibrated": float(cal_probs[i]),
                "decision": decisions_with[i][0],
                "top_features": [{"name": n, "shap": v} for n, v in topk],
            })
        print(f"[eval] SHAP top-k (first 3 test articles):")
        print(json.dumps(top_features_sample, indent=2))
    except Exception as e:
        print(f"[eval] SHAP sample skipped: {e!r}")

    # --- Robustness: paraphrased -----------------------------------------
    para_articles = _paraphrase_articles(test_articles, seed=args.seed)
    X_para, _, _, _ = _build_test_features(para_articles, bundle, seed=args.seed + 100)
    para_raw = mc.predict_proba(X_para)[:, 1]
    para_cal = cal.calibrate(para_raw)
    # Stability metrics.
    delta_p = np.abs(para_cal - cal_probs)
    decisions_para = np.array([decide(p, thresholds)[0] for p in para_cal])
    decisions_orig = np.array([decide(p, thresholds)[0] for p in cal_probs])
    flip_rate = float((decisions_para != decisions_orig).mean())
    robustness_para = {
        "mean_abs_delta_p": float(delta_p.mean()),
        "max_abs_delta_p": float(delta_p.max()),
        "decision_flip_rate": flip_rate,
        "proba_metrics_on_paraphrased": _proba_metrics(y_test, para_cal),
    }
    print(f"[eval] robustness (paraphrased): {json.dumps(robustness_para, indent=2)}")

    # --- Robustness: satirical -------------------------------------------
    satire_articles = _build_satire_articles(n=25, seed=args.seed)
    X_sat, X_tfidf_sat, _, _ = _build_test_features(satire_articles, bundle, seed=args.seed + 200)
    sat_cal = cal.calibrate(mc.predict_proba(X_sat)[:, 1])
    sat_decisions = [decide(p, thresholds)[0] for p in sat_cal]
    sat_labels = np.array([0] * len(satire_articles))  # satire = real
    sat_false_fake_rate = float(sum(1 for l in sat_decisions if l == "FAKE") / len(sat_decisions))
    robustness_satire = {
        "n": len(satire_articles),
        "false_fake_rate": sat_false_fake_rate,  # lower is better
        "review_rate": float(sum(1 for l in sat_decisions if l == "NEEDS_REVIEW") / len(sat_decisions)),
        "real_rate": float(sum(1 for l in sat_decisions if l == "REAL") / len(sat_decisions)),
        "proba_metrics": _proba_metrics(sat_labels, sat_cal),
    }
    print(f"[eval] robustness (satirical): {json.dumps(robustness_satire, indent=2)}")

    # --- Robustness: mixed true/false ------------------------------------
    mixed_articles = _build_mixed_articles(n=25, seed=args.seed)
    X_mix, _, _, _ = _build_test_features(mixed_articles, bundle, seed=args.seed + 300)
    mix_cal = cal.calibrate(mc.predict_proba(X_mix)[:, 1])
    mix_decisions = [decide(p, thresholds)[0] for p in mix_cal]
    mix_labels = np.array([1] * len(mixed_articles))  # mixed = fake
    mix_fake_rate = float(sum(1 for l in mix_decisions if l == "FAKE") / len(mix_decisions))
    robustness_mixed = {
        "n": len(mixed_articles),
        "fake_detection_rate": mix_fake_rate,  # higher is better
        "review_rate": float(sum(1 for l in mix_decisions if l == "NEEDS_REVIEW") / len(mix_decisions)),
        "proba_metrics": _proba_metrics(mix_labels, mix_cal),
    }
    print(f"[eval] robustness (mixed): {json.dumps(robustness_mixed, indent=2)}")

    # --- Summary table --------------------------------------------------
    _print_table(
        "ROBUSTNESS: stability & generalisation",
        [
            ("Paraphrased", robustness_para["proba_metrics_on_paraphrased"]),
            ("Satirical (real)", robustness_satire["proba_metrics"]),
            ("Mixed (fake)", robustness_mixed["proba_metrics"]),
        ],
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "ece"],
    )

    # --- Reliability plot ------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_path = out_dir / "reliability.png"
    ok = plot_reliability(y_test, cal_probs, str(rel_path), n_bins=10,
                          title=f"Reliability ({mc.kind} + {cal.mode})")
    print(f"[eval] reliability plot: {rel_path} (ok={ok})")

    # --- Save eval.json --------------------------------------------------
    eval_payload = {
        "dataset": args.dataset,
        "n_test": int(len(test_articles)),
        "model_kind": mc.kind,
        "calibration_mode": cal.mode,
        "thresholds": thresholds,
        "core_metrics": core,
        "ablation": {
            "jev_only": jev_metrics,
            "tfidf_only": tfidf_metrics,
            "hybrid_uncalibrated": hybrid_uncal,
            "hybrid_calibrated": hybrid_metrics,
        },
        "stage3_evidence": stage3_metrics,
        "shap_sample": top_features_sample,
        "robustness": {
            "paraphrased": robustness_para,
            "satirical": robustness_satire,
            "mixed_true_false": robustness_mixed,
        },
        "reliability_data": {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in reliability_data(y_test, cal_probs, n_bins=10).items()
        },
        "reliability_plot": str(rel_path),
        "has_shap": bool(HAS_SHAP),
    }
    eval_path = out_dir / "eval.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2, default=str)
    print(f"[eval] saved results -> {eval_path}")
    print("[eval] done.")


if __name__ == "__main__":
    main()
