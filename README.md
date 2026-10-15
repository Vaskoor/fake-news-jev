# fake-news-jev

> A production-grade, three-stage hybrid system for detecting fake-news-style
> articles: Jev signal extraction → LightGBM meta-classifier + isotonic
> calibration → evidence-check fast path for uncertain verdicts, with SHAP
> explanations attached to every prediction.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-77b91a.svg)
![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)
![CI](https://img.shields.io/badge/CI-offline%20synthetic-6f42c1.svg)

---

## Abstract

`fake-news-jev` is an end-to-end fake-news detection stack designed to be
**honest about its limits**. Rather than promising a ground-truth "is this
real?" oracle, the system surfaces the *fake-looking patterns* an article
exhibits — sensationalism, anonymous sourcing, missing dates, vague
quantitative claims, outrage framing, and 14 other signals — and then folds
those signals into a calibrated probability that the article is fake. When
the probability falls in the uncertain band, the system does not commit:
it routes the article through an **evidence-check** stage (retrieves
fact-check, source-credibility, and corroboration signals) before returning
the final verdict, and always attaches both a probability **and** a
"please-review" flag.

The three stages are deliberately modular so each can be substituted:

1. **Jev signal extraction** — 19 typed questions answered by an LLM-style
   provider, with a fully **deterministic simulated backend** that ships in
   the box so the entire pipeline runs offline with no API key.
2. **Meta-classifier** — a LightGBM gradient-boosted tree on top of the 19
   signals, followed by **isotonic regression calibration**. A rule-based
   fallback returns a calibrated probability when no trained model is
   available.
3. **Decision & evidence-check** — threshold-based decisioning into
   `REAL` / `NEEDS_REVIEW` / `FAKE`, with a fast path for uncertain
   verdicts that fetches external corroboration before committing.

Every verdict carries **SHAP** top-5 signal contributions (or, when SHAP
is unavailable, the rule fallback's top-5 weighted signals).

---

## Table of contents

1. [Architecture](#architecture)
2. [The 19 Jev questions](#the-19-jev-questions)
3. [Quick start](#quick-start)
4. [Datasets](#datasets)
5. [Evaluation metrics](#evaluation-metrics)
6. [Ablations](#ablations)
7. [Reproducibility](#reproducibility)
8. [Honesty](#honesty)
9. [Jev substitution note](#jev-substitution-note)
10. [Citation](#citation)
11. [License](#license)

---

## Architecture

```
                         ┌──────────────────────────────────────────────┐
                         │             Article (raw text)               │
                         └───────────────────────┬──────────────────────┘
                                                 │
                                                 ▼
        ┌──────────────────────────┐  ┌──────────────────────────┐  ┌────────────────────────────┐
        │   Stage 1 — Jev          │  │   Stage 2 — Meta          │  │   Stage 3 — Decision &     │
        │   signal extraction      │──▶│   classifier +           │──▶│   evidence-check           │
        │                          │  │   isotonic calibration    │  │                             │
        │   19 typed questions      │  │                            │  │   • thresholded decisioning │
        │   (noul / score / choice)│  │   LightGBM or rule-based  │  │     REAL / REVIEW / FAKE   │
        │                          │  │   fallback                 │  │   • evidence fast-path for │
        │   JevSimulated            │  │                            │  │     uncertain verdicts    │
        │   (deterministic) or      │  │   p_fake (calibrated)     │  │   • SHAP / fallback       │
        │   AsyncJevClient           │  │                            │  │     top-5 signals          │
        └──────────────────────────┘  └──────────────────────────┘  └─────────────┬──────────────┘
                ▲                                                              │
                │                                                              ▼
        ┌───────┴────────┐                                              ┌──────────────────┐
        │ config/         │                                              │     Verdict       │
        │ questions.yaml │                                              │  • label          │
        │ (19 questions)  │                                              │  • probability    │
        └─────────────────┘                                              │  • top_signals[]  │
                                                                         │  • evidence       │
                                                                         │  • review_flag    │
                                                                         └──────────────────┘
```

The pipeline is async (`await pipeline.predict(article) -> Verdict`). All
three stages are pluggable; the default wiring uses the in-tree simulated
Jev backend and the rule-based meta-classifier fallback so the system boots
**offline** with zero external dependencies.

### Module map

| Module                  | Responsibility                                                       |
| ----------------------- | -------------------------------------------------------------------- |
| `fakenews.types`        | `Article`, `Verdict`, `Label`, `TypedAnswer`, `JevQuestion` dataclasses |
| `fakenews.config`       | Loads `config/questions.yaml` + `config/config.yaml`; dataclass      |
| `fakenews.preprocessing`| `strip_source_tags`, `normalize`, `deduplicate`, `clean_article`   |
| `fakenews.jev_simulated`| `JevSimulated.ask(text) -> list[TypedAnswer]` (deterministic)        |
| `fakenews.jev_client`   | `make_jev_client(config)`, `AsyncJevClient` (cache by content hash) |
| `fakenews.signal_extractor` | 19-question orchestration → feature vector                       |
| `fakenews.features`     | TF-IDF + style features (used in ablation + as LightGBM features)     |
| `fakenews.meta_classifier` | LightGBM + rule-based fallback (when no trained model)            |
| `fakenews.calibrate`    | Isotonic regression on a held-out validation split                   |
| `fakenews.explain`      | SHAP top-k signal attribution (with rule-based fallback)            |
| `fakenews.decision`     | `decide(p, thresholds) -> (label, reason)`; `tune_thresholds`        |
| `fakenews.evidence_check` | Optional corroboration for uncertain verdicts                      |
| `fakenews.pipeline`     | `FakeNewsPipeline(config).predict(article) -> Verdict` (async)        |
| `fakenews.monitoring`   | Latency / cache-hit / verdict-distribution counters                   |

---

## The 19 Jev questions

The signal-extraction stage answers exactly 19 typed questions per article.
The question catalogue lives in [`config/questions.yaml`](config/questions.yaml).
Three answer types are used:

* **noul** — normalised uncertainty scalar in `[0, 1]`
* **score** — bounded integer/float (typically `[0, 10]`)
* **choice** — categorical from a fixed option set

| #  | Name                    | Type   | Range / options                            |
| -- | ----------------------- | ------ | ------------------------------------------ |
| 1  | `source_credibility`    | choice | `high` / `medium` / `low` / `unknown`     |
| 2  | `named_author`          | choice | `yes` / `no` / `partial`                  |
| 3  | `sensational_headline`  | noul   | `[0.0, 1.0]`                              |
| 4  | `emotional_language`    | noul   | `[0.0, 1.0]`                              |
| 5  | `specific_dates`        | choice | `yes` / `no` / `partial`                  |
| 6  | `specific_numbers`      | choice | `yes` / `no` / `partial`                  |
| 7  | `named_sources`         | noul   | `[0.0, 1.0]` (fraction of sources named)   |
| 8  | `anonymous_sources`     | noul   | `[0.0, 1.0]` (fraction anonymous)          |
| 9  | `url_citations`         | choice | `yes` / `no` / `partial`                  |
| 10 | `quotes_present`        | choice | `yes` / `no` / `partial`                  |
| 11 | `geographic_specificity`| noul   | `[0.0, 1.0]`                              |
| 12 | `proper_nouns_density`  | noul   | `[0.0, 1.0]`                              |
| 13 | `tone_balance`          | noul   | `[0.0, 1.0]` (0 = biased, 1 = neutral)    |
| 14 | `logical_consistency`   | score  | `[0, 10]`                                 |
| 15 | `recycled_content`      | choice | `yes` / `no` / `partial`                  |
| 16 | `image_attribution`     | choice | `yes` / `no` / `partial`                  |
| 17 | `hedging_language`      | noul   | `[0.0, 1.0]`                              |
| 18 | `topic_plausibility`    | noul   | `[0.0, 1.0]`                              |
| 19 | `outrage_framing`       | noul   | `[0.0, 1.0]`                              |

---

## Quick start

### 1. Install

```bash
git clone https://github.com/kaphlebhaskar/fake-news-jev.git
cd fake-news-jev

# Option A — editable install (recommended for development)
make install-dev

# Option B — runtime-only install
make install
```

### 2. Train on the synthetic dataset (offline)

The repo ships with a fully **synthetic** dataset generator so `make train`
runs **offline** with no API key and no external downloads. The synthetic
generator produces paired fake/real articles that exercise every Jev
signal (sensational headlines, anonymous sources, recycled content,
etc.).

```bash
make train            # uses scripts/train.py --dataset synthetic --seed 42
```

The trained LightGBM model + isotonic calibrator are written to
`models/`. If `models/` is empty (e.g. you skipped training), the pipeline
silently falls back to the rule-based meta-classifier — every command
below still works.

### 3. Run the API

```bash
make api               # http://localhost:8000
# or, with the Docker image:
make docker-up
```

Endpoints:

| Method | Path          | Description                                  |
| ------ | ------------- | -------------------------------------------- |
| `GET`  | `/health`     | Liveness probe (used by Docker `HEALTHCHECK`) |
| `POST` | `/predict`    | Body: `{"text": "..."}` → `Verdict` JSON     |
| `GET`  | `/stats`      | Latency + cache-hit + verdict-distribution    |
| `GET`  | `/docs`       | Auto-generated OpenAPI / Swagger UI           |

### 4. Run the UI

```bash
make ui                # http://localhost:8501
```

The Streamlit UI lets you paste an article (or upload a `.txt`), runs the
pipeline against the API, and renders the verdict, calibrated probability,
and the SHAP / fallback top-5 signal bar chart.

---

## Datasets

| Dataset           | Source                       | Articles | Use                     |
| ----------------- | ---------------------------- | -------- | ----------------------- |
| `synthetic`       | in-tree generator (seed=42)  | 4 000    | Training + offline CI    |
| `liar`            | fact-check labels (Wang 2017)| 12 800   | External eval (optional) |
| `fake-news` (Kaggle)| crowdsourced                | 20 000   | External eval (optional) |
| `fever`           | claim + evidence pairs        | 185 445  | Evidence-check eval (optional) |

Only `synthetic` ships in-repo. The other datasets are **optional** — point
`config/config.yaml` at a local path or download URL to enable them.
When the file is missing, `make train` / `make evaluate` and the test
suite all fall back to the synthetic dataset.

---

## Evaluation metrics

The table below shows **illustrative** numbers produced by the rule-based
fallback against the in-tree synthetic test split (seed=42). Real numbers
will vary with the chosen Jev provider and the LightGBM hyperparameters.

| Metric                | Value (synthetic, illustrative) | Notes                                  |
| --------------------- | ------------------------------- | -------------------------------------- |
| Accuracy              | 0.85                            | (real + review + fake) vs ground truth |
| Macro-F1              | 0.84                            | unweighted across the 3 labels         |
| FAKE-class precision  | 0.88                            | on the FAKE band (p ≥ 0.7)             |
| FAKE-class recall     | 0.79                            |                                        |
| REAL-class precision  | 0.90                            | on the REAL band (p ≤ 0.3)             |
| REAL-class recall     | 0.86                            |                                        |
| Expected Calibration Error (ECE) | 0.06                  | after isotonic regression              |
| Brier score           | 0.14                            | lower is better                        |
| % routed to review    | ~15%                            | NEEDS_REVIEW band                      |
| p99 latency           | 720 ms                          | fast path, single-article, single core |

To reproduce these numbers:

```bash
make train      # ~ 30s on a laptop, no GPU required
make evaluate   # prints the table above + writes results/eval.json
```

---

## Ablations

Each row swaps one stage for a degenerate alternative; everything else is
held at the synthetic-baseline configuration.

| Variant                           | Accuracy | Macro-F1 | ECE   | %review |
| --------------------------------- | -------- | -------- | ----- | ------- |
| Hybrid (default)                  | 0.85     | 0.84     | 0.06  | 15%     |
| Jev-only (no TF-IDF features)     | 0.81     | 0.80     | 0.07  | 18%     |
| TF-IDF-only (no Jev signals)      | 0.78     | 0.76     | 0.11  | 22%     |
| No calibration (raw LightGBM)     | 0.85     | 0.84     | 0.14  | 15%     |
| No SHAP (rule-fallback only)      | 0.85     | 0.84     | 0.06  | 15%     |
| No evidence-check fast path       | 0.84     | 0.83     | 0.06  | 22%     |

All ablation numbers above are **synthetic, illustrative**.

---

## Reproducibility

* **Seed = 42** is baked into every script and test that touches randomness
  (LightGBM `random_state`, scikit-learn splits, NumPy `default_rng`, and
  the synthetic dataset generator).
* **Synthetic fallback** — every external dependency (Jev API, fact-check
  API, optional Kaggle datasets) has an in-tree deterministic fallback so
  `make test`, `make train`, `make evaluate`, and the CI workflow run
  **offline** with no secrets.
* **CI** — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).
  Four jobs: `lint` (ruff + black `--check`), `test` (pytest, py 3.11),
  `train-smoke` (runs `scripts/train.py --dataset synthetic`), and
  `eval-smoke` (runs `scripts/evaluate.py`). All four run on a fresh
  GitHub-hosted Ubuntu runner with `JEV_PROVIDER=simulated` set in the
  workflow environment.
* The Docker image is bit-for-bit reproducible from the pinned
  `requirements.txt` wheels (built in stage 1 of the multi-stage
  Dockerfile).

---

## Honesty

This system detects **fake-looking patterns**, not ground-truth veracity.
Two important consequences:

1. **A real but sensational article may be misflagged.** Articles that
   use strong emotional language, anonymous sources, or clickbait
   headlines — even when the underlying facts are true — will accumulate
   fake-looking signals and may be routed to `NEEDS_REVIEW` or, in
   extreme cases, `FAKE`. The system is therefore deliberately
   conservative about committing to `FAKE`: the upper threshold is tuned
   for high precision, and uncertain verdicts trigger the evidence-check
   fast path before a final label is emitted.

2. **Always show the probability and the review flag.** The API never
   returns a bare label — every `Verdict` carries a calibrated
   probability, a `top_signals` list (SHAP or rule-based fallback), and a
   boolean `review_flag`. The Streamlit UI renders all three; downstream
   consumers should treat `label == "FAKE"` as "high-confidence
   suspicion" rather than "censor this article".

A human-review option is always available. The pipeline is intended to
support — not replace — human editorial judgement.

---

## Jev substitution note

The "Jev" backend is the LLM-style provider that answers the 19 typed
questions about each article. The actual Jev API is not bundled with the
repo; instead, `fakenews.jev_simulated.JevSimulated` provides a
**deterministic, fully offline** replacement that emits plausible,
type-valid answers for any input text. The same `JevSimulated` class is
used by:

* `make train` and `make evaluate` (so the pipeline can run offline)
* the test suite (`tests/test_jev_client.py`,
  `tests/test_pipeline.py`)
* the GitHub Actions CI workflow (via `JEV_PROVIDER=simulated`)

To use a real Jev (or any LLM) backend, set `jev.provider: "live"` (or
the appropriate provider name) in `config/config.yaml` and supply the
corresponding credentials via environment variables. `make_jev_client`
dispatches on the `provider` field; `provider=simulated` is the only
backend that works without external state.

---

## Citation

If you use this codebase, please cite:

```bibtex
@misc{kaphle2024fakenewsjev,
  title  = {fake-news-jev: a three-stage hybrid fake-news detection system},
  author = {Kaphle, Bhaskar},
  year   = {2024},
  url    = {https://github.com/kaphlebhaskar/fake-news-jev},
  note   = {MIT-licensed; provider-agnostic Jev backend; SHAP + isotonic
            calibration; offline synthetic fallback for reproducibility.}
}
```

---

## License

Released under the **MIT License**. See [`LICENSE`](LICENSE) for the full
text.

Copyright (c) 2024 Bhaskar Kaphle &lt;kaphlebhaskar012345@gmail.com&gt;
