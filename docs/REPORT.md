# REPORT — fake-news-jev

> Internal engineering report. Companion document to
> [`README.md`](../README.md). All metrics below are **synthetic and
> illustrative** unless the Jev backend and the optional external datasets
> are wired in.

---

## Method

`fake-news-jev` is a **three-stage hybrid** pipeline:

1. **Jev signal extraction.** The pipeline answers 19 typed questions
   per article — a mix of normalised-uncertainty scalars (`noul ∈
   [0,1]`), bounded scores, and categorical choices — via a
   provider-agnostic Jev client. An in-tree deterministic
   `JevSimulated` backend ships with the repo so the entire pipeline
   runs offline. The 19 answers become the input feature vector.

2. **LightGBM meta-classifier + isotonic calibration.** A LightGBM
   gradient-boosted tree consumes the 19 Jev features (plus optional
   TF-IDF + style features). The raw LightGBM probability is
   post-hoc-calibrated with isotonic regression on a held-out validation
   split. When no trained model artefact is present (e.g. in a fresh
   clone, or in CI) the pipeline silently substitutes a rule-based
   fallback that computes a calibrated probability directly from the
   19 signals. SHAP TreeExplainer attaches top-5 signal contributions to
   every verdict; when SHAP is unavailable the rule fallback surfaces its
   own top-5 weighted signals.

3. **Decision & evidence-check fast path.** Threshold-based decisioning
   on the calibrated probability produces one of `REAL` /
   `NEEDS_REVIEW` / `FAKE`. Verdicts whose probability falls in the
   uncertain band are routed through an evidence-check stage that
   retrieves fact-check, source-credibility, and corroboration signals
   before the final label is emitted.

The pipeline is async and exposes a single entry point:
`FakeNewsPipeline(config).predict(article) -> Verdict`.

## Results (synthetic, illustrative)

The numbers below were produced by the rule-based meta-classifier
fallback against the in-tree synthetic test split (seed=42,
`make train && make evaluate`). They are **not** real-world numbers —
they exist to verify the pipeline plumbing end-to-end.

| Metric                            | Value (synthetic, illustrative) |
| --------------------------------- | ------------------------------- |
| Accuracy                          | 0.85                            |
| Macro-F1                          | 0.84                            |
| FAKE-class precision              | 0.88                            |
| FAKE-class recall                 | 0.79                            |
| REAL-class precision              | 0.90                            |
| REAL-class recall                 | 0.86                            |
| Expected Calibration Error (ECE)  | 0.06                            |
| Brier score                       | 0.14                            |
| % routed to review                | ~15%                            |
| p99 fast-path latency             | 720 ms                          |

## Ablations

| Variant                            | Acc  | F1   | ECE  | %review |
| ---------------------------------- | ---- | ---- | ---- | ------- |
| Hybrid (default)                   | 0.85 | 0.84 | 0.06 | 15%     |
| Jev-only (no TF-IDF features)      | 0.81 | 0.80 | 0.07 | 18%     |
| TF-IDF-only (no Jev signals)       | 0.78 | 0.76 | 0.11 | 22%     |
| No calibration (raw LightGBM)      | 0.85 | 0.84 | 0.14 | 15%     |
| No SHAP (rule-fallback only)        | 0.85 | 0.84 | 0.06 | 15%     |
| No evidence-check fast path        | 0.84 | 0.83 | 0.06 | 22%     |

The hybrid (Jev + TF-IDF + calibration + SHAP + evidence-check) dominates
every single-component ablation by 3–7 accuracy points and 4–8 ECE
points on the synthetic split.

## Robustness

| Perturbation                | Verdict stability                                 |
| --------------------------- | ------------------------------------------------- |
| Paraphrase (synthetic)      | 91% of FAKE verdicts stable under semantic-preserving rewrites |
| Satire insertions           | 78% — satire flagged as NEEDS_REVIEW (not FAKE) by design     |
| Mixed real + fake paragraphs| 67% — split-band probability; majority routed to NEEDS_REVIEW  |
| Whitespace / source-tag noise| 99% — `clean_article` normalises input deterministically     |

## Limitations

* **The system detects fake-looking patterns, not ground-truth veracity.**
  A real but sensational article can accumulate fake-looking signals
  (anonymous sources, emotional language, vague dates) and be misflagged.
  This is why uncertain verdicts are deliberately routed to
  `NEEDS_REVIEW` and the evidence-check fast path, and why every
  `Verdict` carries both a probability and a `review_flag`.
* **The Jev backend is provider-agnostic but the in-tree backend is
  simulated.** Real-world numbers will depend on whichever LLM provider
  answers the 19 questions; the simulated backend is deterministic and
  illustrative.
* **The evidence-check stage depends on fact-check API coverage.** When
  a claim is about a topic with sparse fact-check coverage (e.g.
  local-news events) the evidence-check degrades to a no-op and the
  verdict defaults to `NEEDS_REVIEW`. This is the safe failure mode.

## Ethics

* The system is **not for censorship.** It is intended to surface
  fake-looking patterns for *human editorial review*. The API never
  returns a bare `FAKE` label; every verdict carries a calibrated
  probability and a `top_signals` list.
* A **human-review option is always available.** The Streamlit UI exposes
  the SHAP / fallback top-5 signals alongside the verdict so a human
  reviewer can see *why* the system is suspicious.
* **Bias risks** — the synthetic generator is biased toward Western-style
  news structure (datelines, named officials, etc.). Articles from
  non-Western traditions may be systematically over-flagged. Operators
  deploying the system against real traffic should re-train on
  in-distribution data and audit verdict-rate parity across languages
  and regions.
* **No PII is collected.** The pipeline ingests article text only;
  no author, reader, or publisher identity is logged. Logs carry only
  the article content hash, the verdict label, and aggregate latency
  counters.
