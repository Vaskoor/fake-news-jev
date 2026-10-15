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
"""Fake News Detector — Streamlit UI.

A single-page Streamlit app that calls the FastAPI ``/predict`` endpoint
via :mod:`requests`. When the API is unreachable (e.g. the user launched
the UI standalone without the API), the app falls back to an in-process
:class:`fakenews.pipeline.FakeNewsPipeline` so demos still work end-to-end.

Run with::

    streamlit run ui/app.py

If Streamlit is not installed (the default sandbox state), invoking the
file directly prints clear install instructions instead of crashing:

    python ui/app.py

The matplotlib SHAP-style barh chart is *guarded* — if matplotlib is not
available the chart is simply skipped, but the rest of the UI still
renders.
"""

from __future__ import annotations

import os
import sys
import asyncio
from typing import Any, Optional

# Make ``fakenews`` importable when the UI is launched from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_PATH = os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

# ---------------------------------------------------------------------------
# Guarded imports — streamlit + matplotlib + requests
# ---------------------------------------------------------------------------
try:
    import streamlit as st  # type: ignore
    _HAS_STREAMLIT = True
except ImportError:  # spec: streamlit NOT installed in sandbox — graceful no-op
    st = None  # type: ignore[assignment]
    _HAS_STREAMLIT = False

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # type: ignore
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False

try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except ImportError:  # pragma: no cover
    requests = None  # type: ignore[assignment]
    _HAS_REQUESTS = False


# ---------------------------------------------------------------------------
# API endpoint configuration (env-overridable)
# ---------------------------------------------------------------------------
_API_BASE = os.environ.get("FAKENEWS_API_BASE", "http://localhost:8000")
_API_TIMEOUT = float(os.environ.get("FAKENEWS_API_TIMEOUT", "5.0"))

# Colour palette for verdict badges.
_BADGE_COLORS = {
    "REAL": "#1f9d55",          # green
    "FAKE": "#e3342f",          # red
    "NEEDS_REVIEW": "#f6993f",  # amber
}
_BADGE_LABEL = {
    "REAL": "REAL",
    "FAKE": "FAKE",
    "NEEDS_REVIEW": "NEEDS REVIEW",
}


# ---------------------------------------------------------------------------
# Predict — tries the API first, falls back to an in-process pipeline.
# ---------------------------------------------------------------------------
def predict_via_api(headline: str, body: str, source: str) -> Optional[dict]:
    """POST /predict on the FastAPI service. Returns ``None`` if unreachable."""
    if not _HAS_REQUESTS:
        return None
    payload = {
        "headline": headline,
        "body": body,
        "source": source or "",
        "url": "",
        "date": "",
    }
    try:
        r = requests.post(  # type: ignore[union-attr]
            f"{_API_BASE}/predict",
            json=payload,
            timeout=_API_TIMEOUT,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def predict_in_process(headline: str, body: str, source: str) -> Optional[dict]:
    """Build a pipeline in-process and run predict synchronously.

    Used as the fallback when the FastAPI service is not reachable so the
    UI is still useful for offline demos.
    """
    try:
        from fakenews.pipeline import build_default_pipeline  # type: ignore
        from fakenews.types import Article  # type: ignore
    except ImportError as exc:
        print(f"[ui] fakenews not importable: {exc}", file=sys.stderr)
        return None
    try:
        pipe = getattr(predict_in_process, "_pipe", None)  # type: ignore[arg-type]
        if pipe is None:
            pipe = build_default_pipeline()
            predict_in_process._pipe = pipe  # type: ignore[attr-defined]
        article = Article(
            headline=headline or "",
            body=body or "",
            source=source or "",
        )
        # streamlit runs in its own thread; the pipeline's JevSimulated is
        # async so we drive it via asyncio.run on a fresh event loop.
        verdict = asyncio.run(pipe.predict(article))
    except Exception as exc:
        print(f"[ui] in-process predict failed: {exc}", file=sys.stderr)
        return None
    # Serialise the Verdict into the same dict shape /predict returns.
    return _verdict_to_dict_local(verdict)


def _verdict_to_dict_local(verdict: Any) -> dict:
    """Serialise a Verdict dataclass to the same JSON shape returned by the API."""
    signals = []
    for s in (getattr(verdict, "signals", None) or []):
        signals.append({
            "name": str(getattr(s, "name", "") or ""),
            "value": getattr(s, "value", 0.0),
            "probability": float(getattr(s, "probability", 0.0) or 0.0),
            "type": str(getattr(getattr(s, "type", None), "value", "") or ""),
            "contribution": float(getattr(s, "contribution", 0.0) or 0.0),
        })
    evidence = getattr(verdict, "evidence", None)
    ev_dict = None
    if evidence is not None:
        ev_dict = {
            "claims": list(getattr(evidence, "claims", None) or []),
            "findings": list(getattr(evidence, "findings", None) or []),
            "source": str(getattr(evidence, "source", "") or ""),
            "adjusted_probability": getattr(evidence, "adjusted_probability", None),
        }
    return {
        "label": str(getattr(verdict, "label", "NEEDS_REVIEW") or "NEEDS_REVIEW"),
        "probability": float(getattr(verdict, "probability", 0.0) or 0.0),
        "signals": signals,
        "explanation": str(getattr(verdict, "explanation", "") or ""),
        "evidence": ev_dict,
        "latency_ms": float(getattr(verdict, "latency_ms", 0.0) or 0.0),
        "model_version": str(getattr(verdict, "model_version", "") or ""),
    }


def render_verdict(verdict: dict) -> None:
    """Render a Verdict dict using Streamlit primitives.

    Called from :func:`main` — has access to the ``st`` module global.
    """
    label = verdict.get("label", "NEEDS_REVIEW")
    prob = float(verdict.get("probability", 0.0) or 0.0)
    signals = verdict.get("signals", []) or []
    evidence = verdict.get("evidence")
    explanation = verdict.get("explanation", "")
    latency_ms = float(verdict.get("latency_ms", 0.0) or 0.0)
    model_version = verdict.get("model_version", "")

    # ----- Verdict badge -----
    color = _BADGE_COLORS.get(label, "#888888")
    badge_text = _BADGE_LABEL.get(label, label)
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:1rem;margin:0.5rem 0 1rem 0;">
          <span style="
              background:{color};color:white;font-weight:700;font-size:1.25rem;
              padding:0.4rem 1.0rem;border-radius:0.5rem;
              letter-spacing:0.05em;">{badge_text}</span>
          <span style="color:#555;font-size:0.95rem;">
            p(fake) = <b>{prob:.3f}</b> · latency {latency_ms:.1f} ms · {model_version}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Probability gauge -----
    st.write("**Probability gauge**")
    # Streamlit's progress bar is [0..1]; colour is fixed so we just use it
    # as a magnitude indicator. The badge carries the verdict colour.
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(
        "Disclaimer: the probability and verdict are an automated estimate. "
        "Articles in the NEEDS REVIEW band should always be inspected by a "
        "human before any consequential action is taken."
    )

    # ----- Top-5 signal breakdown -----
    st.write("**Top-5 signal breakdown** (contribution to p(fake))")
    if not signals:
        st.info("No signals returned.")
    else:
        # Show as a table of horizontal bars (Streamlit native).
        max_contrib = max((abs(float(s.get("contribution", 0.0))) for s in signals), default=1.0) or 1.0
        for s in signals[:5]:
            name = s.get("name", "")
            contrib = float(s.get("contribution", 0.0) or 0.0)
            val = s.get("value", 0.0)
            ttype = s.get("type", "")
            pct = min(100.0, abs(contrib) / max_contrib * 100.0) if max_contrib else 0.0
            sign = "+" if contrib >= 0 else "-"
            st.markdown(
                f"""
                <div style="margin:0.25rem 0;font-family:monospace;font-size:0.9rem;">
                  <div style="display:flex;justify-content:space-between;margin-bottom:0.2rem;">
                    <span><b>{name}</b> <span style="color:#888;">({ttype})</span></span>
                    <span>{sign}{abs(contrib):.4f} · value={val!r}</span>
                  </div>
                  <div style="background:#eee;border-radius:0.25rem;height:0.7rem;overflow:hidden;">
                    <div style="background:{'#e3342f' if contrib >= 0 else '#1f9d55'};width:{pct:.1f}%;height:100%;"></div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ----- SHAP-style matplotlib chart (guarded) -----
    if _HAS_MPL and signals:
        try:
            _render_shap_chart(signals)
        except Exception as exc:  # pragma: no cover
            st.info(f"(SHAP chart skipped: {exc})")
    elif not _HAS_MPL:
        st.caption("_matplotlib not installed — SHAP-style chart skipped._")

    # ----- Explanation + evidence findings -----
    if explanation:
        st.write("**Explanation**")
        st.caption(explanation)
    if evidence:
        st.write("**Evidence findings**")
        findings = evidence.get("findings") or []
        if not findings:
            st.info(
                f"source={evidence.get('source', '')} · "
                f"{len(evidence.get('claims', []))} claim(s) extracted, "
                "0 fact-check matches."
            )
        else:
            for i, f in enumerate(findings, start=1):
                if isinstance(f, dict):
                    st.markdown(
                        f"**{i}.** _rating_: `{f.get('rating', '')}`  ·  "
                        f"_claimant_: `{f.get('claimant', '')}`  ·  "
                        f"_claim_: {f.get('claim', '')}"
                    )
                    if f.get("url"):
                        st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;[source]({f['url']})")
                else:
                    st.write(f"{i}. {f}")


def _render_shap_chart(signals: list[dict]) -> None:
    """Render a horizontal bar chart of |SHAP| contributions via matplotlib."""
    names = [s.get("name", "") for s in signals[:5]]
    contribs = [float(s.get("contribution", 0.0) or 0.0) for s in signals[:5]]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(7, max(2.0, 0.5 * len(names) + 1)))
    # barh with positive/negative colouring.
    colors = ["#e3342f" if c >= 0 else "#1f9d55" for c in contribs]
    y_pos = list(range(len(names)))
    ax.barh(y_pos, contribs, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # top contribution at the top.
    ax.axvline(0.0, color="#888", linewidth=0.8)
    ax.set_xlabel("Contribution to p(fake)  (+ pushes toward FAKE, − toward REAL)")
    ax.set_title("Top-5 signal contributions (SHAP-style)")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    """Render the Streamlit app (only called when streamlit is available)."""
    st.set_page_config(
        page_title="Fake News Detector",
        page_icon="🔍",
        layout="centered",
    )

    st.title("Fake News Detector (Jev-powered)")
    st.caption(
        "3-stage pipeline: Jev signal extraction → meta-classifier → "
        "optional evidence check. Powered by the typed-decision Jev backend."
    )

    # ----- Input form -----
    with st.form("predict_form"):
        headline = st.text_input(
            "Headline",
            value="",
            placeholder="e.g. Senate Passes Bipartisan Infrastructure Bill",
        )
        body = st.text_area(
            "Article body",
            value="",
            height=200,
            placeholder="Paste the full article text here...",
        )
        source = st.text_input(
            "Source (optional)",
            value="",
            placeholder="e.g. AP, Reuters, blog.example.com",
        )
        submitted = st.form_submit_button("Analyze")
        if submitted:
            if not body.strip():
                st.error("`body` is required.")
                return
            with st.spinner("Running pipeline…"):
                verdict = predict_via_api(headline, body, source)
                if verdict is None:
                    st.warning(
                        f"FastAPI at {_API_BASE} unreachable — "
                        "falling back to in-process pipeline."
                    )
                    verdict = predict_in_process(headline, body, source)
                if verdict is None:
                    st.error(
                        "Neither the FastAPI service nor the in-process "
                        "pipeline could produce a verdict. Check that the "
                        "FastAPI service is running (uvicorn api.main:app) "
                        "or that the fakenews package is importable."
                    )
                    return
            render_verdict(verdict)

    # ----- Footer -----
    st.divider()
    st.caption(
        f"API: `{_API_BASE}/predict`  ·  matplotlib: "
        f"{'✓' if _HAS_MPL else '✗'}  ·  requests: "
        f"{'✓' if _HAS_REQUESTS else '✗'}  ·  Set "
        f"`FAKENEWS_API_BASE` to point at a different API host."
    )


# ---------------------------------------------------------------------------
# Offline-safe entry points
# ---------------------------------------------------------------------------
def _print_offline_instructions() -> None:
    """Print install/run instructions when neither streamlit nor the API work."""
    print("=" * 60)
    print("  Fake News Detector — UI offline")
    print("=" * 60)
    print()
    print("This UI requires Streamlit, which is not installed.")
    print()
    print("Install + run:")
    print("    pip install streamlit matplotlib requests")
    print("    streamlit run ui/app.py")
    print()
    print("Or use the FastAPI service directly:")
    print("    uvicorn api.main:app --port 8000")
    print("    curl -X POST http://localhost:8000/predict \\")
    print("         -H 'Content-Type: application/json' \\")
    print("         -d '{\"headline\":\"...\",\"body\":\"...\"}'")
    print()
    print("Repo root: " + _REPO_ROOT)
    print("=" * 60)


if __name__ == "__main__":
    if not _HAS_STREAMLIT:
        # Streamlit not installed — print install instructions and exit.
        _print_offline_instructions()
        sys.exit(0)
    # ``streamlit run ui/app.py`` re-executes this file with ``__name__``
    # == "__main__" (streamlit's runner uses runpy), so this branch is the
    # entry point under both ``streamlit run`` and direct ``python ui/app.py``.
    main()
