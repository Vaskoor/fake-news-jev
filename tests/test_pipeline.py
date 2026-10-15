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
# Tests for the end-to-end `FakeNewsPipeline` (simulated Jev backend, rule
# based meta-classifier fallback — no trained model, fully offline).

"""End-to-end tests for `fakenews.pipeline.FakeNewsPipeline`.

The pipeline is exercised against two synthetic articles: a clearly fake
article (sensationalist, anonymous sources, no dates) and a clearly real
article (Reuters wire, named officials, dated, balanced). With the simulated
Jev backend and rule-based meta-classifier fallback (no trained LightGBM
model required), the pipeline must produce:

* FAKE article → label FAKE or NEEDS_REVIEW with probability > 0.5
* REAL article → label REAL with probability < 0.3
* Fast-path latency < 1s per article
* A top-5 signal list (SHAP or rule-based fallback) attached to the verdict

No async-test plugin is required: ``asyncio.run`` is used to drive the
coroutine from a synchronous test body.
"""

import asyncio
import os
import sys
import time

# Make the src-layout package importable without an installed wheel.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fakenews.config import load_config  # noqa: E402
from fakenews.pipeline import FakeNewsPipeline  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")

FAKE_ARTICLE = (
    "SHOCKING EXPOSE: Insiders reveal the TRUTH about what the government "
    "is hiding from you!!!\n\n"
    "Anonymous sources familiar with the situation have confirmed that "
    "something absolutely devastating is happening behind closed doors. "
    "You won't believe what they don't want you to know. Share this before "
    "they take it down! Wake up, sheeple! This is the story mainstream "
    "media refuses to cover.\n\n"
    "According to a high-ranking official who asked not to be named, "
    "'Everything you have been told is a lie.' No further details, dates, "
    "or locations were provided at the time of publication."
)

REAL_ARTICLE = (
    "WASHINGTON (Reuters) - The Federal Reserve on Wednesday raised its "
    "benchmark interest rate by 25 basis points to a target range of "
    "5.25%-5.50%, the highest level since 2007, citing persistent "
    "inflation pressures in the services sector.\n\n"
    "\"The Committee will carefully assess incoming data and the evolving "
    "outlook,\" Fed Chair Jerome Powell said in a press conference "
    "following the two-day Federal Open Market Committee meeting. Powell "
    "added that future moves would be data-dependent and that officials "
    "remain committed to returning inflation to the 2 percent target.\n\n"
    "The decision was unanimous among the 12 voting members, according to "
    "the statement published at 2:00 p.m. EDT on July 26, 2023."
)


def _config():
    """Load (or build) the config object; falls back to a minimal dict if
    the YAML is unavailable or load_config() does not accept a path so
    tests still run in a freshly-cloned repo.
    """
    if os.path.exists(CONFIG_PATH):
        try:
            return load_config(CONFIG_PATH)
        except Exception:
            pass
    try:
        return load_config()  # type: ignore[call-arg]
    except Exception:
        pass
    # Minimal dict fallback that mirrors the shape of config/config.yaml.
    return {
        "jev": {"provider": "simulated"},
        "thresholds": {"lower": 0.3, "upper": 0.7},
        "meta": {"backend": "rule", "seed": 42},
    }


def _label_str(label) -> str:
    return str(label).upper().replace("LABEL.", "").replace(".", "")


def _probability(verdict) -> float:
    """Pull a scalar probability out of the Verdict regardless of attribute
    name (probability / prob / p_fake / score).
    """
    for attr in ("probability", "prob", "p_fake", "p", "score"):
        v = getattr(verdict, attr, None)
        if isinstance(v, (int, float)):
            return float(v)
    raise AssertionError(
        f"No scalar probability field on verdict: {verdict!r}"
    )


def _signals(verdict):
    """Pull the top signal list from the Verdict regardless of name."""
    for attr in ("top_signals", "signals", "shap_signals", "features", "evidence"):
        v = getattr(verdict, attr, None)
        if v:
            return v
    return []


# ---------------------------------------------------------------------------
# Pipeline smoke + verdict correctness
# ---------------------------------------------------------------------------

def test_pipeline_returns_verdict_for_fake_article():
    async def _run():
        pipeline = FakeNewsPipeline(_config())
        return await pipeline.predict(FAKE_ARTICLE)

    verdict = asyncio.run(_run())
    assert verdict is not None
    label = _label_str(getattr(verdict, "label", verdict))
    prob = _probability(verdict)
    # FAKE article → label FAKE or NEEDS_REVIEW with probability > 0.5
    assert ("FAKE" in label) or ("REVIEW" in label) or ("NEEDS" in label), (
        f"expected FAKE/NEEDS_REVIEW for synthetic fake article, got {label}"
    )
    assert prob > 0.5, f"expected probability > 0.5 for fake article, got {prob}"


def test_pipeline_returns_verdict_for_real_article():
    async def _run():
        pipeline = FakeNewsPipeline(_config())
        return await pipeline.predict(REAL_ARTICLE)

    verdict = asyncio.run(_run())
    assert verdict is not None
    label = _label_str(getattr(verdict, "label", verdict))
    prob = _probability(verdict)
    # REAL article → label REAL with probability < 0.3
    assert "REAL" in label or "TRUE" in label, (
        f"expected REAL for synthetic real article, got {label}"
    )
    assert prob < 0.3, f"expected probability < 0.3 for real article, got {prob}"


# ---------------------------------------------------------------------------
# Latency — fast path must complete within 1 second
# ---------------------------------------------------------------------------

def test_pipeline_fast_path_latency_under_one_second():
    config = _config()

    async def _run():
        pipeline = FakeNewsPipeline(config)
        # Run twice — second call exercises the warm path / caches.
        await pipeline.predict(REAL_ARTICLE)
        t0 = time.perf_counter()
        v = await pipeline.predict(REAL_ARTICLE)
        elapsed = time.perf_counter() - t0
        return v, elapsed

    _, elapsed = asyncio.run(_run())
    assert elapsed < 1.0, (
        f"fast-path latency {elapsed*1000:.0f}ms exceeds 1000ms budget"
    )


# ---------------------------------------------------------------------------
# Explainability — top-5 signals must be present on the verdict
# ---------------------------------------------------------------------------

def test_pipeline_returns_top_signals():
    async def _run():
        pipeline = FakeNewsPipeline(_config())
        return await pipeline.predict(FAKE_ARTICLE)

    verdict = asyncio.run(_run())
    signals = _signals(verdict)
    assert len(signals) >= 5, (
        f"expected >=5 top signals, got {len(signals)}: {signals!r}"
    )
    # Each signal should have at least a name and a contribution/value.
    for s in signals[:5]:
        if isinstance(s, (tuple, list)):
            assert len(s) >= 2, f"signal tuple malformed: {s!r}"
        elif isinstance(s, dict):
            assert "name" in s or "feature" in s, f"signal dict malformed: {s!r}"
        else:
            # Object form — must have a name attribute.
            assert hasattr(s, "name") or hasattr(s, "feature"), (
                f"signal object malformed: {s!r}"
            )


def test_pipeline_real_article_signals_lower_than_fake():
    """Cross-check: signal magnitudes on the REAL article should be smaller
    (or oppositely-signed) than on the FAKE article. This is a sanity check
    that the rule fallback is actually responsive to input content.
    """
    config = _config()

    async def _run():
        pipeline = FakeNewsPipeline(config)
        fake = await pipeline.predict(FAKE_ARTICLE)
        real = await pipeline.predict(REAL_ARTICLE)
        return fake, real

    fake, real = asyncio.run(_run())
    assert _probability(fake) > _probability(real), (
        "rule-based fallback did not differentiate fake vs real articles"
    )
