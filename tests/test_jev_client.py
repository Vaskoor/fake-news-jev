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
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# Tests for `fakenews.jev_simulated.JevSimulated` + `fakenews.jev_client`
# (factory + async wrapper + content-hash cache).

"""Tests for the Jev signal-extraction backend.

Verifies:
* ``JevSimulated.ask(text)`` returns exactly 19 TypedAnswer objects whose
  values are type-valid (noul in [0,1], score in configured range, choice
  in the configured option set).
* The simulated backend is deterministic — identical input → identical
  output (across two independent instances).
* ``make_jev_client(config)`` returns an async client whose
  ``ask(text)`` delegates to the simulated backend when configured with
  ``provider = simulated``.
* Repeating a request with the same content hash triggers a cache hit
  (observable via the client's stats counter; falls back to asserting
  identical responses on cache miss).

Runs offline. ``asyncio.run`` drives async calls from synchronous tests
(no pytest-asyncio dependency).
"""

import asyncio
import os
import sys

# Make the src-layout package importable without an installed wheel.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import yaml  # noqa: E402  (PyYAML is a core dependency)

from fakenews.jev_client import make_jev_client  # noqa: E402
from fakenews.jev_simulated import JevSimulated  # noqa: E402


# ---------------------------------------------------------------------------
# Question-spec fixtures — load config/questions.yaml to know the 19 names,
# types, ranges, and option sets. Falls back to a built-in canonical list
# if the YAML is absent.
# ---------------------------------------------------------------------------

QUESTIONS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "config", "questions.yaml"
)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")


_CANONICAL_QUESTIONS = [
    {"name": "source_credibility", "type": "choice", "options": ["high", "medium", "low", "unknown"]},
    {"name": "named_author", "type": "choice", "options": ["yes", "no", "partial"]},
    {"name": "sensational_headline", "type": "noul", "range": [0.0, 1.0]},
    {"name": "emotional_language", "type": "noul", "range": [0.0, 1.0]},
    {"name": "specific_dates", "type": "choice", "options": ["yes", "no", "partial"]},
    {"name": "specific_numbers", "type": "choice", "options": ["yes", "no", "partial"]},
    {"name": "named_sources", "type": "noul", "range": [0.0, 1.0]},
    {"name": "anonymous_sources", "type": "noul", "range": [0.0, 1.0]},
    {"name": "url_citations", "type": "choice", "options": ["yes", "no", "partial"]},
    {"name": "quotes_present", "type": "choice", "options": ["yes", "no", "partial"]},
    {"name": "geographic_specificity", "type": "noul", "range": [0.0, 1.0]},
    {"name": "proper_nouns_density", "type": "noul", "range": [0.0, 1.0]},
    {"name": "tone_balance", "type": "noul", "range": [0.0, 1.0]},
    {"name": "logical_consistency", "type": "score", "range": [0, 10]},
    {"name": "recycled_content", "type": "choice", "options": ["yes", "no", "partial"]},
    {"name": "image_attribution", "type": "choice", "options": ["yes", "no", "partial"]},
    {"name": "hedging_language", "type": "noul", "range": [0.0, 1.0]},
    {"name": "topic_plausibility", "type": "noul", "range": [0.0, 1.0]},
    {"name": "outrage_framing", "type": "noul", "range": [0.0, 1.0]},
]


def _load_questions():
    if os.path.exists(QUESTIONS_PATH):
        try:
            with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            qs = data.get("questions") if isinstance(data, dict) else data
            if isinstance(qs, list) and len(qs) == 19:
                return qs
        except Exception:
            pass
    return _CANONICAL_QUESTIONS


def _config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            pass
    return {"jev": {"provider": "simulated"}}


QUESTIONS = _load_questions()
EXPECTED_QUESTION_COUNT = 19  # Jev signal-extraction backend exposes 19 questions


def _answer_value(ans):
    """Extract the scalar/array value from a TypedAnswer regardless of name."""
    for attr in ("value", "v", "answer", "score", "raw"):
        v = getattr(ans, attr, None)
        if v is not None:
            return v
    raise AssertionError(f"no value field on {ans!r}")


def _answer_type(ans):
    for attr in ("type", "qtype", "kind"):
        v = getattr(ans, attr, None)
        if v is not None:
            return str(v)
    raise AssertionError(f"no type field on {ans!r}")


def _answer_question_name(ans):
    for attr in ("question_name", "name", "question", "q"):
        v = getattr(ans, attr, None)
        if v is not None:
            return str(v)
    raise AssertionError(f"no question_name on {ans!r}")


def _answer_options(ans):
    for attr in ("options", "choices", "allowed"):
        v = getattr(ans, attr, None)
        if v is not None:
            return list(v)
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "WASHINGTON (Reuters) - The Federal Reserve raised its benchmark "
    "interest rate by 25 basis points on Wednesday, citing persistent "
    "inflation pressures. Fed Chair Jerome Powell said future moves would "
    "be data-dependent. The decision was unanimous among the 12 voting "
    "members, according to the statement published at 2:00 p.m. EDT."
)


def test_jev_simulated_ask_returns_exactly_19_answers():
    sim = JevSimulated()
    answers = list(asyncio.run(sim.ask(SAMPLE_TEXT)).values())
    assert isinstance(answers, (list, tuple))
    assert len(answers) == EXPECTED_QUESTION_COUNT, (
        f"expected {EXPECTED_QUESTION_COUNT} TypedAnswers, got {len(answers)}"
    )


def test_jev_simulated_answer_values_are_type_valid():
    sim = JevSimulated()
    answers = list(asyncio.run(sim.ask(SAMPLE_TEXT)).values())

    by_name = {_answer_question_name(a): a for a in answers}

    # Build expected spec by name from the canonical / YAML question list.
    spec_by_name = {q["name"]: q for q in QUESTIONS}
    assert len(spec_by_name) == 19, "question spec list must have 19 entries"

    for qname, qspec in spec_by_name.items():
        assert qname in by_name, f"missing answer for question {qname!r}"
        ans = by_name[qname]
        qtype = _answer_type(ans).lower()
        value = _answer_value(ans)

        if qspec["type"] == "noul":
            assert isinstance(value, (int, float)), (
                f"{qname}: noul value must be numeric, got {type(value)}"
            )
            assert 0.0 <= float(value) <= 1.0, (
                f"{qname}: noul value {value} outside [0,1]"
            )
        elif qspec["type"] == "score":
            assert isinstance(value, (int, float)), (
                f"{qname}: score value must be numeric, got {type(value)}"
            )
            lo, hi = qspec.get("range", [0, 10])
            assert lo <= float(value) <= hi, (
                f"{qname}: score {value} outside [{lo},{hi}]"
            )
        elif qspec["type"] == "choice":
            opts = _answer_options(ans) or qspec.get("options", [])
            assert value in opts, (
                f"{qname}: choice {value!r} not in options {opts!r}"
            )
        else:
            raise AssertionError(
                f"unknown question type {qspec['type']!r} for {qname!r}"
            )


def test_jev_simulated_is_deterministic_same_input_same_output():
    sim_a = JevSimulated()
    sim_b = JevSimulated()
    out_a = list(asyncio.run(sim_a.ask(SAMPLE_TEXT)).values())
    out_b = list(asyncio.run(sim_b.ask(SAMPLE_TEXT)).values())
    assert len(out_a) == len(out_b) == EXPECTED_QUESTION_COUNT

    vals_a = [_answer_value(a) for a in out_a]
    vals_b = [_answer_value(a) for a in out_b]
    assert vals_a == vals_b, (
        "JevSimulated.ask() is not deterministic for identical inputs"
    )

    # A second call against the SAME instance should also match.
    out_c = list(asyncio.run(sim_a.ask(SAMPLE_TEXT)).values())
    vals_c = [_answer_value(a) for a in out_c]
    assert vals_c == vals_a, (
        "JevSimulated.ask() is not stable across repeated calls on same instance"
    )


def test_async_jev_client_with_simulated_provider_delegates_correctly():
    config = _config()
    # Force the simulated provider regardless of what config has.
    if isinstance(config, dict):
        jev_cfg = dict(config.get("jev", {}))
        jev_cfg["provider"] = "simulated"
        config = {**config, "jev": jev_cfg}

    client = make_jev_client(config)

    async def _run():
        return list((await client.ask(SAMPLE_TEXT)).values())

    answers = asyncio.run(_run())
    assert isinstance(answers, (list, tuple))
    assert len(answers) == EXPECTED_QUESTION_COUNT, (
        f"AsyncJevClient(simulated) returned {len(answers)} answers, "
        f"expected {EXPECTED_QUESTION_COUNT}"
    )

    # The async client's output must agree with the bare simulated backend.
    sim = JevSimulated()
    sim_answers = list(asyncio.run(sim.ask(SAMPLE_TEXT)).values())
    sim_vals = [_answer_value(a) for a in sim_answers]
    client_vals = [_answer_value(a) for a in answers]
    assert client_vals == sim_vals, (
        "AsyncJevClient(provider=simulated) does not match JevSimulated output"
    )


def test_async_jev_client_cache_hits_on_repeat_content_hash():
    config = _config()
    if isinstance(config, dict):
        jev_cfg = dict(config.get("jev", {}))
        jev_cfg["provider"] = "simulated"
        config = {**config, "jev": jev_cfg}

    client = make_jev_client(config)

    async def _run():
        a1 = await client.ask(SAMPLE_TEXT)
        a2 = await client.ask(SAMPLE_TEXT)  # identical content → cache hit
        a3 = await client.ask(SAMPLE_TEXT + " ")  # different hash → miss
        return a1, a2, a3

    a1, a2, a3 = asyncio.run(_run())

    # Identical inputs must produce identical outputs (deterministic) — this
    # is the minimum bar; the cache makes it fast, not different.
    v1 = [_answer_value(a) for a in a1.values()]
    v2 = [_answer_value(a) for a in a2.values()]
    assert v1 == v2, "cache-miss semantics broke output equality"

    # Probe for a public cache-stats API on the client.
    hits = None
    for attr in ("cache_info", "cache_stats", "stats", "metrics"):
        if not hasattr(client, attr):
            continue
        info = getattr(client, attr)
        if callable(info):
            try:
                info = info()
            except Exception:
                continue
        if isinstance(info, dict):
            hits = info.get("hits") or info.get("hit_count") or info.get("hits_count")
        else:
            hits = (
                getattr(info, "hits", None)
                or getattr(info, "hit_count", None)
                or getattr(info, "hit", None)
            )
        if hits is not None:
            break

    if hits is not None:
        # At least one cache hit must have been recorded for the repeated call.
        assert int(hits) >= 1, (
            f"expected >=1 cache hit for repeat content; stats={info!r}"
        )
    # If no public stats API exists, we still assert output equality above.
