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
"""
Shared typed primitives — the CONTRACT for the fake-news-jev pipeline.

Every other module (config, jev_client, jev_simulated, signal_extractor,
downstream meta-classifier / evidence / verdict stages) imports these
dataclasses. They MUST stay backwards-compatible across versions; new fields
get default values so existing call sites keep working.

The contract mirrors what the real Jev typed-decision model returns:

    state (unstructured text) + typed Questions
        -> JevClient.ask
        -> {question_name: TypedAnswer}
        -> mapped to Signal[]
        -> meta-classifier
        -> Evidence + Verdict

See `questions.yaml` for the canonical, versioned question set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class AnswerType(str, Enum):
    """Kind of answer a Jev question produces.

    - NOUL:    yes/no style, `value` is a float in [0,1] (probability of "yes").
    - SCORE:   numeric, `value` is a float in `[min, max]` (e.g. 0..10, -5..5).
    - CHOICE:   categorical, `value` is a string from `Question.options`.
    """

    NOUL = "noul"
    SCORE = "score"
    CHOICE = "choice"


@dataclass
class Question:
    """A versioned, atomic, unambiguous question posed to Jev.

    Attributes:
        name:     stable slug; downstream signals are keyed by this.
        type:     one of AnswerType.
        prompt:   the atomic question text shown to the model.
        version:  bump when the prompt semantics change (A/B testable).
        min/max:  required for SCORE questions (e.g. -5..5, 0..10).
        options:  required for CHOICE questions (closed vocabulary).
    """

    name: str
    type: AnswerType
    prompt: str
    version: int = 1
    min: Optional[float] = None
    max: Optional[float] = None
    options: Optional[list[str]] = None


@dataclass
class TypedAnswer:
    """What Jev returns for one question.

    Attributes:
        name:        matches `Question.name`.
        type:        matches `Question.type`.
        value:       float (NOUL/SCORE) or str (CHOICE).
        probability: calibrated confidence in [0,1] that the value is correct.
                     For NOUL questions this is also the "yes" probability.
    """

    name: str
    type: AnswerType
    value: Any
    probability: float


@dataclass
class Article:
    """Raw article input for the pipeline.

    Only `headline` and `body` are required; the rest are passed through to
    downstream evidence/verdict stages for citation and provenance.
    """

    headline: str
    body: str
    source: str = ""
    url: str = ""
    date: str = ""


@dataclass
class Signal:
    """A typed, Jev-derived feature for the meta-classifier.

    Attributes:
        name:         matches the originating `Question.name`.
        value:        float (NOUL/SCORE) or str (CHOICE) — copied from
                      the TypedAnswer.
        probability:  calibrated confidence.
        type:         AnswerType of the originating question.
        contribution: feature-importance-ish weight filled in by the
                      meta-classifier stage (0.0 by default).
    """

    name: str
    value: Any
    probability: float
    type: AnswerType
    contribution: float = 0.0


@dataclass
class Evidence:
    """Audit trail that adjusts the calibrated probability toward a Verdict.

    Attributes:
        claims:               list of verifiable claims extracted from article.
        findings:             per-claim verification findings (supports/refutes).
        adjusted_probability: post-evidence calibrated P(fake).
        source:               provenance tag (e.g. "jev+simulated+grounding").
    """

    claims: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    adjusted_probability: Optional[float] = None
    source: str = ""


@dataclass
class Verdict:
    """Final pipeline output for one Article.

    Attributes:
        label:          REAL | NEEDS_REVIEW | FAKE (per config thresholds).
        probability:    calibrated P(fake) in [0,1].
        signals:        list[Signal] feeding/exiting the meta-classifier.
        explanation:    human-readable justification (model-generated).
        evidence:        optional Evidence from the grounding stage.
        latency_ms:     end-to-end wall-clock latency in milliseconds.
        model_version:  e.g. "jev-simulated-v1+lgbm-v1" for reproducibility.
    """

    label: str
    probability: float
    signals: list = field(default_factory=list)
    explanation: str = ""
    evidence: Optional[Evidence] = None
    latency_ms: float = 0.0
    model_version: str = ""


__all__ = [
    "AnswerType",
    "Question",
    "TypedAnswer",
    "Article",
    "Signal",
    "Evidence",
    "Verdict",
]
