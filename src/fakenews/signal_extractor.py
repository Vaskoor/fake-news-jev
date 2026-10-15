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
Stage 1 — Signal extractor.

Holds a `JevClient` and the canonical `questions` list, and turns an
`Article` into a `list[Signal]` by:

  1. Building `state = f"{article.headline}\n\n{article.body}"`.
  2. Making ONE `await client.ask(state, questions)` call (the client
     handles all parallelism/caching/rate-limiting internally).
  3. Mapping each `TypedAnswer` 1:1 onto a `Signal` (same name, value,
     probability, type; `contribution` left at the default 0.0 for the
     meta-classifier stage to fill in).

Fast path: with `JevSimulated` the entire `ask` call is synchronous compute
wrapped in an asyncio coroutine — sub-millisecond on a 1KB article, well
under the 1s budget. With a real provider the budget is `jev.timeout_s`
(default 5s) × `jev.max_retries` (default 3); the cache + rate-limiter
prevent thundering herds under load.
"""

from __future__ import annotations

import time
from typing import Any

from .types import Article, AnswerType, Question, Signal, TypedAnswer
from .jev_client import JevClient
from .utils import get_logger, truncate


class SignalExtractor:
    """Stage 1 — Article -> list[Signal] via one JevClient.ask call."""

    def __init__(
        self,
        client: JevClient,
        questions: list[Question],
        logger=None,
    ) -> None:
        if not questions:
            raise ValueError("SignalExtractor requires a non-empty questions list")
        self._client = client
        self._questions = list(questions)
        # Index for O(1) metadata lookup when mapping answer -> Signal.
        self._q_by_name: dict[str, Question] = {q.name: q for q in self._questions}
        self._log = logger or get_logger(__name__, level="INFO", fmt="json")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def extract(self, article: Article) -> list[Signal]:
        """Extract typed signals for `article`.

        Args:
            article: Article with at least `headline` and `body` set.

        Returns:
            list[Signal] — one per Question in `self._questions`,
            in the SAME order as the loaded questions.

        Raises:
            JevError: if the underlying client raises after all retries.
        """
        state = self._build_state(article)
        t0 = time.monotonic()

        answers: dict[str, TypedAnswer] = await self._client.ask(
            state, self._questions
        )
        signals = self._map_answers(answers)

        dt_ms = (time.monotonic() - t0) * 1000.0
        if self._log:
            self._log.info(
                "signal_extracted",
                extra={
                    "headline": truncate(article.headline, 60),
                    "n_signals": len(signals),
                    "latency_ms": round(dt_ms, 2),
                },
            )
        return signals

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_state(article: Article) -> str:
        """Combine headline + body into the canonical state string."""
        headline = (article.headline or "").strip()
        body = (article.body or "").strip()
        if headline and body:
            return f"{headline}\n\n{body}"
        return headline or body or ""

    def _map_answers(
        self, answers: dict[str, TypedAnswer]
    ) -> list[Signal]:
        """Map TypedAnswer[] -> Signal[] in the questions' canonical order.

        Missing answers (which the contract says MUST NOT happen) are
        tolerated by emitting a neutral Signal so the meta-classifier
        still receives a full-length feature vector.
        """
        signals: list[Signal] = []
        for q in self._questions:
            ta = answers.get(q.name)
            if ta is None:
                signals.append(
                    Signal(
                        name=q.name,
                        value=_neutral_value(q),
                        probability=0.5,
                        type=q.type,
                        contribution=0.0,
                    )
                )
                continue
            signals.append(
                Signal(
                    name=ta.name,
                    value=ta.value,
                    probability=ta.probability,
                    type=ta.type,
                    contribution=0.0,
                )
            )
        return signals


def _neutral_value(q: Question) -> Any:
    """Return a neutral placeholder value for a missing answer."""
    if q.type is AnswerType.SCORE:
        if q.min is not None and q.max is not None:
            return (q.min + q.max) / 2.0
        return 0.0
    if q.type is AnswerType.CHOICE:
        if q.options:
            return q.options[0]
        return "none"
    return 0.5  # NOUL


__all__ = ["SignalExtractor"]
