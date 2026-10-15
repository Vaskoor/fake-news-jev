# -*- coding: utf-8 -*-
"""
Stage 3 — fact-check evidence layer.

When the calibrated ``p(fake)`` falls inside the *review band* (between
``decision.evidence_check.low`` and ``decision.evidence_check.high``), the
pipeline routes the article through this layer. ``EvidenceChecker``:

  1. Extracts factual claims from the article text
     (:meth:`EvidenceChecker.extract_claims`).
  2. Queries the Google Fact Check Tools API for each claim
     (:meth:`EvidenceChecker.check_claims`). This call is *guarded* — if
     the API key is missing, the network is down, or the dependency
     ``requests`` is not installed, the checker returns an empty
     ``Evidence`` with ``source="unavailable"`` rather than crashing.
  3. Adjusts ``p(fake)`` based on the rating of matched fact-checks
     (:meth:`EvidenceChecker.adjust_probability`). This is a simple,
     deterministic rule, NOT a learned calibrator — see the docstring.

The module also exposes a small ``Evidence`` dataclass with a local
fallback so the file is importable even when ``fakenews.types`` (Task
fn-A) is not yet present in the sandbox.

MIT License — Copyright (c) 2024 fake-news-jev contributors.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import numpy as np


# --- guarded import of shared types -----------------------------------------
try:                                   # Task fn-A ships these dataclasses.
    from .types import Evidence, Article  # type: ignore
    HAS_TYPES = True
except (ImportError, ModuleNotFoundError):  # Local fallbacks keep this file usable.
    from dataclasses import dataclass, field

    @dataclass
    class Evidence:
        """Container for fact-check findings.

        Fields
        ------
        claims : list[str]
            Claims extracted from the article.
        findings : list[dict]
            Each dict has keys ``claim``, ``claimant``, ``rating``, ``url``.
        source : str
            ``"google_fact_check"`` on success, ``"unavailable"`` if the
            API/network/key was unavailable.
        """

        claims: list = field(default_factory=list)
        findings: list = field(default_factory=list)
        source: str = "unavailable"

    Article = Any  # type: ignore
    HAS_TYPES = False


# --- regex patterns for claim extraction ------------------------------------
_FACTUAL_MARKERS = [
    r"according to",
    r"\bsaid\b",
    r"\bsays\b",
    r"\bstated\b",
    r"\breported\b",
    r"\bclaimed\b",
    r"\bannounced\b",
    r"\bdeclared\b",
    r"\bconfirmed\b",
    r"\brevealed\b",
    r"\bdisclosed\b",
    r"\bcited\b",
]
_FACTUAL_RX = re.compile("|".join(_FACTUAL_MARKERS), re.IGNORECASE)
_NUMBER_RX = re.compile(r"\b(?:\d+(?:,\d{3})*(?:\.\d+)?|\d+\s*(?:%|percent|million|billion|thousand))\b", re.IGNORECASE)
# Crude named-entity hint: Capitalised multiword tokens.
_NE_RX = re.compile(r"\b(?:[A-Z][a-zA-Z'’]+\s+){1,3}[A-Z][a-zA-Z'’]+\b")
# Sentence splitter — keep it simple so it is deterministic and offline.
_SENT_SPLIT_RX = re.compile(r"(?<=[.!?])\s+|[\n\r]+")


class EvidenceChecker:
    """Stage-3 evidence checker.

    Parameters
    ----------
    config : dict, optional
        The full app config dict. Reads ``decision.evidence_check.low`` and
        ``decision.evidence_check.high`` to determine the review band.
    api_key : str, optional
        Google Fact Check Tools API key. If not provided, falls back to the
        ``GOOGLE_FACT_CHECK_API_KEY`` environment variable. If still
        missing, all checks return ``source="unavailable"``.
    """

    FACT_CHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1.1/claimsearch:search"

    # Rating tokens that indicate the claim is FALSE (push p(fake) up).
    FALSE_MARKERS = (
        "false", "pants on fire", "pants fire", "mostly false",
        "incorrect", "untrue", "fake", "misleading", "fabricated",
        "wrong", "disputed",
    )
    # Rating tokens that indicate the claim is TRUE (push p(fake) down).
    TRUE_MARKERS = (
        "true", "mostly true", "correct", "accurate", "half true",
        "verified",
    )

    def __init__(self, config: Optional[Dict[str, Any]] = None, api_key: Optional[str] = None) -> None:
        self.config = config or {}
        ec = (self.config.get("decision", {}) or {}).get("evidence_check", {}) or {}
        self.low = float(ec.get("low", 0.30))
        self.high = float(ec.get("high", 0.70))
        self.api_key = api_key or os.environ.get("GOOGLE_FACT_CHECK_API_KEY")
        # Per-claim adjustment magnitude.
        self.delta_per_finding = 0.10
        self.max_delta = 0.30

    # ------------------------------------------------------------------
    # gating
    # ------------------------------------------------------------------
    def should_check(self, p_fake: float, config: Optional[Dict[str, Any]] = None) -> bool:
        """Return ``True`` if the calibrated ``p_fake`` is in the review band.

        Honours overrides via ``config["decision"]["evidence_check"]`` if
        provided.
        """
        ec_cfg = (config or {}).get("decision", {}).get("evidence_check", {}) if config else {}
        low = float(ec_cfg.get("low", self.low))
        high = float(ec_cfg.get("high", self.high))
        p = float(p_fake)
        return low <= p < high

    # ------------------------------------------------------------------
    # claim extraction
    # ------------------------------------------------------------------
    def extract_claims(self, article: "Article") -> List[str]:
        """Heuristic claim extraction.

        Strategy
        --------
        Splits the article body (``article.text`` / ``article.body`` /
        ``article.content`` + ``article.title``) into sentences and keeps the
        ones containing:

          * a factual-marker verb (``said``, ``according to``, ``reported``,
            ``claimed``, etc.), OR
          * a number/percentage/quantity, OR
          * a multi-token capitalised named-entity hint.

        Returns at most 5 claims (truncated to 240 chars) to keep the
        Google Fact Check API queries tractable.
        """
        text = (
            getattr(article, "text", None)
            or getattr(article, "body", None)
            or getattr(article, "content", None)
            or ""
        )
        title = getattr(article, "title", None) or ""
        full = f"{title}. {text}" if title else text
        if not full.strip():
            return []

        sentences = []
        for chunk in _SENT_SPLIT_RX.split(full):
            chunk = chunk.strip()
            if len(chunk) < 8:
                continue
            if len(chunk) > 240:
                chunk = chunk[:237] + "..."
            sentences.append(chunk)

        claims: List[str] = []
        seen = set()
        for s in sentences:
            if s in seen:
                continue
            if _FACTUAL_RX.search(s) or _NUMBER_RX.search(s) or _NE_RX.search(s):
                claims.append(s)
                seen.add(s)
            if len(claims) >= 5:
                break
        return claims

    # ------------------------------------------------------------------
    # fact-check lookup
    # ------------------------------------------------------------------
    def check_claims(self, claims: List[str], api_key: Optional[str] = None) -> Evidence:
        """Query the Google Fact Check Tools API for each claim.

        Guarded
        -------
        * No API key (no ``--api-key`` and no ``GOOGLE_FACT_CHECK_API_KEY``
          env var) -> returns ``Evidence(claims, [], "unavailable")``.
        * ``requests`` not installed -> same.
        * Any HTTP/JSON error or timeout -> same.

        Returns
        -------
        Evidence
            ``findings`` is a list of dicts with keys ``claim``,
            ``claimant``, ``rating``, ``url`` (one entry per matched
            fact-check, capped at 3 per claim).
        """
        key = api_key or self.api_key
        if not key:
            return Evidence(claims=list(claims), findings=[], source="unavailable")

        try:
            import requests  # type: ignore
        except ImportError:
            return Evidence(claims=list(claims), findings=[], source="unavailable")

        findings: List[Dict[str, str]] = []
        try:
            for claim_text in claims[:5]:  # cap to keep latency bounded
                try:
                    resp = requests.get(
                        self.FACT_CHECK_ENDPOINT,
                        params={"query": claim_text[:1000], "key": key},
                        timeout=5,
                    )
                except Exception:
                    continue
                if resp.status_code != 200:
                    continue
                try:
                    data = resp.json()
                except ValueError:
                    continue
                for claim_obj in (data.get("claims") or [])[:3]:
                    claimant = claim_obj.get("claimant", "") or ""
                    text_val = claim_obj.get("text", claim_text) or claim_text
                    for review in (claim_obj.get("claimReview") or [])[:1]:
                        rating = review.get("textualRating", "") or ""
                        url = review.get("url", "") or ""
                        findings.append({
                            "claim": text_val[:500],
                            "claimant": claimant,
                            "rating": rating,
                            "url": url,
                        })
        except Exception:
            return Evidence(claims=list(claims), findings=[], source="unavailable")

        return Evidence(
            claims=list(claims),
            findings=findings,
            source="google_fact_check" if findings else "no_match",
        )

    # ------------------------------------------------------------------
    # probability adjustment
    # ------------------------------------------------------------------
    def adjust_probability(self, p_fake: float, evidence: "Evidence") -> float:
        """Heuristic adjustment of ``p_fake`` based on fact-check findings.

        Rule
        ----
        For each finding:

          * Rating contains a FALSE marker (``"false"``, ``"pants on fire"``,
            ``"misleading"``, …) -> push ``p_fake`` UP by ``delta_per_finding``.
          * Rating contains a TRUE marker (``"true"``, ``"correct"`` …) ->
            push ``p_fake`` DOWN by ``delta_per_finding``.
          * Otherwise -> unchanged.

        The net delta is clipped to ``[-max_delta, +max_delta]`` and the
        final probability is clipped to ``[0, 1]``.

        .. note::
            This is a *simple deterministic rule*, **NOT** a learned
            calibrator. It exists to give the pipeline a defensible behaviour
            when the meta-classifier is uncertain, and to provide a hook for
            future learned adjustment modules. Production deployments should
            replace this with a calibrated secondary model trained on
            (claim-text + fact-check-rating + meta-probability) -> label.
        """
        if not getattr(evidence, "findings", None):
            return float(p_fake)
        delta = 0.0
        for f in evidence.findings:
            rating = str(f.get("rating", "") or "").lower()
            if not rating:
                continue
            if any(m in rating for m in self.FALSE_MARKERS):
                delta += self.delta_per_finding
            elif any(m in rating for m in self.TRUE_MARKERS):
                delta -= self.delta_per_finding
        delta = float(np.clip(delta, -self.max_delta, self.max_delta))
        return float(np.clip(float(p_fake) + delta, 0.0, 1.0))


__all__ = ["EvidenceChecker", "Evidence"]
