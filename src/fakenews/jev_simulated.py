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
Jev substitution — if the real Jev API is unavailable, this deterministic
heuristic backend implements the identical `ask() -> {name: TypedAnswer}`
contract. Set provider=typesafe/cloudflare/openrouter + JEV_API_KEY for the
real endpoint.

DETERMINISM
-----------
- No `random` calls are used for output values. All counts, lexicon hits,
  and style heuristics are pure functions of `state` text.
- Tie-breaks (e.g. two topic categories with equal keyword counts) use
  `utils.content_hash` (sha256-truncated) — stable across runs.
- Calling `ask(state, questions)` twice on the same input MUST return the
  same `dict[str, TypedAnswer]`.

CALIBRATION NOTE
----------------
- NOUL `value` is the YES probability in [0,1].
- NOUL/SCORE/CHOICE `probability` is the model's confidence in `value`,
  deliberately constrained to [0.5, 0.95] (low signal → 0.5; strong signal
  near a boundary → 0.95). It is NOT a perfectly calibrated probability —
  the downstream `model.calibration: isotonic` stage handles that.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .jev_client import JevClient
from .types import AnswerType, Question, TypedAnswer
from .utils import content_hash

# ---------------------------------------------------------------------------
# Lexicons (lowercased; matched as substring on lowercased state text)
# ---------------------------------------------------------------------------

_CLICKBAIT_PHRASES: tuple[str, ...] = (
    "you won't believe",
    "you wont believe",
    "this one trick",
    "shocking",
    "mind-blowing",
    "what happens next",
    "will blow your mind",
    "is unbelievable",
    "number 7 will",
    "they didn't want you to see",
    "doctors hate",
    "the truth about",
)

_ANON_SOURCE_PHRASES: tuple[str, ...] = (
    "anonymous sources said",
    "anonymous sources",
    "unnamed sources",
    "sources claim",
    "sources said",
    "according to sources",
    "sources familiar with",
    "an insider",
    "anonymous insider",
    "officials speaking on condition of anonymity",
)

_VERIFIABLE_SOURCE_PHRASES: tuple[str, ...] = (
    "(reuters)",
    "(ap)",
    "(associated press)",
    "(afp)",
    "(bloomberg)",
    "reuters reported",
    "according to a study",
    "according to a study published",
    "according to research",
    "the report states",
    "the report found",
    "according to a report",
    "according to a pew research",
    "according to the world health organization",
    "according to the cdc",
    "according to the fda",
    "department of",
    "ministry of",
    "according to documents obtained",
    "in an email",
    "press release",
    "the data show",
)

_SATIRE_MARKERS: tuple[str, ...] = (
    "the onion",
    "babylon bee",
    "nottheonion",
    "not the onion",
    "the hard times",
    "clickhole",
    "reductress",
    "the borowitz report",
    "weekly world news",
    "satire",
    "parody",
    "this is a joke",
)

_CONSPIRACY_PHRASES: tuple[str, ...] = (
    "they don't want you to know",
    "they dont want you to know",
    "deep state",
    "wake up",
    "wake up, america",
    "wake up america",
    "cover-up",
    "cover up",
    "the truth is being hidden",
    "mainstream media won't",
    "mainstream media wont",
    "shadow government",
    "new world order",
    "globalists",
    "the elites",
    "they are hiding",
    "secretly orchestrated",
    "plandemic",
    "false flag",
)

_OPINION_PHRASES: tuple[str, ...] = (
    "i think",
    "i believe",
    "in my opinion",
    "in my view",
    "we should",
    "we must",
    "we ought to",
    "ought to",
    "it is wrong that",
    "it is right that",
    "personally,",
    "to my mind",
    "i feel that",
    "the right thing to do",
    "in this columnist's view",
    "as a columnist",
    "editorial",
    "op-ed",
)

_AUTHORITY_IMPERSONATION_PHRASES: tuple[str, ...] = (
    "experts say",
    "expert says",
    "officials confirmed",
    "doctors warn",
    "a high-ranking official",
    "anonymous official",
    "officials speaking on condition",
    "scientists confirm",
    "the government admitted",
    "the pentagon said",
    "the white house said",
    "top scientist says",
)

_FEAR_PHRASES: tuple[str, ...] = (
    "danger",
    "dangerous",
    "threat",
    "crisis",
    "panic",
    "warning",
    "deadly",
    "catastrophe",
    "catastrophic",
    "alarming",
    "terrifying",
)

_OUTRAGE_PHRASES: tuple[str, ...] = (
    "outrage",
    "disgusting",
    "scandal",
    "shameful",
    "fury",
    "furious",
    "sick of",
    "appalling",
    "despicable",
    "deplorable",
)

_CHERRY_PICK_PHRASES: tuple[str, ...] = (
    "only",
    "despite",
    "even though",
    "while ignoring",
    "conveniently",
    "handpicked",
    "selectively",
)

_EMOTION_WORDS: tuple[str, ...] = (
    "amazing",
    "horrific",
    "terrifying",
    "shocking",
    "outrageous",
    "stunning",
    "devastating",
    "unbelievable",
    "incredible",
    "explosive",
    "bombshell",
    "cataclysmic",
    "nightmare",
    "horror",
    "furious",
    "panic",
    "fury",
    "horrified",
    "meltdown",
)

_INFORMAL_MARKERS: tuple[str, ...] = (
    "lol",
    "lmao",
    "rofl",
    "omg",
    "btw",
    "wont believe",
    "u ",
    "ur ",
    "thru",
    "tho",
    "kinda",
    "gonna",
    "wanna",
    "y'all",
    "ya'll",
    "fwiw",
    "smh",
    "tbh",
    "ikr",
    "idk",
)

_LEFT_BIAS_WORDS: tuple[str, ...] = (
    "workers",
    "unions",
    "inequality",
    "racism",
    "oppression",
    "privilege",
    "systemic",
    "progressive",
    "progressives",
    "solidarity",
    "capitalism",
    "exploitation",
    "disenfranchised",
    "marginalized",
    "police brutality",
    "climate justice",
    "social justice",
)

_RIGHT_BIAS_WORDS: tuple[str, ...] = (
    "freedom",
    "patriot",
    "patriots",
    "liberty",
    "constitution",
    "second amendment",
    "first amendment",
    "big government",
    "tax cuts",
    "family values",
    "traditional",
    "woke",
    "globalist",
    "illegal aliens",
    "illegal immigrants",
    "law and order",
    "deep state",
    "elites",
)

_TOPIC_LEXICONS: dict[str, tuple[str, ...]] = {
    "politics": (
        "government", "election", "senate", "senator", "congress",
        "congressman", "congresswoman", "president", "democrat", "republican",
        "politician", "campaign", "ballot", "vote", "voter", "primary",
        "legislation", "bill", "lawmaker", "governor", "white house",
    ),
    "health": (
        "vaccine", "vaccination", "cancer", "disease", "doctor",
        "hospital", "pandemic", "virus", "outbreak", "patient",
        "treatment", "drug", "fda", "cdc", "who", "infection",
        "symptom", "mental health", "wellness",
    ),
    "finance": (
        "stock", "stocks", "market", "markets", "dollar", "economy",
        "investor", "investors", "bank", "banking", "crypto",
        "cryptocurrency", "bitcoin", "prices", "inflation", "fed",
        "interest rate", "wall street", "recession", "gdp", "trade",
    ),
    "science": (
        "study", "studies", "research", "researcher", "researchers",
        "nasa", "scientist", "scientists", "experiment", "theory",
        "physics", "chemistry", "biology", "genetics", "genome",
        "climate", "ecosystem", "astronomer", "telescope", "orbit",
    ),
    "sports": (
        "game", "games", "team", "teams", "league", "player",
        "players", "score", "scores", "match", "matches", "season",
        "championship", "playoffs", "quarterback", "goal", "touchdown",
        "inning", "tournament", "rivalry", "coach",
    ),
}

_PREDICTION_MARKERS: tuple[str, ...] = (
    "will", "going to", "predict", "forecast", "next year",
    "tomorrow will", "soon will", "expected to",
)
_OPINION_CLAIM_MARKERS: tuple[str, ...] = (
    "i think", "i believe", "in my opinion", "in my view",
    "should", "ought to", "we must", "personally",
)
_RUMOR_MARKERS: tuple[str, ...] = (
    "reportedly", "rumor has it", "allegedly", "sources say",
    "sources claim", "word is", "buzz is", "whispers",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _caps_ratio(text: str) -> float:
    """Fraction of alphabetic chars that are uppercase."""
    if not text:
        return 0.0
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _exclamation_density(text: str) -> float:
    """Exclamation marks per 100 chars."""
    if not text:
        return 0.0
    return 100.0 * text.count("!") / max(1, len(text))


def _count_phrase_hits(haystack_lower: str, phrases: tuple[str, ...]) -> int:
    return sum(1 for p in phrases if p in haystack_lower)


def _count_word_hits(haystack_lower: str, words: tuple[str, ...]) -> int:
    """Count whole-word matches (word-boundary aware)."""
    n = 0
    for w in words:
        # \b doesn't always behave for hyphenated words; use simple split.
        n += len(re.findall(r"\b" + re.escape(w) + r"\b", haystack_lower))
    return n


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid for confidence shaping."""
    if x >= 0:
        z = -x
        return 1.0 / (1.0 + 2.718281828459045 ** z)
    z = x
    return 2.718281828459045 ** z / (1.0 + 2.718281828459045 ** z)


def _prob_from_strength(strength: float, tie_break: int = 0) -> float:
    """Return calibrated confidence in [0.5, 0.95].

    - `strength` in [0,1] is how strongly the evidence supports the YES
      (or high-value) answer.
    - Confidence is lowest near strength=0.5 and saturates at 0.95 for
      strength near 0 or 1.
    - A tiny deterministic tie-break jitter (±0.005) is added so two
      near-identical inputs don't share a probability spuriously.

    Args:
        strength:  evidence strength.
        tie_break: small deterministic int from `content_hash`.

    Returns:
        Confidence in [0.5, 0.95].
    """
    s = _clamp(strength, 0.0, 1.0)
    conf = 0.5 + 0.45 * (abs(s - 0.5) / 0.5)
    jitter = ((tie_break % 11) - 5) / 1000.0  # ±0.005
    return _clamp(conf + jitter, 0.5, 0.95)


def _hash_tie_break(state: str) -> int:
    """Stable integer tie-break from `state` text."""
    h = content_hash(state)
    return int(h, 16)


# ---------------------------------------------------------------------------
# Per-question handlers
# ---------------------------------------------------------------------------


def _h_is_clickbait(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    """Clickbait: ALL-CAPS words in headline, '!!', curiosity-gap phrases."""
    s = 0.0
    if headline:
        words = headline.split()
        cap_words = sum(
            1
            for w in words
            if len(w) >= 3 and w.isupper() and any(c.isalpha() for c in w)
        )
        total = max(1, len(words))
        cap_ratio = cap_words / total
        s += 0.3 * _clamp(cap_ratio / 0.3, 0.0, 1.0)
        if "!!" in headline:
            s += 0.2
        s += 0.5 * _clamp(_count_phrase_hits(state_lower, _CLICKBAIT_PHRASES) / 2.0, 0.0, 1.0)
    s = _clamp(s, 0.0, 1.0)
    return s, _prob_from_strength(s, _hash_tie_break(state))


def _h_uses_anonymous_sources(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    n = _count_phrase_hits(state_lower, _ANON_SOURCE_PHRASES)
    s = _clamp(n / 2.0, 0.0, 1.0)
    return s, _prob_from_strength(s, _hash_tie_break(state))


def _h_cites_verifiable_sources(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    n = _count_phrase_hits(state_lower, _VERIFIABLE_SOURCE_PHRASES)
    # Multiple distinct sources reinforce; treat 3 as saturation.
    s = _clamp(n / 3.0, 0.0, 1.0)
    return s, _prob_from_strength(s, _hash_tie_break(state))


def _h_is_satire(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    n = _count_phrase_hits(state_lower, _SATIRE_MARKERS)
    s = _clamp(n / 1.0, 0.0, 1.0)
    return s, _prob_from_strength(s, _hash_tie_break(state))


def _h_contains_conspiracy_framing(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    n = _count_phrase_hits(state_lower, _CONSPIRACY_PHRASES)
    s = _clamp(n / 2.0, 0.0, 1.0)
    return s, _prob_from_strength(s, _hash_tie_break(state))


def _h_is_opinion_not_news(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    n = _count_phrase_hits(state_lower, _OPINION_PHRASES)
    s = _clamp(n / 2.0, 0.0, 1.0)
    # Hedging: opinion pieces tend to use first-person "I"
    first_person = len(re.findall(r"\bi\b", state_lower))
    s += 0.15 * _clamp(first_person / 5.0, 0.0, 1.0)
    s = _clamp(s, 0.0, 1.0)
    return s, _prob_from_strength(s, _hash_tie_break(state))


def _h_has_statistical_claims(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    pct = len(re.findall(r"\d+\s*%", state_lower))
    ratios = len(re.findall(r"\d+\s*out of\s*\d+", state_lower))
    studies_claim = len(re.findall(r"studies? show", state_lower))
    n = pct + ratios + studies_claim
    # If stats present but no verifiable-source attribution → high.
    verifiable = _count_phrase_hits(state_lower, _VERIFIABLE_SOURCE_PHRASES)
    base = _clamp(n / 2.0, 0.0, 1.0)
    if n > 0 and verifiable == 0:
        base = min(1.0, base + 0.2)
    s = base
    return s, _prob_from_strength(s, _hash_tie_break(state))


def _h_impersonates_authority(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    n = _count_phrase_hits(state_lower, _AUTHORITY_IMPERSONATION_PHRASES)
    s = _clamp(n / 2.0, 0.0, 1.0)
    return s, _prob_from_strength(s, _hash_tie_break(state))


def _h_has_unverified_claims(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    """Unverified claims = factual claims minus verifiable-source coverage.

    Heuristic: any sentence with a factual verb (is/are/was/were/said)
    counts as a claim; subtract sentences explicitly attributed to a
    verifiable source.
    """
    sentences = re.split(r"(?<=[.!?])\s+", state_lower)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    claim_re = re.compile(r"\b(is|are|was|were|said|says|reported|found|showed)\b")
    claim_count = sum(1 for s in sentences if claim_re.search(s))
    verifiable = _count_phrase_hits(state_lower, _VERIFIABLE_SOURCE_PHRASES)
    unverified = max(0, claim_count - verifiable)
    s = _clamp(unverified / 3.0, 0.0, 1.0)
    return s, _prob_from_strength(s, _hash_tie_break(state))


def _h_headline_contradicts_body(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    """Low content overlap between headline words and body → contradiction.

    Returns high if headline has many words NOT present in body AND the
    body is non-trivial. Negation words ("not", "no") in either also
    raise the score.
    """
    if not headline or not body_lower:
        s = 0.0
        return s, _prob_from_strength(s, _hash_tie_break(state))
    hl_words = set(re.findall(r"[a-z]+", headline.lower()))
    # Drop stopwords from headline token set.
    stop = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
            "with", "is", "are", "was", "were", "be", "by", "at", "as"}
    hl_words -= stop
    if not hl_words:
        s = 0.0
        return s, _prob_from_strength(s, _hash_tie_break(state))
    body_words = set(re.findall(r"[a-z]+", body_lower))
    missing = hl_words - body_words
    overlap_ratio = 1.0 - (len(missing) / len(hl_words))
    # Low overlap → high contradiction score.
    s = _clamp(1.0 - overlap_ratio, 0.0, 1.0)
    # Negation in headline but not body is a strong contradiction signal.
    if (" not " in headline.lower() or " no " in headline.lower()) and " not " not in body_lower:
        s = min(1.0, s + 0.25)
    return s, _prob_from_strength(s, _hash_tie_break(state))


def _h_is_fake_overall(
    state: str,
    headline: str,
    body_lower: str,
    state_lower: str,
    sub_answers: dict[str, float],
) -> tuple[float, float]:
    """Aggregate: weighted combination of the other NOUL signals.

    `sub_answers` is the per-question YES probability already computed for
    the 10 sibling NOUL questions (excluding this one).
    """
    weights = {
        "is_clickbait": 0.12,
        "has_unverified_claims": 0.15,
        "uses_anonymous_sources": 0.10,
        "contains_conspiracy_framing": 0.15,
        "cites_verifiable_sources": -0.18,  # negative: sources REDUCE fake
        "is_satire": -0.05,  # satire != fake; treat as soft anti-fake signal
        "is_opinion_not_news": 0.05,
        "has_statistical_claims": 0.05,
        "impersonates_authority": 0.15,
        "headline_contradicts_body": 0.10,
    }
    s = 0.0
    for name, w in weights.items():
        s += w * sub_answers.get(name, 0.0)
    # Style priors: low professionalism and high sensationalism raise fake.
    caps_ratio = _caps_ratio(headline or state)
    if caps_ratio > 0.3:
        s += 0.10
    exclaim = _exclamation_density(state)
    if exclaim > 1.0:
        s += 0.05
    s = _clamp(s, 0.0, 1.0)
    return s, _prob_from_strength(s, _hash_tie_break(state))


# ----- SCORE handlers -------------------------------------------------------


def _h_emotional_intensity(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    words = _count_word_hits(state_lower, _EMOTION_WORDS)
    excl = _exclamation_density(state)
    caps = _caps_ratio(headline or state)
    raw = words * 1.5 + excl * 5.0 + caps * 20.0
    value = _clamp(raw, 0.0, 10.0)
    s = value / 10.0
    return value, _prob_from_strength(s, _hash_tie_break(state))


def _h_sensationalism(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    cb = _count_phrase_hits(state_lower, _CLICKBAIT_PHRASES)
    caps = _caps_ratio(headline or state)
    excl = _exclamation_density(headline or state)
    raw = cb * 2.5 + caps * 20.0 + excl * 8.0
    value = _clamp(raw, 0.0, 10.0)
    s = value / 10.0
    return value, _prob_from_strength(s, _hash_tie_break(state))


def _h_factual_specificity(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    """Higher = more specific & verifiable.

    Boosts: numbers, percentages, dates, named sources.
    """
    pct = len(re.findall(r"\d+\s*%", state_lower))
    nums = len(re.findall(r"\b\d+\b", state_lower))
    dates = len(re.findall(r"\b(19|20)\d{2}\b", state_lower))
    sources = _count_phrase_hits(state_lower, _VERIFIABLE_SOURCE_PHRASES)
    raw = pct * 1.0 + nums * 0.4 + dates * 1.5 + sources * 2.0
    value = _clamp(raw, 0.0, 10.0)
    s = value / 10.0
    return value, _prob_from_strength(s, _hash_tie_break(state))


def _h_political_bias(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    """Right-positive, left-negative scale (-5..+5)."""
    left = _count_word_hits(state_lower, _LEFT_BIAS_WORDS)
    right = _count_word_hits(state_lower, _RIGHT_BIAS_WORDS)
    delta = right - left
    value = _clamp(float(delta), -5.0, 5.0)
    s = abs(value) / 5.0
    return value, _prob_from_strength(s, _hash_tie_break(state))


def _h_writing_professionalism(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[float, float]:
    """Higher = more professional. Inverse of informal markers + caps/exclaim."""
    informal = _count_phrase_hits(state_lower, _INFORMAL_MARKERS)
    excl = _exclamation_density(state)
    caps = _caps_ratio(headline or state)
    raw = 10.0 - (informal * 1.5 + excl * 4.0 + caps * 25.0)
    value = _clamp(raw, 0.0, 10.0)
    s = value / 10.0
    return value, _prob_from_strength(s, _hash_tie_break(state))


# ----- CHOICE handlers ------------------------------------------------------


def _h_topic(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[str, float]:
    counts: dict[str, int] = {}
    for topic, lex in _TOPIC_LEXICONS.items():
        counts[topic] = _count_word_hits(state_lower, lex)
    best_topic = "other"
    best_count = 0
    for topic, c in counts.items():
        if c > best_count or (c == best_count and c > 0):
            best_count = c
            best_topic = topic
    # Deterministic tie-break among equal leaders using stable hash.
    leaders = [t for t, c in counts.items() if c == best_count and c > 0]
    if len(leaders) > 1:
        tb = _hash_tie_break(state)
        best_topic = leaders[tb % len(leaders)]
    strength = _clamp(best_count / 3.0, 0.0, 1.0) if best_topic != "other" else 0.05
    return best_topic, _prob_from_strength(strength, _hash_tie_break(state))


def _h_claim_type(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[str, float]:
    prediction = _count_phrase_hits(state_lower, _PREDICTION_MARKERS)
    opinion = _count_phrase_hits(state_lower, _OPINION_CLAIM_MARKERS)
    rumor = _count_phrase_hits(state_lower, _RUMOR_MARKERS)
    scores = {
        "prediction": prediction,
        "opinion": opinion,
        "rumor": rumor,
        "fact": 1,  # default weak prior
    }
    best = max(scores, key=lambda k: scores[k])
    leaders = [k for k, v in scores.items() if v == scores[best] and v > 0]
    if len(leaders) > 1:
        tb = _hash_tie_break(state)
        best = leaders[tb % len(leaders)]
    if scores[best] == 0:
        best = "fact"
    strength = _clamp(scores[best] / 2.0, 0.0, 1.0)
    return best, _prob_from_strength(strength, _hash_tie_break(state))


def _h_manipulation_tactic(state: str, headline: str, body_lower: str, state_lower: str) -> tuple[str, float]:
    fear = _count_word_hits(state_lower, _FEAR_PHRASES)
    outrage = _count_word_hits(state_lower, _OUTRAGE_PHRASES)
    false_auth = _count_phrase_hits(state_lower, _AUTHORITY_IMPERSONATION_PHRASES)
    cherry = _count_word_hits(state_lower, _CHERRY_PICK_PHRASES)
    scores = {
        "fear": fear,
        "outrage": outrage,
        "false_authority": false_auth,
        "cherry_picking": cherry,
        "none": 0,
    }
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "none", _prob_from_strength(0.1, _hash_tie_break(state))
    leaders = [k for k, v in scores.items() if v == scores[best] and v > 0]
    if len(leaders) > 1:
        tb = _hash_tie_break(state)
        best = leaders[tb % len(leaders)]
    strength = _clamp(scores[best] / 3.0, 0.0, 1.0)
    return best, _prob_from_strength(strength, _hash_tie_break(state))


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_NOUL_HANDLERS: dict[str, Callable[..., tuple[float, float]]] = {
    "is_clickbait": _h_is_clickbait,
    "uses_anonymous_sources": _h_uses_anonymous_sources,
    "cites_verifiable_sources": _h_cites_verifiable_sources,
    "is_satire": _h_is_satire,
    "contains_conspiracy_framing": _h_contains_conspiracy_framing,
    "is_opinion_not_news": _h_is_opinion_not_news,
    "has_statistical_claims": _h_has_statistical_claims,
    "impersonates_authority": _h_impersonates_authority,
    "has_unverified_claims": _h_has_unverified_claims,
    "headline_contradicts_body": _h_headline_contradicts_body,
}

_SCORE_HANDLERS: dict[str, Callable[..., tuple[float, float]]] = {
    "emotional_intensity": _h_emotional_intensity,
    "sensationalism": _h_sensationalism,
    "factual_specificity": _h_factual_specificity,
    "political_bias": _h_political_bias,
    "writing_professionalism": _h_writing_professionalism,
}

_CHOICE_HANDLERS: dict[str, Callable[..., tuple[Any, float]]] = {
    "topic": _h_topic,
    "claim_type": _h_claim_type,
    "manipulation_tactic": _h_manipulation_tactic,
}


# ---------------------------------------------------------------------------
# JevSimulated
# ---------------------------------------------------------------------------


class JevSimulated(JevClient):
    """Deterministic simulated backend implementing the JevClient contract.

    See module docstring for the substitution contract and determinism
    guarantees. The class is stateless except for an optional logger, so
    it is safe to share across asyncio tasks.
    """

    def __init__(self, logger=None) -> None:
        self._log = logger
        # Synchronous compute is wrapped in asyncio to satisfy the
        # async contract; no real network I/O happens here.

    async def ask(
        self,
        state: str,
        questions: "Optional[list[Question]]" = None,
    ) -> dict[str, TypedAnswer]:
        if questions is None:
            try:
                from fakenews.config import load_questions
                questions = load_questions()
            except Exception:
                questions = []
        # Stage A: compute every non-aggregate NOUL/SCORE/CHOICE first
        # so the aggregator `is_fake_overall` can read them.
        headline, body = _split_headline_body(state)
        state_lower = state.lower()
        body_lower = body.lower()
        ctx = (state, headline, body_lower, state_lower)

        answers: dict[str, TypedAnswer] = {}
        sub_yes: dict[str, float] = {}  # for is_fake_overall aggregation

        for q in questions:
            if q.name in _NOUL_HANDLERS and q.name != "is_fake_overall":
                val, prob = _NOUL_HANDLERS[q.name](*ctx)
                answers[q.name] = TypedAnswer(
                    name=q.name, type=AnswerType.NOUL, value=val, probability=prob
                )
                sub_yes[q.name] = val
            elif q.name in _SCORE_HANDLERS:
                val, prob = _SCORE_HANDLERS[q.name](*ctx)
                # Clip to declared bounds if present.
                if q.min is not None:
                    val = max(q.min, val)
                if q.max is not None:
                    val = min(q.max, val)
                answers[q.name] = TypedAnswer(
                    name=q.name, type=AnswerType.SCORE, value=val, probability=prob
                )
            elif q.name in _CHOICE_HANDLERS:
                val, prob = _CHOICE_HANDLERS[q.name](*ctx)
                # Coerce to a valid option if `options` declared.
                if q.options:
                    if val not in q.options:
                        # fallback: deterministic first option (still stable)
                        val = q.options[0]
                answers[q.name] = TypedAnswer(
                    name=q.name, type=AnswerType.CHOICE, value=val, probability=prob
                )

        # Stage B: aggregate `is_fake_overall` last (if it was requested).
        for q in questions:
            if q.name == "is_fake_overall":
                val, prob = _h_is_fake_overall(state, headline, body_lower, state_lower, sub_yes)
                answers[q.name] = TypedAnswer(
                    name=q.name, type=AnswerType.NOUL, value=val, probability=prob
                )

        # Guarantee every question is present per contract.
        for q in questions:
            if q.name not in answers:
                # Unknown question → emit a conservative neutral NOUL answer
                # (kept type-correct) so downstream code keeps working.
                answers[q.name] = TypedAnswer(
                    name=q.name,
                    type=q.type,
                    value=_neutral_value(q),
                    probability=0.5,
                )
        return answers


def _neutral_value(q: Question) -> Any:
    """Return a neutral placeholder value for an unrecognised question."""
    if q.type is AnswerType.SCORE:
        if q.min is not None and q.max is not None:
            return (q.min + q.max) / 2.0
        return 0.0
    if q.type is AnswerType.CHOICE:
        if q.options:
            return q.options[0]
        return "none"
    return 0.5  # NOUL


def _split_headline_body(state: str) -> tuple[str, str]:
    """Split `state` into (headline, body) on the first blank line.

    The signal extractor joins them as `f"{headline}\n\n{body}"`, so the
    canonical split point is a `\n\n`. If absent, treat the first line as
    the headline and the remainder as body.
    """
    if not state:
        return "", ""
    if "\n\n" in state:
        headline, _, body = state.partition("\n\n")
        return headline.strip(), body.strip()
    if "\n" in state:
        headline, _, body = state.partition("\n")
        return headline.strip(), body.strip()
    return state.strip(), ""


__all__ = ["JevSimulated"]
