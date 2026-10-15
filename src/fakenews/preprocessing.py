# -*- coding: utf-8 -*-
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
"""Text preprocessing for the fake-news-detection pipeline.

Leakage-aware text cleaning for news articles:

* ``strip_source_tags``   — remove parenthetical source tags ("(Reuters)"),
                            location prefixes ("WASHINGTON - "), and common
                            newsletter/clickbait boilerplate ("Sign up for
                            our newsletter", "Read more:", ...). Regex-based.
* ``normalize_text``       — NFC unicode, collapse whitespace, strip URLs,
                            HTML tags, HTML entities and email addresses.
                            Optional lowercase flag for the TF-IDF featurizer
                            to inherit.
* ``clean_article``       — apply strip + normalize to an Article's headline
                            and body and return a fresh Article (uses
                            ``dataclasses.replace`` so any extra fields
                            added later by fn-A pass through untouched).
* ``deduplicate``         — drop exact and near-duplicate articles by a
                            normalized-text fingerprint (BLAKE2b-16). Used
                            after loading to prevent train/test leakage from
                            duplicated wire copy.
* ``preprocess_pipeline`` — convenience alias for ``clean_article`` (the
                            per-batch deduplicate is exposed separately so
                            callers can apply it at the corpus level, after
                            splits, to avoid leakage between folds).

The strip + normalize order is intentional: source-tag regexes match BEFORE
whitespace is collapsed so multi-line boilerplate like "Sign up for our
newsletter\\nGet the best of ... in your inbox" is caught as a unit.

No network, no side effects, no third-party imports beyond stdlib. The
downstream ``data.py`` module wires these into every loader.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import replace
from typing import Iterable

from .types import Article

__all__ = [
    "strip_source_tags",
    "normalize_text",
    "clean_article",
    "deduplicate",
    "preprocess_pipeline",
]


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Parenthetical source tags: "(Reuters)", "(AP)", "(CNN)", "(BBC)", plus a
# handful of common Western outlets that wire copy is attributed to. Matched
# case-insensitively so "(reuters)" / "(REUTERS)" both survive the cut.
_SOURCE_TAG_PATTERN = re.compile(
    r"\(\s*(?:"
    r"Reuters|AP|Associated Press|CNN|BBC|Fox News|Fox|"
    r"NBC News|NBC|CBS News|CBS|ABC News|ABC|AFP|Bloomberg|"
    r"The Guardian|The Washington Post|The New York Times|NYTimes|"
    r"NPR|USA Today|Politico|The Hill|The Telegraph|Daily Mail|"
    r"Instabit|BuzzFeed|HuffPost|Huffington Post|TIME|Newsweek|"
    r"Wall Street Journal|WSJ|Financial Times|FT"
    r")\s*\)\s*",
    re.IGNORECASE,
)

# Dateline / location prefixes:
#   "WASHINGTON - ", "WASHINGTON — ", "WASHINGTON, April 3 - ",
#   "LONDON — ", "PARIS (AP) - " (source tag already removed above).
# Match 1+ Capitalized/UPPERCASE words optionally followed by a state code
# or date phrase, then a dash separator (en-dash, em-dash, hyphen, or comma).
# Anchor to start-of-line so we only strip the *lede* dateline (a paragraph
# in the middle of the body that begins with "LONDON - " is rare and the
# downside of stripping it is tiny; we still keep the rest).
_LOCATION_PREFIX_PATTERN = re.compile(
    r"""^\s*
        (?P<loc>
            (?:[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4})  # city name(s)
            (?:\s*,\s*[A-Z]{2})?                                # optional state code
            (?:\s*\([^)]*\))?                                   # optional (AP) tag
        )
        \s*(?:-{1,2}|—|–|·)\s*                                  # separator
    """,
    re.MULTILINE | re.VERBOSE | re.IGNORECASE,
)

# Boilerplate phrases commonly prepended/appended by CMSes (WordPress,
# Substack, Mailchimp, BuzzFeed-style "Image: ..."). All matched
# case-insensitively and anchored to line boundaries where appropriate.
_BOILERPLATE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\s*Image\s*:?\s*[^\n]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Photo\s*:?\s*[^\n]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Getty Images[^\n]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*\(AP Photo\)\s*[^\n]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Screenshot\s*:?\s*[^\n]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"Sign up for (?:our )?newsletter[^\n]*", re.IGNORECASE),
    re.compile(r"Subscribe to (?:our |the )?[^\n]*newsletter[^\n]*", re.IGNORECASE),
    re.compile(r"Read more:?[^\n]*", re.IGNORECASE),
    re.compile(r"Related (?:articles?|coverage|stories?|videos?):?[^\n]*", re.IGNORECASE),
    re.compile(r"Recommended (?:reading|stories|articles):?[^\n]*", re.IGNORECASE),
    re.compile(r"Click here to[^\n]*", re.IGNORECASE),
    re.compile(r"Follow us on[^\n]*", re.IGNORECASE),
    re.compile(r"Share this (?:article|story|post)[^\n]*", re.IGNORECASE),
    re.compile(r"^\s*Trending now:?\s*[^\n]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Editor'?s note:?\s*[^\n]*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*Advertisement:?\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*More from (?:our )?[^\n]*$", re.IGNORECASE | re.MULTILINE),
]

# --- normalization patterns ------------------------------------------------
_URL_PATTERN = re.compile(
    r"""(?:https?://|www\.|ftp://|mailto:)    # scheme
        [^\s<>"'()]+                          # body
        |                                     # or
        \b[a-z0-9][a-z0-9-]{1,61}[a-z0-9]     # domain
        \.(?:com|org|net|gov|edu|io|co|us|uk|info|biz|me|tv|news|ai)
        (?:/[^\s<>"']*)?                       # optional path
    """,
    re.IGNORECASE | re.VERBOSE,
)
_HTML_TAG_PATTERN = re.compile(r"<[a-zA-Z][^>]*>|</[a-zA-Z][^>]*>")
_HTML_ENTITY_PATTERN = re.compile(r"&#?\w+;")
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_NON_PRINTABLE_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def strip_source_tags(text: str) -> str:
    """Remove parenthetical source tags, dateline prefixes, and boilerplate.

    Strips, in order:

    1. Parenthetical outlet tags like "(Reuters)", "(AP)", "(CNN)".
       These are a *leakage* signal — the Kaggle Fake/Real News dataset uses
       "(Reuters)" as the verbatim tag on every real article, so models that
       see it learn "tag present => real" rather than anything substantive.
    2. Dateline / location prefixes like "WASHINGTON - " or "LONDON — ".
       Strong dataset-of-origin signal for the Kaggle Fake/Real set (which
       starts every real article with one). Removing it forces the model to
       look at content.
    3. Boilerplate appended by CMSes ("Sign up for our newsletter",
       "Read more: ...", "Image: ...", "Getty Images", "Advertisement",
       "Trending now:", "Editor's note:"). These add noise without signal and
       are usually dataset-specific.

    Whitespace is collapsed *after* all removals so multi-line boilerplate
    patterns don't leave dangling newlines.

    Args:
        text: Raw article headline or body (may be empty / None-tolerant).

    Returns:
        Cleaned string with source tags, datelines, and boilerplate removed.
        Always a str (never raises on bad input).
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)

    out = _SOURCE_TAG_PATTERN.sub(" ", text)
    out = _LOCATION_PREFIX_PATTERN.sub("", out)
    for pat in _BOILERPLATE_PATTERNS:
        out = pat.sub(" ", out)
    out = _WHITESPACE_PATTERN.sub(" ", out).strip()
    return out


def normalize_text(text: str, lowercase: bool = False) -> str:
    """NFC-normalize, strip URLs/HTML/emails, and collapse whitespace.

    Pipeline (in order):

    1. NFC unicode normalization — folds combining-character forms so that
       ``"cafe\u0301"`` and ``"café"`` compare equal. Important for any
       dataset that's been scraped from the open web.
    2. Drop control characters (everything in ``\x00-\x1F`` except ``\t\n``).
    3. Strip HTML tags, HTML entities (``&amp;``, ``&#39;``), URLs (http,
       https, ftp, mailto, plus bare domain + TLD), and email addresses.
    4. Collapse all whitespace runs (incl. tabs, newlines, NBSPs) to a single
       ASCII space. Leading/trailing whitespace is stripped.
    5. Optional lowercase for bag-of-words consumers (TF-IDF).

    Args:
        text:      Raw input string. None / non-str are stringified.
        lowercase: If True, returns the lower-cased text. Defaults to False
                   because the metadata featurizer computes ``caps_ratio``
                   from the cleaned text — folding case here would zero
                   that signal out.

    Returns:
        Normalized string. Always a str (never raises on bad input).
    """
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)
    text = _NON_PRINTABLE_PATTERN.sub(" ", text)
    text = _HTML_TAG_PATTERN.sub(" ", text)
    text = _HTML_ENTITY_PATTERN.sub(" ", text)
    text = _URL_PATTERN.sub(" ", text)
    text = _EMAIL_PATTERN.sub(" ", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def clean_article(article) -> Article:
    """Apply ``strip_source_tags + normalize_text`` to headline and body.

    Accepts an Article object OR a plain string (treated as the body).

    Returns a *new* Article instance (the input is never mutated).
    Uses ``dataclasses.replace`` so any extra fields added to Article by a
    future revision of fn-A are preserved unchanged. The ``source`` field is
    also normalized (it's often "Reuters" vs "reuters" inconsistently across
    datasets) but NOT source-tag-stripped — the source field carries real
    metadata we want to keep for the ``source_reputation`` feature.

    Args:
        article: Article with raw (possibly dirty) headline/body/source.

    Returns:
        New Article with cleaned fields. ``url`` and ``date`` pass through.
    """
    if isinstance(article, str):
        # string in -> cleaned string out (preserve the simple contract)
        return normalize_text(strip_source_tags(article))
    headline = normalize_text(strip_source_tags(article.headline or ""))
    body = normalize_text(strip_source_tags(article.body or ""))
    source = normalize_text(article.source or "").strip() if article.source else ""
    return replace(
        article,
        headline=headline,
        body=body,
        source=source,
    )


def _fingerprint(text: str) -> str:
    """Return a 16-char BLAKE2b hex digest of `text`.

    BLAKE2b is faster than SHA-256 and is cryptographically adequate for
    dedup keys at our scale (~75k articles per dataset). The 16-byte digest
    (128 bits) gives ~negligible collision probability well past the dataset
    sizes we'll ever see.
    """
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def deduplicate(articles, threshold: float = 0.9) -> list:

    """Drop exact and near-duplicate articles by normalized-text fingerprint.

    The fingerprint is computed over the *normalized* concatenation of the
    article's headline + body (after source-tag stripping + whitespace
    normalization). This catches both:

    * Exact duplicates — same text byte-for-byte.
    * Near-duplicates that differ only in whitespace, HTML, URLs, source
      tags, or unicode normalization form. (Two scraped copies of the same
      AP wire story commonly differ in exactly these ways.)

    The *first* occurrence wins; later duplicates are dropped. Callers
    should run ``deduplicate`` AFTER train/test splitting (or operate on
    each split separately) so that a duplicate that crosses the train/test
    boundary is caught — that's the leakage case that matters.

    Args:
        articles: Iterable of Article objects.

    Returns:
        New list with duplicates removed, in first-seen order.
    """
    seen: set[str] = set()
    out: list[Article] = []
    for art in articles:
        if isinstance(art, str):
            norm = normalize_text(strip_source_tags(art))
        else:
            norm = normalize_text(
                strip_source_tags((art.headline or "") + " " + (art.body or ""))
        )
        fp = _fingerprint(norm)
        if fp in seen:
            continue
        seen.add(fp)
        out.append(art)
    # Near-duplicate pass: token-Jaccard >= threshold vs an earlier kept entry.
    def _tok(t):
        return set(re.sub(r'[^a-z0-9\s]', ' ', t.lower()).split())
    def _txt(a):
        return a if isinstance(a, str) else ((getattr(a,'headline','') or '') + ' ' + (getattr(a,'body','') or ''))
    kept = []
    for art in out:
        tk = _tok(normalize_text(strip_source_tags(_txt(art))))
        if not any((lambda kt: len(tk & kt) / (len(tk | kt) or 1) >= threshold)(_tok(normalize_text(strip_source_tags(_txt(k))))) for k in kept):
            kept.append(art)
    return kept


def preprocess_pipeline(article: Article) -> Article:
    """Full per-article preprocessing pipeline (alias for ``clean_article``).

    Provided for callers that want a single function name to wire into a
    higher-level pipeline. The corpus-level ``deduplicate`` step is exposed
    separately because it operates on a list, not on one article.

    Args:
        article: Raw Article.

    Returns:
        Cleaned Article.
    """
    return clean_article(article)

# Backwards-compat alias (test + external callers may use the short name).
normalize = normalize_text
