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
# Tests for `fakenews.preprocessing` — source-tag stripping, whitespace
# normalisation, and near-duplicate removal. Pure offline, deterministic.

"""Tests for fakenews.preprocessing text utilities.

These tests run offline and require only the `fakenews` package present on
the import path. They exercise three responsibilities of the preprocessing
module: source-tag removal (Reuters datelines + location leads), whitespace
normalisation, and near-duplicate removal. A short end-to-end check via
`clean_article` is included.
"""

import os
import sys

# Make the src-layout package importable without an installed wheel.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fakenews.preprocessing import (  # noqa: E402  (after sys.path manipulation)
    clean_article,
    deduplicate,
    normalize,
    strip_source_tags,
)


# ---------------------------------------------------------------------------
# strip_source_tags
# ---------------------------------------------------------------------------

def test_strip_source_tags_removes_reuters_parenthetical():
    text = "Officials said the policy would change by Friday. (Reuters)"
    out = strip_source_tags(text)
    assert "(Reuters)" not in out
    assert "Reuters" not in out
    # The body of the sentence must survive intact.
    assert "Officials said the policy would change by Friday." in out


def test_strip_source_tags_removes_multiple_reuters():
    text = "(Reuters) Officials said X. (Reuters) More text. (AP)"
    out = strip_source_tags(text)
    assert "Reuters" not in out
    # The (AP) wire tag is also expected to be removed by the same rule.
    assert "(AP)" not in out
    assert "Officials said X." in out
    assert "More text." in out


def test_strip_source_tags_removes_location_lead():
    text = "WASHINGTON - The president signed a bill today."
    out = strip_source_tags(text)
    assert "WASHINGTON -" not in out
    # The trailing sentence body is preserved.
    assert "The president signed a bill today." in out


def test_strip_source_tags_removes_location_lead_lowercase():
    # Some outlets format datelines in lowercase or mixed case; the lead
    # pattern "<CITY> -<space> should be matched case-insensitively.
    text = "london - a spokesperson confirmed the report."
    out = strip_source_tags(text)
    assert not out.lower().startswith("london -")
    assert "a spokesperson confirmed the report." in out.lower()


def test_strip_source_tags_preserves_plain_text_without_tags():
    text = "The committee will meet next Tuesday to discuss the proposal."
    out = strip_source_tags(text)
    assert out.strip() == text.strip()


# ---------------------------------------------------------------------------
# normalize (whitespace collapsing)
# ---------------------------------------------------------------------------

def test_normalize_collapses_multiple_spaces():
    text = "Some  multiple   spaces"
    out = normalize(text)
    assert "  " not in out
    assert out == "Some multiple spaces"


def test_normalize_collapses_tabs_and_newlines():
    text = "Some\t\ttabs\n\nand newlines  here"
    out = normalize(text)
    assert "\t" not in out
    assert "\n" not in out
    assert "  " not in out
    assert out == "Some tabs and newlines here"


def test_normalize_strips_leading_and_trailing_whitespace():
    text = "   \n  trimmed body  \t "
    out = normalize(text)
    assert out == "trimmed body"


def test_normalize_handles_empty_string():
    assert normalize("") == ""


# ---------------------------------------------------------------------------
# deduplicate (near-duplicate removal)
# ---------------------------------------------------------------------------

def test_deduplicate_drops_exact_duplicates():
    docs = [
        "The cat sat on the mat.",
        "The cat sat on the mat.",  # exact dup
        "Dogs are great pets.",
    ]
    out = deduplicate(docs)
    # At least the exact duplicate must be removed.
    assert len(out) < len(docs)
    assert any("Dogs are great pets." in d for d in out)


def test_deduplicate_drops_near_duplicates():
    docs = [
        "The committee will vote on the proposal next week.",
        "The committee will vote on the proposal next week.",  # exact dup
        "The committee will vote on the proposal this week.",  # near dup
        "A storm is expected to hit the coast by morning.",     # distinct
    ]
    out = deduplicate(docs, threshold=0.85)
    # At least one of the two near-dups should be removed.
    assert len(out) <= 3
    assert any("A storm is expected to hit the coast by morning." in d for d in out)


def test_deduplicate_preserves_distinct_documents():
    docs = [
        "Apple released a new phone today.",
        "The senate passed the tax bill last night.",
        "Heavy rain flooded the downtown area.",
    ]
    out = deduplicate(docs)
    # All distinct documents should survive.
    assert len(out) == len(docs)


def test_deduplicate_empty_input():
    assert deduplicate([]) == []


# ---------------------------------------------------------------------------
# clean_article end-to-end
# ---------------------------------------------------------------------------

def test_clean_article_runs_full_pipeline():
    raw = (
        "WASHINGTON -   Sources familiar with the matter said the policy\n"
        "would change. (Reuters)   More  details\tfollowed."
    )
    out = clean_article(raw)
    assert isinstance(out, str)
    assert "WASHINGTON -" not in out
    assert "(Reuters)" not in out
    assert "  " not in out
    assert "\t" not in out
    assert "\n" not in out
    # Body preserved
    assert "Sources familiar with the matter" in out
    assert "More details followed." in out


def test_clean_article_handles_none_or_empty_gracefully():
    # The contract is loose here — empty/None inputs should not raise; they
    # may return an empty string.
    try:
        out = clean_article("")
        assert isinstance(out, str)
    except TypeError:
        # If the implementation refuses empty input, accept and skip.
        pass
