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
"""Dataset loaders + leak-proof splits for the fake-news pipeline.

Four public datasets, each normalized to the canonical schema:

    id      title   text    label   source   date   needs_review

* ``label`` is the binary target: 1 = fake, 0 = real.
* ``needs_review`` is True only for LIAR rows whose 6-class label is
  genuinely ambiguous ("half-true", "mostly-true"). For all other rows
  (Fake/Real, FakeNewsNet, WELFake) it is False.

Datasets:

* ``fake_real``  — Kaggle "Fake and real news dataset" (Clement Bisaillon):
                   Fake.csv + True.csv combined. Two known schema variants
                   are supported (single combined CSV, or a directory with
                   Fake.csv + True.csv).
* ``liar``       — LIAR TSV (Politifact + raw fact-checks, 6-class).
* ``fakenewsnet``— FakeNewsNet (BuzzFeed + PolitiFact subset CSV exports).
* ``welfake``    — WELFake (Kaggle, 72k 0/1-labeled rows).

Splits (leakage-proof):

* ``time_based_split``    — sort by ``date`` (parsed) and take the last
                            ``test_frac`` rows chronologically. No future
                            leakage. Rows whose date cannot be parsed are
                            dropped with a logged warning.
* ``cross_dataset_split`` — train on one dataset, test on another. The
                            real generalization test: distribution shift
                            between sources/topics/eras.

Plus a fully deterministic synthetic dataset generator for tests, demos,
and bootstrapping the pipeline before any real data has been downloaded:

* ``make_synthetic_dataset`` — Real articles use Reuters/AP source tags,
  location prefixes, named officials, verifiable stats. Fake articles use
  ALL-CAPS headlines, clickbait ("you won't believe"), anonymous sources
  ("according to insiders"), conspiracy framing ("they don't want you to
  know"), and high emotional intensity ("!!!"). Seeded.

This module does NOT touch the network — pass in a path and it reads CSV /
TSV / directory contents. The only file in the repo that may call out to
the network is ``scripts/download_data.py``.
"""
from __future__ import annotations

import calendar
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .preprocessing import clean_article, normalize_text, strip_source_tags
from .types import Article
from .utils import get_logger

__all__ = [
    "DATASETS",
    "DatasetName",
    "load_dataset",
    "to_articles",
    "extract_labels",
    "time_based_split",
    "cross_dataset_split",
    "load_fake_real",
    "load_liar",
    "load_fakenewsnet",
    "load_welfake",
    "make_synthetic_dataset",
    "default_dataset_path",
]

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical schema
# ---------------------------------------------------------------------------

CANONICAL_COLUMNS: list[str] = [
    "id", "title", "text", "label", "source", "date", "needs_review",
]


DatasetName = str  # literal alias: "fake_real" | "liar" | "fakenewsnet" | "welfake"


# ---------------------------------------------------------------------------
# DATASETS — per-dataset metadata, expected columns, label mappings
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict[str, Any]] = {
    "fake_real": {
        "name": "Kaggle Fake and Real News",
        "loader": "load_fake_real",
        "format": "csv",  # Fake.csv + True.csv OR single combined CSV
        "expected_columns": ["title", "text", "subject", "date"],
        "label_field": "label",  # set by the loader; 0/1
        "label_map": {0: 0, 1: 1},  # identity, since loader sets labels
        "needs_review": False,
        "default_path": "data/raw/fake_real",  # dir or combined CSV
        "notes": (
            "Clement Bisaillon's 'Fake and real news dataset'. Distributed as "
            "Fake.csv + True.csv. Wire-tag '(Reuters)' is on every real "
            "article — strip_source_tags removes it before model sees text."
        ),
    },
    "liar": {
        "name": "LIAR (Politifact fact-checks)",
        "loader": "load_liar",
        "format": "tsv",
        "expected_columns": [
            "statement_id", "label", "statement", "subject", "speaker",
            "job", "state", "party", "barely_true_counts",
            "false_counts", "half_true_counts", "mostly_true_counts",
            "pants_on_fire_counts", "context", "venue",
        ],
        "label_field": "label",
        # LIAR's 6-class taxonomy -> 0/1. pants-fire / false are clearly fake.
        # true is clearly real. half-true / mostly-true / barely-true get the
        # needs_review flag and are mapped to the *closest* class:
        #   mostly-true -> 0 (mostly accurate)
        #   half-true   -> 0 (signal: borderline real)
        #   barely-true -> 1 (mostly false)
        # Callers can drop needs_review rows for binary training or keep them
        # for a 3-way verdict model (REAL / NEEDS_REVIEW / FAKE).
        "label_map": {
            "pants-fire": 1,
            "false": 1,
            "barely-true": 1,
            "half-true": 0,
            "mostly-true": 0,
            "true": 0,
        },
        "needs_review_labels": ["half-true", "mostly-true", "barely-true"],
        "needs_review": True,
        "default_path": "data/raw/liar/train.tsv",
        "notes": (
            "UCSB's Politifact-based 6-class taxonomy. `statement` becomes "
            "title; `subject` + `speaker` + `context` become the body."
        ),
    },
    "fakenewsnet": {
        "name": "FakeNewsNet (BuzzFeed + PolitiFact)",
        "loader": "load_fakenewsnet",
        "format": "csv",  # pre-flattened from JSON; we accept CSV
        "expected_columns": ["id", "title", "text", "label", "source", "date"],
        "label_field": "label",
        "label_map": {"real": 0, "fake": 1, 0: 0, 1: 1},
        "needs_review": False,
        "default_path": "data/raw/fakenewsnet.csv",
        "notes": (
            "FakeNewsNet ships as JSON files per article. Most downstream "
            "pipelines pre-flatten to CSV with columns [id, title, text, "
            "label, source, date] (label in {real, fake}). We accept that "
            "CSV directly."
        ),
    },
    "welfake": {
        "name": "WELFake (Kaggle, 72k articles)",
        "loader": "load_welfake",
        "format": "csv",
        "expected_columns": ["id", "title", "text", "label"],
        "label_field": "label",  # already 0/1, 1 = fake
        "label_map": {0: 0, 1: 1},
        "needs_review": False,
        "default_path": "data/raw/welfake.csv",
        "notes": (
            "Trained-on amalgam of Kaggle Fake/Real, BuzzFeed, PolitiFact, "
            "and McIntosh. 0 = real, 1 = fake. Great generalization benchmark "
            "because its source mix is hidden from the model."
        ),
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the absolute path to the fake-news-jev repo root."""
    return Path(__file__).resolve().parents[2]


def default_dataset_path(name: str) -> Path:
    """Return the default on-disk path for dataset `name`.

    Resolves relative to the repo root (``/home/z/Desktop/fake-news-jev`` on
    this machine). The path may or may not exist; callers should check.
    """
    spec = DATASETS.get(name)
    if spec is None:
        raise KeyError(f"Unknown dataset name: {name!r}")
    p = Path(spec["default_path"])
    if not p.is_absolute():
        p = _repo_root() / p
    return p


def _empty_canonical(n: int = 0) -> pd.DataFrame:
    """Return an empty DataFrame with the canonical column order."""
    return pd.DataFrame({c: pd.Series(dtype="object") for c in CANONICAL_COLUMNS})


def _to_int_label(value: Any) -> int:
    """Coerce a label value to int {0, 1}."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and not pd.isna(value):
        return int(value)
    s = str(value).strip().lower()
    truthy = {"1", "true", "fake", "yes", "y", "t", "pants-fire", "false",
              "barely-true"}
    falsy = {"0", "real", "no", "n", "f", "true", "mostly-true", "half-true"}
    # Note: LIAR's "true" is ambiguous w.r.t. boolean True; we always treat
    # the raw string "true" as REAL (=0) per the LIAR label_map. "false" -> 1.
    if s in {"1", "fake", "yes", "y", "t", "pants-fire", "false", "barely-true"}:
        return 1
    if s in {"0", "real", "no", "n", "f", "true", "mostly-true", "half-true"}:
        return 0
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return 0


def _coerce_str(value: Any) -> str:
    """Coerce a value to str; NaN/None -> empty string."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _parse_date(value: Any) -> Optional[pd.Timestamp]:
    """Parse a date value into a pandas Timestamp, or None if unparseable."""
    if value is None:
        return None
    s = _coerce_str(value).strip()
    if not s:
        return None
    try:
        ts = pd.to_datetime(s, errors="coerce", utc=False)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

def load_dataset(name: str, path: Optional[str | os.PathLike] = None) -> pd.DataFrame:
    """Load and normalize any of the four datasets to the canonical schema.

    Args:
        name: One of "fake_real", "liar", "fakenewsnet", "welfake".
        path: Path to the dataset. If None, falls back to the dataset's
              ``default_path`` under the repo root.

    Returns:
        A pandas DataFrame with columns in CANONICAL_COLUMNS order:
        ``[id, title, text, label, source, date, needs_review]``.
        ``label`` is int {0, 1} (1 = fake). ``needs_review`` is bool.
        Rows with no text (empty after coercion) are dropped.

    Raises:
        KeyError: unknown dataset name.
        FileNotFoundError: path does not exist.
    """
    spec = DATASETS.get(name)
    if spec is None:
        raise KeyError(
            f"Unknown dataset name {name!r}; expected one of "
            f"{sorted(DATASETS)}"
        )
    loader_name = spec["loader"]
    loader = {
        "load_fake_real": load_fake_real,
        "load_liar": load_liar,
        "load_fakenewsnet": load_fakenewsnet,
        "load_welfake": load_welfake,
    }[loader_name]
    resolved_path: Optional[Path]
    if path is None:
        resolved_path = None
    else:
        resolved_path = Path(path)
    _log.info(f"loading dataset={name} path={resolved_path or '<default>'}")
    return loader(resolved_path) if resolved_path is not None else loader(None)


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_fake_real(path: Optional[str | os.PathLike] = None) -> pd.DataFrame:
    """Load the Kaggle Fake/Real News dataset.

    Accepts either:

    * A directory containing ``Fake.csv`` and ``True.csv`` (the original
      Kaggle distribution). ``Fake.csv`` rows get ``label=1`` and source
      set to "fake-kaggle"; ``True.csv`` rows get ``label=0`` and source
      "reuters" (the dataset's wire tag).
    * A single CSV with columns including ``title``, ``text``, ``subject``,
      ``date`` and either ``label`` (0/1) or ``class`` ("fake"/"real").

    Args:
        path: Directory or CSV path. Defaults to repo ``data/raw/fake_real``.

    Returns:
        DataFrame in canonical schema.
    """
    p = Path(path) if path is not None else default_dataset_path("fake_real")
    if not p.exists():
        raise FileNotFoundError(
            f"fake_real dataset not found at {p}; run scripts/download_data.py"
        )
    frames: list[pd.DataFrame] = []
    if p.is_dir():
        for fname, label, source in [
            ("Fake.csv", 1, "fake-kaggle"),
            ("True.csv", 0, "reuters"),
        ]:
            fpath = p / fname
            if not fpath.exists():
                continue
            df = pd.read_csv(fpath)
            df["_auto_label"] = label
            df["_auto_source"] = source
            frames.append(df)
        if not frames:
            raise FileNotFoundError(
                f"No Fake.csv or True.csv found in directory {p}"
            )
        raw = pd.concat(frames, ignore_index=True, sort=False)
    else:
        raw = pd.read_csv(p)
    # Title/text normalization
    title_col = _first_present(raw, ["title", "headline", "Title"])
    text_col = _first_present(raw, ["text", "content", "Text"])
    date_col = _first_present(raw, ["date", "Date"])
    subject_col = _first_present(raw, ["subject", "Subject"])
    # Label: prefer explicit column, fall back to auto_label.
    if "label" in raw.columns:
        label_series = raw["label"].apply(_to_int_label)
    elif "class" in raw.columns:
        label_series = raw["class"].apply(_to_int_label)
    elif "_auto_label" in raw.columns:
        label_series = raw["_auto_label"].astype(int)
    else:
        raise ValueError("fake_real: no label/class column and not a 2-file dir")
    source_series = (
        raw["_auto_source"]
        if "_auto_source" in raw.columns
        else raw.get("source", pd.Series([""] * len(raw)))
    )
    # The "subject" column on Kaggle Fake/Real is a topic tag
    # (e.g. "politicsNews", "left-news") — strong dataset-of-origin signal,
    # but useful as body prefix for topic models. We append it to text so
    # the body featurizer sees it without leaking it as a separate column.
    body = (raw[text_col].fillna("").astype(str)).copy()
    if subject_col is not None:
        subj = raw[subject_col].fillna("").astype(str)
        body = body.where(subj.str.len() == 0, subj + " . " + body)
    out = pd.DataFrame({
        "id":           raw.get("id", pd.Series(np.arange(len(raw)))).astype(str),
        "title":        raw[title_col].fillna("").astype(str) if title_col else "",
        "text":         body,
        "label":        label_series.astype(int).values,
        "source":       source_series.fillna("").astype(str).values,
        "date":         raw[date_col].fillna("").astype(str).values if date_col else "",
        "needs_review": False,
    })
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    _log.info(f"fake_real loaded: {len(out)} rows ({int(out['label'].sum())} fake)")
    return out[CANONICAL_COLUMNS]


def load_liar(path: Optional[str | os.PathLike] = None) -> pd.DataFrame:
    """Load the LIAR TSV (Politifact fact-checks).

    LIAR ships as a TSV with 14 columns; the most important are:
    ``statement_id, label, statement, subject, speaker, job, state, party,
    context, venue``. The ``label`` column is one of six strings:
    ``pants-fire, false, barely-true, half-true, mostly-true, true``.

    Mapping (per the task spec):
    * ``pants-fire`` -> 1 (fake)
    * ``false``     -> 1 (fake)
    * ``barely-true``-> 1 (fake, needs_review)
    * ``half-true``  -> 0 (real-ish, needs_review)
    * ``mostly-true``-> 0 (real-ish, needs_review)
    * ``true``       -> 0 (real)

    Title becomes the ``statement`` (the claim). Body is composed from
    ``subject`` + ``speaker`` + ``job`` + ``state`` + ``party`` +
    ``context`` + ``venue`` — LIAR doesn't ship article bodies, so the
    metadata is the article.

    Args:
        path: TSV path. Defaults to ``data/raw/liar/train.tsv``.

    Returns:
        DataFrame in canonical schema; ``needs_review`` True for ambiguous
        LIAR labels.
    """
    p = Path(path) if path is not None else default_dataset_path("liar")
    if not p.exists():
        raise FileNotFoundError(
            f"liar dataset not found at {p}; run scripts/download_data.py"
        )
    raw = pd.read_csv(p, sep="\t", header=None, dtype=str, keep_default_na=False,
                     on_bad_lines="skip", engine="python")
    # LIAR TSV has no header — assign canonical column names.
    expected = DATASETS["liar"]["expected_columns"]
    cols = expected + [f"_extra_{i}" for i in range(len(raw.columns) - len(expected))]
    raw.columns = cols[: len(raw.columns)]
    label_map = DATASETS["liar"]["label_map"]
    needs_review_labels = set(DATASETS["liar"]["needs_review_labels"])

    def _map_row(label_value: str) -> tuple[int, bool]:
        s = str(label_value).strip().lower()
        y = label_map.get(s)
        if y is None:
            return 0, True  # unknown -> default to real + needs_review
        return y, (s in needs_review_labels)

    mapped = raw["label"].apply(_map_row)
    labels = mapped.apply(lambda t: t[0]).astype(int)
    needs_review = mapped.apply(lambda t: t[1]).astype(bool)

    # Body composition — LIAR has no article body. Concatenate metadata.
    def _col(name: str) -> pd.Series:
        return raw[name].fillna("").astype(str) if name in raw.columns else pd.Series([""] * len(raw))

    body_parts = pd.Series([""] * len(raw))
    for name in ["subject", "speaker", "job", "state", "party", "context", "venue"]:
        v = _col(name)
        body_parts = body_parts.where(v.str.strip().eq(""), body_parts + v + " . ")
    out = pd.DataFrame({
        "id":           raw["statement_id"].astype(str) if "statement_id" in raw else pd.Series(np.arange(len(raw))).astype(str),
        "title":        _col("statement"),
        "text":         body_parts,
        "label":        labels.values,
        "source":       _col("speaker"),
        "date":         "",  # LIAR TSV doesn't carry per-statement dates
        "needs_review": needs_review.values,
    })
    out = out[out["title"].str.len() > 0].reset_index(drop=True)
    _log.info(
        f"liar loaded: {len(out)} rows ({int(labels.sum())} fake, "
        f"{int(needs_review.sum())} needs_review)"
    )
    return out[CANONICAL_COLUMNS]


def load_fakenewsnet(path: Optional[str | os.PathLike] = None) -> pd.DataFrame:
    """Load the FakeNewsNet dataset (pre-flattened CSV form).

    FakeNewsNet is distributed as per-article JSON files in the
    `Kaggle FakeNewsNet` repository. Most working pipelines flatten it to
    a CSV with columns ``[id, title, text, label, source, date]`` where
    ``label`` is in ``{real, fake}`` or ``{0, 1}``. We accept that CSV
    directly. If the path points to a directory, we look for any
    ``*.csv`` file (the first one alphabetically) and load it.

    Args:
        path: CSV path or directory. Defaults to ``data/raw/fakenewsnet.csv``.

    Returns:
        DataFrame in canonical schema; ``needs_review`` is always False.
    """
    p = Path(path) if path is not None else default_dataset_path("fakenewsnet")
    if not p.exists():
        raise FileNotFoundError(
            f"fakenewsnet dataset not found at {p}; run scripts/download_data.py"
        )
    if p.is_dir():
        csvs = sorted(p.glob("*.csv"))
        if not csvs:
            raise FileNotFoundError(f"No CSV files in directory {p}")
        p = csvs[0]
    raw = pd.read_csv(p, dtype=str, keep_default_na=False)
    label_map = DATASETS["fakenewsnet"]["label_map"]
    def _to_label(v: str) -> int:
        s = str(v).strip().lower()
        if s in label_map:
            return int(label_map[s])
        return _to_int_label(s)
    title_col = _first_present(raw, ["title", "headline", "Title"]) or "title"
    text_col = _first_present(raw, ["text", "content", "body"]) or "text"
    source_col = _first_present(raw, ["source", "source_url", "outlet"])
    date_col = _first_present(raw, ["date", "publish_date", "timestamp"])
    id_col = _first_present(raw, ["id", "_id", "article_id"]) or "id"
    out = pd.DataFrame({
        "id":           raw[id_col].astype(str) if id_col in raw else pd.Series(np.arange(len(raw))).astype(str),
        "title":        raw[title_col].astype(str) if title_col in raw else "",
        "text":         raw[text_col].astype(str) if text_col in raw else "",
        "label":        raw["label"].apply(_to_label).astype(int).values if "label" in raw else np.zeros(len(raw), dtype=int),
        "source":       raw[source_col].astype(str).values if source_col and source_col in raw else "",
        "date":         raw[date_col].astype(str).values if date_col and date_col in raw else "",
        "needs_review": False,
    })
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    _log.info(f"fakenewsnet loaded: {len(out)} rows ({int(out['label'].sum())} fake)")
    return out[CANONICAL_COLUMNS]


def load_welfake(path: Optional[str | os.PathLike] = None) -> pd.DataFrame:
    """Load the WELFake dataset (Kaggle, 72k articles, 0/1 labels).

    WELFake ships as a CSV with columns ``[id, title, text, label]`` where
    ``label`` is already 0 (real) / 1 (fake). We just coerce types and
    add the missing canonical columns (``source``, ``date``, ``needs_review``).

    Args:
        path: CSV path. Defaults to ``data/raw/welfake.csv``.

    Returns:
        DataFrame in canonical schema.
    """
    p = Path(path) if path is not None else default_dataset_path("welfake")
    if not p.exists():
        raise FileNotFoundError(
            f"welfake dataset not found at {p}; run scripts/download_data.py"
        )
    raw = pd.read_csv(p, dtype={"id": str, "title": str, "text": str, "label": int},
                     keep_default_na=False)
    out = pd.DataFrame({
        "id":           raw["id"].astype(str) if "id" in raw else pd.Series(np.arange(len(raw))).astype(str),
        "title":        raw.get("title", "").astype(str),
        "text":         raw.get("text", "").astype(str),
        "label":        raw["label"].apply(_to_int_label).astype(int).values,
        "source":       "welfake",  # WELFake obscures provenance — uniform tag
        "date":         "",          # WELFake has no date column
        "needs_review": False,
    })
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    _log.info(f"welfake loaded: {len(out)} rows ({int(out['label'].sum())} fake)")
    return out[CANONICAL_COLUMNS]


def _first_present(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name from `candidates` present in `df`."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Articles <-> DataFrame bridge
# ---------------------------------------------------------------------------

def to_articles(df: pd.DataFrame) -> list[Article]:
    """Convert rows of a canonical-schema DataFrame to Article objects.

    Each row is run through `clean_article` (strip_source_tags +
    normalize_text) so downstream stages see clean text. The label column is
    NOT carried on the Article (per the shared-types contract: Article only
    has headline/body/source/url/date); use `extract_labels` to get the
    matching y vector.

    Args:
        df: Canonical-schema DataFrame (from `load_dataset`).

    Returns:
        List of Article objects, one per row, in df order.
    """
    if df is None or len(df) == 0:
        return []
    out: list[Article] = []
    for _, row in df.iterrows():
        art = Article(
            headline=_coerce_str(row.get("title", "")),
            body=_coerce_str(row.get("text", "")),
            source=_coerce_str(row.get("source", "")),
            url="",  # canonical schema doesn't carry url separately
            date=_coerce_str(row.get("date", "")),
        )
        out.append(clean_article(art))
    return out


def extract_labels(df: pd.DataFrame) -> np.ndarray:
    """Return the per-row label vector (1 = fake, 0 = real) as int8 ndarray.

    Convenience helper: Article objects don't carry labels (per the
    shared-types contract), so the meta-classifier stage needs a parallel
    y vector. This pulls the canonical ``label`` column out and casts to
    int8 for memory efficiency.
    """
    if df is None or len(df) == 0 or "label" not in df.columns:
        return np.zeros(0, dtype=np.int8)
    return df["label"].astype(int).to_numpy(dtype=np.int8)


# ---------------------------------------------------------------------------
# Splits (leakage-proof)
# ---------------------------------------------------------------------------

def time_based_split(
    df: pd.DataFrame,
    test_frac: float = 0.2,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split — no future leakage.

    Sorts rows by parsed date ascending; the last ``test_frac`` rows form
    the test set, the rest the train set. This is the canonical anti-leakage
    split for any time-indexed dataset: the model is trained only on the
    past and evaluated only on the future.

    Rows whose ``date`` value can't be parsed to a real timestamp are
    dropped with a logged warning (they'd otherwise collapse to ``NaT`` and
    sort inconsistently across pandas versions).

    Args:
        df:         Canonical-schema DataFrame.
        test_frac:  Fraction in (0, 1) of the chronological tail to hold
                    out for testing. Defaults to 0.2.
        date_column: Column to parse for the timestamp. Defaults to "date".

    Returns:
        (train_df, test_df) — both canonical-schema, both reset_index'd.
        train_df comes before test_df in time.

    Raises:
        ValueError: test_frac not in (0, 1) or no rows have a parseable date.
    """
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac!r}")
    if df is None or len(df) == 0:
        raise ValueError("time_based_split: empty DataFrame")
    if date_column not in df.columns:
        raise ValueError(f"time_based_split: column {date_column!r} not in df")

    ts = df[date_column].apply(_parse_date)
    n_dropped = int(ts.isna().sum())
    if n_dropped:
        _log.warning(
            f"time_based_split: dropping {n_dropped}/{len(df)} rows with "
            f"unparseable dates"
        )
    mask = ts.notna()
    if not mask.any():
        raise ValueError(
            "time_based_split: no rows have a parseable date — use a "
            "random split or supply a different date column"
        )
    df_dated = df.loc[mask].copy()
    df_dated["_ts"] = ts.loc[mask]
    df_dated = df_dated.sort_values("_ts", kind="mergesort").reset_index(drop=True)
    df_dated = df_dated.drop(columns=["_ts"])
    n_test = max(1, int(len(df_dated) * test_frac))
    n_train = len(df_dated) - n_test
    train_df = df_dated.iloc[:n_train].reset_index(drop=True)
    test_df = df_dated.iloc[n_train:].reset_index(drop=True)
    _log.info(
        f"time_based_split: train={len(train_df)} ({train_df['date'].iloc[0] if len(train_df) else '∅'}"
        f" → {train_df['date'].iloc[-1] if len(train_df) else '∅'}); "
        f"test={len(test_df)} ({test_df['date'].iloc[0] if len(test_df) else '∅'}"
        f" → {test_df['date'].iloc[-1] if len(test_df) else '∅'})"
    )
    return train_df[CANONICAL_COLUMNS], test_df[CANONICAL_COLUMNS]


def cross_dataset_split(
    train_name: str,
    test_name: str,
    paths: Optional[dict[str, str | os.PathLike]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on one dataset, test on another — the real generalization test.

    This is the most stringent split available: the train and test
    distributions differ in source, topic mix, era, and stylistic
    conventions. A model that scores well on a cross-dataset split has
    learned something general about fake vs. real news; a model that
    scores well only on a random-within-dataset split may have just
    memorized source tags or topic vocabulary.

    Args:
        train_name: Dataset key for the train set ("fake_real" | "liar" |
                    "fakenewsnet" | "welfake").
        test_name:  Dataset key for the test set.
        paths:      Optional mapping ``{name: path}`` to override each
                    dataset's default path. Missing keys fall back to the
                    default.

    Returns:
        (train_df, test_df) — both canonical-schema, ready for
        ``to_articles`` / ``build_feature_matrix``.
    """
    paths = paths or {}
    train_path = paths.get(train_name) or default_dataset_path(train_name)
    test_path = paths.get(test_name) or default_dataset_path(test_name)
    train_df = load_dataset(train_name, train_path)
    test_df = load_dataset(test_name, test_path)
    _log.info(
        f"cross_dataset_split: train={train_name} ({len(train_df)} rows) "
        f"-> test={test_name} ({len(test_df)} rows)"
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Synthetic dataset generator (fully deterministic; seeded)
# ---------------------------------------------------------------------------

# Topics that mix cleanly across real/fake framings. Each topic has a
# tuple of (subject, named_outlet, named_official, location_prefix).
_TOPICS = [
    ("politics", "Reuters", "the White House press secretary", "WASHINGTON"),
    ("politics", "AP", "a senior State Department official", "WASHINGTON"),
    ("health", "BBC", "an HHS spokesperson", "GENEVA"),
    ("health", "Reuters", "a CDC epidemiologist", "ATLANTA"),
    ("finance", "Bloomberg", "a Treasury Department official", "NEW YORK"),
    ("finance", "Reuters", "an SEC spokesperson", "WASHINGTON"),
    ("science", "AP", "a NASA scientist", "CAPE CANAVERAL"),
    ("science", "Reuters", "an IPCC climate scientist", "GENEVA"),
    ("world", "Reuters", "a foreign ministry official", "BRUSSELS"),
    ("world", "AP", "a UN humanitarian coordinator", "UNITED NATIONS"),
]

# Real-article templates — wire-style, low-sensational, named sourcing.
_REAL_HEADLINE_TEMPLATES = [
    "{official} confirms {number} {nountopic} in {location}",
    "{location} reports {number} {nountopic}, {official} says",
    "{outlet}: {official} announces {action} on {topic}",
    "{location} - {outlet} - {official} reports {number} {nountopic}",
    "{official}: {action} on {topic} expected by {month}",
]
_REAL_BODY_TEMPLATES = [
    (
        "{location} (Reuters) - {official} on {weekday} confirmed that "
        "{number} {nountopic} were reported in {location} during the "
        "previous {days}-day period, according to data released by the "
        "{agency}. The figures represent a {percent}% change from the "
        "prior period. \"{quote_real},\" {official} said. The announcement "
        "comes amid ongoing efforts to {action}."
    ),
    (
        "{location} - {outlet} - {official} said {weekday} that {action} "
        "on {topic} would proceed as scheduled, citing {percent}% "
        "support from stakeholders. \"{quote_real},\" they said in a "
        "statement. The {agency} is expected to publish a full report "
        "by {month}."
    ),
    (
        "{location} (AP) - A {agency} report dated {month} {day} found "
        "that {number} {nountopic} had been recorded, a figure "
        "{official} described as \"{quote_real}.\" The data was "
        "independently reviewed by {outlet}."
    ),
]
_REAL_QUOTES = [
    "we are taking this very seriously",
    "the data speaks for itself",
    "our priority remains the public interest",
    "we are committed to transparency",
    "this is consistent with our prior guidance",
]

# Fake-article templates — ALL-CAPS, clickbait, conspiracy, anonymous sources.
_FAKE_HEADLINE_TEMPLATES = [
    "BREAKING: {outlet_Enemies} HIDING THE TRUTH ABOUT {noun_caps}!!!",
    "YOU WON'T BELIEVE WHAT {official_caps} JUST ADMITTED ABOUT {noun_caps}!",
    "EXPOSED: {outlet_Enemies} DON'T WANT YOU TO KNOW THIS ONE TRICK ABOUT {noun_caps}",
    "SHOCKING: INSIDERS REVEAL THE {noun_caps} COVER-UP!!!",
    "URGENT: {official_caps} CAUGHT LYING ABOUT {noun_caps}!!",
    "THE TRUTH ABOUT {noun_caps} THEY'RE HIDING FROM YOU!!!",
]
_FAKE_BODY_TEMPLATES = [
    (
        "According to insiders who spoke on condition of anonymity, "
        "{outlet_Enemies} have been deliberately suppressing the real "
        "story about {nountopic}. They don't want you to know that "
        "{quote_fake}!!! Wake up, people — this is the {noun_caps} "
        "cover-up the mainstream media won't report. Share this before "
        "they take it down!!!"
    ),
    (
        "BOMBSHELL!!! Sources close to {official_caps} are now revealing "
        "what {outlet_Enemies} refused to cover: the {noun_caps} "
        "conspiracy is REAL. According to one whistleblower, "
        "\"{quote_fake},\" and the evidence is undeniable. Why is no "
        "one talking about this??? Spread the word!!!"
    ),
    (
        "BREAKING NEWS!!! Anonymous sources are now confirming what many "
        "have long suspected: {outlet_Enemies} have been lying to you "
        "about {nountopic}. \"They don't want you to know the truth,\" "
        "one insider revealed. This changes EVERYTHING!!! Don't let "
        "them bury this story — share it everywhere!!!"
    ),
]
_FAKE_QUOTES = [
    "they're hiding the truth about what really happened",
    "this goes all the way to the top",
    "the numbers don't add up and they know it",
    "we were told to keep quiet but enough is enough",
    "the real story is being suppressed",
]
_NOUN_BY_TOPIC = {
    "politics":   ("votes", "VOTES"),
    "health":     ("cases", "CASES"),
    "finance":    ("transactions", "TRANSACTIONS"),
    "science":    ("measurements", "MEASUREMENTS"),
    "world":      ("incidents", "INCIDENTS"),
}
_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
_MONTHS = list(calendar.month_name)[1:]  # ['January', ..., 'December']
_OUTLET_ENEMIES = ["THE MAINSTREAM MEDIA", "BIG TECH", "THE DEEP STATE",
                   "THE ESTABLISHMENT", "THE FACT-CHECKERS"]
_AGENCIES = ["CDC", "FDA", "SEC", "FTC", "EPA", "NASA", "UN", "WHO"]
_ACTIONS = ["tighten oversight", "release new guidance",
            "announce reforms", "publish the data",
            "convene a working group", "expand the program"]


def make_synthetic_dataset(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a deterministic synthetic fake/real news dataset.

    Used by tests, demos, and the bootstrap path before real datasets are
    downloaded. Real articles mimic AP/Reuters wire copy: dateline prefix,
    "(Reuters)"/"(AP)" source tags, named officials, verifiable numbers,
    professional quotes, low sensationalism. Fake articles mimic tabloid
    clickbait: ALL-CAPS headlines, exclamation chains, "you won't believe"
    phrasing, anonymous sources ("according to insiders"), conspiracy
    framing ("they don't want you to know", "wake up"), high emotional
    intensity.

    The mix of topics (politics/health/finance/science/world) ensures the
    metadata featurizer sees variety and the synthetic labels are
    learnable from the text alone (the JEV signal extractor should also
    produce a high-AUC ordering on this set, since the real/fake templates
    are designed to maximize separation on every JEV question).

    Args:
        n:    Number of articles to generate. Half are fake, half real.
              The last item is rounded down to a multiple of 2.
        seed: Integer seed for the Python `random.Random` instance used to
              fill in the templates. Two runs with the same (n, seed) yield
              byte-identical output.

    Returns:
        DataFrame in canonical schema with columns
        ``[id, title, text, label, source, date, needs_review]``.
        Dates span Jan 1 2023 -> Dec 31 2023, evenly spaced.
    """
    if n < 1:
        return _empty_canonical()
    rng = random.Random(seed)  # local, deterministic, no global state
    n = max(2, n - (n % 2))  # even number of rows for balanced split
    half = n // 2

    rows: list[dict[str, Any]] = []
    # Evenly-spaced dates over 2023 so time_based_split is meaningful.
    start = datetime(2023, 1, 1, tzinfo=timezone.utc)
    span_seconds = (datetime(2024, 1, 1, tzinfo=timezone.utc) - start).total_seconds()
    def _date_at(i: int) -> str:
        ts = start.timestamp() + (i * span_seconds / max(n - 1, 1))
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    for i in range(half):
        topic, outlet, official, location = rng.choice(_TOPICS)
        weekday = rng.choice(_WEEKDAYS)
        month = rng.choice(_MONTHS)
        day = rng.randint(1, 28)
        number = rng.randint(10, 9999)
        percent = rng.randint(1, 99)
        days = rng.randint(3, 90)
        agency = rng.choice(_AGENCIES)
        action = rng.choice(_ACTIONS)
        noun_topic, noun_caps = _NOUN_BY_TOPIC[topic]
        quote = rng.choice(_REAL_QUOTES)
        headline_tpl = rng.choice(_REAL_HEADLINE_TEMPLATES)
        body_tpl = rng.choice(_REAL_BODY_TEMPLATES)
        headline = headline_tpl.format(
            official=official, number=number, nountopic=noun_topic,
            location=location, outlet=outlet, action=action, topic=topic,
            month=month,
        )
        body = body_tpl.format(
            location=location, official=official, weekday=weekday,
            number=number, nountopic=noun_topic, days=days, agency=agency,
            percent=percent, action=action, topic=topic, outlet=outlet,
            quote_real=quote, month=month, day=day,
        )
        rows.append({
            "id":           f"syn-real-{i:05d}",
            "title":        headline,
            "text":         body,
            "label":        0,
            "source":       outlet.lower(),
            "date":         _date_at(i),
            "needs_review": False,
        })

    for j in range(half):
        topic, outlet, official, location = rng.choice(_TOPICS)
        noun_topic, noun_caps = _NOUN_BY_TOPIC[topic]
        quote = rng.choice(_FAKE_QUOTES)
        headline_tpl = rng.choice(_FAKE_HEADLINE_TEMPLATES)
        body_tpl = rng.choice(_FAKE_BODY_TEMPLATES)
        headline = headline_tpl.format(
            outlet_Enemies=rng.choice(_OUTLET_ENEMIES),
            noun_caps=noun_caps,
            official_caps=official.upper(),
        )
        body = body_tpl.format(
            outlet_Enemies=rng.choice(_OUTLET_ENEMIES),
            nountopic=noun_topic, noun_caps=noun_caps,
            official_caps=official.upper(), quote_fake=quote,
        )
        rows.append({
            "id":           f"syn-fake-{j:05d}",
            "title":        headline,
            "text":         body,
            "label":        1,
            "source":       "anonymous-blog",
            "date":         _date_at(half + j),
            "needs_review": False,
        })

    # IMPORTANT: shuffle with the same seeded rng so the returned DataFrame
    # is not trivially partitionable by row index (otherwise a leak-prone
    # `iloc[:n//2]` test would see all-real then all-fake). The shuffle is
    # deterministic given the seed.
    rng.shuffle(rows)
    df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS).reset_index(drop=True)
    return df
