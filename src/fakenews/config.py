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
"""Configuration loading: YAML config + versioned questions.

- `load_config(path=None)` loads `config/config.yaml` with sane defaults
  if the file is missing or pyyaml is not installed.
- `load_questions(path=None)` parses `config/questions.yaml` into validated
  `Question` dataclasses.
- `get_logger(name)` re-exports `utils.get_logger` for convenience.

Both loaders accept an explicit path; if omitted, they fall back to
`<repo_root>/config/<file>.yaml` resolved relative to this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

from .types import AnswerType, Question
from .utils import get_logger, set_seed

# Repo root = parent of `src/`
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.yaml"
_DEFAULT_QUESTIONS_PATH = _REPO_ROOT / "config" / "questions.yaml"

# Embedded fallback config — used if config.yaml is missing or pyyaml
# is unavailable. Matches config.yaml exactly.
_DEFAULT_CONFIG: dict[str, Any] = {
    "project": {"name": "fake-news-jev", "version": "0.1.0", "seed": 42},
    "jev": {
        "provider": "simulated",
        "api_key_env": "JEV_API_KEY",
        "endpoint": "",
        "timeout_s": 5,
        "max_retries": 3,
        "rate_limit_per_min": 60,
        "cache_enabled": True,
    },
    "decision": {
        "real_max": 0.3,
        "review_min": 0.3,
        "fake_min": 0.8,
        "evidence_check": {"enabled": True, "low": 0.3, "high": 0.8},
    },
    "model": {
        "meta_classifier": "lightgbm",
        "calibration": "isotonic",
        "tfidf_max_features": 5000,
        "ngram_range": [1, 2],
    },
    "logging": {"level": "INFO", "format": "json"},
}


def _import_yaml() -> Optional[Any]:
    """Import pyyaml lazily; return None if unavailable."""
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def load_config(path: Optional[str | os.PathLike] = None) -> dict[str, Any]:
    """Load YAML config, merging over built-in defaults.

    Missing files or unavailable pyyaml are non-fatal: the caller always
    receives a complete dict (the `_DEFAULT_CONFIG` above, possibly
    shallowly overridden by what was readable from disk).

    Args:
        path: optional explicit path; defaults to repo_root/config/config.yaml.

    Returns:
        A dict with keys: project, jev, decision, model, logging.
    """
    cfg_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    cfg = {k: dict(v) if isinstance(v, dict) else v for k, v in _DEFAULT_CONFIG.items()}
    # Deep-ish copy of nested dicts so caller mutations don't poison defaults.
    for k, v in list(cfg.items()):
        if isinstance(v, dict):
            cfg[k] = {kk: (list(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}

    yaml = _import_yaml()
    if yaml is None or not cfg_path.exists():
        # No pyyaml or no file: return defaults verbatim.
        if not cfg_path.exists() and yaml is not None:
            # file missing but pyyaml available — still return defaults silently.
            pass
        return cfg

    with cfg_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    # Shallow-merge top-level sections (loaded values win).
    for k, v in loaded.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            merged = dict(cfg[k])
            merged.update(v)
            cfg[k] = merged
        else:
            cfg[k] = v
    return cfg


def load_questions(path: Optional[str | os.PathLike] = None) -> list[Question]:
    """Parse questions.yaml into validated `Question` dataclasses.

    Validation rules:
    - `version` (top-level) is recorded but per-question `version` is the
      authoritative one used downstream.
    - Each question must have a unique `name`.
    - SCORE questions must have both `min` and `max`.
    - CHOICE questions must have a non-empty `options` list.
    - `type` must be one of AnswerType.

    Args:
        path: optional explicit path; defaults to repo_root/config/questions.yaml.

    Returns:
        List[Question] in file order.

    Raises:
        ValueError: on any validation error (duplicate name, bad type,
            missing min/max/options).
    """
    q_path = Path(path) if path else _DEFAULT_QUESTIONS_PATH
    yaml = _import_yaml()
    if yaml is None or not q_path.exists():
        return _builtin_questions()

    with q_path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}

    questions_raw = loaded.get("questions", []) if isinstance(loaded, dict) else []
    questions: list[Question] = []
    seen_names: set[str] = set()

    for i, q in enumerate(questions_raw):
        if not isinstance(q, dict):
            raise ValueError(f"question #{i} is not a mapping: {q!r}")
        name = q.get("name")
        if not name:
            raise ValueError(f"question #{i} missing `name`")
        if name in seen_names:
            raise ValueError(f"duplicate question name: {name!r}")
        seen_names.add(name)

        type_str = q.get("type")
        try:
            qtype = AnswerType(type_str)
        except ValueError:
            raise ValueError(
                f"question {name!r}: type {type_str!r} not in "
                f"{[t.value for t in AnswerType]}"
            )

        prompt = q.get("prompt", "")
        version = int(q.get("version", 1))
        mn = q.get("min")
        mx = q.get("max")
        opts = q.get("options")

        if qtype is AnswerType.SCORE and (mn is None or mx is None):
            raise ValueError(
                f"SCORE question {name!r} requires both `min` and `max`"
            )
        if qtype is AnswerType.CHOICE and not opts:
            raise ValueError(
                f"CHOICE question {name!r} requires a non-empty `options` list"
            )

        questions.append(
            Question(
                name=name,
                type=qtype,
                prompt=prompt,
                version=version,
                min=float(mn) if mn is not None else None,
                max=float(mx) if mx is not None else None,
                options=list(opts) if opts else None,
            )
        )

    if not questions:
        # File existed but had no questions — fall back to built-ins.
        return _builtin_questions()
    return questions


def _builtin_questions() -> list[Question]:
    """Return the hard-coded 19-question set as a last-resort fallback.

    Kept in sync with `config/questions.yaml`; used only if pyyaml is
    unavailable or the file is missing. The on-disk YAML is the source of
    truth — this is a safety net.
    """
    from .types import AnswerType as AT  # local alias for brevity

    specs = [
        # Each tuple: (name, type, prompt, min, max, options)
        # --- 11 NOUL ---
        ("is_fake_overall", AT.NOUL, "Is this article overall misinformation / fake news?", None, None, None),
        ("is_clickbait", AT.NOUL, "Does the headline use clickbait phrasing to drive engagement?", None, None, None),
        ("has_unverified_claims", AT.NOUL, "Does the article make factual claims not attributable to a verifiable source?", None, None, None),
        ("uses_anonymous_sources", AT.NOUL, "Does the article rely on anonymous or unnamed sources for its central claims?", None, None, None),
        ("contains_conspiracy_framing", AT.NOUL, "Does the article frame events as a hidden conspiracy or cover-up?", None, None, None),
        ("cites_verifiable_sources", AT.NOUL, "Does the article cite identifiable, verifiable sources (named outlets, studies, documents)?", None, None, None),
        ("is_satire", AT.NOUL, "Is the article clearly satirical or parody?", None, None, None),
        ("is_opinion_not_news", AT.NOUL, "Is this primarily an opinion piece rather than reported news?", None, None, None),
        ("has_statistical_claims", AT.NOUL, "Does the article assert numeric statistics without a cited dataset?", None, None, None),
        ("impersonates_authority", AT.NOUL, "Does the article impersonate or fabricate an official authority figure or institution?", None, None, None),
        ("headline_contradicts_body", AT.NOUL, "Does the headline materially contradict the body of the article?", None, None, None),
        # --- 5 SCORE ---
        ("emotional_intensity", AT.SCORE, "Rate the emotional intensity of the language 0-10.", 0.0, 10.0, None),
        ("sensationalism", AT.SCORE, "Rate the sensationalism of the headline+lede 0-10.", 0.0, 10.0, None),
        ("factual_specificity", AT.SCORE, "Rate how specific and verifiable the factual claims are 0-10.", 0.0, 10.0, None),
        ("political_bias", AT.SCORE, "Rate political bias on a -5 (left) to +5 (right) scale.", -5.0, 5.0, None),
        ("writing_professionalism", AT.SCORE, "Rate the writing professionalism 0-10.", 0.0, 10.0, None),
        # --- 3 CHOICE ---
        ("topic", AT.CHOICE, "Classify the primary topic of the article.", None, None,
         ["politics", "health", "finance", "science", "sports", "other"]),
        ("claim_type", AT.CHOICE, "Classify the dominant type of claim made.", None, None,
         ["fact", "opinion", "prediction", "rumor"]),
        ("manipulation_tactic", AT.CHOICE, "Identify the dominant manipulation tactic, if any.", None, None,
         ["none", "fear", "outrage", "false_authority", "cherry_picking"]),
    ]
    out: list[Question] = []
    for name, qtype, prompt, mn, mx, opts in specs:
        out.append(
            Question(
                name=name,
                type=qtype,
                prompt=prompt,
                version=1,
                min=mn,
                max=mx,
                options=list(opts) if opts else None,
            )
        )
    return out


def apply_config_seeds(cfg: dict[str, Any]) -> None:
    """Seed RNGs from `cfg.project.seed` if present."""
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_seed(seed)


__all__ = [
    "load_config",
    "load_questions",
    "get_logger",
    "apply_config_seeds",
]
