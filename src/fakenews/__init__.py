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
"""fake-news-jev — Jev-typed-decision fake-news detection core.

Public surface (the contract):

  from fakenews import (
      AnswerType, Question, TypedAnswer,
      Article, Signal, Evidence, Verdict,
      JevClient, AsyncJevClient, JevSimulated, make_jev_client,
      SignalExtractor,
      load_config, load_questions, get_logger,
      content_hash, set_seed, ensure_dir, truncate,
  )

Stage-1 wiring (canonical):

    cfg = load_config()
    qs  = load_questions()
    set_seed(cfg["project"]["seed"])
    client   = make_jev_client(cfg)               # simulated | typesafe | cloudflare | openrouter
    extractor = SignalExtractor(client, qs)
    signals = await extractor.extract(article)   # ONE Jev call per article
"""

from __future__ import annotations

__version__ = "0.1.0"

# Shared types (the CONTRACT)
from .types import (
    AnswerType,
    Article,
    Evidence,
    Question,
    Signal,
    TypedAnswer,
    Verdict,
)

# Configuration & utilities
from .config import (
    apply_config_seeds,
    get_logger,
    load_config,
    load_questions,
)
from .utils import content_hash, ensure_dir, set_seed, truncate

# Jev clients
from .jev_client import AsyncJevClient, JevClient, JevError, make_jev_client
from .jev_simulated import JevSimulated

# Stage 1
from .signal_extractor import SignalExtractor

__all__ = [
    "__version__",
    # types
    "AnswerType",
    "Question",
    "TypedAnswer",
    "Article",
    "Signal",
    "Evidence",
    "Verdict",
    # config & utils
    "load_config",
    "load_questions",
    "get_logger",
    "apply_config_seeds",
    "content_hash",
    "set_seed",
    "ensure_dir",
    "truncate",
    # clients
    "JevClient",
    "AsyncJevClient",
    "JevSimulated",
    "JevError",
    "make_jev_client",
    # stage 1
    "SignalExtractor",
]
