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
Jev client — the typed-decision bridge.

`JevClient` is the abstract contract every Stage-1 caller holds:

    answers: dict[str, TypedAnswer] = await client.ask(state, questions)

There are two concrete implementations:

  1. `JevSimulated` (in jev_simulated.py) — a deterministic lexicon-driven
     heuristic backend that ships with this repo. It is the *substitution*
     used when the real Jev endpoint is unavailable. See its module
     docstring for the exact substitution contract.

  2. `AsyncJevClient` (here) — wraps a real Jev HTTP provider with:

       - exponential-backoff retries (stdlib `asyncio`, no third-party dep)
       - per-call timeout via `asyncio.wait_for`
       - token-bucket rate limiter (configurable requests-per-minute)
       - in-memory LRU cache keyed by sha256(state + question names)

It auto-routes the request body shape per `provider`:
  - `typesafe`   : {"state": <str>, "questions": [{"name","type","prompt"}]}
  - `cloudflare` : {"state": <str>, "qs": [{"n","t","p"}]} + X-Jev-Sig header
  - `openrouter` : OpenAI chat-completions wrapper
                   {"model": "...", "messages": [{"role":"system","content": ...},
                                                 {"role":"user","content": ...}],
                    "response_format": {"type": "json_object"}}

  All three return responses normalised to the same `dict[str, TypedAnswer]`
  shape so callers can swap providers without touching downstream code.

Use `make_jev_client(config)` to obtain the right concrete client based on
`config['jev']['provider']`.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Optional

from .types import AnswerType, Question, TypedAnswer
from .utils import content_hash, get_logger

try:  # httpx is optional for the simulated-only path
    import httpx  # type: ignore

    _HAS_HTTPX = True
except Exception:  # pragma: no cover
    _HAS_HTTPX = False


class JevClient(ABC):
    """Abstract typed-decision client.

    Subclasses MUST implement `ask`. Callers should hold a `JevClient`
    reference and never the concrete subclass, so swapping simulated ↔
    real is a config change.
    """

    @abstractmethod
    async def ask(
        self,
        state: str,
        questions: "Optional[list[Question]]" = None,
    ) -> dict[str, TypedAnswer]:
        """Pose `questions` against `state` and return typed, calibrated answers.

        If ``questions`` is None, implementations should default to the
        canonical 19 questions (loaded from config/questions.yaml).

        Args:
            state:     unstructured text the questions are posed against
                       (typically article headline + "\n\n" + body).
            questions: the canonical, versioned question list.

        Returns:
            Mapping {question.name -> TypedAnswer}. Every question in
            `questions` MUST be present in the returned mapping.

        Raises:
            JevError: on transport / auth / parse / timeout failures
                after all retries are exhausted.
        """
        raise NotImplementedError


class JevError(RuntimeError):
    """Raised when the real Jev endpoint is unreachable or returns garbage."""


# ---------------------------------------------------------------------------
# In-memory LRU cache + token-bucket rate limiter (stdlib only)
# ---------------------------------------------------------------------------


class _LRU:
    """Tiny stdlib LRU keyed by string."""

    __slots__ = ("_d", "_max")

    def __init__(self, maxsize: int = 512) -> None:
        self._d: "OrderedDict[str, dict[str, TypedAnswer]]" = OrderedDict()
        self._max = maxsize

    def get(self, key: str) -> Optional[dict[str, TypedAnswer]]:
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key: str, val: dict[str, TypedAnswer]) -> None:
        if key in self._d:
            self._d.move_to_end(key)
        self._d[key] = val
        if len(self._d) > self._max:
            self._d.popitem(last=False)


class _TokenBucket:
    """Async token-bucket rate limiter.

    Tokens accrue at `rate_per_min / 60` per second, capped at
    `rate_per_min`. `acquire()` waits (via asyncio sleep) until at least
    one token is available, then consumes one.
    """

    __slots__ = ("_rate", "_capacity", "_tokens", "_last", "_lock")

    def __init__(self, rate_per_min: int) -> None:
        self._rate = max(1, rate_per_min) / 60.0  # tokens / second
        self._capacity = float(max(1, rate_per_min))
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Block the caller until at least one token is available, then consume it."""
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(
                    self._capacity, self._tokens + elapsed * self._rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                # Need to wait for the deficit to replenish.
                wait = (1.0 - self._tokens) / self._rate
                await asyncio.sleep(wait)


# ---------------------------------------------------------------------------
# AsyncJevClient — real-provider implementation
# ---------------------------------------------------------------------------


class AsyncJevClient(JevClient):
    """Async, resilient HTTP client for the real Jev endpoint.

    Construction is via `make_jev_client(config)` (or direct kwargs).
    The HTTP transport is `httpx.AsyncClient` (optional dep — falls back
    to stdlib `urllib.request`-over-thread-executor if httpx missing).

    Request body shapes per `provider`:

      **typesafe** (Jev native HTTP API):
        POST {endpoint}
        Headers: {Authorization: Bearer <api_key>, Content-Type: application/json}
        Body:    {
          "state":     "<article state>",
          "questions": [{"name": ..., "type": "noul|score|choice",
                         "prompt": ...}, ...]
        }
        200 OK response body:
        {
          "answers": [
            {"name": ..., "type": ..., "value": <float|str>, "probability": <float>},
            ...
          ]
        }

      **cloudflare** (Jev worker binding):
        POST {endpoint}
        Headers: {X-Jev-Sig: <hmac(api_key, body)>, Content-Type: application/json}
        Body:    {"state": ..., "qs": [{"n": ..., "t": ..., "p": ...}, ...]}
        200 OK response body:
        {"answers": [{"n": ..., "t": ..., "v": <float|str>, "p": <float>}, ...]}

      **openrouter** (OpenAI chat-completions wrapper):
        POST https://openrouter.ai/api/v1/chat/completions
        Headers: {Authorization: Bearer <api_key>,
                  Content-Type: application/json,
                  HTTP-Referer: https://fake-news-jev.local,
                  X-Title: fake-news-jev}
        Body:    {
          "model": "jev/jev-v1",   # or any OpenAI-compatible model id
          "messages": [
            {"role": "system", "content":
              "You are a typed-decision oracle. Answer ONLY with a JSON object."},
            {"role": "user", "content":
              "<state>\n\nFor each of these questions answer with a JSON object
              {name,type,value,probability}. Questions: <q1>, <q2>, ..."}
          ],
          "response_format": {"type": "json_object"},
          "temperature": 0
        }
        200 OK response body:
        {"choices":[{"message":{"content":"{\"answers\":[...]}"}}]}

    All three are normalised to `dict[str, TypedAnswer]`.
    """

    def __init__(
        self,
        provider: str,
        endpoint: str,
        api_key: str,
        timeout_s: float = 5.0,
        max_retries: int = 3,
        rate_limit_per_min: int = 60,
        cache_enabled: bool = True,
        cache_size: int = 512,
        model: Optional[str] = None,
    ) -> None:
        if provider not in {"typesafe", "cloudflare", "openrouter"}:
            raise ValueError(f"AsyncJevClient: unknown provider {provider!r}")
        self.provider = provider
        self.endpoint = endpoint or _default_endpoint(provider)
        self.api_key = api_key
        self.timeout_s = timeout_s
        self.max_retries = max(0, max_retries)
        self._bucket = _TokenBucket(rate_limit_per_min)
        self._cache = _LRU(cache_size) if cache_enabled else None
        self.model = model or _default_model(provider)
        self._log = get_logger(__name__, level="INFO", fmt="json")

    # ---------- public API ----------

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
        if questions is None:
            try:
                from fakenews.config import load_questions
                questions = load_questions()
            except Exception:
                questions = []
        cache_key = content_hash(state + "|" + ",".join(q.name for q in questions))
        if self._cache is not None:
            hit = self._cache.get(cache_key)
            if hit is not None:
                self._log.debug("jev_cache_hit", extra={"key": cache_key})
                return hit

        body, headers = self._build_request(state, questions)

        last_exc: Optional[BaseException] = None
        for attempt in range(self.max_retries + 1):
            try:
                await self._bucket.acquire()
                raw = await asyncio.wait_for(
                    self._post(body, headers), timeout=self.timeout_s
                )
                answers = _normalise_response(self.provider, raw, questions)
                if self._cache is not None:
                    self._cache.put(cache_key, answers)
                return answers
            except asyncio.TimeoutError as e:
                last_exc = JevError(f"timeout after {self.timeout_s}s")
                self._log.warning(
                    f"jev_timeout attempt={attempt} provider={self.provider}"
                )
            except JevError as e:
                last_exc = e
                self._log.warning(
                    f"jev_error attempt={attempt} provider={self.provider} err={e}"
                )
            except Exception as e:  # noqa: BLE001
                last_exc = JevError(str(e))
                self._log.warning(
                    f"jev_unexpected attempt={attempt} provider={self.provider} err={e}"
                )
            # exponential backoff: 0.1 * 2^attempt, capped at 5s
            if attempt < self.max_retries:
                await asyncio.sleep(min(5.0, 0.1 * (2 ** attempt)))

        raise JevError(
            f"jev ask failed after {self.max_retries + 1} attempts: {last_exc}"
        )

    # ---------- request builders ----------

    def _build_request(
        self,
        state: str,
        questions: "Optional[list[Question]]" = None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        """Return (json_body, http_headers) for the configured provider."""
        if self.provider == "typesafe":
            body = {
                "state": state,
                "questions": [
                    {"name": q.name, "type": q.type.value, "prompt": q.prompt}
                    for q in questions
                ],
            }
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            return body, headers

        if self.provider == "cloudflare":
            body = {
                "state": state,
                "qs": [
                    {"n": q.name, "t": q.type.value, "p": q.prompt}
                    for q in questions
                ],
            }
            sig = _hmac_b64(self.api_key, json.dumps(body, separators=(",", ":")))
            headers = {
                "X-Jev-Sig": sig,
                "Content-Type": "application/json",
            }
            return body, headers

        # openrouter (OpenAI chat-completions wrapper)
        q_block = "\n".join(
            f"- {q.name} ({q.type.value}): {q.prompt}" for q in questions
        )
        sys_msg = (
            "You are a typed-decision oracle. Read the STATE then answer "
            "EVERY listed QUESTION. Respond with a single JSON object: "
            '{"answers":[{"name":<str>,"type":"noul|score|choice",'
            '"value":<float|str>,"probability":<float 0..1>}]}'
        )
        user_msg = f"STATE:\n{state}\n\nQUESTIONS:\n{q_block}"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://fake-news-jev.local",
            "X-Title": "fake-news-jev",
        }
        return body, headers

    # ---------- transport ----------

    async def _post(
        self, body: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        """POST JSON and return parsed JSON. Uses httpx if available."""
        if _HAS_HTTPX:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    self.endpoint, json=body, headers=headers
                )
                if resp.status_code >= 400:
                    raise JevError(
                        f"HTTP {resp.status_code}: {resp.text[:200]}"
                    )
                return resp.json()
        # Fallback: run urllib in a thread (sync lib under async shell).
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, _urllib_post, self.endpoint, body, headers
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_endpoint(provider: str) -> str:
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1/chat/completions"
    # typesafe / cloudflare endpoints must be supplied by config.
    return ""


def _default_model(provider: str) -> str:
    if provider == "openrouter":
        return "openai/gpt-4o-mini"
    return "jev-v1"


def _hmac_b64(secret: str, msg: str) -> str:
    import base64
    import hashlib
    import hmac

    return base64.b64encode(
        hmac.new(
            secret.encode("utf-8"),
            msg.encode("utf-8"),
            hashlib.sha256,
        ).digest()
    ).decode("ascii")


def _urllib_post(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    import urllib.request
    import urllib.error

    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise JevError(f"HTTP {e.code}: {e.read()[:200]!r}") from e


def _normalise_response(
    provider: str,
    raw: dict[str, Any],
    questions: "Optional[list[Question]]" = None,
) -> dict[str, TypedAnswer]:
    """Parse a provider response into `dict[str, TypedAnswer]`.

    Raises JevError if a required question is missing or a value/type
    doesn't match the declared Question.
    """
    out: dict[str, TypedAnswer] = {}
    answers_raw: list[dict[str, Any]] = []

    if provider == "openrouter":
        try:
            content = raw["choices"][0]["message"]["content"]
            if isinstance(content, str):
                raw = json.loads(content)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise JevError(f"openrouter: unparseable response: {e}") from e

    if isinstance(raw, dict) and isinstance(raw.get("answers"), list):
        answers_raw = raw["answers"]
    else:
        raise JevError(f"{provider}: missing top-level 'answers' list")

    q_by_name = {q.name: q for q in questions}

    for a in answers_raw:
        # Accept both verbose (typesafe) and short (cloudflare) keys.
        name = a.get("name") or a.get("n")
        if not name or name not in q_by_name:
            continue  # tolerate extra answers from the model
        q = q_by_name[name]
        atype_str = a.get("type") or a.get("t") or q.type.value
        try:
            atype = AnswerType(atype_str)
        except ValueError:
            atype = q.type
        value = a.get("value") if "value" in a else a.get("v")
        prob = a.get("probability") if "probability" in a else a.get("p")
        if value is None or prob is None:
            raise JevError(f"{provider}: answer {name!r} missing value/probability")
        try:
            prob_f = float(prob)
        except (TypeError, ValueError) as e:
            raise JevError(f"{provider}: bad probability for {name!r}: {prob!r}") from e
        # Coerce score/noul values to float; leave choice as str.
        if atype in (AnswerType.NOUL, AnswerType.SCORE):
            try:
                value = float(value)
            except (TypeError, ValueError):
                pass
        else:
            value = str(value)
        out[name] = TypedAnswer(
            name=name, type=atype, value=value, probability=prob_f
        )

    # Ensure every question is present.
    missing = [q.name for q in questions if q.name not in out]
    if missing:
        raise JevError(f"{provider}: missing answers for: {missing}")
    return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_jev_client(config: dict[str, Any]) -> JevClient:
    """Build the right `JevClient` for `config['jev']['provider']`.

    - `simulated`  -> `JevSimulated` (no key/endpoint required).
    - `typesafe` | `cloudflare` | `openrouter` -> `AsyncJevClient`
      with endpoint/key/timeout/retries/rate-limit/cache from config.
      The API key is read from `os.environ[config['jev']['api_key_env']]`
      and MUST be present (else JevError).
    """
    # Import locally to avoid a circular import at module load time.
    from .jev_simulated import JevSimulated

    jc = config.get("jev", {})
    provider = str(jc.get("provider", "simulated")).lower()

    if provider == "simulated":
        return JevSimulated()

    if provider in {"typesafe", "cloudflare", "openrouter"}:
        env_var = jc.get("api_key_env", "JEV_API_KEY")
        api_key = os.environ.get(env_var, "").strip()
        if not api_key:
            raise JevError(
                f"provider={provider!r} requires ${env_var} to be set in the environment"
            )
        return AsyncJevClient(
            provider=provider,
            endpoint=str(jc.get("endpoint", "")),
            api_key=api_key,
            timeout_s=float(jc.get("timeout_s", 5)),
            max_retries=int(jc.get("max_retries", 3)),
            rate_limit_per_min=int(jc.get("rate_limit_per_min", 60)),
            cache_enabled=bool(jc.get("cache_enabled", True)),
        )

    raise JevError(f"unknown jev.provider in config: {provider!r}")


__all__ = [
    "JevClient",
    "AsyncJevClient",
    "JevError",
    "make_jev_client",
]
