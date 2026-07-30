"""Nebius (OpenAI-compatible) chat client with provenance logging.

Thin wrapper over the OpenAI SDK that (a) targets the Nebius Token Factory
endpoint, (b) retries transient failures with backoff, (c) handles
reasoning-class models that return their trace in a non-standard ``reasoning``
field, and (d) returns a fully self-describing ``CallResult`` so every row in
the result store can be reproduced and audited.

Concurrency is intentionally left to the caller (run.py drives a thread pool);
this class is a simple, synchronous, thread-safe unit of work — the OpenAI
client is safe to share across threads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from . import config
from .config import ModelSpec

# Errors worth retrying: transient network / server / rate-limit conditions.
_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


@dataclass
class CallResult:
    """Everything needed to reproduce and audit one model call."""

    model_id: str
    text: str | None
    reasoning: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    temperature: float
    seed: int | None
    attempts: int
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.text is not None


class LLMClient:
    """Reusable client for the configured endpoint."""

    def __init__(
        self,
        *,
        max_retries: int = 4,
        backoff_base: float = 1.5,
        timeout: float = 120.0,
    ) -> None:
        self._client = OpenAI(
            base_url=config.base_url(),
            api_key=config.api_key(),
            timeout=timeout,
            max_retries=0,  # we own the retry loop for full provenance
        )
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def complete(
        self,
        messages: list[dict[str, str]],
        model: ModelSpec,
        *,
        temperature: float = 0.0,
        seed: int | None = 42,
        max_tokens: int | None = None,
    ) -> CallResult:
        """One chat completion, with retries. Never raises for API errors —
        a failed call is returned as a ``CallResult`` with ``error`` set so the
        runner can record it and move on."""
        budget = max_tokens or model.max_tokens
        last_err: str | None = None
        t0 = time.time()

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=model.model_id,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=budget,
                )
            except _RETRYABLE as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.max_retries:
                    time.sleep(self.backoff_base ** attempt)
                    continue
            except Exception as e:  # non-retryable (bad request, auth, etc.)
                return CallResult(
                    model_id=model.model_id, text=None, reasoning=None,
                    finish_reason=None, prompt_tokens=None, completion_tokens=None,
                    latency_s=round(time.time() - t0, 3), temperature=temperature,
                    seed=seed, attempts=attempt, error=f"{type(e).__name__}: {e}",
                )
            else:
                msg = resp.choices[0].message
                reasoning = (msg.model_extra or {}).get("reasoning")
                usage = resp.usage
                return CallResult(
                    model_id=model.model_id,
                    text=msg.content,
                    reasoning=reasoning,
                    finish_reason=resp.choices[0].finish_reason,
                    prompt_tokens=getattr(usage, "prompt_tokens", None),
                    completion_tokens=getattr(usage, "completion_tokens", None),
                    latency_s=round(time.time() - t0, 3),
                    temperature=temperature,
                    seed=seed,
                    attempts=attempt,
                    raw={"id": resp.id, "model": resp.model,
                         "finish_reason": resp.choices[0].finish_reason},
                )

        return CallResult(
            model_id=model.model_id, text=None, reasoning=None, finish_reason=None,
            prompt_tokens=None, completion_tokens=None,
            latency_s=round(time.time() - t0, 3), temperature=temperature,
            seed=seed, attempts=self.max_retries, error=last_err or "unknown",
        )
