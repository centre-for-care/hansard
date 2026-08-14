"""Phrase embeddings for semantic comparison of free-text sub-themes.

Because sub-themes are open vocabulary ("staff shortages" vs "workforce
crisis"), agreement must be measured in embedding space, not by string match.
This module wraps the hosted Qwen3 embedding model and caches every phrase to
disk, so iterating on the metrics never re-embeds (or re-bills) the same text.

``embed_texts`` is the single low-level entry point for every embedding call
in the package (phrase cache here, speech/chunk/query caches in retrieve.py):
it retries transient failures with backoff, sorts the response by index (the
API may return items out of order — silently scrambling assignments if
trusted), and verifies the count before returning.
"""

from __future__ import annotations

import json
import time

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from . import config

EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
_CACHE_VECS = config.ARTIFACTS_DIR / "embeddings.npy"
_CACHE_KEYS = config.ARTIFACTS_DIR / "embeddings_keys.json"
_BATCH = 128

# One cache write per this many texts, so a crash mid-run loses at most one
# slice rather than the whole batch of new embeddings.
_SAVE_EVERY = 512

_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError,
              RateLimitError)


def make_client() -> OpenAI:
    return OpenAI(base_url=config.base_url(), api_key=config.api_key())


def embed_texts(
    texts: list[str],
    client: OpenAI,
    *,
    model: str = EMBED_MODEL,
    batch: int = _BATCH,
    max_retries: int = 4,
    backoff_base: float = 1.5,
    verbose: bool = False,
) -> np.ndarray:
    """Embed ``texts`` -> unit-normalised (n, d) float32 matrix, in order."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        # The API rejects empty strings; a single space keeps row alignment.
        chunk = [t if t and t.strip() else " " for t in texts[i: i + batch]]
        resp = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = client.embeddings.create(model=model, input=chunk)
                break
            except _RETRYABLE as e:
                if attempt == max_retries:
                    raise
                if verbose:
                    print(f"  embed retry {attempt}: {type(e).__name__}")
                time.sleep(backoff_base ** attempt)
        data = sorted(resp.data, key=lambda d: d.index)
        if len(data) != len(chunk):
            raise RuntimeError(
                f"embedding response has {len(data)} items for "
                f"{len(chunk)} inputs (model={model})")
        out.extend(d.embedding for d in data)
        if verbose:
            print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}",
                  flush=True)
    arr = np.asarray(out, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return arr


class EmbeddingCache:
    """Disk-backed cache mapping phrase -> unit-normalised vector."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._index: dict[str, int] = {}
        self._mat: np.ndarray | None = None
        self._client: OpenAI | None = None
        self._load()

    def _load(self) -> None:
        if _CACHE_VECS.exists() and _CACHE_KEYS.exists():
            self._mat = np.load(_CACHE_VECS)
            self._keys = json.loads(_CACHE_KEYS.read_text(encoding="utf-8"))
            self._index = {k: i for i, k in enumerate(self._keys)}

    def _save(self) -> None:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(_CACHE_VECS, self._mat)
        _CACHE_KEYS.write_text(json.dumps(self._keys), encoding="utf-8")

    def _client_(self) -> OpenAI:
        if self._client is None:
            self._client = make_client()
        return self._client

    def ensure(self, phrases: list[str]) -> None:
        """Embed and cache any phrases not already present. Saves after each
        slice so an interrupted run keeps everything embedded so far."""
        missing = sorted({p for p in phrases if p and p not in self._index})
        if not missing:
            return
        cli = self._client_()
        for i in range(0, len(missing), _SAVE_EVERY):
            block = missing[i: i + _SAVE_EVERY]
            vecs = embed_texts(block, cli)
            self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])
            for p in block:
                self._index[p] = len(self._keys)
                self._keys.append(p)
            self._save()

    def get(self, phrase: str) -> np.ndarray | None:
        i = self._index.get(phrase)
        return None if i is None else self._mat[i]

    def matrix(self, phrases: list[str]) -> np.ndarray:
        """(n, d) matrix for the given phrases (must be cached already)."""
        return np.vstack([self._mat[self._index[p]] for p in phrases])
