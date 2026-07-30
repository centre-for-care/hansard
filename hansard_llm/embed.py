"""Phrase embeddings for semantic comparison of free-text sub-themes.

Because sub-themes are open vocabulary ("staff shortages" vs "workforce
crisis"), agreement must be measured in embedding space, not by string match.
This module wraps the hosted Qwen3 embedding model and caches every phrase to
disk, so iterating on the metrics never re-embeds (or re-bills) the same text.
"""

from __future__ import annotations

import json

import numpy as np
from openai import OpenAI

from . import config

EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"
_CACHE_VECS = config.ARTIFACTS_DIR / "embeddings.npy"
_CACHE_KEYS = config.ARTIFACTS_DIR / "embeddings_keys.json"
_BATCH = 128


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
            self._client = OpenAI(base_url=config.base_url(), api_key=config.api_key())
        return self._client

    def _embed_raw(self, phrases: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        cli = self._client_()
        for i in range(0, len(phrases), _BATCH):
            chunk = phrases[i : i + _BATCH]
            resp = cli.embeddings.create(model=EMBED_MODEL, input=chunk)
            out.extend(d.embedding for d in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr

    def ensure(self, phrases: list[str]) -> None:
        """Embed and cache any phrases not already present."""
        missing = sorted({p for p in phrases if p and p not in self._index})
        if not missing:
            return
        vecs = self._embed_raw(missing)
        self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])
        for p in missing:
            self._index[p] = len(self._keys)
            self._keys.append(p)
        self._save()

    def get(self, phrase: str) -> np.ndarray | None:
        i = self._index.get(phrase)
        return None if i is None else self._mat[i]

    def matrix(self, phrases: list[str]) -> np.ndarray:
        """(n, d) matrix for the given phrases (must be cached already)."""
        return np.vstack([self._mat[self._index[p]] for p in phrases])
