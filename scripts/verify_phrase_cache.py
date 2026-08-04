"""One-off: integrity check for the phrase embedding cache.

The pre-hardening ``embed.EmbeddingCache`` did not sort API responses by
index, so if the endpoint ever returned a batch out of order the phrase->vector
assignment in ``embeddings.npy`` would be silently scrambled. This re-embeds a
random sample of cached phrases with the hardened client and compares cosine
similarity against the cached vectors: ~1.0 everywhere means the cache is
sound; low values on many phrases mean it must be rebuilt (delete
``embeddings.npy`` + ``embeddings_keys.json`` and re-run the metrics).

Usage:  python scripts/verify_phrase_cache.py [--n 200] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hansard_llm.embed import EmbeddingCache, embed_texts, make_client  # noqa: E402


def main() -> None:
    """Re-embed a random sample of cached phrases and compare cosine
    similarity against the cache, exiting non-zero on mismatches."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=0.99,
                    help="cosine below this counts as a mismatch")
    args = ap.parse_args()

    cache = EmbeddingCache()
    if cache._mat is None or not cache._keys:
        sys.exit("phrase cache is empty — nothing to verify")

    rng = np.random.default_rng(args.seed)
    n = min(args.n, len(cache._keys))
    picks = rng.choice(len(cache._keys), size=n, replace=False)
    phrases = [cache._keys[i] for i in picks]
    cached = cache._mat[picks]

    fresh = embed_texts(phrases, make_client(), verbose=True)
    cos = (cached * fresh).sum(axis=1)

    bad = cos < args.tol
    print(f"\nchecked {n} phrases: cosine min={cos.min():.4f} "
          f"mean={cos.mean():.4f}; {int(bad.sum())} below {args.tol}")
    if bad.any():
        print("\nMISMATCHES (cache likely scrambled — rebuild it):")
        for i in np.where(bad)[0][:20]:
            print(f"  {cos[i]:.3f}  {phrases[i][:80]!r}")
        sys.exit(1)
    print("cache looks sound.")


if __name__ == "__main__":
    main()
