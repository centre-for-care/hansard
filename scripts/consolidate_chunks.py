"""One-off/maintenance: fold per-speech chunk embedding files into the
consolidated store (chunk_embeddings.npy + chunk_index.parquet).

Reads the existing consolidated store (if any) plus every per-speech .npy in
speech_chunk_embeddings/, writes a fresh consolidated pair atomically, then
verifies a sample of speeches round-trips identically via ChunkEmbeddingCache
before deleting the per-speech files.

Usage:  python scripts/consolidate_chunks.py [--keep-files]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hansard_llm import retrieve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-files", action="store_true",
                    help="do not delete per-speech files after verification")
    args = ap.parse_args()

    blocks: list[np.ndarray] = []
    keys: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0

    def add(key: str, arr: np.ndarray) -> None:
        nonlocal pos
        keys.append(key)
        spans.append((pos, pos + len(arr)))
        blocks.append(arr)
        pos += len(arr)

    # existing consolidated store first (so re-runs are cumulative)
    if retrieve._CHUNK_CONSOLIDATED.exists() and retrieve._CHUNK_INDEX.exists():
        mat = np.load(retrieve._CHUNK_CONSOLIDATED)
        idx = pd.read_parquet(retrieve._CHUNK_INDEX)
        for r in idx.itertuples():
            add(r.key, mat[r.start: r.end])
        print(f"carried over {len(idx)} consolidated speeches")

    seen = set(keys)
    files = sorted(retrieve._CHUNK_DIR.glob("*.npy"))
    n_new = 0
    for f in files:
        key = f.stem.replace("_", "|", 1)  # speech_id_hash -> speech_id|hash
        # speech ids can be negative; only the FIRST underscore is the sep —
        # but negative ids start with '-', not '_', so replace-first is safe.
        if key in seen:
            continue
        add(key, np.load(f))
        n_new += 1
    print(f"added {n_new} speeches from {len(files)} per-speech files")

    if not keys:
        sys.exit("nothing to consolidate")

    full = np.vstack(blocks).astype(np.float32)
    index = pd.DataFrame({"key": keys,
                          "start": [s for s, _ in spans],
                          "end": [e for _, e in spans]})

    tmp_mat = retrieve._CHUNK_CONSOLIDATED.with_suffix(".npy.tmp.npy")
    tmp_idx = retrieve._CHUNK_INDEX.with_suffix(".parquet.tmp")
    np.save(tmp_mat, full)
    index.to_parquet(tmp_idx, index=False)
    tmp_mat.replace(retrieve._CHUNK_CONSOLIDATED)
    tmp_idx.replace(retrieve._CHUNK_INDEX)
    print(f"wrote {len(index)} speeches / {len(full)} chunk vectors "
          f"({full.nbytes / 1e6:.0f} MB) -> {retrieve._CHUNK_CONSOLIDATED.name}")

    # verify: fresh cache (no per-file fallback needed) returns identical mats
    rng = np.random.default_rng(0)
    sample_files = [files[i] for i in
                    rng.choice(len(files), size=min(50, len(files)),
                               replace=False)] if files else []
    cache = retrieve.ChunkEmbeddingCache()
    bad = 0
    for f in sample_files:
        key = f.stem.replace("_", "|", 1)
        span = cache._spans.get(key)
        got = (np.asarray(cache._consolidated[span[0]: span[1]])
               if span else None)
        want = np.load(f)
        if got is None or not np.array_equal(got, want):
            bad += 1
            print(f"  MISMATCH {key}")
    print(f"verified {len(sample_files)} sampled speeches: {bad} mismatches")
    if bad:
        sys.exit("verification failed — per-speech files NOT deleted")

    if not args.keep_files:
        for f in files:
            f.unlink()
        print(f"deleted {len(files)} per-speech files")


if __name__ == "__main__":
    main()
