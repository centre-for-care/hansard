"""Reproducible taxonomy of discovered sub-themes (the brief's topic map).

The original 152-cluster table in the theme brief was produced by a scratchpad
script that no longer exists, with an unrecorded clustering threshold. This
script replaces it: the arm, the pool filter, and the threshold are all named
here, and a sidecar manifest records the code version, so the table can always
be regenerated.

Arm: expert_hc_sc / v1_nocap / json / role=none / temp0, pilot pool only —
the shipping-default configuration the brief's map was built from. Note this
is deliberately NOT metrics._core (which is the definition="current" nuisance
grid).

Usage:  python -m hansard_llm.docs.build_taxonomy [--threshold 0.25]
Writes: docs/taxonomy_clusters.csv (+ .manifest.json)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hansard_llm import config, metrics, provenance, run
from hansard_llm.embed import EmbeddingCache

OUT = Path(__file__).resolve().parent / "taxonomy_clusters.csv"

# Agglomerative cosine-distance threshold. The artifact trail
# (phrase_clusters_025.npy) and the README's 0.22-0.25 range point to 0.25 for
# the published map; metrics.discover_taxonomy's default 0.35 was never
# calibrated for it.
DEFAULT_THRESHOLD = 0.25

ARM = dict(definition="expert_hc_sc", task="v1_nocap",
           output_format="json", role="none", condition="temp0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--min-cluster", type=int, default=2)
    args = ap.parse_args()

    df = run.load_legacy()
    sel = df[(df["pool"] == "pilot")]
    for col, val in ARM.items():
        sel = sel[sel[col] == val]
    print(f"{len(sel)} rows in the taxonomy arm "
          f"({sel['speech_id'].nunique()} speeches, "
          f"{sel['model_id'].nunique()} models)")

    cache = EmbeddingCache()
    rows = [(r.speech_id, p) for r in sel.itertuples()
            if isinstance(r.subthemes, list) for p in r.subthemes]
    sp_ids, phrases = zip(*rows)
    uniq = sorted(set(phrases))
    cache.ensure(list(uniq))
    print(f"{len(rows)} theme emissions, {len(uniq)} unique phrases")

    # Reuse the clustering in metrics.discover_taxonomy by handing it a frame
    # shaped like its input (it reads speech_id + subthemes via _core, so we
    # bypass _core and call the clustering directly on our arm).
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    M = cache.matrix(list(uniq))
    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=args.threshold,
    ).fit_predict(M)

    pidx = {p: i for i, p in enumerate(uniq)}
    out = []
    for c in sorted(set(labels)):
        members = [uniq[i] for i in range(len(uniq)) if labels[i] == c]
        if len(members) < args.min_cluster:
            continue
        mi = [pidx[p] for p in members]
        centroid = M[mi].mean(axis=0)
        medoid = members[int(np.argmax(M[mi] @ centroid))]
        mset = set(members)
        speeches = {sp for sp, p in zip(sp_ids, phrases) if p in mset}
        out.append({"cluster": int(c), "label": medoid,
                    "n_phrases": len(members), "n_speeches": len(speeches),
                    "members": "; ".join(sorted(members)[:30])})
    tab = (pd.DataFrame(out).sort_values("n_speeches", ascending=False)
           .reset_index(drop=True))
    tab.to_csv(OUT, index=False)
    provenance.write_manifest(OUT.parent, {
        "artifact": OUT.name,
        "arm": ARM,
        "pool": "pilot",
        "distance_threshold": args.threshold,
        "min_cluster": args.min_cluster,
        "n_clusters": len(tab),
        "n_phrases": len(uniq),
        "n_emissions": len(rows),
        "embed_model": "Qwen/Qwen3-Embedding-8B",
    }, filename=f"{OUT.stem}.manifest.json")

    print(f"wrote {len(tab)} clusters -> {OUT}")
    print(tab.head(25)[["label", "n_phrases", "n_speeches"]].to_string())


if __name__ == "__main__":
    main()
