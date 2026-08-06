"""Consistency, variance decomposition, and estimand robustness.

The pilot's question is not "what is the right answer" (no gold yet) but "how
much does the answer move when we perturb things that *shouldn't* matter".
This module quantifies that at three levels:

    presence (binary)   - Krippendorff's alpha + % agreement across the grid,
                          a self-consistency baseline (temp>0 reps), and a
                          per-factor decomposition of where disagreement comes
                          from (role / task / format / model).

    sub-themes (free)    - semantic agreement via embeddings (soft Jaccard with
                          bipartite phrase matching), plus a post-hoc discovered
                          taxonomy (cluster all emitted themes) with prevalence.

    estimand (aggregate) - topic prevalence per grid cell and its spread: does
                          the headline number survive the perturbations even
                          when individual labels flip?

Everything is computed on a *reparsed* results frame (run.load_results), so it
always reflects the current parser. A slot for gold-standard validity metrics
is left for when the hand-labelled set arrives.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

from .embed import EmbeddingCache
from .prompts import TASK_UNCAPPED

# The model factor is model_id, not family: the qwen family holds several
# different scales (4B / 14B / 30B-A3B / 32B), so grouping by family would leave
# within-family size differences inside every "all else fixed" cell and
# attribute them to whichever factor was being varied.
FACTORS = ("role", "task", "output_format", "model_id")
DEFAULT_TAU = 0.72  # cosine threshold for "same theme" (see embed smoke test)


# --------------------------------------------------------------------------
# Binary / nominal agreement
# --------------------------------------------------------------------------
def krippendorff_alpha(units: list[list]) -> float:
    """Nominal Krippendorff's alpha. ``units`` is one list of ratings per item
    (None entries = missing). Items with <2 ratings are ignored.

    alpha = 1 means perfect agreement; 0 means agreement at chance; <0 means
    systematic disagreement.
    """
    units = [[x for x in u if x is not None] for u in units]
    units = [u for u in units if len(u) >= 2]
    if not units:
        return float("nan")

    values = sorted({x for u in units for x in u})
    vidx = {v: i for i, v in enumerate(values)}
    V = len(values)
    o = np.zeros((V, V))
    for u in units:
        m = len(u)
        cnt = Counter(u)
        for a, ca in cnt.items():
            for b, cb in cnt.items():
                o[vidx[a], vidx[b]] += (ca * (cb - (1 if a == b else 0))) / (m - 1)

    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n < 2:
        return float("nan")
    do = o.sum() - np.trace(o)                       # observed off-diagonal
    de = ((n_c.sum() ** 2 - (n_c ** 2).sum())) / (n - 1)  # expected off-diagonal
    if de == 0:
        return 1.0
    return 1.0 - do / de


def _pairwise_disagreement(labels: list) -> float | None:
    """Probability two distinct ratings (ignoring None) disagree."""
    vals = [x for x in labels if x is not None]
    m = len(vals)
    if m < 2:
        return None
    cnt = Counter(vals)
    same = sum(c * (c - 1) for c in cnt.values())
    total = m * (m - 1)
    return 1.0 - same / total


# --------------------------------------------------------------------------
# Presence robustness
# --------------------------------------------------------------------------
def _core(df: pd.DataFrame) -> pd.DataFrame:
    """The deterministic core grid (single rep at temp 0).

    Excludes the two experimental arms, which are separate experiments rather
    than part of the robustness grid and must not leak into the topic map,
    presence agreement, or factor decomposition:

        * the uncapped no-cap arm (``task == TASK_UNCAPPED``)
        * the definition-sensitivity arm (``definition != "current"``)

    Analyse each on its own selection instead.
    """
    core = df[df["condition"] == "temp0"]
    if "task" in core.columns:
        core = core[core["task"] != TASK_UNCAPPED]
    if "definition" in core.columns:
        # Pilot robustness grid was labeled under the ``current`` wording.
        core = core[core["definition"] == "current"]
    return core.copy()


def presence_agreement(df: pd.DataFrame) -> dict:
    """Agreement on ``mentions_topic`` across all grid cells, per speech.

    Each (variant x model) cell is a 'rater'; speeches are the items.

    Caveat: the 32 cells are not independent raters — each model contributes 8
    of them, so ratings within a model are correlated and alpha here overstates
    the effective number of independent judgements. Read it as a descriptive
    grid-stability index, not a literal inter-rater coefficient; for a
    model-as-rater alpha, compute it over one cell per model.
    """
    core = _core(df)
    units = [list(g["mentions_topic"]) for _, g in core.groupby("speech_id")]
    pair = [d for u in units if (d := _pairwise_disagreement(u)) is not None]
    return {
        "n_speeches": core["speech_id"].nunique(),
        "n_raters_per_speech": core.groupby("speech_id").size().median(),
        "krippendorff_alpha": round(krippendorff_alpha(units), 4),
        "mean_pairwise_agreement": round(1 - np.mean(pair), 4) if pair else float("nan"),
    }


def factor_decomposition(df: pd.DataFrame, factors=FACTORS) -> pd.DataFrame:
    """How much each factor alone perturbs the presence judgement.

    For factor F, hold the speech and every *other* factor fixed, then measure
    disagreement across F's levels. Averaging over all such groups isolates F's
    marginal contribution to instability, comparably across factors.
    """
    core = _core(df)
    rows = []
    for f in factors:
        others = [c for c in FACTORS if c != f] + ["speech_id"]
        dis = []
        for _, g in core.groupby(others):
            d = _pairwise_disagreement(list(g["mentions_topic"]))
            if d is not None:
                dis.append(d)
        rows.append({
            "factor": f,
            "n_levels": core[f].nunique(),
            "mean_disagreement": round(float(np.mean(dis)), 4) if dis else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("mean_disagreement", ascending=False)


def self_consistency(df: pd.DataFrame) -> dict:
    """Run-to-run agreement at temperature > 0 (the realistic deployment
    setting). If the self-consistency probe was not run, returns an empty dict.
    """
    sc = df[df["condition"] == "temp07"]
    if sc.empty:
        return {}
    units = [list(g["mentions_topic"])
             for _, g in sc.groupby(["speech_id", "prompt_hash", "model_id"])]
    pair = [d for u in units if (d := _pairwise_disagreement(u)) is not None]
    return {
        "n_cells": len(units),
        "krippendorff_alpha": round(krippendorff_alpha(units), 4),
        "mean_pairwise_agreement": round(1 - np.mean(pair), 4) if pair else float("nan"),
    }


# --------------------------------------------------------------------------
# Estimand robustness
# --------------------------------------------------------------------------
def prevalence_spread(df: pd.DataFrame) -> dict:
    """Topic prevalence per grid cell and how far it spreads.

    The estimand we report downstream is an aggregate rate; what matters is
    whether it survives perturbation even when item labels flip.
    """
    core = _core(df)
    cell = (core.groupby(["variant_id", "model_id"])["mentions_topic"]
            .apply(lambda s: s.dropna().mean()))
    return {
        "mean_prevalence": round(float(cell.mean()), 4),
        "min_prevalence": round(float(cell.min()), 4),
        "max_prevalence": round(float(cell.max()), 4),
        "sd_prevalence": round(float(cell.std()), 4),
        "spread": round(float(cell.max() - cell.min()), 4),
    }


# --------------------------------------------------------------------------
# Sub-theme semantic agreement
# --------------------------------------------------------------------------
def _soft_jaccard(a: list[str], b: list[str], cache: EmbeddingCache,
                  tau: float) -> float:
    """Greedy bipartite matching of two phrase sets at cosine threshold tau."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    A, B = cache.matrix(a), cache.matrix(b)
    sims = A @ B.T
    used_b: set[int] = set()
    matches = 0
    for i in range(len(a)):
        order = np.argsort(-sims[i])
        for j in order:
            if sims[i, j] < tau:
                break
            if j not in used_b:
                used_b.add(int(j))
                matches += 1
                break
    return matches / (len(a) + len(b) - matches)


def theme_agreement(df: pd.DataFrame, cache: EmbeddingCache | None = None,
                    *, tau: float = DEFAULT_TAU) -> dict:
    """Mean semantic agreement of emitted sub-themes across grid cells.

    Only cells that judged the topic present (with parsed, non-empty themes)
    contribute. Phrases are embedded once (cached) then compared with soft
    Jaccard.
    """
    core = _core(df)
    cache = cache or EmbeddingCache()

    # Embed every phrase up front.
    all_phrases = sorted({p for subs in core["subthemes"] if isinstance(subs, list)
                          for p in subs})
    cache.ensure(all_phrases)

    per_speech = []
    for _, g in core.groupby("speech_id"):
        sets = [list(s) for s in g["subthemes"]
                if isinstance(s, list) and len(s) > 0]
        if len(sets) < 2:
            continue
        scores = [_soft_jaccard(a, b, cache, tau)
                  for a, b in combinations(sets, 2)]
        if scores:
            per_speech.append(float(np.mean(scores)))

    return {
        "n_speeches_scored": len(per_speech),
        "n_unique_phrases": len(all_phrases),
        "tau": tau,
        "mean_theme_agreement": round(float(np.mean(per_speech)), 4) if per_speech else float("nan"),
    }


def discover_taxonomy(df: pd.DataFrame, cache: EmbeddingCache | None = None,
                      *, distance_threshold: float = 0.35,
                      min_cluster: int = 2) -> pd.DataFrame:
    """Cluster all emitted sub-themes into a discovered taxonomy.

    Agglomerative clustering on cosine distance; each cluster is labelled by its
    medoid (the phrase nearest the cluster centroid). Prevalence is the number
    of distinct speeches with at least one theme in that cluster — the
    exploratory "what is there" view.
    """
    from sklearn.cluster import AgglomerativeClustering

    core = _core(df)
    rows = [(r.speech_id, p) for r in core.itertuples()
            if isinstance(r.subthemes, list) for p in r.subthemes]
    if not rows:
        return pd.DataFrame(columns=["cluster", "label", "n_phrases", "n_speeches"])
    sp_ids, phrases = zip(*rows)
    uniq = sorted(set(phrases))
    cache = cache or EmbeddingCache()
    cache.ensure(list(uniq))
    M = cache.matrix(list(uniq))

    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=distance_threshold,
    ).fit_predict(M)

    pidx = {p: i for i, p in enumerate(uniq)}
    out = []
    for c in sorted(set(labels)):
        members = [uniq[i] for i in range(len(uniq)) if labels[i] == c]
        if len(members) < min_cluster:
            continue
        mi = [pidx[p] for p in members]
        centroid = M[mi].mean(axis=0)
        medoid = members[int(np.argmax(M[mi] @ centroid))]
        speeches = {sp for sp, p in zip(sp_ids, phrases) if p in set(members)}
        out.append({"cluster": int(c), "label": medoid,
                    "n_phrases": len(members), "n_speeches": len(speeches)})
    return (pd.DataFrame(out).sort_values("n_speeches", ascending=False)
            .reset_index(drop=True))


# --------------------------------------------------------------------------
# Headline summary
# --------------------------------------------------------------------------
def summarize(df: pd.DataFrame, *, with_themes: bool = True) -> None:
    """Print the headline robustness numbers."""
    print("PRESENCE robustness (temp 0 grid)")
    for k, v in presence_agreement(df).items():
        print(f"  {k:28s} {v}")
    print("\nPER-FACTOR disagreement (presence)")
    print(factor_decomposition(df).to_string(index=False))
    sc = self_consistency(df)
    if sc:
        print("\nSELF-CONSISTENCY (temp 0.7 reps)")
        for k, v in sc.items():
            print(f"  {k:28s} {v}")
    print("\nESTIMAND (prevalence) spread")
    for k, v in prevalence_spread(df).items():
        print(f"  {k:28s} {v}")
    if with_themes:
        print("\nSUB-THEME semantic agreement")
        for k, v in theme_agreement(df).items():
            print(f"  {k:28s} {v}")
