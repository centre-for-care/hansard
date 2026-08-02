"""Definition-as-query semantic retrieval experiment (Stage-1 filter pilot).

Tests whether cosine similarity between a speech embedding and the H&SC
construct definition recovers on-topic speeches better than the keyword seed.

Two pools
---------
* Labeled pilot (270) — rank against LLM majority presence (proxy gold).
* Filter pool (~3k, era-stratified, not seed-balanced) — score distributions
  and a small LLM spot-check on top / bottom / mid bands.

Document representations: whole-speech (truncated to ``run.MAX_SPEECH_CHARS``)
and max-over-sentence-chunks (RAG-style).

Usage::

    python -m hansard_llm.retrieve --embed --evaluate --spotcheck --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config, run, sample
from .config import ModelSpec
from .embed import EMBED_MODEL, embed_texts, make_client
from .prompts import TASK_UNCAPPED, build_definition_variants

# --------------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------------
FILTER_POOL_PATH = config.ARTIFACTS_DIR / "retrieval_filter_pool.parquet"
SCORES_PATH = config.ARTIFACTS_DIR / "retrieval_scores.parquet"
METRICS_PATH = config.ARTIFACTS_DIR / "retrieval_metrics.json"
SUMMARY_PATH = Path(__file__).resolve().parent / "docs" / "retrieval_summary.md"
SPOTCHECK_IDS_PATH = config.ARTIFACTS_DIR / "retrieval_spotcheck_ids.json"

_WHOLE_VECS = config.ARTIFACTS_DIR / "speech_embeddings_whole.npy"
_WHOLE_KEYS = config.ARTIFACTS_DIR / "speech_embeddings_whole_keys.json"
_CHUNK_DIR = config.ARTIFACTS_DIR / "speech_chunk_embeddings"
_QUERY_VECS = config.ARTIFACTS_DIR / "retrieval_query_embeddings.npy"
_QUERY_KEYS = config.ARTIFACTS_DIR / "retrieval_query_keys.json"

_BATCH = 64
_FILTER_POOL_N = 3000
_FILTER_SEED = 20260731
_CHUNK_MIN = 40
_CHUNK_MAX = 400
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

SPOTCHECK_MODELS: tuple[ModelSpec, ...] = tuple(
    m for m in config.CORE_MODELS
    if m.model_id in (
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "meta-llama/Llama-3.3-70B-Instruct",
    )
)


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------
def query_texts() -> dict[str, str]:
    """Construct-definition queries for the retrieval arm."""
    return {
        "expert_hc_sc": config.HSC_DEFINITIONS["expert_hc_sc"].description,
        "expert_sc_hc": config.HSC_DEFINITIONS["expert_sc_hc"].description,
        "current": config.HSC_DEFINITIONS["current"].description,
        "era_neutral": config.HSC_DEFINITIONS["era_neutral"].description,
        "expert_hc_only": config._EXPERT_HEALTHCARE,
        "expert_sc_only": config._EXPERT_SOCIAL_CARE,
        "name_only": "health and social care",
    }


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def _content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def truncate_speech(text: str | None, n: int = run.MAX_SPEECH_CHARS) -> str:
    return (text or "")[:n]


def split_chunks(text: str) -> list[str]:
    """Crude sentence chunks in [_CHUNK_MIN, _CHUNK_MAX] chars."""
    raw = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]
    chunks: list[str] = []
    buf = ""
    for s in raw:
        if len(s) > _CHUNK_MAX:
            if buf:
                chunks.append(buf)
                buf = ""
            # hard-split long sentences
            for i in range(0, len(s), _CHUNK_MAX):
                piece = s[i: i + _CHUNK_MAX].strip()
                if len(piece) >= _CHUNK_MIN:
                    chunks.append(piece)
            continue
        candidate = f"{buf} {s}".strip() if buf else s
        if len(candidate) <= _CHUNK_MAX:
            buf = candidate
        else:
            if len(buf) >= _CHUNK_MIN:
                chunks.append(buf)
            buf = s
    if buf and len(buf) >= _CHUNK_MIN:
        chunks.append(buf)
    if not chunks and text.strip():
        chunks = [text.strip()[:_CHUNK_MAX]]
    return chunks


# --------------------------------------------------------------------------
# Embedding clients / caches
# --------------------------------------------------------------------------
def _embed_client() -> OpenAI:
    return make_client()


def _embed_raw(texts: list[str], client: OpenAI | None = None) -> np.ndarray:
    """Retrying, index-sorted, unit-normalised embedding (see embed.embed_texts)."""
    return embed_texts(texts, client or _embed_client(), batch=_BATCH,
                       verbose=True)


class SpeechEmbeddingCache:
    """Disk-backed whole-speech embedding cache keyed by speech_id|content_hash."""

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._index: dict[str, int] = {}
        self._mat: np.ndarray | None = None
        self._client: OpenAI | None = None
        self._load()

    def _load(self) -> None:
        if _WHOLE_VECS.exists() and _WHOLE_KEYS.exists():
            self._mat = np.load(_WHOLE_VECS)
            self._keys = json.loads(_WHOLE_KEYS.read_text(encoding="utf-8"))
            self._index = {k: i for i, k in enumerate(self._keys)}

    def _save(self) -> None:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(_WHOLE_VECS, self._mat)
        _WHOLE_KEYS.write_text(json.dumps(self._keys), encoding="utf-8")

    @staticmethod
    def key(speech_id, text: str) -> str:
        return f"{speech_id}|{_content_hash(text)}|whole"

    def ensure(self, items: list[tuple[object, str]]) -> None:
        """``items`` is a list of (speech_id, truncated_text). Saves after
        every slice so an interrupted run keeps its progress on disk."""
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        for sid, text in items:
            k = self.key(sid, text)
            if k not in self._index:
                missing_keys.append(k)
                missing_texts.append(text if text.strip() else " ")
        if not missing_texts:
            return
        print(f"embedding {len(missing_texts)} whole speeches…")
        if self._client is None:
            self._client = _embed_client()
        save_every = 512
        for i in range(0, len(missing_texts), save_every):
            vecs = _embed_raw(missing_texts[i: i + save_every], self._client)
            self._mat = (vecs if self._mat is None
                         else np.vstack([self._mat, vecs]))
            for k in missing_keys[i: i + save_every]:
                self._index[k] = len(self._keys)
                self._keys.append(k)
            self._save()

    def matrix(self, items: list[tuple[object, str]]) -> np.ndarray:
        self.ensure(items)
        idx = [self._index[self.key(sid, text)] for sid, text in items]
        return self._mat[idx]


class ChunkEmbeddingCache:
    """Per-speech chunk matrices on disk under ``speech_chunk_embeddings/``."""

    def __init__(self) -> None:
        _CHUNK_DIR.mkdir(parents=True, exist_ok=True)
        self._client: OpenAI | None = None
        self._mem: dict[str, np.ndarray] = {}

    @staticmethod
    def key(speech_id, text: str) -> str:
        return f"{speech_id}|{_content_hash(text)}"

    def _path(self, key: str) -> Path:
        # filesystem-safe
        safe = key.replace("|", "_")
        return _CHUNK_DIR / f"{safe}.npy"

    def get(self, speech_id, text: str) -> np.ndarray | None:
        k = self.key(speech_id, text)
        if k in self._mem:
            return self._mem[k]
        p = self._path(k)
        if p.exists():
            arr = np.load(p)
            self._mem[k] = arr
            return arr
        return None

    def ensure(self, items: list[tuple[object, str]]) -> None:
        todo: list[tuple[str, object, str, list[str]]] = []
        for sid, text in items:
            k = self.key(sid, text)
            if self.get(sid, text) is not None:
                continue
            chunks = split_chunks(text)
            todo.append((k, sid, text, chunks))
        if not todo:
            return
        print(f"embedding chunks for {len(todo)} speeches…")
        if self._client is None:
            self._client = _embed_client()
        # flatten, embed, scatter back
        flat: list[str] = []
        spans: list[tuple[str, int, int]] = []
        for k, _sid, _text, chunks in todo:
            start = len(flat)
            flat.extend(c if c.strip() else " " for c in chunks)
            spans.append((k, start, len(flat)))
        vecs = _embed_raw(flat, self._client)
        for k, start, end in spans:
            mat = vecs[start:end]
            np.save(self._path(k), mat)
            self._mem[k] = mat


class QueryEmbeddingCache:
    """Query vectors keyed by ``query_id|content_hash``.

    Keying by content hash (not id alone) means editing a definition text in
    config.py automatically invalidates its stale vector — the failure mode of
    the original id-only cache, which would silently reuse the old embedding.
    """

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._index: dict[str, int] = {}
        self._mat: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if _QUERY_VECS.exists() and _QUERY_KEYS.exists():
            self._mat = np.load(_QUERY_VECS)
            self._keys = json.loads(_QUERY_KEYS.read_text(encoding="utf-8"))
            self._index = {k: i for i, k in enumerate(self._keys)}

    def _save(self) -> None:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(_QUERY_VECS, self._mat)
        _QUERY_KEYS.write_text(json.dumps(self._keys), encoding="utf-8")

    @staticmethod
    def key(query_id: str, text: str) -> str:
        return f"{query_id}|{_content_hash(text)}"

    def ensure(self, queries: dict[str, str]) -> None:
        missing = [(qid, text) for qid, text in queries.items()
                   if self.key(qid, text) not in self._index]
        if not missing:
            return
        print(f"embedding {len(missing)} queries…")
        vecs = _embed_raw([text for _, text in missing])
        self._mat = vecs if self._mat is None else np.vstack([self._mat, vecs])
        for qid, text in missing:
            self._index[self.key(qid, text)] = len(self._keys)
            self._keys.append(self.key(qid, text))
        self._save()

    def vector(self, query_id: str, text: str) -> np.ndarray:
        return self._mat[self._index[self.key(query_id, text)]]

    def matrix(self, queries: dict[str, str]) -> np.ndarray:
        return np.vstack([self.vector(qid, text)
                          for qid, text in queries.items()])


# --------------------------------------------------------------------------
# Filter pool
# --------------------------------------------------------------------------
def draw_filter_pool(
    n: int = _FILTER_POOL_N,
    *,
    seed: int = _FILTER_SEED,
    write: bool = True,
    exclude_ids: set | None = None,
) -> pd.DataFrame:
    """Era-stratified draw from the full corpus (not seed-balanced)."""
    topic = config.DEFAULT_TOPIC
    design = sample.SampleDesign()
    con = sample._connect()
    sample._build_meta_table(con, design, topic)

    eras = [label for label, _, _ in sample.ERA_BUCKETS]
    per_era = max(1, n // len(eras))
    id_frames = []
    for label in eras:
        ids = con.execute(
            f"""
            SELECT * FROM (
                SELECT speech_id, era, seed_present FROM meta
                WHERE era = '{label}'
            ) USING SAMPLE reservoir({per_era} ROWS) REPEATABLE ({seed})
            """
        ).df()
        id_frames.append(ids)
    picked = pd.concat(id_frames, ignore_index=True).drop_duplicates("speech_id")
    if exclude_ids:
        picked = picked[~picked["speech_id"].isin(exclude_ids)]
    # top up if exclusions thinned a bucket (simple: take more overall)
    if len(picked) < n:
        need = n - len(picked) + 50
        extra = con.execute(
            f"""
            SELECT * FROM (
                SELECT speech_id, era, seed_present FROM meta
            ) USING SAMPLE reservoir({need} ROWS) REPEATABLE ({seed + 1})
            """
        ).df()
        if exclude_ids:
            extra = extra[~extra["speech_id"].isin(exclude_ids)]
        picked = (pd.concat([picked, extra], ignore_index=True)
                  .drop_duplicates("speech_id")
                  .head(n))

    con.register("picked_ids", picked[["speech_id"]])
    df = con.execute(
        """
        SELECT e.speech_id, e.year, e.decade, e.chamber, e.speech_type,
               e.word_count, e.section_title, e.speech_text
        FROM enriched e
        JOIN picked_ids p USING (speech_id)
        """
    ).df()
    df = df.merge(picked.drop_duplicates("speech_id"), on="speech_id", how="left")
    con.close()
    df = df.drop_duplicates(subset="speech_id").reset_index(drop=True)
    if write:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(FILTER_POOL_PATH, index=False)
        print(f"wrote filter pool {len(df)} -> {FILTER_POOL_PATH}")
    return df


def load_filter_pool() -> pd.DataFrame:
    if not FILTER_POOL_PATH.exists():
        raise FileNotFoundError(
            f"No filter pool at {FILTER_POOL_PATH}. Run with --embed first."
        )
    return pd.read_parquet(FILTER_POOL_PATH)


# --------------------------------------------------------------------------
# Gold labels from pilot LLM results
# --------------------------------------------------------------------------
def pilot_majority_gold(
    definition: str = "expert_hc_sc",
) -> pd.DataFrame:
    """Speech-level majority ``mentions_topic`` under the shipping default.

    Restricted to the pilot pool: the legacy log also holds filter-pool rows
    from the spot-check under identical grid labels, rated by only 2 models
    instead of 4 — pooling them would silently mix different rater counts.
    """
    df = run.load_legacy()
    sel = df[
        (df["pool"] == "pilot")
        & (df["condition"] == "temp0")
        & (df["role"] == "none")
        & (df["task"] == TASK_UNCAPPED)
        & (df["definition"] == definition)
        & (df["output_format"] == "json")
    ].copy()
    g = (sel.groupby("speech_id")["mentions_topic"]
         .agg(rate="mean", n="count")
         .reset_index())
    g["label"] = (g["rate"] >= 0.5).astype(int)
    return g


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def score_pool(
    speeches: pd.DataFrame,
    *,
    queries: dict[str, str] | None = None,
    whole_cache: SpeechEmbeddingCache | None = None,
    chunk_cache: ChunkEmbeddingCache | None = None,
    query_cache: QueryEmbeddingCache | None = None,
    modes: tuple[str, ...] = ("whole", "maxchunk"),
) -> pd.DataFrame:
    """Return long scores: speech_id × query_id × mode."""
    queries = queries or query_texts()
    whole_cache = whole_cache or SpeechEmbeddingCache()
    chunk_cache = chunk_cache or ChunkEmbeddingCache()
    query_cache = query_cache or QueryEmbeddingCache()

    items = [
        (r.speech_id, truncate_speech(r.speech_text))
        for r in speeches.itertuples()
    ]
    query_cache.ensure(queries)
    qids = list(queries)
    Q = query_cache.matrix(queries)  # (Q, d)

    rows: list[dict] = []
    meta = speeches.set_index("speech_id")

    if "whole" in modes:
        print("scoring whole-speech…")
        D = whole_cache.matrix(items)  # (N, d)
        sims = D @ Q.T                 # (N, Q)
        for i, (sid, _text) in enumerate(items):
            m = meta.loc[sid]
            for j, qid in enumerate(qids):
                rows.append({
                    "speech_id": sid,
                    "query_id": qid,
                    "mode": "whole",
                    "score": float(sims[i, j]),
                    "seed_present": bool(m["seed_present"]) if "seed_present" in m.index else None,
                    "era": m["era"] if "era" in m.index else None,
                    "year": int(m["year"]) if "year" in m.index and pd.notna(m["year"]) else None,
                })

    if "maxchunk" in modes:
        print("scoring max-chunk…")
        chunk_cache.ensure(items)
        for sid, text in items:
            mat = chunk_cache.get(sid, text)
            if mat is None or len(mat) == 0:
                continue
            sims = mat @ Q.T  # (C, Q)
            best = sims.max(axis=0)
            m = meta.loc[sid]
            for j, qid in enumerate(qids):
                rows.append({
                    "speech_id": sid,
                    "query_id": qid,
                    "mode": "maxchunk",
                    "score": float(best[j]),
                    "seed_present": bool(m["seed_present"]) if "seed_present" in m.index else None,
                    "era": m["era"] if "era" in m.index else None,
                    "year": int(m["year"]) if "year" in m.index and pd.notna(m["year"]) else None,
                })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    if len(y_true) == 0:
        return float("nan")
    k = min(k, len(y_true))
    order = np.argsort(-scores)[:k]
    return float(y_true[order].mean())


def _recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    total = y_true.sum()
    if total == 0:
        return float("nan")
    k = min(k, len(y_true))
    order = np.argsort(-scores)[:k]
    return float(y_true[order].sum() / total)


def evaluate_ranking(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    ks: tuple[int, ...] = (10, 25, 50),
    thresholds: tuple[float, ...] = tuple(np.round(np.arange(0.20, 0.81, 0.05), 2)),
) -> dict:
    """Evaluate each (query_id, mode) against binary labels; keyword baseline too."""
    lab = labels.set_index("speech_id")["label"]
    out: dict = {"by_query": [], "keyword_baseline": {}, "thresholds": []}

    # attach labels
    s = scores.copy()
    s["label"] = s["speech_id"].map(lab)
    s = s.dropna(subset=["label"])
    s["label"] = s["label"].astype(int)

    # keyword baseline on the same labeled speeches
    # seed_present may be on scores; fall back to pilot sample
    if s["seed_present"].notna().any():
        # one row per speech for baseline
        base = s.drop_duplicates("speech_id")[["speech_id", "label", "seed_present"]]
        y = base["label"].to_numpy()
        seed = base["seed_present"].astype(int).to_numpy()
        out["keyword_baseline"] = {
            "n": int(len(base)),
            "n_pos": int(y.sum()),
            "precision": round(float(y[seed == 1].mean()) if seed.sum() else float("nan"), 4),
            "recall": round(float(y[seed == 1].sum() / y.sum()) if y.sum() else float("nan"), 4),
            "auroc": round(float(roc_auc_score(y, seed)), 4) if len(np.unique(y)) > 1 else float("nan"),
            "average_precision": round(float(average_precision_score(y, seed)), 4)
            if len(np.unique(y)) > 1 else float("nan"),
        }

    for (qid, mode), g in s.groupby(["query_id", "mode"]):
        # one score per speech
        g = g.drop_duplicates("speech_id")
        y = g["label"].to_numpy()
        sc = g["score"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        row = {
            "query_id": qid,
            "mode": mode,
            "n": int(len(g)),
            "n_pos": int(y.sum()),
            "auroc": round(float(roc_auc_score(y, sc)), 4),
            "average_precision": round(float(average_precision_score(y, sc)), 4),
            "score_mean_pos": round(float(sc[y == 1].mean()), 4),
            "score_mean_neg": round(float(sc[y == 0].mean()), 4),
        }
        for k in ks:
            row[f"precision_at_{k}"] = round(_precision_at_k(y, sc, k), 4)
            row[f"recall_at_{k}"] = round(_recall_at_k(y, sc, k), 4)
        out["by_query"].append(row)

        # thresholds for the primary shipping query
        if qid == "expert_hc_sc":
            for t in thresholds:
                keep = sc >= t
                retained = float(keep.mean())
                recall = float(y[keep].sum() / y.sum()) if y.sum() else float("nan")
                precision = float(y[keep].mean()) if keep.sum() else float("nan")
                out["thresholds"].append({
                    "query_id": qid,
                    "mode": mode,
                    "threshold": float(t),
                    "retained_frac": round(retained, 4),
                    "recall": round(recall, 4),
                    "precision": round(precision, 4),
                    "n_kept": int(keep.sum()),
                })

    out["by_query"] = sorted(
        out["by_query"], key=lambda r: (-r["average_precision"], r["query_id"], r["mode"])
    )
    return out


def filter_pool_seed_summary(scores: pd.DataFrame) -> list[dict]:
    """Mean score by seed_present for each query/mode on the filter pool."""
    rows = []
    for (qid, mode), g in scores.groupby(["query_id", "mode"]):
        g = g.dropna(subset=["seed_present"])
        if g.empty:
            continue
        for present, sub in g.groupby("seed_present"):
            rows.append({
                "query_id": qid,
                "mode": mode,
                "seed_present": bool(present),
                "n": int(len(sub)),
                "score_mean": round(float(sub["score"].mean()), 4),
                "score_p50": round(float(sub["score"].median()), 4),
                "score_p90": round(float(sub["score"].quantile(0.9)), 4),
            })
    return rows


# --------------------------------------------------------------------------
# Spot-check
# --------------------------------------------------------------------------
def select_spotcheck_ids(
    filter_scores: pd.DataFrame,
    *,
    query_id: str = "expert_hc_sc",
    mode: str = "whole",
    n_band: int = 50,
    seed: int = _FILTER_SEED,
) -> pd.DataFrame:
    """Top / bottom / random-mid speeches for LLM validation."""
    g = filter_scores[
        (filter_scores["query_id"] == query_id)
        & (filter_scores["mode"] == mode)
    ].drop_duplicates("speech_id").sort_values("score", ascending=False)
    top = g.head(n_band).assign(band="top")
    bottom = g.tail(n_band).assign(band="bottom")
    mid = g.iloc[n_band: len(g) - n_band]
    rng = np.random.default_rng(seed)
    if len(mid) >= n_band:
        pick = rng.choice(mid.index.to_numpy(), size=n_band, replace=False)
        random_mid = mid.loc[pick].assign(band="random_mid")
    else:
        random_mid = mid.assign(band="random_mid")
    out = pd.concat([top, bottom, random_mid], ignore_index=True)
    out = out.drop_duplicates("speech_id")
    return out


def run_spotcheck(
    spot_meta: pd.DataFrame,
    filter_pool: pd.DataFrame,
    *,
    max_workers: int = 16,
    models: tuple[ModelSpec, ...] = SPOTCHECK_MODELS,
) -> pd.DataFrame:
    """LLM-label spot-check speeches with expert_hc_sc / JSON / uncapped."""
    speeches = filter_pool[filter_pool["speech_id"].isin(spot_meta["speech_id"])].copy()
    # attach band for later join
    speeches = speeches.merge(
        spot_meta[["speech_id", "band", "score"]].rename(columns={"score": "retrieval_score"}),
        on="speech_id", how="left",
    )
    topic = config.HSC_DEFINITIONS["expert_hc_sc"]
    variants = build_definition_variants(
        [topic], roles=("none",), formats=("json",), task=TASK_UNCAPPED,
    )
    plan = run.RunPlan(
        speeches=speeches,
        topic=topic,
        variants=variants,
        models=models,
        conditions=(run.CORE,),
        max_workers=max_workers,
        pool="filter_pool",
    )
    # Own experiment, own run directory: spot-check rows must never share a
    # log with pilot rows again (that mixing broke the definition chart and
    # contaminated the majority gold in the pre-provenance store).
    n = run.execute(plan, experiment="retrieval_spotcheck")
    print(f"spotcheck wrote {n} new cells")
    return speeches


def summarize_spotcheck(spot_meta: pd.DataFrame) -> list[dict]:
    # Spot-check rows live in the legacy log (pre-provenance runs) and/or the
    # versioned retrieval_spotcheck experiment; read both.
    frames = [run.load_legacy()]
    try:
        frames.append(run.load_experiment("retrieval_spotcheck"))
    except FileNotFoundError:
        pass
    df = pd.concat(frames, ignore_index=True)
    sel = df[
        (df["condition"] == "temp0")
        & (df["role"] == "none")
        & (df["task"] == TASK_UNCAPPED)
        & (df["definition"] == "expert_hc_sc")
        & (df["output_format"] == "json")
        & (df["speech_id"].isin(spot_meta["speech_id"]))
        & (df["model_id"].isin([m.model_id for m in SPOTCHECK_MODELS]))
    ]
    rate = sel.groupby("speech_id")["mentions_topic"].mean()
    meta = spot_meta.set_index("speech_id")
    rows = []
    for band, g in meta.groupby("band"):
        labels = rate.reindex(g.index).dropna()
        rows.append({
            "band": band,
            "n_speeches": int(len(g)),
            "n_labeled": int(len(labels)),
            "mean_retrieval_score": round(float(g["score"].mean()), 4),
            "llm_positive_rate": round(float((labels >= 0.5).mean()), 4) if len(labels) else None,
            "llm_mean_rate": round(float(labels.mean()), 4) if len(labels) else None,
        })
    return rows


# --------------------------------------------------------------------------
# Summary markdown
# --------------------------------------------------------------------------
def write_summary(metrics: dict, spot: list[dict] | None = None) -> None:
    lines = [
        "# Semantic retrieval experiment — summary",
        "",
        "Definition-as-query ranking with `Qwen3-Embedding-8B`. Proxy gold on the",
        "pilot: speech-level majority of `mentions_topic` under expert HC→SC /",
        "JSON / uncapped / role=none.",
        "",
        "## Keyword baseline (pilot)",
        "",
        "```",
        json.dumps(metrics.get("keyword_baseline", {}), indent=2),
        "```",
        "",
        "## Ranking metrics by query × mode (pilot)",
        "",
        "| Query | Mode | AUROC | AP | P@25 | R@25 | mean+ | mean− |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in metrics.get("by_query", []):
        lines.append(
            f"| {r['query_id']} | {r['mode']} | {r['auroc']:.3f} | "
            f"{r['average_precision']:.3f} | {r.get('precision_at_25', float('nan')):.3f} | "
            f"{r.get('recall_at_25', float('nan')):.3f} | {r['score_mean_pos']:.3f} | "
            f"{r['score_mean_neg']:.3f} |"
        )

    lines += [
        "",
        "## Threshold sweep — `expert_hc_sc` (pilot)",
        "",
        "| Mode | Threshold | Retained | Recall | Precision |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in metrics.get("thresholds", []):
        lines.append(
            f"| {r['mode']} | {r['threshold']:.2f} | {r['retained_frac']:.3f} | "
            f"{r['recall']:.3f} | {r['precision']:.3f} |"
        )

    if metrics.get("filter_pool_by_seed"):
        lines += [
            "",
            "## Filter pool — score by keyword seed",
            "",
            "| Query | Mode | Seed | N | Mean | P50 | P90 |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for r in metrics["filter_pool_by_seed"]:
            if r["query_id"] not in ("expert_hc_sc", "current", "name_only"):
                continue
            lines.append(
                f"| {r['query_id']} | {r['mode']} | {r['seed_present']} | "
                f"{r['n']} | {r['score_mean']:.3f} | {r['score_p50']:.3f} | "
                f"{r['score_p90']:.3f} |"
            )

    if spot:
        lines += [
            "",
            "## LLM spot-check on filter pool (expert HC→SC / JSON)",
            "",
            "| Band | N labeled | Mean retrieval score | LLM positive rate |",
            "|---|---:|---:|---:|",
        ]
        for r in spot:
            lines.append(
                f"| {r['band']} | {r['n_labeled']} | {r['mean_retrieval_score']:.3f} | "
                f"{r['llm_positive_rate']} |"
            )

    # recommendation stub from best AP + shipping query threshold
    best = metrics.get("by_query", [None])[0]
    lines += ["", "## Provisional takeaway", ""]
    kb = metrics.get("keyword_baseline", {})
    if best:
        lines.append(
            f"Best ranking on the labeled pilot: **{best['query_id']}** / "
            f"**{best['mode']}** (AP={best['average_precision']:.3f}, "
            f"AUROC={best['auroc']:.3f}). Keyword baseline AP="
            f"{kb.get('average_precision', 'n/a')} "
            f"(precision={kb.get('precision', 'n/a')}, "
            f"recall={kb.get('recall', 'n/a')})."
        )
    # Prefer shipping query (expert_hc_sc / whole) for the Stage-1 threshold note
    thr = [
        t for t in metrics.get("thresholds", [])
        if t["query_id"] == "expert_hc_sc" and t["mode"] == "whole"
        and t["recall"] >= 0.9
    ]
    if thr:
        t = max(thr, key=lambda x: x["threshold"])
        lines.append(
            f"For the shipping query **expert_hc_sc / whole**, a cosine "
            f"threshold of **{t['threshold']:.2f}** reaches ≥90% recall of "
            f"LLM-positives on the pilot while retaining "
            f"**{100 * t['retained_frac']:.1f}%** of speeches "
            f"(precision ≈ {100 * t['precision']:.1f}%). "
            f"Pilot is keyword-stratified, so retained fractions will be "
            f"lower on a natural corpus."
        )
    if spot:
        by_band = {r["band"]: r for r in spot}
        top = by_band.get("top", {})
        mid = by_band.get("random_mid", {})
        bot = by_band.get("bottom", {})
        lines.append(
            f"On the era-stratified filter pool spot-check (150 speeches, "
            f"expert HC→SC / JSON), LLM-positive rates were "
            f"**{top.get('llm_positive_rate')}** (top-50 by score), "
            f"**{mid.get('llm_positive_rate')}** (random mid), "
            f"**{bot.get('llm_positive_rate')}** (bottom-50) — strong "
            f"enrichment at the top of the ranking outside the keyword-"
            f"stratified pilot."
        )
    lines.append("")
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {SUMMARY_PATH}")


# --------------------------------------------------------------------------
# Pipeline steps
# --------------------------------------------------------------------------
def embed_and_score(*, modes: tuple[str, ...] = ("whole", "maxchunk")) -> pd.DataFrame:
    pilot = sample.load_sample()
    exclude = set(pilot["speech_id"])
    if FILTER_POOL_PATH.exists():
        filt = load_filter_pool()
        print(f"loaded existing filter pool ({len(filt)})")
    else:
        filt = draw_filter_pool(exclude_ids=exclude)

    whole_cache = SpeechEmbeddingCache()
    chunk_cache = ChunkEmbeddingCache()
    query_cache = QueryEmbeddingCache()
    queries = query_texts()

    print("=== pilot pool ===")
    pilot_scores = score_pool(
        pilot, queries=queries, whole_cache=whole_cache, chunk_cache=chunk_cache,
        query_cache=query_cache, modes=modes,
    )
    pilot_scores["pool"] = "pilot"

    print("=== filter pool ===")
    filt_scores = score_pool(
        filt, queries=queries, whole_cache=whole_cache, chunk_cache=chunk_cache,
        query_cache=query_cache, modes=modes,
    )
    filt_scores["pool"] = "filter"

    scores = pd.concat([pilot_scores, filt_scores], ignore_index=True)
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORES_PATH, index=False)
    print(f"wrote {SCORES_PATH} ({len(scores)} rows)")
    return scores


def evaluate_all(scores: pd.DataFrame | None = None) -> dict:
    if scores is None:
        scores = pd.read_parquet(SCORES_PATH)
    gold = pilot_majority_gold()
    pilot_scores = scores[scores["pool"] == "pilot"]
    metrics = evaluate_ranking(pilot_scores, gold)
    filt_scores = scores[scores["pool"] == "filter"]
    metrics["filter_pool_by_seed"] = filter_pool_seed_summary(filt_scores)
    metrics["n_pilot_labeled"] = int(len(gold))
    metrics["n_pilot_pos"] = int(gold["label"].sum())
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote {METRICS_PATH}")
    # print top lines
    print("\nTop configs by AP:")
    for r in metrics["by_query"][:8]:
        print(f"  {r['query_id']:16s} {r['mode']:8s}  AP={r['average_precision']:.3f}  "
              f"AUROC={r['auroc']:.3f}")
    print("Keyword baseline:", metrics["keyword_baseline"])
    return metrics


def spotcheck_all(*, max_workers: int = 16) -> list[dict]:
    scores = pd.read_parquet(SCORES_PATH)
    filt_scores = scores[scores["pool"] == "filter"]
    spot_meta = select_spotcheck_ids(filt_scores)
    SPOTCHECK_IDS_PATH.write_text(
        spot_meta[["speech_id", "band", "score"]].to_json(orient="records"),
        encoding="utf-8",
    )
    print(f"spotcheck bands: {spot_meta.groupby('band').size().to_dict()}")
    filt = load_filter_pool()
    run_spotcheck(spot_meta, filt, max_workers=max_workers)
    summary = summarize_spotcheck(spot_meta)
    print("spotcheck summary:")
    for r in summary:
        print(f"  {r}")
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Definition-as-query retrieval experiment")
    ap.add_argument("--embed", action="store_true",
                    help="draw filter pool (if needed), embed, and score both pools")
    ap.add_argument("--evaluate", action="store_true",
                    help="evaluate pilot rankings vs LLM gold + keyword baseline")
    ap.add_argument("--spotcheck", action="store_true",
                    help="LLM-label top/bottom/mid bands of the filter pool")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--whole-only", action="store_true",
                    help="skip max-chunk arm (faster smoke)")
    args = ap.parse_args(argv)

    if not (args.embed or args.evaluate or args.spotcheck):
        ap.error("pass at least one of --embed / --evaluate / --spotcheck")

    modes = ("whole",) if args.whole_only else ("whole", "maxchunk")
    scores = None
    metrics = None
    spot = None

    if args.embed:
        scores = embed_and_score(modes=modes)
    if args.evaluate:
        metrics = evaluate_all(scores)
    if args.spotcheck:
        spot = spotcheck_all(max_workers=args.workers)
        # refresh metrics file with spotcheck block
        if metrics is None and METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if metrics is not None:
            metrics["spotcheck"] = spot
            METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if metrics is None and METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if metrics is not None:
        write_summary(metrics, spot=spot or metrics.get("spotcheck"))


if __name__ == "__main__":
    main()
