"""Embedder-sensitivity grid (Workstream C1).

Does retrieval change with the embedding model — across sizes within a family,
and across families at comparable size? And how sensitive is it to the query
wording (construct definition) and the document representation?

Grid axes
---------
* model           8 embedders (registry below): Qwen3 0.6B/4B/8B (size axis),
                  BGE base/large (second size pair), GTE-large, E5-large-v2,
                  Nomic v1.5 (family axis). All English. Qwen3-8B is the
                  anchor (the shipping pilot model).
* representation  whole | maxchunk | meanchunk. Chunk arms use overlapping
                  sentence windows from ``retrieve.split_chunks`` (max/overlap
                  recorded in the run manifest); maxchunk is the length-bias
                  arm (max-over-chunks mechanically rewards longer texts),
                  meanchunk the length-bias control.
* query           4 retained definitions (expert_hc_sc, expert_sc_hc,
                  era_neutral, current) + name_only as a free diagnostic.

Backends: ``st`` (sentence-transformers, local GPU/CPU — the cluster path) or
``api`` (OpenAI-compatible /embeddings endpoint — the Nebius path, used for
the serving-stack parity check on Qwen3-8B).

Each (model, backend) run writes scores + a manifest under
``runs/embedder_grid/<run_id>/``; evaluation against gold happens in
``evaluate`` once panel labels exist (leave-one-definition-out — see plan).

Usage::

    python -m hansard_llm.embedder_grid --model BAAI/bge-base-en-v1.5 --backend st
    python -m hansard_llm.embedder_grid --model Qwen/Qwen3-Embedding-8B --backend api
    python -m hansard_llm.embedder_grid --list
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, provenance
from .retrieve import _CHUNK_MAX, _CHUNK_OVERLAP, _CHUNK_SCHEME, split_chunks

# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EmbedderSpec:
    """One embedding model and how to talk to it.

    ``query_template`` wraps the query text in the model's trained query
    format (documents are bare except the ``_DOC_TEMPLATE`` models);
    ``max_tokens`` is the usable context — text beyond it is what the model
    physically cannot read, so "whole" representation means "first
    max_tokens" for short-context models and we record that.
    """

    model_id: str
    family: str
    params_m: int                     # size in millions of parameters
    max_tokens: int
    query_template: str = "{text}"    # applied to queries only
    trust_remote_code: bool = False

    def format_query(self, text: str) -> str:
        return self.query_template.format(text=text)


# Qwen3-Embedding's trained query scaffold (model card): instruction + query;
# documents are embedded bare. Held identical across sizes so the size axis
# is not confounded by prompt wording.
_QWEN_Q = ("Instruct: Given a topic definition, retrieve parliamentary "
           "speeches that substantively discuss the topic\nQuery: {text}")

EMBEDDERS: tuple[EmbedderSpec, ...] = (
    # -- size axis (one family, one training recipe) --
    EmbedderSpec("Qwen/Qwen3-Embedding-0.6B", "qwen3", 600, 32000, _QWEN_Q),
    EmbedderSpec("Qwen/Qwen3-Embedding-4B", "qwen3", 4000, 32000, _QWEN_Q),
    EmbedderSpec("Qwen/Qwen3-Embedding-8B", "qwen3", 8000, 32000, _QWEN_Q),  # anchor
    # -- second size pair + family axis --
    EmbedderSpec("BAAI/bge-base-en-v1.5", "bge", 109, 512,
                 "Represent this sentence for searching relevant passages: {text}"),
    EmbedderSpec("BAAI/bge-large-en-v1.5", "bge", 335, 512,
                 "Represent this sentence for searching relevant passages: {text}"),
    EmbedderSpec("Alibaba-NLP/gte-large-en-v1.5", "gte", 434, 8192,
                 trust_remote_code=True),
    EmbedderSpec("intfloat/e5-large-v2", "e5", 335, 512,
                 "query: {text}"),
    EmbedderSpec("nomic-ai/nomic-embed-text-v1.5", "nomic", 137, 8192,
                 "search_query: {text}", trust_remote_code=True),
)
EMBEDDERS_BY_ID = {e.model_id: e for e in EMBEDDERS}

# Models whose trained scheme also prefixes documents (all others embed bare).
_DOC_TEMPLATE = {
    "nomic-ai/nomic-embed-text-v1.5": "search_document: {text}",
    "intfloat/e5-large-v2": "passage: {text}",
}

# The retained query axis (user decision) + free diagnostic baseline.
QUERY_IDS: tuple[str, ...] = ("expert_hc_sc", "expert_sc_hc", "era_neutral",
                              "current", "name_only")

REPRESENTATIONS: tuple[str, ...] = ("whole", "maxchunk", "meanchunk")

from .sample import EVAL_SAMPLE_PATH, load_eval_sample  # noqa: E402  (canonical loader)

# Untruncated cap for the embedding arm: long-context models read up to 32k
# tokens; 100k chars is beyond any speech in the corpus, i.e. "no truncation",
# while still bounding pathological rows.
MAX_EMBED_CHARS = 100_000

# Full-corpus size, used only for the printed embed-time extrapolation
# (deciding whether the winning embedder is affordable on everything).
_CORPUS_N = 9_196_605


def grid_queries() -> dict[str, str]:
    q = {qid: config.HSC_DEFINITIONS[qid].description
         for qid in QUERY_IDS if qid in config.HSC_DEFINITIONS}
    q["name_only"] = "health and social care"
    return q


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class STBackend:
    """Local sentence-transformers inference (cluster / workstation)."""

    def __init__(self, spec: EmbedderSpec, batch: int = 32) -> None:
        from sentence_transformers import SentenceTransformer
        self.spec = spec
        self.batch = batch
        self.model = SentenceTransformer(
            spec.model_id, trust_remote_code=spec.trust_remote_code)
        self.model.max_seq_length = min(self.model.max_seq_length or 10**9,
                                        spec.max_tokens)

    def embed(self, texts: list[str], *, verbose: bool = False) -> np.ndarray:
        arr = self.model.encode(
            texts, batch_size=self.batch, normalize_embeddings=True,
            show_progress_bar=verbose, convert_to_numpy=True)
        return arr.astype(np.float32)


class APIBackend:
    """OpenAI-compatible /embeddings endpoint (Nebius, or vLLM --task embed)."""

    def __init__(self, spec: EmbedderSpec, batch: int = 64) -> None:
        from .embed import embed_texts, make_client
        self.spec = spec
        self.batch = batch
        self._embed_texts = embed_texts
        self._client = make_client()

    def embed(self, texts: list[str], *, verbose: bool = False) -> np.ndarray:
        return self._embed_texts(texts, self._client, model=self.spec.model_id,
                                 batch=self.batch, verbose=verbose)


def make_backend(spec: EmbedderSpec, backend: str):
    if backend == "st":
        return STBackend(spec)
    if backend == "api":
        return APIBackend(spec)
    raise ValueError(f"unknown backend {backend!r}")


# --------------------------------------------------------------------------
# Scoring one model over the eval subset
# --------------------------------------------------------------------------
def score_model(
    spec: EmbedderSpec,
    backend_name: str,
    speeches: pd.DataFrame,
    *,
    representations: tuple[str, ...] = REPRESENTATIONS,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Long scores: speech_id x query_id x representation for one embedder,
    plus wall-clock timings (for extrapolating full-corpus embedding cost).

    Documents are embedded once per representation (queries are one vector
    each), so adding queries is free — the whole query axis rides on a single
    document pass. Chunk arms call ``split_chunks`` (overlapping windows;
    scheme ``ch{max}o{overlap}``).
    """
    be = make_backend(spec, backend_name)
    queries = grid_queries()
    doc_tpl = _DOC_TEMPLATE.get(spec.model_id, "{text}")

    qids = list(queries)
    Q = be.embed([spec.format_query(queries[q]) for q in qids])

    texts = [(t or "")[:MAX_EMBED_CHARS] for t in speeches["speech_text"]]
    sids = list(speeches["speech_id"])

    timings: dict = {
        "n_docs": len(texts),
        "chunk_scheme": _CHUNK_SCHEME,
        "chunk_max_chars": _CHUNK_MAX,
        "chunk_overlap_chars": _CHUNK_OVERLAP,
    }
    rows: list[dict] = []

    def emit(rep: str, sims: np.ndarray) -> None:
        for i, sid in enumerate(sids):
            for j, qid in enumerate(qids):
                rows.append({"speech_id": sid, "query_id": qid,
                             "representation": rep,
                             "score": float(sims[i, j])})

    if "whole" in representations:
        if verbose:
            print(f"[{spec.model_id}] embedding {len(texts)} whole docs…")
        t0 = time.perf_counter()
        D = be.embed([doc_tpl.format(text=t) for t in texts], verbose=verbose)
        timings["embed_whole_s"] = round(time.perf_counter() - t0, 2)
        emit("whole", D @ Q.T)

    need_chunks = {"maxchunk", "meanchunk"} & set(representations)
    if need_chunks:
        chunk_lists = [split_chunks(t) for t in texts]
        flat = [doc_tpl.format(text=c) for cl in chunk_lists for c in cl]
        spans, start = [], 0
        for cl in chunk_lists:
            spans.append((start, start + len(cl)))
            start += len(cl)
        if verbose:
            print(f"[{spec.model_id}] embedding {len(flat)} chunks…")
        timings["n_chunks"] = len(flat)
        t0 = time.perf_counter()
        C = be.embed(flat, verbose=verbose)
        timings["embed_chunks_s"] = round(time.perf_counter() - t0, 2)
        sims_all = C @ Q.T
        for rep, reduce_fn in (("maxchunk", np.max), ("meanchunk", np.mean)):
            if rep not in representations:
                continue
            sims = np.vstack([
                reduce_fn(sims_all[a:b], axis=0) if b > a
                else np.zeros(len(qids), dtype=np.float32)
                for a, b in spans])
            emit(rep, sims)

    out = pd.DataFrame(rows)
    out["model_id"] = spec.model_id
    out["family"] = spec.family
    out["params_m"] = spec.params_m
    out["backend"] = backend_name
    return out, timings


def run_model(model_id: str, backend: str, *, verbose: bool = True) -> Path:
    """Score one embedder over the eval subset; write scores + manifest."""
    spec = EMBEDDERS_BY_ID[model_id]
    speeches = load_eval_sample()
    scores, timings = score_model(spec, backend, speeches, verbose=verbose)

    try:  # record which GPU produced the timings — they don't transfer
        import torch
        if torch.cuda.is_available():
            timings["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    run_id = provenance.new_run_id()
    out_dir = provenance.run_dir("embedder_grid", run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = model_id.replace("/", "__")
    scores_path = out_dir / f"scores_{safe}.parquet"
    scores.to_parquet(scores_path, index=False)
    provenance.write_manifest(out_dir, {
        "experiment": "embedder_grid",
        "run_id": run_id,
        "model_id": model_id,
        "family": spec.family,
        "params_m": spec.params_m,
        "max_tokens": spec.max_tokens,
        "query_template": spec.query_template,
        "doc_template": _DOC_TEMPLATE.get(model_id, "{text}"),
        "embed_backend": backend,
        "representations": list(REPRESENTATIONS),
        "chunk_max_chars": _CHUNK_MAX,
        "chunk_overlap_chars": _CHUNK_OVERLAP,
        "chunk_scheme": _CHUNK_SCHEME,
        "queries": grid_queries(),
        "eval_sample": str(EVAL_SAMPLE_PATH),
        "n_speeches": int(speeches["speech_id"].nunique()),
        "max_embed_chars": MAX_EMBED_CHARS,
        "n_rows": len(scores),
        "timings": timings,
    })
    if verbose:
        print(f"wrote {len(scores)} score rows -> {scores_path}")
        if timings.get("embed_whole_s"):
            rate = timings["n_docs"] / timings["embed_whole_s"]
            est_h = _CORPUS_N / rate / 3600
            print(f"whole-doc throughput: {rate:.1f} docs/s "
                  f"({timings.get('gpu', backend)}) -> full corpus "
                  f"({_CORPUS_N:,}) ~= {est_h:.1f} GPU-hours")
    return scores_path


def load_all_scores() -> pd.DataFrame:
    """Every embedder_grid score file across runs, concatenated.

    Does not filter on chunk scheme — if pre-overlap and post-overlap runs
    coexist under ``runs/embedder_grid/``, check each run's manifest
    ``chunk_scheme`` before treating scores as comparable.
    """
    files = sorted((config.RUNS_DIR / "embedder_grid").glob("*/scores_*.parquet"))
    if not files:
        raise FileNotFoundError("no embedder_grid score files yet")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


# --------------------------------------------------------------------------
# Cross-embedder comparison (gold-free diagnostics)
# --------------------------------------------------------------------------
def rank_agreement(scores: pd.DataFrame, *, query_id: str = "expert_hc_sc",
                   representation: str = "whole",
                   ks: tuple[int, ...] = (500, 1000, 2000)) -> pd.DataFrame:
    """Between-embedder Spearman and overlap@k on a fixed query/representation.

    Gold-free: measures whether embedders *rank the corpus differently* at
    all — if they agree near-perfectly, the sensitivity question is settled
    without labels; if they diverge, gold decides who is right.
    """
    from scipy.stats import spearmanr

    sel = scores[(scores["query_id"] == query_id)
                 & (scores["representation"] == representation)]
    wide = sel.pivot_table(index="speech_id", columns="model_id",
                           values="score").dropna()
    models = list(wide.columns)
    rows = []
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            rho = float(spearmanr(wide[a], wide[b]).statistic)
            row = {"model_a": a, "model_b": b, "spearman": round(rho, 4)}
            for k in ks:
                ta = set(wide[a].nlargest(k).index)
                tb = set(wide[b].nlargest(k).index)
                row[f"overlap_at_{k}"] = round(len(ta & tb) / k, 4)
            rows.append(row)
    return pd.DataFrame(rows).sort_values("spearman")


def length_bias(scores: pd.DataFrame, speeches: pd.DataFrame,
                *, query_id: str = "expert_hc_sc") -> pd.DataFrame:
    """corr(score, log word_count) per model x representation — the maxchunk
    length-bias diagnostic (pilot: 0.57 maxchunk vs 0.23 whole).

    Overlapping chunks raise n_chunks on long speeches vs the old abutting
    scheme; compare manifests' ``chunk_scheme`` before pooling runs.
    """
    wc = speeches.set_index("speech_id")["word_count"]
    sel = scores[scores["query_id"] == query_id].copy()
    sel["log_wc"] = np.log(sel["speech_id"].map(wc).clip(lower=1))
    return (sel.groupby(["model_id", "representation"])
            .apply(lambda g: round(float(np.corrcoef(g["score"], g["log_wc"])[0, 1]), 4),
                   include_groups=False)
            .rename("corr_score_log_wc").reset_index())


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Embedder sensitivity grid")
    ap.add_argument("--model", help="HF id from the registry")
    ap.add_argument("--backend", choices=("st", "api"), default="st")
    ap.add_argument("--list", action="store_true", help="list registry and exit")
    ap.add_argument("--diagnostics", action="store_true",
                    help="run gold-free cross-embedder diagnostics on all "
                         "scores collected so far")
    args = ap.parse_args(argv)

    if args.list:
        for e in EMBEDDERS:
            print(f"{e.model_id:42s} {e.family:6s} {e.params_m:>5}M "
                  f"ctx={e.max_tokens}")
        return
    if args.diagnostics:
        scores = load_all_scores()
        speeches = load_eval_sample()
        print("\n== rank agreement (expert_hc_sc / whole)")
        print(rank_agreement(scores).to_string(index=False))
        print("\n== length bias")
        print(length_bias(scores, speeches).to_string(index=False))
        return
    if not args.model:
        ap.error("pass --model (see --list) or --diagnostics")
    if args.model not in EMBEDDERS_BY_ID:
        ap.error(f"{args.model} not in registry; see --list")
    run_model(args.model, args.backend)


if __name__ == "__main__":
    main()
