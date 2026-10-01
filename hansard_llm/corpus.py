"""Full-corpus labelling: eligible pool, frozen 8-way shards, one cell per speech.

The eval2k panel is a 4-definition × 2-temperature grid. A production pass over
the ~4.27M eligible speeches is one shipping definition, temp 0, one rep.
Shards are packed decade×chamber cells (Commons 2010 split by ``speech_id % k``
when a cell exceeds ``ceil(N / 8)``). Cluster jobs read the frozen parquet;
they do not re-filter ``full_data_enriched.parquet``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import duckdb
import pandas as pd

from . import config, provenance, run, sample
from .prompts import TASK_UNCAPPED, build_definition_variants

N_SHARDS = 8
POOL = "eligible"

# One experiment name per production model so A/B caches never mix.
EXPERIMENT_BY_MODEL: dict[str, str] = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": "corpus_nemotron",
    "Qwen/Qwen3-30B-A3B-Instruct-2507": "corpus_qwen30",
}

ELIGIBLE_POOL_PATH = config.DATA_DIR / "eligible_pool.parquet"
ELIGIBLE_MANIFEST = config.DATA_DIR / "eligible_pool.manifest.json"

# Columns the runner + analyses need; shard_id is the pack assignment.
POOL_COLUMNS: tuple[str, ...] = (
    "speech_id", "year", "decade_bin", "chamber", "speech_type",
    "word_count", "section_title", "speech_text", "shard_id",
)


@dataclass(frozen=True)
class Piece:
    """One packable unit: a (decade, chamber) cell, or one split of an oversized cell."""

    decade_bin: int
    chamber: str
    part: int
    k: int
    n: int


def experiment_for_model(model_id: str) -> str:
    if model_id in EXPERIMENT_BY_MODEL:
        return EXPERIMENT_BY_MODEL[model_id]
    slug = model_id.rsplit("/", 1)[-1].lower().replace(".", "")
    return f"corpus_{slug}"


def eligible_pool_path():
    return ELIGIBLE_POOL_PATH


def _connect_enriched(*, memory: str = "8GB") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit = '{memory}'")
    con.execute(
        f"CREATE OR REPLACE VIEW enriched AS "
        f"SELECT * FROM read_parquet('{sample._enriched_path()}')"
    )
    return con


def pieces_from_cells(
    cells: pd.DataFrame,
    *,
    n_shards: int = N_SHARDS,
) -> tuple[list[Piece], int, int]:
    """Split cells larger than ``ceil(N / n_shards)`` into k equal estimated parts.

    ``cells`` needs columns ``decade_bin``, ``chamber``, ``n``.
    """
    if cells.empty:
        raise ValueError("no eligible cells to pack")
    total = int(cells["n"].sum())
    cap = (total + n_shards - 1) // n_shards
    pieces: list[Piece] = []
    for r in cells.itertuples():
        n = int(r.n)
        decade = int(r.decade_bin)
        chamber = str(r.chamber)
        if n <= cap:
            pieces.append(Piece(decade, chamber, 0, 1, n))
            continue
        k = (n + cap - 1) // cap
        base, rem = divmod(n, k)
        for i in range(k):
            pieces.append(Piece(
                decade, chamber, i, k, base + (1 if i < rem else 0)))
    return pieces, cap, total


def pack_pieces(
    pieces: list[Piece],
    *,
    n_shards: int = N_SHARDS,
) -> dict[tuple[int, str, int], int]:
    """Greedy: largest piece first, assign to the currently lightest shard.

    Returns ``(decade_bin, chamber, part) -> shard_id``.
    """
    ordered = sorted(pieces, key=lambda p: (-p.n, p.decade_bin, p.chamber, p.part))
    loads = [0] * n_shards
    mapping: dict[tuple[int, str, int], int] = {}
    for p in ordered:
        i = min(range(n_shards), key=lambda s: (loads[s], s))
        loads[i] += p.n
        mapping[(p.decade_bin, p.chamber, p.part)] = i
    return mapping


def nonnegative_mod_sql(value: str, modulus: str) -> str:
    """SQL for ``value % modulus`` in ``0 .. modulus-1`` (Python-style).

    DuckDB's ``%`` keeps the dividend sign, so negative ``speech_id``s become
    ``-1`` for ``% 2`` and miss the Commons-2010 split (parts are only 0/1).
    """
    return f"CAST((({value} % {modulus}) + {modulus}) % {modulus} AS INTEGER)"


def piece_map_frame(
    pieces: list[Piece],
    mapping: dict[tuple[int, str, int], int],
) -> pd.DataFrame:
    rows = [{
        "decade_bin": p.decade_bin,
        "chamber": p.chamber,
        "part": p.part,
        "k": p.k,
        "n_est": p.n,
        "shard_id": mapping[(p.decade_bin, p.chamber, p.part)],
    } for p in pieces]
    return pd.DataFrame(rows)


def _cell_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        f"""
        SELECT (year // 10) * 10 AS decade_bin, chamber, COUNT(*) AS n
        FROM enriched
        WHERE {sample.eligible_where_sql()}
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()


def build_shards(
    *,
    n_shards: int = N_SHARDS,
    write: bool = True,
    path=None,
) -> pd.DataFrame:
    """Filter the enriched parquet, pack 8 shards, write ``eligible_pool.parquet``.

    Returns the piece-map frame (one row per packed piece). Cluster jobs must
    not call this; they read the frozen parquet.
    """
    out = path or ELIGIBLE_POOL_PATH
    con = _connect_enriched()
    cells = _cell_counts(con)
    pieces, cap, total = pieces_from_cells(cells, n_shards=n_shards)
    mapping = pack_pieces(pieces, n_shards=n_shards)
    piece_df = piece_map_frame(pieces, mapping)
    piece_df = piece_df.copy()
    piece_df["chamber"] = piece_df["chamber"].astype(str)
    con.register("piece_map", piece_df[["decade_bin", "chamber", "part", "k",
                                        "shard_id"]])

    sql = f"""
        SELECT
            e.speech_id,
            e.year,
            (e.year // 10) * 10 AS decade_bin,
            e.chamber,
            e.speech_type,
            e.word_count,
            e.section_title,
            e.speech_text,
            p.shard_id
        FROM enriched e
        JOIN piece_map p
          ON ((e.year // 10) * 10) = p.decade_bin
         AND e.chamber = p.chamber
         AND (CASE WHEN p.k = 1 THEN 0
                   ELSE {nonnegative_mod_sql('e.speech_id', 'p.k')} END) = p.part
        WHERE {sample.eligible_where_sql(table="e")}
    """
    if write:
        out.parent.mkdir(parents=True, exist_ok=True)
        # DuckDB COPY avoids pulling 4.27M rows through pandas.
        con.execute(
            f"COPY ({sql}) TO '{out.as_posix()}' (FORMAT PARQUET)"
        )
        shard_n = con.execute(
            f"""
            SELECT shard_id, COUNT(*) AS n
            FROM read_parquet('{out.as_posix()}')
            GROUP BY 1 ORDER BY 1
            """
        ).df()
        n_written = int(shard_n["n"].sum())
        if n_written != total:
            raise RuntimeError(
                f"eligible pool wrote {n_written} rows, expected {total} "
                f"from cell counts — join to piece_map dropped or duplicated rows")
        provenance.write_manifest(out.parent, {
            "artifact": out.name,
            "n_rows": n_written,
            "n_shards": n_shards,
            "cap": cap,
            "filter": sample.eligible_where_sql(),
            "split_rule": "speech_id % k for cells with n > cap",
            "pieces": piece_df.to_dict(orient="records"),
            "shard_n": shard_n.to_dict(orient="records"),
            "duckdb_version": duckdb.__version__,
        }, filename=ELIGIBLE_MANIFEST.name)
    con.close()
    return piece_df


def load_shard(
    shard: int,
    *,
    columns: list[str] | None = None,
    limit: int | None = None,
    path=None,
) -> pd.DataFrame:
    p = path or ELIGIBLE_POOL_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"No eligible pool at {p}. Run "
            f"`python -m hansard_llm.corpus --build-shards` first.")
    if shard < 0 or shard >= N_SHARDS:
        raise ValueError(f"shard must be 0..{N_SHARDS - 1}, got {shard}")
    cols = columns or list(POOL_COLUMNS)
    col_sql = ", ".join(cols)
    lim = f"LIMIT {int(limit)}" if limit is not None else ""
    con = duckdb.connect(":memory:")
    df = con.execute(
        f"""
        SELECT {col_sql}
        FROM read_parquet('{p.as_posix()}')
        WHERE shard_id = {int(shard)}
        ORDER BY speech_id
        {lim}
        """
    ).df()
    con.close()
    if df.empty:
        raise RuntimeError(f"shard {shard} is empty in {p}")
    return df


def _variants():
    return build_definition_variants(
        [config.DEFAULT_TOPIC],
        roles=("none",),
        formats=("json",),
        task=TASK_UNCAPPED,
    )


def corpus_plan(
    model,
    speeches: pd.DataFrame,
    *,
    max_workers: int = 32,
    max_tokens: int | None = None,
) -> run.RunPlan:
    return run.RunPlan(
        speeches=speeches,
        topic=config.DEFAULT_TOPIC,
        variants=_variants(),
        models=(model,),
        conditions=(run.CORE,),
        max_workers=max_workers,
        pool=POOL,
        max_tokens=max_tokens,
    )


def dry_run(plan: run.RunPlan, *, experiment: str, shard: int) -> dict:
    """Count cached vs would-run cells. Does not call the LLM."""
    done = run._experiment_done_keys(experiment, include_legacy=False)
    jobs = run._build_jobs(plan, done)
    stats = run.experiment_cell_stats(experiment)
    summary = {
        "experiment": experiment,
        "shard": shard,
        "n_speeches": int(plan.speeches["speech_id"].nunique()),
        "cells_cached_experiment": len(done),
        "cells_would_run": len(jobs),
        "experiment_rows": stats["n"],
        "experiment_parse_ok": stats["parse_ok"],
    }
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Full-corpus eligible pool + labelling.")
    ap.add_argument("--build-shards", action="store_true",
                    help="filter enriched parquet and write eligible_pool.parquet")
    ap.add_argument("--model", help="model_id from config (required to label/dry-run)")
    ap.add_argument("--shard", type=int, default=None,
                    help="shard index 0..7")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N speech_ids in the shard (smoke only)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print would-run counts; no LLM calls")
    ap.add_argument("--status", action="store_true",
                    help="stream counts for this model's corpus experiment and exit")
    ap.add_argument("--compact", action="store_true",
                    help="write the slim analysis parquet for this experiment and exit")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--rerun", action="store_true",
                    help="ignore the experiment cache and rewrite cells in this plan")
    args = ap.parse_args(argv)

    if args.build_shards:
        piece_df = build_shards()
        print(f"wrote {ELIGIBLE_POOL_PATH}")
        print(piece_df.groupby("shard_id")["n_est"].sum().to_string())
        return

    if not args.model:
        ap.error("pass --model (or --build-shards)")
    spec = config.MODELS_BY_ID.get(args.model)
    if spec is None:
        ap.error(f"unknown model {args.model!r}; known: "
                 f"{sorted(config.MODELS_BY_ID)}")
    experiment = experiment_for_model(spec.model_id)

    if args.status:
        stats = run.experiment_cell_stats(experiment)
        print(f"{experiment}: {stats['n']} rows, parse_ok {stats['parse_ok']}")
        return

    if args.compact:
        dest = config.DATA_DIR / f"{experiment}.parquet"
        run.compact_experiment_to_parquet(experiment, dest)
        print(f"wrote {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return

    if args.shard is None:
        ap.error("pass --shard 0..7")

    cols = list(POOL_COLUMNS)
    if args.dry_run:
        cols = [c for c in cols if c != "speech_text"]
    speeches = load_shard(args.shard, columns=cols, limit=args.limit)
    plan = corpus_plan(spec, speeches, max_workers=args.workers,
                       max_tokens=args.max_tokens)

    if args.dry_run:
        summary = dry_run(plan, experiment=experiment, shard=args.shard)
        for k, v in summary.items():
            print(f"{k}: {v}")
        return

    n = run.execute(
        plan, experiment=experiment, cli_args=vars(args),
        include_legacy_cache=False, rerun=args.rerun)
    print(f"wrote {n} new cells")


if __name__ == "__main__":
    main()
