"""Stratified pilot sampling over the enriched Parquet via DuckDB.

A robustness pilot must not be run on an arbitrary slice. We stratify on the
three dimensions most likely to interact with model behaviour:

    era       - debate language and topic salience shift over two centuries
    length    - very short vs long speeches stress the model differently
    presence  - the existing H&SC regex (seed) so the sample is not ~97%
                negatives; it is a *stratifier*, never a label

Within each stratum cell we draw a reproducible reservoir sample (DuckDB
``REPEATABLE``). Procedural text and near-empty speeches are excluded so every
sampled speech has substantive content to reason about.

The result (one row per speech, with ``speech_id`` and all strata fields) is
written to ``config.SAMPLE_PATH`` and is the fixed input to the grid runner.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from . import config
from .config import Topic

# Era buckets: roughly distinct regimes of parliamentary H&SC language.
# (label, lo_year_inclusive, hi_year_exclusive)
ERA_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("pre_1900", 1803, 1900),
    ("1900_1945", 1900, 1945),
    ("1945_1980", 1945, 1980),   # NHS founded 1948
    ("1980_2005", 1980, 2005),   # community-care reforms
    ("2005_2025", 2005, 2026),   # Care Act era
)

# Length tiers worth contrasting; ultra_short is excluded (too little content).
LENGTH_TIERS: tuple[str, ...] = ("short", "medium", "long")

MIN_WORDS = 40  # below this there is not enough text for a topic judgement


@dataclass
class SampleDesign:
    """How many speeches to draw per stratum cell, and the global filters."""

    per_cell_present: int = 12   # seed-positive speeches per (era x length) cell
    per_cell_absent: int = 6     # seed-negative speeches per (era x length) cell
    min_words: int = MIN_WORDS
    chambers: tuple[str, ...] = ("Commons", "Lords")
    seed: int = 20260629


def _enriched_path() -> str:
    p = config.DATA_DIR / "full_data_enriched.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Expected enriched parquet at {p}")
    return p.as_posix()


def _connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit = '4GB'")
    con.execute(
        f"CREATE OR REPLACE VIEW enriched AS "
        f"SELECT * FROM read_parquet('{_enriched_path()}')"
    )
    return con


def _era_case() -> str:
    whens = " ".join(
        f"WHEN year >= {lo} AND year < {hi} THEN '{label}'"
        for label, lo, hi in ERA_BUCKETS
    )
    return f"CASE {whens} ELSE NULL END"


def _build_meta_table(con: duckdb.DuckDBPyConnection,
                      design: SampleDesign, topic: Topic) -> None:
    """One regex pass over the Parquet -> a small in-memory metadata table
    (no speech_text), tagged with era and the seed-presence flag. This is the
    only expensive scan; all per-cell sampling then runs against this table.
    """
    chambers = ", ".join("'" + c + "'" for c in design.chambers)
    con.execute(
        f"""
        CREATE TEMP TABLE meta AS
        SELECT
            speech_id,
            {_era_case()}                                          AS era,
            speech_type,
            regexp_matches(lower(speech_text), '{topic.seed_regex}') AS seed_present
        FROM enriched
        WHERE speech_text IS NOT NULL
          AND NOT procedural
          AND word_count >= {design.min_words}
          AND speech_type IN ({", ".join("'" + t + "'" for t in LENGTH_TIERS)})
          AND chamber IN ({chambers})
          AND year IS NOT NULL
        """
    )


def draw_sample(
    design: SampleDesign | None = None,
    topic: Topic | None = None,
    *,
    write: bool = True,
) -> pd.DataFrame:
    """Build the stratified pilot sample. Returns the frame and (optionally)
    writes it to ``config.SAMPLE_PATH``.

    Two phases: (1) a single regex scan tags candidate rows with era + seed
    presence into a small metadata table; (2) per-cell reservoir sampling picks
    speech_ids, and full text is fetched only for the few hundred selected.

    Cells that cannot meet their target (e.g. seed-positive H&SC speeches before
    1900 are rare) simply contribute fewer rows; realised counts come from
    :func:`describe`.
    """
    design = design or SampleDesign()
    topic = topic or config.DEFAULT_TOPIC
    con = _connect()
    _build_meta_table(con, design, topic)

    # Population size of every stratum cell, for sampling weights: the draw is
    # deliberately non-proportional (seed-positives are over-sampled ~10x, eras
    # equalised), so any corpus-level rate computed from the sample MUST weight
    # rows by cell_pop / cell_n or the estimate inherits the design, not the
    # corpus.
    cell_pop = con.execute(
        """
        SELECT era, speech_type, seed_present, COUNT(*) AS cell_pop
        FROM meta GROUP BY era, speech_type, seed_present
        """
    ).df()

    id_frames: list[pd.DataFrame] = []
    for label, _lo, _hi in ERA_BUCKETS:
        for tier in LENGTH_TIERS:
            for present, n in ((True, design.per_cell_present),
                               (False, design.per_cell_absent)):
                # Sample the *filtered* subquery: DuckDB's USING SAMPLE binds to
                # the table expression and would otherwise sample before WHERE.
                ids = con.execute(
                    f"""
                    SELECT * FROM (
                        SELECT speech_id, era, seed_present FROM meta
                        WHERE era = '{label}' AND speech_type = '{tier}'
                          AND seed_present = {str(present).lower()}
                    ) USING SAMPLE reservoir({n} ROWS) REPEATABLE ({design.seed})
                    """
                ).df()
                id_frames.append(ids)

    picked = pd.concat(id_frames, ignore_index=True).drop_duplicates("speech_id")
    con.register("picked_ids", picked[["speech_id"]])

    # Fetch full rows (incl. text) only for the selected speeches.
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

    # Attach weights: cell population over realised cell sample size.
    cell_n = (df.groupby(["era", "speech_type", "seed_present"], dropna=False)
              .size().rename("cell_n").reset_index())
    df = (df.merge(cell_pop, on=["era", "speech_type", "seed_present"], how="left")
          .merge(cell_n, on=["era", "speech_type", "seed_present"], how="left"))
    df["sampling_weight"] = df["cell_pop"] / df["cell_n"]

    if write:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.SAMPLE_PATH, index=False)
        _write_sample_manifest(config.SAMPLE_PATH, design, topic, len(df))
    return df


def _write_sample_manifest(path, design: SampleDesign, topic: Topic,
                           n_rows: int) -> None:
    """Sidecar manifest: reservoir REPEATABLE draws are only reproducible on
    the same DuckDB version, so record it (plus seed and design) next to the
    parquet."""
    import duckdb as _duckdb
    from . import provenance
    provenance.write_manifest(path.parent, {
        "artifact": path.name,
        "n_rows": n_rows,
        "design": {"per_cell_present": design.per_cell_present,
                   "per_cell_absent": design.per_cell_absent,
                   "min_words": design.min_words,
                   "chambers": list(design.chambers),
                   "seed": design.seed},
        "topic": topic.name,
        "definition_id": topic.definition_id,
        "duckdb_version": _duckdb.__version__,
    }, filename=f"{path.stem}.manifest.json")


def describe(df: pd.DataFrame) -> pd.DataFrame:
    """Realised cell counts: era x length x seed_present."""
    return (
        df.groupby(["era", "speech_type", "seed_present"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["era", "speech_type", "seed_present"])
        .reset_index(drop=True)
    )


def load_sample() -> pd.DataFrame:
    if not config.SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"No sample at {config.SAMPLE_PATH}. Run sample.draw_sample() first."
        )
    return pd.read_parquet(config.SAMPLE_PATH)


# --------------------------------------------------------------------------
# Evaluation subset (embedder grid + LLM panel; Workstream C0)
# --------------------------------------------------------------------------
EVAL_SAMPLE_PATH = config.ARTIFACTS_DIR / "eval10k_sample.parquet"
EVAL_TARGET_N = 10_000
EVAL_FLOOR_PER_DECADE = 250
EVAL_SEED = 20260802


def load_eval_sample() -> pd.DataFrame:
    if not EVAL_SAMPLE_PATH.exists():
        raise FileNotFoundError(
            f"No eval subset at {EVAL_SAMPLE_PATH}. Run "
            f"`python -m hansard_llm.sample --eval-subset` first.")
    return pd.read_parquet(EVAL_SAMPLE_PATH)


def build_eval_subset(
    *,
    target_n: int = EVAL_TARGET_N,
    floor_per_decade: int = EVAL_FLOOR_PER_DECADE,
    seed: int = EVAL_SEED,
    write: bool = True,
) -> pd.DataFrame:
    """Decade-stratified evaluation subset, drawn WITHOUT the seed regex.

    Unlike the pilot (which over-samples seed-positives 2:1 and equalises
    eras), this is a random draw within each decade of an unenriched pool, so
    threshold/retention estimates transfer to the real corpus. Allocation is
    proportional to decade size with a floor (early decades are small; the
    floor keeps their per-decade recall estimates usable), take-all where a
    decade has fewer than the floor. ``sampling_weight`` = decade population /
    decade draw, for corpus-level rates.
    """
    con = _connect()
    # Same content hygiene as the pilot, minus any seed-regex involvement.
    chambers = ", ".join("'" + c + "'" for c in ("Commons", "Lords"))
    tiers = ", ".join("'" + t + "'" for t in LENGTH_TIERS)
    con.execute(
        f"""
        CREATE TEMP TABLE eval_meta AS
        SELECT speech_id, (year // 10) * 10 AS decade_bin
        FROM enriched
        WHERE speech_text IS NOT NULL
          AND NOT procedural
          AND word_count >= {MIN_WORDS}
          AND speech_type IN ({tiers})
          AND chamber IN ({chambers})
          AND year IS NOT NULL
        """
    )
    pop = con.execute(
        "SELECT decade_bin, COUNT(*) AS pop FROM eval_meta GROUP BY 1 ORDER BY 1"
    ).df()
    total_pop = int(pop["pop"].sum())

    frames = []
    for r in pop.itertuples():
        proportional = round(target_n * r.pop / total_pop)
        n = min(int(r.pop), max(floor_per_decade, proportional))
        ids = con.execute(
            f"""
            SELECT * FROM (
                SELECT speech_id, decade_bin FROM eval_meta
                WHERE decade_bin = {int(r.decade_bin)}
            ) USING SAMPLE reservoir({n} ROWS) REPEATABLE ({seed})
            """
        ).df()
        ids["decade_pop"] = int(r.pop)
        frames.append(ids)
    picked = pd.concat(frames, ignore_index=True).drop_duplicates("speech_id")

    con.register("picked_ids", picked[["speech_id"]])
    df = con.execute(
        """
        SELECT e.speech_id, e.year, e.decade, e.chamber, e.speech_type,
               e.word_count, e.section_title, e.speech_text
        FROM enriched e
        JOIN picked_ids p USING (speech_id)
        """
    ).df()
    con.close()
    df = (df.drop_duplicates(subset="speech_id")
          .merge(picked, on="speech_id", how="left")
          .reset_index(drop=True))
    decade_n = df.groupby("decade_bin")["speech_id"].transform("count")
    df["sampling_weight"] = df["decade_pop"] / decade_n

    if write:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(EVAL_SAMPLE_PATH, index=False)
        _write_sample_manifest(EVAL_SAMPLE_PATH,
                               SampleDesign(seed=seed), config.DEFAULT_TOPIC,
                               len(df))
    return df


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Draw samples.")
    ap.add_argument("--eval-subset", action="store_true",
                    help="draw the decade-stratified ~10k evaluation subset "
                         "(no seed regex) instead of the pilot sample")
    args = ap.parse_args()

    if args.eval_subset:
        d = build_eval_subset()
        print(f"Sampled {len(d)} speeches -> {EVAL_SAMPLE_PATH}")
        print(d.groupby("decade_bin")
              .agg(n=("speech_id", "count"), pop=("decade_pop", "first"),
                   weight=("sampling_weight", "first"))
              .to_string())
    else:
        d = draw_sample()
        print(f"Sampled {len(d)} speeches -> {config.SAMPLE_PATH}")
        print(describe(d).to_string(index=False))
