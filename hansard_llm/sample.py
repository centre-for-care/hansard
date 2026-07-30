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
    if write:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.SAMPLE_PATH, index=False)
    return df


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


if __name__ == "__main__":
    d = draw_sample()
    print(f"Sampled {len(d)} speeches -> {config.SAMPLE_PATH}")
    print(describe(d).to_string(index=False))
