"""Pre-deletion checks for the superseded full_data files in hansard_eda.

Two gates, run before each pre-approved deletion:

  json    full_data.json vs full_data.parquet — same rows/columns/content?
          (the JSON was the pre-conversion source; 12.8 GiB)
  base    full_data.parquet vs full_data_enriched.parquet — is enriched a
          strict column superset with the same speech ids? (4.9 GiB)

Prints PASS/FAIL per check and exits non-zero on any FAIL. Content check
compares md5(speech_text) on a deterministic sample of speech_ids.

Usage:  python scripts/verify_parquet_parity.py [json|base|all]
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

DATA = Path(r"C:\Users\maksimz\Desktop\hansard_eda\data")
JSON_PATH = DATA / "full_data.json"
PARQUET_PATH = DATA / "full_data.parquet"
ENRICHED_PATH = DATA / "full_data_enriched.parquet"

SAMPLE_N = 2000
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def _cols(con, rel: str) -> dict[str, str]:
    return {r[0]: r[1] for r in
            con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()}


def compare_json_vs_parquet(con) -> None:
    print(f"\n== full_data.json vs full_data.parquet")
    con.execute(
        f"CREATE OR REPLACE VIEW j AS SELECT * FROM "
        f"read_json_auto('{JSON_PATH.as_posix()}', maximum_object_size=134217728)")
    con.execute(
        f"CREATE OR REPLACE VIEW p AS SELECT * FROM "
        f"read_parquet('{PARQUET_PATH.as_posix()}')")

    jc, pc = _cols(con, "j"), _cols(con, "p")
    check("columns match", set(jc) == set(pc),
          f"json-only={sorted(set(jc) - set(pc))} parquet-only={sorted(set(pc) - set(jc))}")

    nj = con.execute("SELECT COUNT(*) FROM j").fetchone()[0]
    np_ = con.execute("SELECT COUNT(*) FROM p").fetchone()[0]
    check("row counts equal", nj == np_, f"json={nj:,} parquet={np_:,}")

    dj = con.execute("SELECT COUNT(DISTINCT speech_id) FROM j").fetchone()[0]
    dp = con.execute("SELECT COUNT(DISTINCT speech_id) FROM p").fetchone()[0]
    check("distinct speech_id equal", dj == dp, f"json={dj:,} parquet={dp:,}")

    mismatch = con.execute(f"""
        WITH ids AS (
            SELECT speech_id FROM p USING SAMPLE reservoir({SAMPLE_N} ROWS) REPEATABLE (7)
        ),
        a AS (SELECT j.speech_id, md5(COALESCE(j.speech_text,'')) h
              FROM j JOIN ids USING (speech_id)),
        b AS (SELECT p.speech_id, md5(COALESCE(p.speech_text,'')) h
              FROM p JOIN ids USING (speech_id))
        SELECT COUNT(*) FROM a JOIN b USING (speech_id) WHERE a.h <> b.h
    """).fetchone()[0]
    check(f"speech_text md5 on {SAMPLE_N}-row sample", mismatch == 0,
          f"{mismatch} mismatches")


def compare_base_vs_enriched(con) -> None:
    print(f"\n== full_data.parquet vs full_data_enriched.parquet")
    con.execute(f"CREATE OR REPLACE VIEW p AS SELECT * FROM "
                f"read_parquet('{PARQUET_PATH.as_posix()}')")
    con.execute(f"CREATE OR REPLACE VIEW e AS SELECT * FROM "
                f"read_parquet('{ENRICHED_PATH.as_posix()}')")

    pc, ec = _cols(con, "p"), _cols(con, "e")
    missing = sorted(set(pc) - set(ec))
    check("enriched is a column superset", not missing, f"missing={missing}")
    print(f"       enriched adds: {sorted(set(ec) - set(pc))}")

    np_ = con.execute("SELECT COUNT(*) FROM p").fetchone()[0]
    ne = con.execute("SELECT COUNT(*) FROM e").fetchone()[0]
    check("row counts equal", np_ == ne, f"base={np_:,} enriched={ne:,}")

    only_p = con.execute(
        "SELECT COUNT(*) FROM (SELECT speech_id FROM p EXCEPT "
        "SELECT speech_id FROM e)").fetchone()[0]
    check("no speech_id lost in enriched", only_p == 0, f"{only_p} lost")

    mismatch = con.execute(f"""
        WITH ids AS (
            SELECT speech_id FROM p USING SAMPLE reservoir({SAMPLE_N} ROWS) REPEATABLE (7)
        ),
        a AS (SELECT p.speech_id, md5(COALESCE(p.speech_text,'')) h
              FROM p JOIN ids USING (speech_id)),
        b AS (SELECT e.speech_id, md5(COALESCE(e.speech_text,'')) h
              FROM e JOIN ids USING (speech_id))
        SELECT COUNT(*) FROM a JOIN b USING (speech_id) WHERE a.h <> b.h
    """).fetchone()[0]
    check(f"speech_text md5 on {SAMPLE_N}-row sample", mismatch == 0,
          f"{mismatch} mismatches")


def main() -> None:
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit='8GB'")
    con.execute(f"SET temp_directory='{(DATA.parent / 'duckdb_tmp').as_posix()}'")
    if what in ("json", "all"):
        compare_json_vs_parquet(con)
    if what in ("base", "all"):
        compare_base_vs_enriched(con)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED — do NOT delete anything.")
        sys.exit(1)
    print("all checks passed.")


if __name__ == "__main__":
    main()
