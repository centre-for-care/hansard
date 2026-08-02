"""Streaming parity check for the monolithic full_data.json (12.8 GiB).

The file is ONE json object: {"<sql query>": [ 9.2M row objects ]} — too big
for any document parser. This scans it in 64MB chunks, counts speech_id
occurrences, and collects every Nth speech_id; those are then checked for
existence in full_data.parquet via DuckDB.

Combined with verify_parquet_parity.py's `base` check (parquet == enriched on
rows/columns/sampled content), a PASS here establishes the chain
json -> parquet -> enriched, clearing the json for deletion.

Usage:  python scripts/verify_json_streaming.py
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import duckdb

DATA = Path(r"C:\Users\maksimz\Desktop\hansard_eda\data")
JSON_PATH = DATA / "full_data.json"
PARQUET_PATH = DATA / "full_data.parquet"

CHUNK = 64 * 1024 * 1024
KEEP_EVERY = 25_000   # sample every Nth id -> ~370 ids over 9.2M
_ID_RE = re.compile(rb'"speech_id"\s*:\s*(-?\d+)')


def scan() -> tuple[int, list[int]]:
    n = 0
    sample: list[int] = []
    carry = b""
    t0 = time.time()
    size = JSON_PATH.stat().st_size
    read = 0
    with JSON_PATH.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                # flush: scan whatever tail is left
                for m in _ID_RE.finditer(carry):
                    if n % KEEP_EVERY == 0:
                        sample.append(int(m.group(1)))
                    n += 1
                break
            buf = carry + block
            # hold back a tail so a "speech_id": 123 split across the chunk
            # boundary is neither missed nor double-counted
            scan_to = len(buf) - 64
            for m in _ID_RE.finditer(buf, 0, scan_to):
                if n % KEEP_EVERY == 0:
                    sample.append(int(m.group(1)))
                n += 1
            carry = buf[scan_to:]
            read += len(block)
            if read % (CHUNK * 16) < CHUNK:
                rate = read / (time.time() - t0) / 1e6
                print(f"  {read / 1e9:.1f}/{size / 1e9:.1f} GB "
                      f"({rate:.0f} MB/s), {n:,} ids", flush=True)
    return n, sample


def main() -> None:
    print(f"scanning {JSON_PATH.name}…")
    n_json, sample = scan()
    print(f"json speech_id occurrences: {n_json:,} "
          f"({len(sample)} sampled for membership)")

    con = duckdb.connect(":memory:")
    con.execute("SET memory_limit='6GB'")
    n_parquet = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{PARQUET_PATH.as_posix()}')"
    ).fetchone()[0]

    import pandas as pd
    con.register("sample_ids", pd.DataFrame({"speech_id": sample}))
    missing = con.execute(f"""
        SELECT COUNT(*) FROM sample_ids s
        WHERE NOT EXISTS (
            SELECT 1 FROM read_parquet('{PARQUET_PATH.as_posix()}') p
            WHERE p.speech_id = s.speech_id)
    """).fetchone()[0]

    ok_count = n_json == n_parquet
    ok_member = missing == 0
    print(f"  [{'PASS' if ok_count else 'FAIL'}] row counts: "
          f"json={n_json:,} parquet={n_parquet:,}")
    print(f"  [{'PASS' if ok_member else 'FAIL'}] sampled id membership: "
          f"{missing} of {len(sample)} missing from parquet")
    if not (ok_count and ok_member):
        print("do NOT delete full_data.json")
        sys.exit(1)
    print("all checks passed — full_data.json is safe to delete.")


if __name__ == "__main__":
    main()
