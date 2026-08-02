"""One-off: annotate the frozen legacy results log with pool provenance.

Reads the frozen JSONL (reparsed with the current schema), reconstructs which
population each row came from (pilot sample vs retrieval spot-check filter
pool), and writes ``legacy/pilot_results_annotated.parquet``. Prints the pool
breakdown so the known contamination (270 pilot + 150 filter-pool speeches in
the expert_hc_sc/v1_nocap/json stratum) is visible and checkable.

Usage:  python scripts/annotate_legacy_log.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hansard_llm import config, run  # noqa: E402


def main() -> None:
    df = run.load_legacy(write_annotated=True)
    print(f"annotated {len(df)} rows -> {config.LEGACY_ANNOTATED_PATH}")
    print("\nrows by pool:")
    print(df.groupby("pool").size().to_string())
    strat = df[(df["condition"] == "temp0") & (df["role"] == "none")
               & (df["task"] == "v1_nocap")
               & (df["definition"] == "expert_hc_sc")
               & (df["output_format"] == "json")]
    print("\nexpert_hc_sc / v1_nocap / json stratum (the contaminated one):")
    print(strat.groupby("pool")["speech_id"].nunique().rename("n_speeches")
          .to_string())
    n_unknown = int((df["pool"] == "unknown").sum())
    if n_unknown:
        print(f"\nWARNING: {n_unknown} rows in neither the pilot sample nor "
              f"the spot-check id list — investigate before analysing them.")


if __name__ == "__main__":
    main()
