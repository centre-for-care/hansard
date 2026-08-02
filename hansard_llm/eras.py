"""Canonical era binning for analysis and figures.

One definition, used everywhere, so a boundary year can never land on
different sides in different charts. Bins are left-closed / right-open
(``[lo, hi)``): a speech from 1900 is in "1900-1947", and a speech from 1948 —
the NHS founding year — is in "post-1948". The legacy ad-hoc ``pd.cut`` with
pandas' default ``right=True`` had both boundary years on the wrong side.

Note these three *analysis* eras are coarser than the five *sampling* era
buckets in ``sample.ERA_BUCKETS``; the sampling buckets stratify draws, these
bins report results.
"""

from __future__ import annotations

import pandas as pd

# (label, lo_inclusive, hi_exclusive)
ANALYSIS_ERAS: tuple[tuple[str, int, int], ...] = (
    ("pre-1900", 0, 1900),
    ("1900-1947", 1900, 1948),
    ("post-1948", 1948, 2101),
)

ERA_LABELS: list[str] = [label for label, _, _ in ANALYSIS_ERAS]

_EDGES: list[int] = [lo for _, lo, _ in ANALYSIS_ERAS] + [ANALYSIS_ERAS[-1][2]]


def era_of(years: pd.Series) -> pd.Series:
    """Map a year Series to analysis-era labels (left-closed bins).

    NaN years map to NaN rather than raising, so callers joining year from a
    partial lookup (e.g. a sample parquet that does not cover every speech in
    a results frame) degrade gracefully instead of dying on ``astype(int)``.
    """
    return pd.cut(pd.to_numeric(years, errors="coerce"),
                  bins=_EDGES, labels=ERA_LABELS, right=False)
