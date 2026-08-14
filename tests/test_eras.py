"""Era boundary tests — the exact off-by-one the audit found (1900 and 1948
counted on the wrong side of their boundaries) must never come back."""

import numpy as np
import pandas as pd

from hansard_llm.eras import ERA_LABELS, era_of


def test_boundary_years():
    got = era_of(pd.Series([1899, 1900, 1947, 1948]))
    assert list(got) == ["pre-1900", "1900-1947", "1900-1947", "post-1948"]


def test_nhs_founding_year_is_post_1948():
    assert era_of(pd.Series([1948]))[0] == "post-1948"


def test_range_and_labels():
    got = era_of(pd.Series([1803, 2025]))
    assert list(got) == ["pre-1900", "post-1948"]
    assert list(got.cat.categories) == ERA_LABELS


def test_nan_year_stays_nan():
    got = era_of(pd.Series([1950.0, np.nan, None]))
    assert got[0] == "post-1948"
    assert pd.isna(got[1]) and pd.isna(got[2])
