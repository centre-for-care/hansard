"""C3: settle free-text vs JSON with the fixed parser — no new model calls.

The brief attributed a ~12pp presence gap (free 55% vs JSON 43%) entirely to
prompt design. The audit showed ~half of it was the parser's bullets->presence
fallback. This reparses the frozen legacy log with the flagged parser and
reports the gap three ways:

  raw            free positives as parsed (fallback counted) — the old number
  model-stated   excluding parser-inferred presence — the honest prompt effect
  sensitivity    inferred rows counted as positive (upper bound)

Plus the recovered negative-verdict themes and parse-failure asymmetry, i.e.
the actual costs of the free format.

Usage:  python scripts/analyze_format_effect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hansard_llm import run  # noqa: E402


def rate(df, num_mask=None) -> tuple[int, int, float]:
    ok = df[df["parse_ok"]]
    pos = ok if num_mask is None else ok[num_mask.reindex(ok.index, fill_value=False)]
    n_pos = int(pos["mentions_topic"].fillna(False).astype(bool).sum()
                if num_mask is None else len(pos))
    return n_pos, len(ok), n_pos / len(ok) if len(ok) else float("nan")


def main() -> None:
    df = run.load_legacy()
    core = df[(df["pool"] == "pilot") & (df["condition"] == "temp0")
              & (df["definition"] == "current")
              & (df["task"].isin(["v1", "v2"]))].copy()

    print(f"core grid rows: {len(core)} "
          f"({core['speech_id'].nunique()} speeches, "
          f"{core['model_id'].nunique()} models, "
          f"tasks v1+v2, roles both, definition=current)\n")

    out = []
    for fmt, g in core.groupby("output_format"):
        ok = g[g["parse_ok"]].copy()
        mt = ok["mentions_topic"].fillna(False).astype(bool)
        inferred = ok["presence_inferred"].fillna(False).astype(bool)
        raw_rate = mt.mean()
        stated_rate = (mt & ~inferred).sum() / max((~inferred).sum(), 1)
        themes_neg = int(((~mt) & (ok["subthemes_raw"].str.len() > 0)).sum())
        out.append({
            "format": fmt,
            "n_rows": len(g),
            "parse_fail": round(1 - len(ok) / len(g), 4),
            "positive_raw": round(float(raw_rate), 4),
            "n_inferred_pos": int((mt & inferred).sum()),
            "positive_model_stated": round(float(stated_rate), 4),
            "themes_under_neg_verdict": themes_neg,
        })
    import pandas as pd
    tab = pd.DataFrame(out).set_index("format")
    print(tab.to_string())

    gap_raw = tab.loc["free", "positive_raw"] - tab.loc["json", "positive_raw"]
    gap_stated = (tab.loc["free", "positive_model_stated"]
                  - tab.loc["json", "positive_model_stated"])
    print(f"\nfree - json presence gap:")
    print(f"  raw (old accounting)          {100 * gap_raw:+.1f}pp")
    print(f"  model-stated only (honest)    {100 * gap_stated:+.1f}pp")
    print(f"  parser fallback accounts for  "
          f"{100 * (gap_raw - gap_stated):.1f}pp of the reported gap")


if __name__ == "__main__":
    main()
