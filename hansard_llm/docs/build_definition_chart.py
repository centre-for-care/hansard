"""Figure 3 for the brief: presence rate by era under the two definitions.

Neither definition is styled as the correct one (contrast with the cap figure,
where grey deliberately marks the artefactual arm): the point of this chart is
that the choice moves the era profile, not that one arm is right.

Usage:  python -m hansard_llm.docs.build_definition_chart
Writes: hansard_llm/docs/fig_definition_era.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hansard_llm import run, sample

OUT = Path(__file__).resolve().parent / "fig_definition_era.png"

CUR = "#B98F4E"   # warm ochre  : current definition
ERA = "#3B6E8A"   # cool blue   : era-neutral definition
INK, MUTED = "#1c1c1c", "#6b6b6b"
ERAS = ["pre-1900", "1900-1947", "post-1948"]


def main() -> None:
    df = run.load_results(to_parquet=False)
    sel = df[(df["condition"] == "temp0") & (df["role"] == "none")
             & (df["task"] == "v1_nocap")
             & (df["definition"].isin(["current", "era_neutral"]))
             & (df["output_format"].isin(["json", "free"]))].copy()
    yr = sample.load_sample().set_index("speech_id")["year"]
    sel["year"] = sel["speech_id"].map(yr)
    sel["era"] = pd.cut(sel["year"].astype(int), [0, 1900, 1948, 2100], labels=ERAS)

    piv = sel.pivot_table(index="era", columns="definition",
                          values="mentions_topic", aggfunc="mean",
                          observed=True).reindex(ERAS) * 100

    x, w = np.arange(len(ERAS)), 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.0), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    b1 = ax.bar(x - w/2, piv["current"], w, color=CUR, zorder=3,
                label="Current definition (names the NHS)")
    b2 = ax.bar(x + w/2, piv["era_neutral"], w, color=ERA, zorder=3,
                label="Era-neutral definition")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.2,
                    f"{r.get_height():.0f}%", ha="center", va="bottom",
                    fontsize=8.5, color=MUTED)

    ax.set_xticks(x); ax.set_xticklabels(ERAS, fontsize=9.5, color=INK)
    ax.set_ylabel("Speeches judged H&SC (%)", fontsize=9.5, color=MUTED)
    ax.set_ylim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#cccccc"); ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#ededed", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")

    grad = lambda c: piv.loc["post-1948", c] - piv.loc["pre-1900", c]
    print(f"wrote {OUT}")
    print(piv.round(1))
    print(f"gradient current {grad('current'):+.1f}pp, "
          f"era_neutral {grad('era_neutral'):+.1f}pp")


if __name__ == "__main__":
    main()
