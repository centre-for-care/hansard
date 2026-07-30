"""Figure for the brief: presence rate by era under construct definitions.

Shows the original current / era-neutral contrast together with the two expert
order variants, so the reader can see that expert wording tracks current on the
era profile while era-neutral flattens the slope.

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

DEFS = ("current", "expert_hc_sc", "expert_sc_hc", "era_neutral")
COLORS = {
    "current": "#B98F4E",
    "expert_hc_sc": "#6B8F71",
    "expert_sc_hc": "#8FAE95",
    "era_neutral": "#3B6E8A",
}
LABELS = {
    "current": "Current (names the NHS)",
    "expert_hc_sc": "Expert HC→SC",
    "expert_sc_hc": "Expert SC→HC",
    "era_neutral": "Era-neutral",
}
INK, MUTED = "#1c1c1c", "#6b6b6b"
ERAS = ["pre-1900", "1900-1947", "post-1948"]


def main() -> None:
    df = run.load_results(to_parquet=False)
    sel = df[(df["condition"] == "temp0") & (df["role"] == "none")
             & (df["task"] == "v1_nocap")
             & (df["definition"].isin(DEFS))
             & (df["output_format"].isin(["json", "free"]))].copy()
    yr = sample.load_sample().set_index("speech_id")["year"]
    sel["year"] = sel["speech_id"].map(yr)
    sel["era"] = pd.cut(sel["year"].astype(int), [0, 1900, 1948, 2100], labels=ERAS)

    piv = sel.pivot_table(index="era", columns="definition",
                          values="mentions_topic", aggfunc="mean",
                          observed=True).reindex(ERAS)[list(DEFS)] * 100

    n = len(DEFS)
    x = np.arange(len(ERAS))
    w = 0.18
    offsets = (np.arange(n) - (n - 1) / 2) * w
    fig, ax = plt.subplots(figsize=(9.4, 4.2), dpi=200)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    for i, d in enumerate(DEFS):
        bars = ax.bar(x + offsets[i], piv[d], w, color=COLORS[d], zorder=3,
                      label=LABELS[d])
        for r in bars:
            ax.text(r.get_x() + r.get_width()/2, r.get_height() + 1.0,
                    f"{r.get_height():.0f}", ha="center", va="bottom",
                    fontsize=7.5, color=MUTED)

    ax.set_xticks(x); ax.set_xticklabels(ERAS, fontsize=9.5, color=INK)
    ax.set_ylabel("Speeches judged H&SC (%)", fontsize=9.5, color=MUTED)
    ax.set_ylim(0, 100)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#cccccc"); ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors=MUTED, length=0, labelsize=8.5)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#ededed", linewidth=0.8, zorder=0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")

    print(f"wrote {OUT}")
    print(piv.round(1))
    for d in DEFS:
        g = piv.loc["post-1948", d] - piv.loc["pre-1900", d]
        print(f"gradient {d} {g:+.1f}pp")


if __name__ == "__main__":
    main()
