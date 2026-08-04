"""The unified LLM panel (Workstream C2): one run, three uses.

Every panel cell asks the shipping-default extraction question (uncapped, JSON,
role=none, temp 0) of an eval-subset speech under one of two definitions
(``config.PANEL_DEFINITIONS``). From the same rows we get:

1. **Model sensitivity** — pairwise agreement / alpha across models (one cell
   per model, so raters are independent), size vs family via the extended
   axis on a 2k subsample.
2. **Retrieval gold** — majority vote with leave-one-out discipline
   (:func:`panel_gold`): scoring retrieval under definition *d* excludes *d*'s
   cells; evaluating model *m* against gold excludes *m*'s cells. Expert-
   permutation queries are automatically non-circular (the panel never uses
   expert definitions).
3. **Sub-topic comparison** — the same rows carry themes for cross-model
   taxonomy work.

Riders: a temp-0 repeat condition measures the (claimed, never verified)
determinism of served models; a Nebius-vs-vLLM parity slice measures
serving-stack sensitivity.

Run on the cluster (see cluster/run_grid.sbatch), one model per job::

    python -m hansard_llm.panel --model Qwen/Qwen3-32B
    python -m hansard_llm.panel --model Qwen/Qwen3-14B --extended
    python -m hansard_llm.panel --model Qwen/Qwen3-32B --determinism
"""

from __future__ import annotations

import argparse

import pandas as pd

from . import config, run, sample
from .config import ModelSpec
from .prompts import TASK_UNCAPPED, build_definition_variants

EXPERIMENT = "panel10k"
EXPERIMENT_EXT = "panel_extended2k"
EXPERIMENT_DET = "panel_determinism"

EXTENDED_N = 2000       # stratified head of the eval subset for the size axis
DETERMINISM_N = 200     # speeches for the temp-0 repeat check
DETERMINISM_REPS = 3

# Temp-0 with repetitions: if serving were deterministic these would be
# byte-identical; measured disagreement IS the serving noise floor.
DET_CONDITION = run.Condition(temperature=0.0, seed=42,
                              n_reps=DETERMINISM_REPS, label="temp0rep")


def _eval_speeches(n: int | None = None) -> pd.DataFrame:
    """Eval-subset speeches; if ``n`` is set, a deterministic decade-stratified head."""
    df = sample.load_eval_sample()
    if n is not None:
        # deterministic stratified head: sort by (decade, speech_id) then take
        # every k-th row so all decades stay represented
        df = (df.sort_values(["decade_bin", "speech_id"])
              .reset_index(drop=True))
        step = max(1, len(df) // n)
        df = df.iloc[::step].head(n).copy()
    return df


def _variants():
    """Prompt variants: the panel definitions under role=none / JSON / uncapped."""
    topics = [config.HSC_DEFINITIONS[d] for d in config.PANEL_DEFINITIONS]
    return build_definition_variants(
        topics, roles=("none",), formats=("json",), task=TASK_UNCAPPED)


def panel_plan(model: ModelSpec, *, extended: bool = False,
               max_workers: int = 32) -> run.RunPlan:
    """RunPlan for one model on the full eval subset (or the 2k extended slice)."""
    return run.RunPlan(
        speeches=_eval_speeches(EXTENDED_N if extended else None),
        topic=config.DEFAULT_TOPIC,
        variants=_variants(),
        models=(model,),
        conditions=(run.CORE,),
        max_workers=max_workers,
        pool="eval10k",
    )


def determinism_plan(model: ModelSpec, *, max_workers: int = 32) -> run.RunPlan:
    """RunPlan for the temp-0 repeat rider (one definition, DETERMINISM_N speeches)."""
    return run.RunPlan(
        speeches=_eval_speeches(DETERMINISM_N),
        topic=config.DEFAULT_TOPIC,
        variants=_variants()[:1],   # one definition suffices for the rider
        models=(model,),
        conditions=(DET_CONDITION,),
        max_workers=max_workers,
        pool="eval10k",
    )


# --------------------------------------------------------------------------
# Gold with leave-one-out discipline
# --------------------------------------------------------------------------
def load_panel(*, include_extended: bool = False) -> pd.DataFrame:
    """Load the panel run, optionally concatenating the extended run if it exists."""
    frames = [run.load_experiment(EXPERIMENT)]
    if include_extended:
        try:
            frames.append(run.load_experiment(EXPERIMENT_EXT))
        except FileNotFoundError:
            pass
    return pd.concat(frames, ignore_index=True)


def panel_gold(
    df: pd.DataFrame | None = None,
    *,
    exclude_definition: str | None = None,
    exclude_model: str | None = None,
    min_votes: int = 3,
) -> pd.DataFrame:
    """Majority ``mentions_topic`` per speech from panel cells.

    ``exclude_definition``: pass the retrieval query's definition when scoring
    retrieval (leave-one-definition-out); expert queries need no exclusion.
    ``exclude_model``: pass the evaluated model when scoring a model against
    gold (leave-one-model-out).
    Rows with parser-inferred presence are never counted as votes.
    """
    df = df if df is not None else load_panel()
    sel = df[(df["condition"] == "temp0")
             & (df["output_format"] == "json")
             & (df["parse_ok"])
             & (~df.get("presence_inferred", False).astype(bool))]
    if exclude_definition is not None:
        sel = sel[sel["definition"] != exclude_definition]
    if exclude_model is not None:
        sel = sel[sel["model_id"] != exclude_model]
    g = (sel.groupby("speech_id")["mentions_topic"]
         .agg(rate="mean", n="count")
         .reset_index())
    g = g[g["n"] >= min_votes]
    g["label"] = (g["rate"] >= 0.5).astype(int)
    return g


def model_agreement(df: pd.DataFrame | None = None,
                    *, definition: str = "current") -> pd.DataFrame:
    """Pairwise model agreement on presence, one cell per model (independent
    raters — unlike the pilot's 32-correlated-cells alpha)."""
    from itertools import combinations
    df = df if df is not None else load_panel(include_extended=True)
    sel = df[(df["condition"] == "temp0") & (df["definition"] == definition)
             & (df["parse_ok"])]
    wide = sel.pivot_table(index="speech_id", columns="model_id",
                           values="mentions_topic", aggfunc="first")
    rows = []
    for a, b in combinations(wide.columns, 2):
        pair = wide[[a, b]].dropna()
        if len(pair) < 30:
            continue
        rows.append({"model_a": a, "model_b": b, "n": len(pair),
                     "agreement": round(float((pair[a] == pair[b]).mean()), 4)})
    return pd.DataFrame(rows).sort_values("agreement")


def determinism_report(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Byte- and label-level agreement across temp-0 repetitions per model."""
    df = df if df is not None else run.load_experiment(EXPERIMENT_DET)
    sel = df[df["condition"] == DET_CONDITION.label]
    rows = []
    for mid, g in sel.groupby("model_id"):
        per_speech = g.groupby(["speech_id", "prompt_hash"])
        n_cells = 0
        byte_same = 0
        label_same = 0
        for _, cell in per_speech:
            if len(cell) < 2:
                continue
            n_cells += 1
            byte_same += int(cell["raw_text"].nunique(dropna=False) == 1)
            label_same += int(cell["mentions_topic"].nunique(dropna=False) == 1)
        if n_cells:
            rows.append({"model_id": mid, "n_speeches": n_cells,
                         "byte_identical": round(byte_same / n_cells, 4),
                         "label_stable": round(label_same / n_cells, 4)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """CLI entry point: run panel / extended / determinism for one model, or --report."""
    ap = argparse.ArgumentParser(description="Unified LLM panel")
    ap.add_argument("--model", help="model_id from config (panel or extended)")
    ap.add_argument("--extended", action="store_true",
                    help="run the 2k extended-axis slice instead of the full "
                         "eval subset")
    ap.add_argument("--determinism", action="store_true",
                    help="run the temp-0 repeat rider instead of the panel")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--report", action="store_true",
                    help="print agreement + determinism reports from stored "
                         "runs and exit")
    args = ap.parse_args(argv)

    if args.report:
        try:
            print("\n== pairwise model agreement (definition=current)")
            print(model_agreement().to_string(index=False))
        except FileNotFoundError:
            print("no panel runs yet")
        try:
            print("\n== temp-0 determinism")
            print(determinism_report().to_string(index=False))
        except FileNotFoundError:
            print("no determinism runs yet")
        return

    if not args.model:
        ap.error("pass --model or --report")
    spec = config.MODELS_BY_ID.get(args.model)
    if spec is None:
        ap.error(f"unknown model {args.model!r}; known: "
                 f"{sorted(config.MODELS_BY_ID)}")

    if args.determinism:
        plan, experiment = determinism_plan(spec, max_workers=args.workers), EXPERIMENT_DET
    elif args.extended:
        plan, experiment = panel_plan(spec, extended=True,
                                      max_workers=args.workers), EXPERIMENT_EXT
    else:
        plan, experiment = panel_plan(spec, max_workers=args.workers), EXPERIMENT
    n = run.execute(plan, experiment=experiment, cli_args=vars(args))
    print(f"wrote {n} new cells")


if __name__ == "__main__":
    main()
