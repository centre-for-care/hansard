"""Idempotent grid runner + result store.

Executes the experimental grid

    speech  x  prompt-variant  x  model  x  condition(temperature, seed, rep)

against the LLM, parses each response, and records one fully-provenanced row
per call. Two design properties make it safe to run, kill, and resume:

    * Append-only JSONL log is the source of truth. Each completed cell is
      flushed immediately, so an interrupted run loses nothing.
    * Idempotent cache. Cells already present in the log (keyed by
      speech_id|prompt_hash|model|temperature|seed|rep) are skipped, so re-runs
      never re-bill completed work and partial runs converge.

Conditions separate the two questions the pilot asks:
    * temp 0 (1 rep)      - deterministic; isolates prompt/model sensitivity.
    * temp > 0 (N reps)   - the self-consistency baseline (run-to-run variance).
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import config, provenance, schema
from .client import LLMClient
from .config import ModelSpec, Topic
from .prompts import (PromptVariant, TASK_UNCAPPED, build_definition_variants,
                      build_uncapped_variants, build_variants)

# --- No-cap experiment (see prompts.TASK_UNCAPPED) ---
# When a variant drops the "at most N sub-topics" instruction we must widen two
# things that would otherwise silently re-impose the cap: the parse-time cap
# (normalize_themes truncates to this) and the completion token budget (long
# lists would hit the 512 default and truncate mid-answer). Both are applied
# only to the uncapped arm, keyed off the task label so fresh parses and
# reparses of the shared log stay consistent.
UNCAPPED_MAX_SUBTHEMES = 50   # generous ceiling: effectively "no cap", but bounds pathological output
UNCAPPED_MAX_TOKENS = 1024    # up from the 512 default so long lists are not cut off

# The pre-provenance single log, frozen 2026-08-02. Read-only: new runs write
# under runs/<experiment>/<run_id>/ instead (see execute / provenance.py).
RESULTS_LOG = config.LEGACY_RESULTS_LOG


@dataclass(frozen=True)
class Condition:
    """A temperature/seed setting and how many repetitions to draw."""

    temperature: float
    seed: int | None
    n_reps: int
    label: str


# Default conditions: deterministic core grid + a stochastic self-consistency
# probe at temp 0.7.
CORE = Condition(temperature=0.0, seed=42, n_reps=1, label="temp0")
SELFCONSISTENCY = Condition(temperature=0.7, seed=None, n_reps=3, label="temp07")


@dataclass
class RunPlan:
    """What to execute. Defaults to the full core grid (temp 0) only; the
    self-consistency probe is opt-in and typically run on a subset.

    ``pool`` labels which speech population the plan draws from (``pilot``,
    ``filter_pool``, ``eval10k``, …) and is stamped on every row, so different
    populations can never again be pooled silently under identical labels.
    """

    speeches: pd.DataFrame
    topic: Topic
    variants: list[PromptVariant]
    models: tuple[ModelSpec, ...]
    conditions: tuple[Condition, ...] = (CORE,)
    max_workers: int = 16
    pool: str = "pilot"


def _cache_key(speech_id, prompt_hash, model_id, temperature, seed, rep) -> str:
    return f"{speech_id}|{prompt_hash}|{model_id}|{temperature}|{seed}|{rep}"


def _load_done_keys(log_path: Path) -> set[str]:
    if not log_path.exists():
        return set()
    keys: set[str] = set()
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add(_cache_key(r["speech_id"], r["prompt_hash"], r["model_id"],
                                r["temperature"], r["seed"], r["rep"]))
    return keys


def _experiment_logs(experiment: str) -> list[Path]:
    """All results files already written for an experiment, oldest first."""
    exp_dir = config.RUNS_DIR / experiment
    if not exp_dir.exists():
        return []
    return sorted(exp_dir.glob("*/results.jsonl"))


def _experiment_done_keys(experiment: str, *,
                          include_legacy: bool = True) -> set[str]:
    """Cache keys of every cell already completed for this experiment.

    Scans all prior run directories, plus (by default) the frozen legacy log —
    keys are exact (speech|prompt|model|temp|seed|rep), so consulting the
    legacy log can only prevent double-billing, never wrongly skip new work.
    """
    done: set[str] = set()
    for p in _experiment_logs(experiment):
        done |= _load_done_keys(p)
    if include_legacy:
        done |= _load_done_keys(config.LEGACY_RESULTS_LOG)
    return done


@dataclass
class _Job:
    speech_id: int
    speech_text: str
    variant: PromptVariant
    model: ModelSpec
    cond: Condition
    rep: int


def _build_jobs(plan: RunPlan, done: set[str]) -> list[_Job]:
    jobs: list[_Job] = []
    for row in plan.speeches.itertuples():
        text = row.speech_text or ""
        for v in plan.variants:
            for m in plan.models:
                for cond in plan.conditions:
                    for rep in range(cond.n_reps):
                        key = _cache_key(row.speech_id, v.prompt_hash,
                                         m.model_id, cond.temperature,
                                         cond.seed, rep)
                        if key in done:
                            continue
                        jobs.append(_Job(row.speech_id, text, v, m, cond, rep))
    return jobs


def _run_one(client: LLMClient, job: _Job, topic: Topic) -> dict:
    msgs = job.variant.render(job.speech_text)
    uncapped = job.variant.task == TASK_UNCAPPED
    max_tokens = UNCAPPED_MAX_TOKENS if uncapped else None  # None -> model default
    # Read the cap off the variant's own topic, not the plan's: a definition
    # arm puts several Topic objects in one plan, and only the variant knows
    # which one produced this cell.
    parse_n = UNCAPPED_MAX_SUBTHEMES if uncapped else job.variant.topic.max_subthemes
    res = client.complete(
        msgs, job.model,
        temperature=job.cond.temperature, seed=job.cond.seed,
        max_tokens=max_tokens,
    )
    ex = (schema.parse(res.text, job.variant.output_format, parse_n)
          if res.text is not None
          else schema.Extraction(parse_ok=False, parse_error=res.error or "no text"))
    return {
        "speech_id": int(job.speech_id),
        "variant_id": job.variant.variant_id,
        "role": job.variant.role,
        "task": job.variant.task,
        "output_format": job.variant.output_format,
        "definition": job.variant.definition,
        "prompt_hash": job.variant.prompt_hash,
        "model_id": job.model.model_id,
        "family": job.model.family,
        "condition": job.cond.label,
        "temperature": job.cond.temperature,
        "seed": job.cond.seed,
        "rep": job.rep,
        "mentions_topic": ex.mentions_topic,
        "subthemes": ex.subthemes,
        "subthemes_raw": ex.subthemes_raw,
        "presence_inferred": ex.presence_inferred,
        "evidence_quote": ex.evidence_quote,
        "parse_ok": ex.parse_ok,
        "parse_error": ex.parse_error,
        "n_chars_sent": len(job.speech_text),
        "truncated": False,
        "prompt_tokens": res.prompt_tokens,
        "completion_tokens": res.completion_tokens,
        "latency_s": res.latency_s,
        "attempts": res.attempts,
        "finish_reason": res.finish_reason,
        "error": res.error,
        "raw_text": res.text,
        "reasoning": res.reasoning,
        "ts": time.time(),
    }


def _write_run_manifest(plan: RunPlan, *, experiment: str, run_id: str,
                        run_dir: Path, cli_args: dict | None = None) -> None:
    provenance.write_manifest(run_dir, {
        "experiment": experiment,
        "run_id": run_id,
        "pool": plan.pool,
        "n_speeches": int(plan.speeches["speech_id"].nunique()),
        "models": [{"model_id": m.model_id, "family": m.family,
                    "tier": m.tier, "reasoning": m.reasoning}
                   for m in plan.models],
        "conditions": [{"label": c.label, "temperature": c.temperature,
                        "seed": c.seed, "n_reps": c.n_reps}
                       for c in plan.conditions],
        # Full prompt text per hash: a 12-char hash alone cannot tell you what
        # was actually sent once the prompt code has moved on.
        "variants": [{"variant_id": v.variant_id, "definition": v.definition,
                      "prompt_hash": v.prompt_hash, "template": v._template()}
                     for v in plan.variants],
        "cli_args": cli_args or {},
    })


def execute(plan: RunPlan, *, experiment: str, verbose: bool = True,
            include_legacy_cache: bool = True,
            cli_args: dict | None = None) -> int:
    """Run all not-yet-cached cells in ``plan`` under a fresh run directory
    ``runs/<experiment>/<run_id>/``. Returns the number of new cells written.

    Safe to call repeatedly: the idempotent cache spans every prior run of the
    same experiment (and the frozen legacy log), so re-runs only execute cells
    that are genuinely missing. Each invocation that has work to do creates its
    own run directory with a manifest; an invocation with nothing to do
    creates nothing.
    """
    done = _experiment_done_keys(experiment,
                                 include_legacy=include_legacy_cache)
    jobs = _build_jobs(plan, done)
    total = len(jobs)
    if verbose:
        print(f"[{experiment}] {len(done)} cells already done; "
              f"{total} new cells to run ({plan.max_workers} workers)")
    if not total:
        return 0

    run_id = provenance.new_run_id()
    run_dir = provenance.run_dir(experiment, run_id)
    _write_run_manifest(plan, experiment=experiment, run_id=run_id,
                        run_dir=run_dir, cli_args=cli_args)
    log_path = run_dir / "results.jsonl"
    extras = {
        "experiment": experiment,
        "run_id": run_id,
        "pool": plan.pool,
        "code_version": provenance.git_sha(),
        "backend": config.backend_name(),
    }
    if verbose:
        print(f"[{experiment}] run {run_id} -> {log_path}")

    client = LLMClient()
    write_lock = threading.Lock()
    n_done = 0
    t0 = time.time()

    with log_path.open("a", encoding="utf-8") as fh, \
            ThreadPoolExecutor(max_workers=plan.max_workers) as pool:
        futs = {pool.submit(_run_one, client, j, plan.topic): j for j in jobs}
        for fut in as_completed(futs):
            row = {**fut.result(), **extras}
            with write_lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                n_done += 1
                if verbose and (n_done % 25 == 0 or n_done == total):
                    rate = n_done / (time.time() - t0)
                    eta = (total - n_done) / rate if rate else 0
                    print(f"  {n_done}/{total}  ({rate:.1f}/s, eta {eta:.0f}s)")
    return n_done


def reparse_results(df: pd.DataFrame, topic: Topic | None = None) -> pd.DataFrame:
    """Recompute parsed fields from stored ``raw_text`` using the current
    schema. The parse-time snapshot in the log can go stale when the parser is
    improved; analysis should always reparse so parser changes are reflected
    without any re-billing of the (expensive) model calls.
    """
    topic = topic or config.DEFAULT_TOPIC
    capped_n = topic.max_subthemes

    def _mn(r) -> int:
        # Uncapped rows must not be re-truncated to the 5-topic cap at reparse.
        return (UNCAPPED_MAX_SUBTHEMES
                if getattr(r, "task", None) == TASK_UNCAPPED else capped_n)

    def _row(r):
        # A failed call (timeout, refusal, transport error) logs raw_text as
        # JSON null, which pandas hands back as NaN rather than None. Test for
        # "is a string" instead: an `is None` check misses the NaN and the parse
        # then dies on the whole frame because of one bad cell.
        if not isinstance(r.raw_text, str):
            err = r.error if isinstance(r.error, str) else "no text"
            return schema.Extraction(parse_ok=False, parse_error=err)
        return schema.parse(r.raw_text, r.output_format, _mn(r))

    parsed = [_row(r) for r in df.itertuples()]
    out = df.copy()
    out["mentions_topic"] = [e.mentions_topic for e in parsed]
    out["subthemes"] = [e.subthemes for e in parsed]
    out["subthemes_raw"] = [e.subthemes_raw for e in parsed]
    out["presence_inferred"] = [e.presence_inferred for e in parsed]
    out["evidence_quote"] = [e.evidence_quote for e in parsed]
    out["parse_ok"] = [e.parse_ok for e in parsed]
    out["parse_error"] = [e.parse_error for e in parsed]
    return out


def load_results(log_path: Path = RESULTS_LOG, *, to_parquet: bool = False,
                 reparse: bool = True) -> pd.DataFrame:
    """Compile one JSONL log into a DataFrame.

    With ``reparse=True`` (default) the parsed columns are recomputed from
    ``raw_text`` via the current schema, so the returned frame always reflects
    the latest parser. ``to_parquet`` defaults to False: overwriting the shared
    snapshot must be an explicit choice, not a side effect of loading.
    """
    if not log_path.exists():
        raise FileNotFoundError(f"No results log at {log_path}")
    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    # Rows logged before the definition arm existed all used the default
    # definition; without this backfill they would read as NaN and drop out of
    # any groupby on the column.
    if "definition" not in df.columns:
        # Pre-definition-arm rows used the pilot ``current`` wording.
        df["definition"] = "current"
    else:
        df["definition"] = df["definition"].fillna("current")
    if reparse:
        df = reparse_results(df)
    if to_parquet:
        df.to_parquet(config.RESULTS_PATH, index=False)
    return df


def load_experiment(experiment: str, *, reparse: bool = True) -> pd.DataFrame:
    """Concatenate every run of an experiment from the versioned store."""
    logs = _experiment_logs(experiment)
    if not logs:
        raise FileNotFoundError(
            f"No runs for experiment {experiment!r} under {config.RUNS_DIR}")
    frames = [load_results(p, reparse=False) for p in logs]
    df = pd.concat(frames, ignore_index=True)
    if reparse:
        df = reparse_results(df)
    return df


def _legacy_pool_labels(df: pd.DataFrame) -> pd.Series:
    """Reconstruct which population each legacy row came from.

    The legacy log mixed two populations under identical grid labels: the
    270-speech pilot sample and the 150 filter-pool speeches labelled by the
    retrieval spot-check. Membership is recoverable from the sample parquet
    and the spot-check id list (the pools were drawn disjoint).
    """
    from . import sample
    pilot_ids = set(sample.load_sample()["speech_id"])
    spot_path = config.ARTIFACTS_DIR / "retrieval_spotcheck_ids.json"
    spot_ids: set = set()
    if spot_path.exists():
        spot_ids = {r["speech_id"] for r in
                    json.loads(spot_path.read_text(encoding="utf-8"))}
    return pd.Series(
        ["pilot" if sid in pilot_ids
         else "filter_pool" if sid in spot_ids
         else "unknown"
         for sid in df["speech_id"]],
        index=df.index, dtype="object")


def load_legacy(*, reparse: bool = True,
                write_annotated: bool = False) -> pd.DataFrame:
    """The frozen pre-provenance log, with the provenance columns backfilled.

    Adds ``pool`` (pilot | filter_pool | unknown — see ``_legacy_pool_labels``)
    plus constant ``experiment``/``run_id``/``code_version``/``backend`` markers
    so legacy rows can be concatenated with versioned-store rows. Analyses of
    the pilot must filter ``pool == "pilot"``.
    """
    df = load_results(config.LEGACY_RESULTS_LOG, reparse=reparse)
    df["pool"] = _legacy_pool_labels(df)
    df["experiment"] = "legacy_pilot"
    df["run_id"] = "legacy"
    df["code_version"] = "pre-provenance"
    df["backend"] = "nebius"
    if write_annotated:
        config.LEGACY_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.LEGACY_ANNOTATED_PATH, index=False)
    return df


def default_plan(
    *,
    n_speeches: int | None = None,
    models: tuple[ModelSpec, ...] = config.CORE_MODELS,
    conditions: tuple[Condition, ...] = (CORE,),
    max_workers: int = 16,
) -> RunPlan:
    """Convenience builder: load the pilot sample and the full 8-variant grid.

    ``n_speeches`` (if given) takes the first N rows of the sample for a quick
    end-to-end smoke run before committing to the full grid.
    """
    from . import sample
    df = sample.load_sample()
    if n_speeches is not None:
        df = df.head(n_speeches).copy()
    topic = config.DEFAULT_TOPIC
    return RunPlan(
        speeches=df,
        topic=topic,
        variants=build_variants(topic),
        models=models,
        conditions=conditions,
        max_workers=max_workers,
    )


def uncapped_plan(
    *,
    n_speeches: int | None = None,
    models: tuple[ModelSpec, ...] = config.CORE_MODELS,
    conditions: tuple[Condition, ...] = (CORE,),
    max_workers: int = 16,
) -> RunPlan:
    """The focused no-cap arm: the uncapped task (role=none, both formats) over
    all models. Only these cells are new; the matching capped cells
    (role=none, task=v1, same formats/models) are already in the log, so this
    adds 2 variants x len(models) cells per speech and nothing else.
    """
    from . import sample
    df = sample.load_sample()
    if n_speeches is not None:
        df = df.head(n_speeches).copy()
    topic = config.DEFAULT_TOPIC
    return RunPlan(
        speeches=df,
        topic=topic,
        variants=build_uncapped_variants(topic),
        models=models,
        conditions=conditions,
        max_workers=max_workers,
    )


def definition_plan(
    *,
    definitions: tuple[str, ...] = config.ALT_DEFINITIONS,
    n_speeches: int | None = None,
    models: tuple[ModelSpec, ...] = config.CORE_MODELS,
    conditions: tuple[Condition, ...] = (CORE,),
    max_workers: int = 16,
) -> RunPlan:
    """The definition-sensitivity arm: alternative construct definitions at the
    new default configuration (role=none, uncapped, both formats, all models).

    Sizing: len(definitions) x 2 formats x len(models) cells per speech.
    Default definitions are ``config.ALT_DEFINITIONS`` (expert orders, current,
    name-only).
    """
    from . import sample
    df = sample.load_sample()
    if n_speeches is not None:
        df = df.head(n_speeches).copy()
    topics = [config.HSC_DEFINITIONS[d] for d in definitions]
    return RunPlan(
        speeches=df,
        topic=config.DEFAULT_TOPIC,   # nominal; each variant carries its own
        variants=build_definition_variants(topics),
        models=models,
        conditions=conditions,
        max_workers=max_workers,
    )


def uncapped_role_plan(
    *,
    n_speeches: int | None = None,
    models: tuple[ModelSpec, ...] = config.CORE_MODELS,
    conditions: tuple[Condition, ...] = (CORE,),
    max_workers: int = 16,
) -> RunPlan:
    """Role add-on: the uncapped task at role=expert, so role invariance can be
    re-confirmed under the configuration we actually ship rather than under the
    retired capped default. Pairs against the cached role=none uncapped cells.
    """
    from . import sample
    df = sample.load_sample()
    if n_speeches is not None:
        df = df.head(n_speeches).copy()
    topic = config.DEFAULT_TOPIC
    return RunPlan(
        speeches=df,
        topic=topic,
        variants=build_uncapped_variants(topic, roles=("expert",)),
        models=models,
        conditions=conditions,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run the LLM extraction grid.")
    ap.add_argument("--n-speeches", type=int, default=None,
                    help="limit to first N sampled speeches (smoke run)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--self-consistency", action="store_true",
                    help="also run the temp>0 repetition probe")
    ap.add_argument("--no-cap", action="store_true",
                    help="run the focused uncapped arm (role=none, v1 without the "
                         "5-topic limit, json+free, all models) to test whether "
                         "the cap inflates the topic count")
    ap.add_argument("--definitions", nargs="*", default=None,
                    metavar="ID",
                    help=f"run the definition-sensitivity arm. With no ids, runs "
                         f"all uncached alternatives {config.ALT_DEFINITIONS}; "
                         f"otherwise the named ones. Choices: "
                         f"{tuple(config.HSC_DEFINITIONS)}")
    ap.add_argument("--role-check", action="store_true",
                    help="run the role add-on (role=expert, uncapped) to re-confirm "
                         "role invariance under the current default config")
    args = ap.parse_args()

    conds = (CORE, SELFCONSISTENCY) if args.self_consistency else (CORE,)
    if args.definitions is not None:
        ids = tuple(args.definitions) or config.ALT_DEFINITIONS
        unknown = [d for d in ids if d not in config.HSC_DEFINITIONS]
        if unknown:
            ap.error(f"unknown definition(s) {unknown}; "
                     f"choose from {tuple(config.HSC_DEFINITIONS)}")
        plan = definition_plan(definitions=ids, n_speeches=args.n_speeches,
                               conditions=conds, max_workers=args.workers)
        experiment = "pilot_definitions"
    else:
        builder, experiment = (
            (uncapped_role_plan, "pilot_role") if args.role_check
            else (uncapped_plan, "pilot_uncapped") if args.no_cap
            else (default_plan, "pilot_core"))
        plan = builder(n_speeches=args.n_speeches, conditions=conds,
                       max_workers=args.workers)
    n = execute(plan, experiment=experiment, cli_args=vars(args))
    print(f"wrote {n} new cells")
