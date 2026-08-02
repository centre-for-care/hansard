# hansard_llm — robustness-aware LLM topic extraction

A reproducible pilot pipeline that asks open-source LLMs (hosted on Nebius
Token Factory) a targeted question about each parliamentary speech, for a
**fixed topic** (default: *health and social care*):

1. Does this speech **substantively discuss** the topic? *(presence — binary)*
2. If so, which **sub-topics** does it raise? *(free-text, inductive)*
3. A supporting **verbatim quote**. *(for human spot-checks / future gold)*

The goal is not a single answer but a **robustness study**: how much do the
answers move when we vary things that *shouldn't* matter — prompt role, task
wording, output format, and model? The pipeline measures that and attributes
instability to each factor.

## Experimental design

The grid is the cartesian product of nuisance factors we want invariance to:

| Factor | Levels | Where |
|---|---|---|
| `role` | none / domain-expert | `prompts.ROLE_LEVELS` |
| `task` | two paraphrases | `prompts.TASK_LEVELS` |
| `output_format` | json / free (no format instruction) | `prompts.FORMAT_LEVELS` |
| `model` | Qwen3-30B-A3B / gemma-3-27b / Llama-3.3-70B / Qwen3-235B (reference) | `config.CORE_MODELS` |
| `condition` | temp 0 (1 rep) / temp 0.7 (N reps) | `run.CORE`, `run.SELFCONSISTENCY` |

8 prompt variants × 4 models = 32 cells per speech at temp 0. The temp-0.7
condition is the **self-consistency baseline**. (Temp-0 byte-determinism is an
*assumption, not a guarantee* for hosted MoE serving with dynamic batching —
the model-grid plan includes a repeat-at-temp-0 check to measure it.)

## Pipeline

```
sample.draw_sample()    # stratified pilot sample (era × length × seed-presence)
                        # + per-cell sampling weights for corpus-level rates
run.execute(plan, experiment="...")   # grid → runs/<experiment>/<run_id>/
                        # results.jsonl + manifest.json (idempotent, resumable)
run.load_experiment("...")            # versioned store → DataFrame (reparsed)
run.load_legacy()       # frozen pre-provenance pilot log, pool-annotated
metrics.summarize(df)   # presence α, factor decomposition, prevalence spread,
                        # semantic theme agreement
metrics.discover_taxonomy(df)   # cluster emitted themes → "what's there"
```

### Results store

New runs write under `artifacts/llm/runs/<experiment>/<run_id>/` — one
append-only `results.jsonl` plus a `manifest.json` recording the git SHA, the
full prompt text per `prompt_hash`, models, conditions, and pool. Rows carry
`experiment / run_id / pool / code_version / backend`. The pre-provenance
single log is frozen at `artifacts/llm/legacy/` and is read through
`run.load_legacy()`, which reconstructs the `pool` column (the legacy log mixed
pilot and retrieval spot-check rows under identical labels — pilot analyses
must filter `pool == "pilot"`).

### Quick start

```bash
# 1. credentials — copy env.txt to .env and fill in (both are gitignored)
# 2. draw the sample (one regex pass over the enriched Parquet)
python -m hansard_llm.sample
# 3. smoke run (first 3 speeches), then the full grid
python -m hansard_llm.run --n-speeches 3
python -m hansard_llm.run --workers 32
# 4. add the self-consistency probe on a subset
python -m hansard_llm.run --n-speeches 40 --self-consistency
```

```python
from hansard_llm import run, metrics
df = run.load_results()
metrics.summarize(df)
```

## Design choices that make it reproducible

- **Raw text is the source of truth.** Parsed fields are derived; `load_results`
  reparses from `raw_text`, so improving the parser never re-bills the model.
- **Idempotent cache.** Cells are keyed by
  `speech_id|prompt_hash|model|temperature|seed|rep`; re-runs skip completed work.
- **Full provenance** per call (model, params, tokens, latency, attempts, finish
  reason) in the JSONL log.
- **Pinned, declarative config.** Models, topic, and sample design live in
  `config.py` / `sample.py`.
- **Open vocabulary, semantic comparison.** Sub-themes are free text; agreement
  is measured in embedding space (`Qwen3-Embedding-8B`, cached on disk), not by
  string match.

## What is measured

- **Presence:** Krippendorff's α + mean pairwise agreement across the grid;
  per-factor marginal disagreement (which factor destabilises most).
- **Self-consistency:** the same, across temp-0.7 repetitions.
- **Sub-themes:** mean soft-Jaccard (bipartite phrase matching at cosine τ);
  a clustered discovered taxonomy with per-cluster speech prevalence.
- **Estimand:** topic prevalence per cell and its spread — does the headline
  number survive perturbation even when item labels flip?

## Tunables / open items

- `metrics.DEFAULT_TAU` (0.72) — same-theme cosine threshold. Qwen embeddings
  have a high baseline (~0.47 even for unrelated phrases); the reproducible
  taxonomy (`docs/build_taxonomy.py`) uses `distance_threshold=0.25` and
  records it in a sidecar manifest. Note: at 0.25 the pilot arm yields 285
  clusters, not the 152 in the original brief — that number came from a
  since-deleted script with unrecorded settings and should not be cited.
- Reasoning models (`config.REASONING_MODELS`: Kimi-K2.6, gpt-oss-120b) are a
  deferred separate axis — they emit a reasoning trace and need a larger token
  budget; mixing them into the core grid would confound reasoning with family.
- **Validity (not just consistency)** is deferred until the hand-labelled gold
  set exists; `metrics` has a slot for it.
