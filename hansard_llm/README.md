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
| `model` | Qwen3-235B / Llama-3.3-70B / GLM-5.2 | `config.CORE_MODELS` |
| `condition` | temp 0 (1 rep) / temp 0.7 (N reps) | `run.CORE`, `run.SELFCONSISTENCY` |

8 prompt variants × 3 models = 24 cells per speech at temp 0. The temp-0.7
condition is the **self-consistency baseline** (models are byte-deterministic
at temp 0, so run-to-run variance only appears with temperature).

## Pipeline

```
sample.draw_sample()    # stratified pilot sample (era × length × seed-presence)
run.execute(plan)       # grid → append-only JSONL (idempotent, resumable)
run.load_results()      # JSONL → DataFrame, reparsed from raw_text
metrics.summarize(df)   # presence α, factor decomposition, prevalence spread,
                        # semantic theme agreement
metrics.discover_taxonomy(df)   # cluster emitted themes → "what's there"
```

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
  have a high baseline (~0.47 even for unrelated phrases); the taxonomy
  `distance_threshold` likely wants ~0.22–0.25, not 0.35. Calibrate once on the
  full run.
- Reasoning models (`config.REASONING_MODELS`: Kimi-K2.6, gpt-oss-120b) are a
  deferred separate axis — they emit a reasoning trace and need a larger token
  budget; mixing them into the core grid would confound reasoning with family.
- **Validity (not just consistency)** is deferred until the hand-labelled gold
  set exists; `metrics` has a slot for it.
