# hansard_llm — LLM panel + embedder retrieval

Ask open-weight models whether each parliamentary speech **substantively
discusses health and social care**, list free-text **sub-topics**, and give a
supporting **quote**. The same labels feed a **definition-as-query** retrieval
experiment (embed speeches, rank by cosine to construct definitions).

**Default shipping definition:** `expert_hc_sc` (expert healthcare then social
care). Speeches are sent in full (no char truncation).

Primary compute path: **Oxford BMRC cluster** with local **vLLM** (OpenAI-
compatible client). See `../cluster/README.md` for env setup and sbatch.

---

## 0. Evaluation sample

Panel and embedder grid share **`eval2k_sample.parquet`**
(`sample.build_eval_subset` / `python -m hansard_llm.sample --eval-subset`).

**How it is drawn**

- Pool: enriched Hansard parquet, non-procedural, `word_count ≥ 40`,
  Commons/Lords, length tiers short/medium/long, with usable year.
- **No keyword/seed oversampling** — random within decade so retention /
  threshold estimates transfer to the natural corpus (unlike the pilot).
- **Decade-stratified:** target ~2k total; per decade, allocate proportional
  to decade size with a **floor of 50** (take-all if the decade is smaller).
- Reservoir sample with fixed seed (`EVAL_SEED = 20260802`).
- `sampling_weight` = decade population / decade draw (for corpus-level rates).

Realised size is a bit above 2k because of floors + rounding. A larger draw
(e.g. 10k) is optional later for tighter corpus-level rates; sensitivity
questions do not need it.

**Pilot sample** (`pilot_sample.parquet`, `python -m hansard_llm.sample`) is
separate: stratified on era × length × **seed-regex presence** (2:1
present:absent). Used for older pilot arms / spot-checks, not the cluster panel.

```bash
python -m hansard_llm.sample --eval-subset   # write eval2k
python -m hansard_llm.sample                 # write pilot sample (legacy path)
```

---

## 1. LLM panel (`panel2k`)

One job per model. Fixed prompt shape; vary **definition** and **sampling**.

| Axis | Default |
|------|---------|
| Speeches | `eval2k_sample.parquet` (see §0) |
| Definitions | `expert_hc_sc`, `expert_sc_hc`, `current`, `name_only` |
| Role / task / format | `none` / `v1_nocap` / `json` |
| Sampling | `temp0` (T=0, seed=42, 1 rep) **and** `temp07` (T=0.7, no seed, 1 rep) |
| Models | `config.PANEL_MODELS` (= `CORE_MODELS`) — one per job |
| Pool tag | `eval2k` |
| Store | `artifacts/llm/runs/panel2k/<run_id>/` |

**Panel models:** Qwen3-30B-A3B, Gemma-3-27B, Nemotron-Super-49B (FP8 on A100-80GB), Qwen3-32B.

Per model ≈ 2k × 4 defs × 2 temps ≈ **~16k cells** (watch determinism cells/s to extrapolate).

**Run `--determinism` first** (before full `panel2k` or the embedder grid):
smoke-test serve→client wiring **and** measure cells/s so you can extrapolate
wall time. Use the same `MODEL` (and `VLLM_ARGS`) you intend for the real
panel. Details: `../cluster/README.md` sanity ladder.

**Optional add-ons** (same prompt shape; not required for the main panel labels)

| Flag | Experiment | Why it exists | What it runs |
|------|------------|---------------|--------------|
| `--determinism` | `panel_determinism` | Smoke + timing; also measure T=0 serving noise (stacks are **not** guaranteed byte-stable) before reading model disagreement as real | 200 speeches × 1 definition × **3** temp-0 reps (**same** seed) |
| `--extended` | `panel_extended2k` | Ask whether **size / family** moves answers — same eval2k speeches, stored separately for `EXTENDED_MODELS` (4B / 14B / Mistral-Small-24B) | same grid as main panel |

Gold and model-agreement analyses still use the main `panel2k` **`temp0`** rows.

**Uses of the same rows**

1. Model sensitivity (agreement across models; gold/agreement use **`temp0` only**).
2. Retrieval gold via leave-one-definition-out: `panel.panel_gold(exclude_definition=…)`.
3. Cross-model sub-topic / taxonomy work.

```bash
# Cluster — always pass -A (scripts do not set --account); see ../cluster/README.md
# Determinism first (smoke + timing; ~600 cells):
sbatch -A gpu_<group>.prj --export=ALL,MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507,RUN_ARGS="--determinism" \
       cluster/run_grid.sbatch
sbatch -A gpu_<group>.prj --export=ALL,MODEL=nvidia/Llama-3_3-Nemotron-Super-49B-v1_5,\
VLLM_ARGS="--quantization fp8",RUN_ARGS="--determinism" cluster/run_grid.sbatch

# Then full panel (eval2k × 4 defs × temp0+temp07):
sbatch -A gpu_<group>.prj --export=ALL,MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 cluster/run_grid.sbatch
sbatch -A gpu_<group>.prj --export=ALL,MODEL=nvidia/Llama-3_3-Nemotron-Super-49B-v1_5,VLLM_ARGS="--quantization fp8" \
       cluster/run_grid.sbatch
sbatch -A gpu_<group>.prj --export=ALL,MODEL=Qwen/Qwen3-14B,RUN_ARGS="--extended" cluster/run_grid.sbatch

# Or against an already-running endpoint:
export LLM_BASE_URL=… LLM_API_KEY=… LLM_BACKEND_NAME=vllm-…
python -m hansard_llm.panel --model Qwen/Qwen3-32B --determinism   # smoke first
python -m hansard_llm.panel --model Qwen/Qwen3-32B --workers 32
```

Each cell stores **`raw_text`** (generation) plus parsed fields, tokens,
`latency_s`, and provenance. Parse is CPU-only, inline after each completion.
`load_experiment` can **reparse** from `raw_text` if the parser improves.

Deferred: `REFERENCE_MODELS` (Qwen3-235B — not downloaded / not in default plans),
`REASONING_MODELS`. Older pilot nuisance grids live under `python -m hansard_llm.run`
(`--definitions`, `--no-cap`, …) and are not the cluster default.

---

## 2. Embedder grid (retrieval sensitivity)

Does ranking change with **embedder**, **query wording**, and **document
representation**?

| Axis | Levels |
|------|--------|
| Model | 8 embedders in `embedder_grid.EMBEDDERS` (Qwen3 0.6B/4B/8B size axis; BGE base/large; GTE-large; E5-large; Nomic v1.5) |
| Representation | `whole` / `maxchunk` / `meanchunk` (chunks from `retrieve.split_chunks`) |
| Query | same ids as `PANEL_DEFINITIONS` |
| Speeches | same `eval2k_sample.parquet` |
| Backend | `st` (sentence-transformers, cluster) or `api` (OpenAI-compatible embeddings) |
| Store | `artifacts/llm/runs/embedder_grid/<run_id>/` (`scores_*.parquet` + manifest) |

```bash
sbatch -A gpu_<group>.prj cluster/embed_grid.sbatch   # array 0–7; one model: --array=N
python -m hansard_llm.embedder_grid --list
python -m hansard_llm.embedder_grid --model Qwen/Qwen3-Embedding-8B --backend st
python -m hansard_llm.embedder_grid --diagnostics   # gold-free rank agreement / length bias
```

**Eval against LLM gold (LODO):** once `panel2k` exists, score each query with
`retrieve.gold_for_query(qid)` / `panel.panel_gold(exclude_definition=qid)` so a
definition is never scored against labels produced from that same wording.
`retrieve.evaluate_all` does this per query when panel rows are available
(falls back to pilot majority until then).

Related: `python -m hansard_llm.retrieve` — earlier pilot/filter-pool embedding
path (whole + maxchunk, keyword baseline, spot-check). Prefer the embedder grid
+ panel LODO for the eval-subset sensitivity study.

---

## 3. Shared data & results layout

Paths resolve from env (`HANSARD_LLM_*` / `~/.config/hansard_llm.env` on cluster);
defaults point at sibling `hansard_eda/artifacts/llm/`.

| Artifact | Role |
|----------|------|
| `eval2k_sample.parquet` | Eval speeches for panel + embedder grid (§0) |
| `pilot_sample.parquet` | Older stratified pilot (legacy / spot-check) |
| `runs/<experiment>/<run_id>/manifest.json` | Git SHA, prompts, models, conditions, backend |
| `runs/<experiment>/<run_id>/results.jsonl` | LLM cells (append-only, resumable) |
| `runs/embedder_grid/<run_id>/` | Embedding scores + manifest |
| `legacy/` | Frozen pre-provenance pilot JSONL |

Cache key for LLM cells: `speech_id|prompt_hash|model|temperature|seed|rep`.

```python
from hansard_llm import run, panel
df = run.load_experiment("panel2k")
gold = panel.panel_gold(exclude_definition="expert_hc_sc")  # LODO for that query
```

---

## 4. Design notes

- **Raw text is source of truth**; parsed columns are a snapshot.
- **Idempotent resume** across run directories of the same experiment.
- **Open vocabulary** for sub-themes; compare in embedding space downstream.
- **A100 + FP8:** weight-only FP8 (W8A16) for Nemotron-49B — not native FP8 compute (Hopper/GH200).
- Cluster wiring: `cluster/run_grid.sbatch` → `vllm serve` + `python -m hansard_llm.panel`;
  `cluster/embed_grid.sbatch` → embedder array; details in `../cluster/README.md`.
