# Cluster setup — Oxford BMRC (SLURM + vLLM)

OpenAI-compatible client → point `LLM_BASE_URL` at a local `vllm serve`.
Rows record `backend=vllm-<cluster>`. Experiment design lives in
`../hansard_llm/README.md`; this file is how to run it on BMRC.

Cluster docs: <https://www.medsci.ox.ac.uk/for-staff/resources/bmrc>
(esp. *GPU Resources 2026*, *Using the BMRC Cluster*, *Python on the BMRC
Cluster*). Help: bmrc-help@medsci.ox.ac.uk — quote your job ID.

## What you run here

| Job | Script | Entry | Notes |
|-----|--------|-------|-------|
| LLM panel | `run_grid.sbatch` | `python -m hansard_llm.panel --model $MODEL` | serve + label eval10k (4 defs × temp0 + temp07) |
| Embedder grid | `embed_grid.sbatch` | `python -m hansard_llm.embedder_grid` | array 0–7, one embedder each |
| Interactive LLM | `serve_llm.sbatch` | `vllm serve` only | write `logs/vllm-endpoint-<jobid>`; client elsewhere |

**Panel models** (`PANEL_MODELS`): Qwen3-30B-A3B, Gemma-3-27B, Nemotron-Super-49B, Qwen3-32B.  
**Not downloaded / not scheduled by default:** Qwen3-235B (`REFERENCE_MODELS`).  
**Nemotron on A100-80GB:** `VLLM_ARGS="--quantization fp8"` (weight-only FP8 / W8A16 — A100 has no native FP8 compute).

## Placement strategy

- **LLM serving → `gpu_gh200_144gb`** once the partition is up (discovery
  2026-08-03: administratively DOWN). On 144GB every panel model including
  Nemotron-49B fits **bf16 on one card** (no FP8). GH200 = ARM →
  `01b_setup_env_gh200.sh` (steps 2b/2c).
- **Until then: LLM → `gpu_a100_80gb`** (current `run_grid` / `serve_llm`
  default). Nemotron needs FP8 as above; other panel models fit bf16.
- **Embedders → prefer `gpu_a100_40gb` / RTX 48GB** when available (≤8B);
  current `embed_grid.sbatch` still requests `gpu_a100_80gb` — edit the
  partition line if you want to free 80GB cards for LLM jobs.

Templates pick x86 vs ARM venv via `uname -m`.

## BMRC facts the scripts rely on

- **Login**: `ssh <username>@cluster1.bmrc.ox.ac.uk` (cluster1–4), Oxford
  network / VPN. Password + 6-digit authenticator; **no SSH keys**.
- **Compute nodes have no internet** — download weights on login nodes into
  `/well`; jobs run offline (`HF_HUB_OFFLINE` via env file).
- **Storage**: `/well/<group>/users/<username>/` (`HANSARD_SCRATCH`); `$HOME`
  is tiny. Budget ~400GB.
- **Accounts/QOS**: `-A gpu_<group>.prj` (sbatch files use `gpu_mills.prj`);
  `--qos gpu_bmrc_24hr` (or `_4hr`) for priority. Partition max 60h;
  project cap 24 GPUs; `gpu_interactive` for debugging.
- **Data transfer**: FileZilla SFTP to `cluster2.bmrc.ox.ac.uk` (1 connection,
  600s timeout) or rsync from WSL with `ControlMaster`; Globus `bmrc#upload23`
  for large transfers.
- **Login-node etiquette**: no heavy compute; LLM downloads in tmux/stages.

## Order of operations

| Step | Script | Where | Notes |
|------|--------|-------|-------|
| 1 | `00_discovery.sh` | login | quota, queues, modules |
| 2 | `01_setup_env.sh` | login | scratch + x86 venv + `~/.config/hansard_llm.env` |
| 2b/2c | `01b_setup_env_gh200.sh` | login then GH200 | ARM venv — only when GH200 partition is up |
| 3 | upload | laptop | **required:** `eval10k_sample.parquet` → `$HANSARD_SCRATCH/artifacts/llm/`. Optional: `pilot_sample.parquet`, `legacy/`, Nebius `.env`, enriched parquet for new samples |
| 4 | `02_download_models.sh` | login | `embedders` first (~20GB), then `llms` (panel + extended; **no 235B**). Gated: `hf auth login` (Gemma, Nemotron) |
| 5 | `embed_grid.sbatch` | sbatch | embedder sensitivity on eval10k |
| 6 | `run_grid.sbatch` | sbatch | one MODEL per job → panel10k (or `--extended` / `--determinism`) |
| 7 | `serve_llm.sbatch` | sbatch | long-lived server if you want a separate client |

## Sanity ladder (after steps 1–4)

```bash
# rung 1 — env (printed at end of 01 / 01b install): torch | vllm | cuda True

# rung 2 — small in-process generate (use A100 until GH200 is up):
srun -A gpu_mills.prj -p gpu_a100_80gb --gres=gpu:1 -t 15 --mem=32G bash -c '
  source ~/.config/hansard_llm.env; source "$VENV_DIR/bin/activate"
  python -c "
from vllm import LLM, SamplingParams
llm = LLM(\"Qwen/Qwen3-4B-Instruct-2507\")
print(llm.generate([\"Say hello.\"], SamplingParams(max_tokens=16))[0].outputs[0].text)"'

# rung 3 — serve + panel path (determinism add-on ≈ 200 speeches × 1 def × 3 reps):
sbatch --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507,RUN_ARGS="--determinism" \
       cluster/run_grid.sbatch
```

Check `squeue -u $USER` and `logs/hansard-grid-<jobid>.out` /
`logs/vllm-<jobid>.log`. Expect `runs/panel_determinism/<run_id>/results.jsonl`
with `backend=vllm-…`.

## Submit the real grids

```bash
# Embedders (array 0–7)
sbatch cluster/embed_grid.sbatch

# Panel — one sbatch per model (default: eval10k × 4 defs × temp0+temp07)
for m in \
  Qwen/Qwen3-30B-A3B-Instruct-2507 \
  google/gemma-3-27b-it \
  Qwen/Qwen3-32B
do
  sbatch --export=ALL,MODEL="$m" cluster/run_grid.sbatch
done
sbatch --export=ALL,MODEL=nvidia/Llama-3_3-Nemotron-Super-49B-v1_5,VLLM_ARGS="--quantization fp8" \
       cluster/run_grid.sbatch

# Extended size axis (2k speeches)
sbatch --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507,RUN_ARGS="--extended" \
       cluster/run_grid.sbatch
sbatch --export=ALL,MODEL=Qwen/Qwen3-14B,RUN_ARGS="--extended" \
       cluster/run_grid.sbatch
sbatch --export=ALL,MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506,RUN_ARGS="--extended" \
       cluster/run_grid.sbatch
```

Outputs land under `$HANSARD_SCRATCH` / `HANSARD_LLM_ARTIFACTS_DIR`:

- `runs/panel10k/<run_id>/{manifest.json,results.jsonl}`
- `runs/panel_extended2k/…`, `runs/panel_determinism/…`
- `runs/embedder_grid/<run_id>/`

Jobs are resumable: re-submit the same `MODEL` and only missing cells run.
