# Cluster setup — Oxford BMRC (SLURM + vLLM)

The pipeline's LLM client is OpenAI-compatible, so a cluster-hosted
`vllm serve` is a drop-in backend: point `LLM_BASE_URL` at it and every row
records `backend=vllm-<cluster>` in provenance. Nebius stays as fallback
(leave its credentials in the `.env` on `/well`).

Cluster docs: <https://www.medsci.ox.ac.uk/for-staff/resources/bmrc>
(esp. *GPU Resources 2026*, *Using the BMRC Cluster*, *Python on the BMRC
Cluster*). Help: bmrc-help@medsci.ox.ac.uk — quote your job ID.

## Placement strategy

- **LLM serving → `gpu_gh200_144gb`** (40 nodes, best GPU in house) *once
  the partition is up* — discovery 2026-08-03 found it administratively
  DOWN (all 40 nodes idle; likely still commissioning — ask
  bmrc-help@medsci.ox.ac.uk). On 144GB every panel model — including
  Nemotron-Super-49B at ~98GB — fits **bf16 on one card** (no FP8
  confound), with tensor-parallel headroom. GH200s are ARM (aarch64) →
  dedicated venv via `01b_setup_env_gh200.sh` (steps 2b/2c — deferred
  until the partition is up).
- **Until then: LLM serving → `gpu_a100_80gb`** (current sbatch default):
  all panel models fit one 80GB card, Nemotron-49B via
  `VLLM_ARGS="--quantization fp8"`. Switching later = one partition line
  per sbatch file; the templates pick the x86 or ARM venv automatically
  via `uname -m`.
- **Embedder grid → `gpu_a100_40gb,gpu_rtx8000_48gb`**: all 8 embedders
  (≤8B) fit 40GB; keeps the big cards free for LLM jobs.

## BMRC facts the scripts rely on

- **Login**: `ssh <username>@cluster1.bmrc.ox.ac.uk` (cluster1–4), from the
  Oxford network / Oxford VPN only. Password + 6-digit authenticator code;
  **SSH keys are not supported**.
- **Compute nodes have no internet** — model weights and aarch64 wheels are
  downloaded on login nodes into `/well` (shared FS), jobs run offline.
- **Storage**: everything in the group area `/well/<group>/users/<username>/`
  (`$HOME` quota is tiny); `HANSARD_SCRATCH` lives there. Budget ~400GB.
  Node-local NVMe at `/flash/scratch` via `--constraint flash` if IO ever
  bottlenecks (not needed initially).
- **Accounts/QOS**: GPU jobs need `-A gpu_<group>.prj`; `--qos gpu_bmrc_24hr`
  (or `_4hr`) trades a shorter time cap for a large priority boost.
  Partition max runtime 60h; project-wide limit 24 GPUs in use;
  `gpu_interactive` (24GB, 12h, 1 GPU/user) for debugging.
- **Data transfer**: no SSH keys ⇒ from Windows use **FileZilla** (SFTP to
  `cluster2.bmrc.ox.ac.uk`, 1 connection, 600s timeout — BMRC's documented
  settings) or rsync from **WSL** with SSH `ControlMaster`; Globus
  (`bmrc#upload23`) for very large/restartable transfers.
- **Login-node etiquette**: no compute-heavy work; downloads are IO-bound
  and fine, but do the 340GB LLM batch in stages (tmux).

## Order of operations

| Step | Script | Where | Notes |
|---|---|---|---|
| 1 | `00_discovery.sh` | login node | read-only; confirms /well quota headroom, GPU queue occupancy, python module + tooling |
| 2 | `01_setup_env.sh` | login node | `HANSARD_SCRATCH=/well/<group>/users/$USER/hansard-scratch`; x86 venv (embedders/analysis) + `~/.config/hansard_llm.env` |
| 2b | `01b_setup_env_gh200.sh download` | login node | cross-downloads aarch64 wheels (fails fast if one is missing) |
| 2c | `01b_setup_env_gh200.sh install` | GH200 node (srun) | offline ARM venv for vLLM; records `VENV_DIR_ARM` |
| 3 | upload data | laptop | scp/FileZilla: `eval10k_sample.parquet` → `artifacts/llm/` is all the experiments need. Optional, when a task demands them: `pilot_sample.parquet` + `legacy/` → `artifacts/llm/` (pilot-referencing analyses), `.env` → scratch root (Nebius fallback), `full_data_enriched.parquet` → `data/` (drawing new samples) |
| 4 | `02_download_models.sh` | login node | `embedders` first (~20GB), `llms` later (~340GB); gated repos (Llama/Gemma) need `hf auth login` |
| 5 | `embed_grid.sbatch` | sbatch | job array 0–7, one embedder per task, 40GB partitions |
| 6 | `run_grid.sbatch` | sbatch | serve model + run experiment in one job, GH200 |
| 7 | `serve_llm.sbatch` | sbatch | standalone server for interactive use |

The GPU project account (`-A gpu_mills.prj`) is already filled in across
the sbatch files and srun examples — nothing left to edit before
submitting.

## Sanity ladder (after steps 1–4, before the full grid)

Each rung proves one new thing; `srun` only for the quick checks, `sbatch`
for anything that runs unattended.

```bash
# rung 1 — env sanity (printed automatically at the end of 01b install):
#   torch <ver> | vllm <ver> | cuda True

# rung 2 — srun one-liner: vLLM loads a small model from the offline /well
# cache and generates. Proves weights + GPU + offline mode; releases the GPU
# on exit (~5 min):
srun -A gpu_mills.prj -p gpu_gh200_144gb --gres=gpu:1 -t 15 --mem=32G bash -c '
  source ~/.config/hansard_llm.env; source "$VENV_DIR_ARM/bin/activate"
  python -c "
from vllm import LLM, SamplingParams
llm = LLM(\"Qwen/Qwen3-4B-Instruct-2507\")
print(llm.generate([\"Say hello.\"], SamplingParams(max_tokens=16))[0].outputs[0].text)"'

# rung 3 — sbatch smoke test: full serve + pipeline loop on 5 speeches:
sbatch --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507,RUN_ARGS="--n-speeches 5" \
       cluster/run_grid.sbatch
```

Check rung 3 with `squeue -u $USER`, then `logs/hansard-grid-<jobid>.out`;
the row schema in the run's `results.jsonl` should match Nebius rows apart
from `backend`. If it passes, submit the real grid with sbatch only.
