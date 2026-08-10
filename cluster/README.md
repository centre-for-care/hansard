# Cluster setup — Oxford BMRC

One linear path: mkdir → clone → setup script → upload data → download
models → `sbatch`. Experiment design:
[`../hansard_llm/README.md`](../hansard_llm/README.md).

**Oxford ARC (HTC) instead?** Use [`arc/README.md`](arc/README.md) — separate
sbatch + `~/.config/hansard_llm.arc.env` so BMRC and ARC do not collide.

Cluster docs: <https://www.medsci.ox.ac.uk/for-staff/resources/bmrc>
(esp. *GPU Resources 2026*, *Using the BMRC Cluster*, *Python on the BMRC
Cluster*). Help: bmrc-help@medsci.ox.ac.uk — quote your job ID.

---

## Before you start

| Need | Details |
|------|---------|
| Login | `ssh <user>@cluster1.bmrc.ox.ac.uk` (cluster1–4), Oxford network / VPN. Password + 6-digit authenticator; **no SSH keys**. |
| Storage | Work under `/well/<group>/users/$USER/` — `$HOME` is tiny. Budget ~400GB for models + caches. |
| GPU account | Always pass `-A gpu_<group>.prj` on `sbatch` / `srun`. Look it up: `sacctmgr show associations where user=$USER format=Account%-30`. `/well` access ≠ Slurm GPU account. |
| Login-node etiquette | No heavy GPU compute on the login node. Model downloads: use `tmux`. |

Substitute your group everywhere below (example: `mills` → paths under
`/well/mills/…` and account `gpu_mills.prj`).

---

## 1. Create the project root

```bash
export HANSARD_SCRATCH=/well/<group>/users/$USER/hansard-scratch
mkdir -p "$HANSARD_SCRATCH"/{hf_cache,artifacts/llm,data}
```

Later steps add `hansard/` (clone) and `hansard-env/` (setup script). Job logs
go under `hansard/logs/` (created by the sbatch scripts).

| Path | Purpose |
|------|---------|
| `hansard-scratch/hansard/` | Git clone of this repo |
| `hansard-scratch/hansard-env/` | Python venv (`VENV_DIR`) |
| `hansard-scratch/hf_cache/` | HF weights (`HF_HOME`) |
| `hansard-scratch/artifacts/llm/` | `eval2k_sample.parquet` + run outputs |
| `hansard-scratch/data/` | Optional `full_data_enriched.parquet` to rebuild samples |
| `hansard-scratch/hansard/logs/` | SLURM / vLLM logs (submit from `hansard/`) |
| `~/.config/hansard_llm.env` | Paths + offline flags; every sbatch sources this |

---

## 2. Clone the repo

```bash
cd "$HANSARD_SCRATCH"
git clone <repo-url> hansard
cd hansard
```

Use whatever branch / remote your team shares. Submit jobs from this directory
so `cluster/….sbatch` and `logs/` resolve correctly.

---

## 3. Run the setup script

On a **login node** (needs internet), from `$HANSARD_SCRATCH/hansard`:

```bash
HANSARD_SCRATCH=/well/<group>/users/$USER/hansard-scratch bash cluster/01_setup_env.sh
```

Requires §§1–2 already done. Creates `hansard-env`, installs vLLM /
sentence-transformers / this package, writes `~/.config/hansard_llm.env`.
Does not mkdir the project root or clone.

**Activate (interactive only)** — downloads, ad-hoc `python -m …`:

```bash
source ~/.config/hansard_llm.env
source "$VENV_DIR/bin/activate"
```

Do **not** activate before `sbatch`. Each job is a fresh shell and runs the
same two lines itself (ARM panel/serve use `VENV_DIR_ARM` after `01b`).

GH200 ARM venv (optional): `cluster/01b_setup_env_gh200.sh` after this step.

---

## 4. Upload eval data (from your laptop)

Required for panel + embedder grids:

`$HANSARD_SCRATCH/artifacts/llm/eval2k_sample.parquet`

```powershell
scp path\to\eval2k_sample.parquet `
  <user>@cluster1.bmrc.ox.ac.uk:/well/<group>/users/<user>/hansard-scratch/artifacts/llm/
```

Optional later: put `full_data_enriched.parquet` in `$HANSARD_SCRATCH/data/` to
rebuild samples (`python -m hansard_llm.sample --eval-subset`). Not needed if
eval2k is already present.

```bash
ls -lh "$HANSARD_SCRATCH/artifacts/llm/eval2k_sample.parquet"
```

---

## 5. Download model weights (login node)

```bash
source ~/.config/hansard_llm.env
source "$VENV_DIR/bin/activate"
# script unsets HF_HUB_OFFLINE for you
bash cluster/02_download_models.sh embedders   # ~20GB — do this first
bash cluster/02_download_models.sh llms        # panel + extended; not 235B
# or: bash cluster/02_download_models.sh all
# or: bash cluster/02_download_models.sh Qwen/Qwen3-Embedding-0.6B
```

Gated models (Gemma, Nemotron): `hf auth login` once beforehand.

---

## 6. Submit jobs

Always pass `-A` (sbatch files do **not** set `--account`):

```bash
sbatch -A gpu_<group>.prj cluster/<script>.sbatch
```

`#SBATCH` in each file sets partition / GPU / array defaults. Override on the
CLI if needed (`-p`, `--gres`, `--array`). Invalid account → ask bmrc-help.

| Job | Submit (cwd = `$HANSARD_SCRATCH/hansard`) |
|-----|------------------------------------------|
| Embedder grid (all 8) | `sbatch -A gpu_<group>.prj cluster/embed_grid.sbatch` |
| Embedder, one model | `sbatch -A gpu_<group>.prj --array=0 cluster/embed_grid.sbatch` (0–7) |
| LLM panel | `sbatch -A gpu_<group>.prj --export=ALL,MODEL=<hf-id> cluster/run_grid.sbatch` |
| Panel smoke / timing | same + `,RUN_ARGS="--determinism"` |
| Extended size axis | same + `,RUN_ARGS="--extended"` |
| Nemotron on A100-80GB | add `VLLM_ARGS="--quantization fp8"` to `--export` |
| vLLM only | `sbatch -A gpu_<group>.prj --export=ALL,MODEL=<hf-id> cluster/serve_llm.sbatch` |

Every job script: `source ~/.config/hansard_llm.env` → activate venv → run
Python (panel also starts `vllm serve` in the background).

Watch: `squeue -u $USER`. Logs: `logs/` under the repo (`hansard/logs/`).

### Recommended order after setup

1. Smoke — 4B determinism.
2. `--determinism` on each panel model you will use (throughput + wiring).
3. Full `embed_grid` and/or `run_grid` panel.

```bash
sbatch -A gpu_<group>.prj \
  --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507,RUN_ARGS="--determinism" \
  cluster/run_grid.sbatch

for m in \
  Qwen/Qwen3-30B-A3B-Instruct-2507 \
  google/gemma-3-27b-it \
  Qwen/Qwen3-32B
do
  sbatch -A gpu_<group>.prj --export=ALL,MODEL="$m",RUN_ARGS="--determinism" \
    cluster/run_grid.sbatch
done
sbatch -A gpu_<group>.prj \
  --export=ALL,MODEL=nvidia/Llama-3_3-Nemotron-Super-49B-v1_5,\
VLLM_ARGS="--quantization fp8",RUN_ARGS="--determinism" \
  cluster/run_grid.sbatch
```

### Full grids (after rates look sane)

```bash
sbatch -A gpu_<group>.prj cluster/embed_grid.sbatch

for m in \
  Qwen/Qwen3-30B-A3B-Instruct-2507 \
  google/gemma-3-27b-it \
  Qwen/Qwen3-32B
do
  sbatch -A gpu_<group>.prj --export=ALL,MODEL="$m" cluster/run_grid.sbatch
done
sbatch -A gpu_<group>.prj \
  --export=ALL,MODEL=nvidia/Llama-3_3-Nemotron-Super-49B-v1_5,VLLM_ARGS="--quantization fp8" \
  cluster/run_grid.sbatch

sbatch -A gpu_<group>.prj --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507,RUN_ARGS="--extended" \
  cluster/run_grid.sbatch
sbatch -A gpu_<group>.prj --export=ALL,MODEL=Qwen/Qwen3-14B,RUN_ARGS="--extended" \
  cluster/run_grid.sbatch
sbatch -A gpu_<group>.prj \
  --export=ALL,MODEL=mistralai/Mistral-Small-3.2-24B-Instruct-2506,RUN_ARGS="--extended" \
  cluster/run_grid.sbatch
```

Outputs under `$HANSARD_LLM_ARTIFACTS_DIR` (`artifacts/llm/`):

- `runs/panel2k/<run_id>/{manifest.json,results.jsonl}`
- `runs/panel_extended2k/…`, `runs/panel_determinism/…`
- `runs/embedder_grid/<run_id>/`

Jobs are resumable: re-submit the same `MODEL` and only missing cells run.

---

## Script map

| Script | When | Role |
|--------|------|------|
| `00_discovery.sh` | optional, login | quota / queues / modules check |
| `01_setup_env.sh` | once, login | venv + `~/.config/hansard_llm.env` |
| `01b_setup_env_gh200.sh` | optional | ARM venv for GH200 |
| `02_download_models.sh` | once (or as needed), login | HF weights into `$HF_HOME` |
| `embed_grid.sbatch` | GPU job | embedder array → `hansard_llm.embedder_grid` |
| `run_grid.sbatch` | GPU job | `vllm serve` + `hansard_llm.panel` |
| `serve_llm.sbatch` | GPU job | `vllm serve` only |

---

## Placement notes

- **LLM serving** → `gpu_a100_80gb` (script default). Nemotron needs FP8 as above;
  other panel models fit bf16. **Not scheduled by default:** Qwen3-235B.
- **Embedders** (≤8B) can use `gpu_a100_40gb` / RTX 48GB if you change the
  `#SBATCH --partition` line in `embed_grid.sbatch`.
- Interactive debug: `gpu_interactive`. Optional QOS: `--qos gpu_bmrc_24hr` (or `_4hr`).
