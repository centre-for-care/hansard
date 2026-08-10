# Cluster setup — Oxford ARC (HTC GPUs)

Parallel path to BMRC (`../README.md`). Same Python package and download
helper; different storage, SLURM headers, and env file so the two clusters
do not overwrite each other.

Docs: [ARC user guide](https://arc-user-guide.readthedocs.io/en/latest/)
([connecting](https://arc-user-guide.readthedocs.io/en/latest/connecting-to-arc.html),
[GPUs](https://arc-user-guide.readthedocs.io/en/latest/job-scheduling.html#gpu-resources),
[storage](https://arc-user-guide.readthedocs.io/en/latest/arc-storage.html)).
Help: support@arc.ox.ac.uk.

Experiment design: [`../../hansard_llm/README.md`](../../hansard_llm/README.md).

---

## Before you start

| Need | Details |
|------|---------|
| Login | `ssh <user>@htc-login.arc.ox.ac.uk` (Oxford net / VPN). From outside: often `gateway.arc.ox.ac.uk` first. |
| Disk | Persistent project root: **`$DATA/hansard_root`** (under `/data/<project>/$USER`, ~5 TiB). `$HOME` is ~15 GiB. |
| Job temp (not our root) | ARC’s **`$SCRATCH` / `$TMPDIR`** are per-job and deleted on exit — unrelated to `hansard_root`. Do not put the venv or HF cache there. |
| Builds | Not on the login node — use `srun -p interactive --pty /bin/bash` (or `--clusters=htc`). |
| GPUs | Only on **htc**. Partitions are time-based (`short` ≤12h, `medium` ≤48h, `long`). |

**GPU choice (queue vs VRAM):** A100s are scarce (~16). Prefer **L40S** (~92, also on medium/long) for embedders; **H100** for larger LLMs but H100 nodes are on **`short` only** (≤12h). See [ARC systems / GPUs](https://arc-user-guide.readthedocs.io/en/latest/arc-systems.html#gpu-resources).

Check: `echo $DATA`; `myquota`.

---

## 1. Project root on `$DATA`

Layout is `$DATA/hansard_root/{hansard,hansard-env,hf_cache,…}` — same role as BMRC’s
`/well/…/hansard-scratch`, different name so it is not confused with ARC `$SCRATCH`.
The env var stays `HANSARD_SCRATCH` (shared with the package / BMRC scripts).

```bash
export HANSARD_SCRATCH="${HANSARD_SCRATCH:-$DATA/hansard_root}"
mkdir -p "$HANSARD_SCRATCH"/{hf_cache,artifacts/llm,data}
```

| Path | Purpose |
|------|---------|
| `$DATA/hansard_root/hansard/` | Git clone |
| `$DATA/hansard_root/hansard-env/` | venv (`VENV_DIR`) |
| `$DATA/hansard_root/hf_cache/` | HF weights (`HF_HOME`) |
| `$DATA/hansard_root/artifacts/llm/` | `eval2k_sample.parquet` + runs |
| `$DATA/hansard_root/data/` | Optional `full_data_enriched.parquet` |
| `~/.config/hansard_llm.arc.env` | ARC paths (BMRC uses `hansard_llm.env`) |

---

## 2. Clone

```bash
cd "$HANSARD_SCRATCH"
git clone <repo-url> hansard
cd hansard
```

Submit jobs from this directory so `cluster/arc/….sbatch` and `logs/` resolve.

---

## 3. Setup script (interactive node)

```bash
srun --clusters=htc -p interactive --pty /bin/bash
cd "$HANSARD_SCRATCH/hansard"
HANSARD_SCRATCH="$HANSARD_SCRATCH" bash cluster/arc/01_setup_env.sh
```

Writes `~/.config/hansard_llm.arc.env` and creates the venv. Does not mkdir/clone.

**Activate (interactive):**

```bash
source ~/.config/hansard_llm.arc.env
source "$VENV_DIR/bin/activate"
```

Do not activate before `sbatch` — job scripts do that.

---

## 4. Upload eval2k

```powershell
scp path\to\eval2k_sample.parquet `
  <user>@htc-login.arc.ox.ac.uk:/data/<project>/<user>/hansard_root/artifacts/llm/
```

(Adjust the `$DATA` path from `echo $DATA` on the cluster.)

---

## 5. Download weights

```bash
source ~/.config/hansard_llm.arc.env
source "$VENV_DIR/bin/activate"
bash cluster/02_download_models.sh embedders   # shared with BMRC
bash cluster/02_download_models.sh llms
```

Run on interactive (or a job with network). Script unsets offline flags; needs `HF_HOME` from the env file.

---

## 6. Submit jobs

From `$HANSARD_SCRATCH/hansard`:

```bash
# Embedders — L40S on short (≤12h)
sbatch cluster/arc/embed_grid.sbatch
sbatch --array=0 cluster/arc/embed_grid.sbatch   # one model

# Panel — H100 on short (≤12h). Override GPU/partition on the CLI if needed.
sbatch --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507,RUN_ARGS="--determinism" \
  cluster/arc/run_grid.sbatch

sbatch --export=ALL,MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 cluster/arc/run_grid.sbatch
```

Defaults in the scripts:

| Script | Cluster | Partition | GPU | Time |
|--------|---------|-----------|-----|------|
| `embed_grid.sbatch` | htc | short | L40S ×1 | 12h |
| `run_grid.sbatch` | htc | short | H100 ×1 | 12h |
| `serve_llm.sbatch` | htc | short | H100 ×1 | 12h |

Overrides:

```bash
# Longer walltime if a job needs it:
sbatch -p medium -t 48:00:00 cluster/arc/embed_grid.sbatch

# Panel on L40S if H100 queue is bad (may need VLLM_ARGS quant for big models):
sbatch --gres=gpu:1 --constraint='gpu_sku:L40S' -p medium -t 48:00:00 \
  --export=ALL,MODEL=...,VLLM_ARGS="--quantization fp8" cluster/arc/run_grid.sbatch
```

If `gpu_sku:L40S` / `H100` is rejected, check live names with ARC docs / `sinfo` and adjust the `#SBATCH --constraint` lines.

Watch: `squeue -u $USER --clusters=htc`. Logs: `logs/` under the repo.

---

## Script map

| Script | Role |
|--------|------|
| `01_setup_env.sh` | venv + `~/.config/hansard_llm.arc.env` |
| `embed_grid.sbatch` | embedder array |
| `run_grid.sbatch` | vLLM + panel |
| `serve_llm.sbatch` | vLLM only |
| `../02_download_models.sh` | HF downloads (shared) |

BMRC originals stay in `cluster/*.sbatch` — do not mix headers.
