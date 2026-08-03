#!/usr/bin/env bash
# One-time environment setup on the cluster (BMRC login node). Everything
# lives in GROUP area — BMRC policy is all data under
# /well/<group>/users/<username>/ ($HOME has a tiny quota). Check the /well
# quota from the discovery output first: models + caches need ~400GB.
#
#   HANSARD_SCRATCH=/well/<group>/users/$USER/hansard-scratch \
#     bash cluster/01_setup_env.sh
#
# Creates:
#   $HANSARD_SCRATCH/hansard-env/     uv virtualenv with vllm + sentence-transformers
#   $HANSARD_SCRATCH/hf_cache/        HuggingFace model cache
#   $HANSARD_SCRATCH/hansard/         clone of the code repo (if REPO_URL set)
#   $HANSARD_SCRATCH/artifacts/llm/   experiment artifacts (results store)
#   ~/.config/hansard_llm.env         env vars sourced by the sbatch templates
#
# NOTE (arch): this builds the x86_64 venv used for the embedder grid
# (A100/RTX8000 partitions) and general analysis. LLM serving runs on the
# GH200 partition (ARM) with its own venv — build that one afterwards with
# 01b_setup_env_gh200.sh ON a GH200 node.

set -euo pipefail

SCRATCH_DIR="${HANSARD_SCRATCH:?set HANSARD_SCRATCH to /well/<group>/users/$USER/hansard-scratch}"
REPO_URL="${REPO_URL:-}"          # optional: git URL; otherwise rsync the repo yourself
REPO_BRANCH="${REPO_BRANCH:-feat/hansard-llm-pipeline}"  # branch to clone (never touches main)
# BMRC module system (EasyBuild); this is the documented default Python.
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

mkdir -p "$SCRATCH_DIR"/{hf_cache,artifacts/llm,data}

module load "$PYTHON_MODULE" 2>/dev/null \
  || echo "module $PYTHON_MODULE not found — continuing with system python3"

# uv: fast, no root, resolves once. Install to ~/.local/bin if absent.
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

ENV_DIR="$SCRATCH_DIR/hansard-env"
# Use the module's interpreter (matches cluster glibc); fall back to any 3.11+.
uv venv "$ENV_DIR" --python "$(command -v python3)"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# vLLM wheels bundle their CUDA runtime; only the node driver matters.
uv pip install vllm sentence-transformers huggingface_hub[hf_transfer]

# The pipeline package (editable, so cluster-side pulls take effect).
if [ -n "$REPO_URL" ]; then
  [ -d "$SCRATCH_DIR/hansard" ] || git clone -b "$REPO_BRANCH" "$REPO_URL" "$SCRATCH_DIR/hansard"
fi
[ -d "$SCRATCH_DIR/hansard" ] && uv pip install -e "$SCRATCH_DIR/hansard"

# Central env file sourced by every sbatch template.
CFG="$HOME/.config/hansard_llm.env"
mkdir -p "$(dirname "$CFG")"
cat > "$CFG" <<EOF
export HANSARD_SCRATCH="$SCRATCH_DIR"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
# Compute nodes have no internet: make HF read straight from the /well cache
# instead of retrying the hub. UNSET these when downloading on a login node
# (02_download_models.sh does this itself).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HANSARD_LLM_DATA_DIR="$SCRATCH_DIR/data"
export HANSARD_LLM_ARTIFACTS_DIR="$SCRATCH_DIR/artifacts/llm"
export HANSARD_LLM_ENV="$SCRATCH_DIR/.env"
export VENV_DIR="$ENV_DIR"
EOF
echo "wrote $CFG"

# Data note: copy the corpus + pilot artifacts up once. BMRC does NOT allow
# SSH keys (password + 2FA on every connection), so from Windows prefer
# FileZilla (SFTP to cluster2.bmrc.ox.ac.uk, 1 connection, 600s timeout — per
# BMRC docs) or run rsync from WSL where ControlMaster lets one login serve
# many transfers. Files to upload:
#   hansard_eda/data/full_data_enriched.parquet      -> $SCRATCH_DIR/data/
#   hansard_eda/artifacts/llm/pilot_sample.parquet   -> $SCRATCH_DIR/artifacts/llm/
#   hansard_eda/artifacts/llm/eval10k_sample.parquet -> $SCRATCH_DIR/artifacts/llm/
#   hansard_eda/artifacts/llm/legacy/                -> $SCRATCH_DIR/artifacts/llm/legacy/
# And create $SCRATCH_DIR/.env with LLM_BASE_URL/LLM_API_KEY (Nebius fallback);
# the vLLM sbatch templates override these per job.

echo "env ready: source $ENV_DIR/bin/activate"
python -c "import vllm; print('vllm', vllm.__version__)"
