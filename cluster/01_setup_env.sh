#!/usr/bin/env bash
# One-time x86_64 env setup on a BMRC login node (walkthrough: cluster/README.md):
#   HANSARD_SCRATCH=/well/<group>/users/$USER/hansard-scratch bash cluster/01_setup_env.sh
# Creates the venv (vllm + sentence-transformers + hansard_llm), HF cache and
# artifact dirs on /well, and ~/.config/hansard_llm.env sourced by sbatch jobs.
# GH200 (ARM) serving needs its own venv: 01b_setup_env_gh200.sh.

set -euo pipefail

SCRATCH_DIR="${HANSARD_SCRATCH:?set HANSARD_SCRATCH to /well/<group>/users/$USER/hansard-scratch}"
REPO_URL="${REPO_URL:-}"
REPO_BRANCH="${REPO_BRANCH:-feat/hansard-llm-pipeline}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

mkdir -p "$SCRATCH_DIR"/{hf_cache,artifacts/llm,data}

module load "$PYTHON_MODULE" 2>/dev/null \
  || echo "module $PYTHON_MODULE not found — continuing with system python3"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR="$SCRATCH_DIR/.uv-cache"   # $HOME quota is tiny

ENV_DIR="$SCRATCH_DIR/hansard-env"
uv venv "$ENV_DIR" --python "$(command -v python3)"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# sentence-transformers <5.6: 5.6 pulls torchcodec -> FFmpeg >=6.1; BMRC has 6.0.
uv pip install vllm 'sentence-transformers~=5.1.0' huggingface_hub

if [ -n "$REPO_URL" ]; then
  [ -d "$SCRATCH_DIR/hansard" ] || git clone -b "$REPO_BRANCH" "$REPO_URL" "$SCRATCH_DIR/hansard"
fi
[ -d "$SCRATCH_DIR/hansard" ] && uv pip install -e "$SCRATCH_DIR/hansard"

CFG="$HOME/.config/hansard_llm.env"
mkdir -p "$(dirname "$CFG")"
cat > "$CFG" <<EOF
# module load supplies the interpreter's runtime libs (libffi etc.).
# Offline mode: compute nodes have no internet; 02_download_models unsets it.
type module >/dev/null 2>&1 && module load $PYTHON_MODULE 2>/dev/null || true
export HANSARD_SCRATCH="$SCRATCH_DIR"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export UV_CACHE_DIR="$SCRATCH_DIR/.uv-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HANSARD_LLM_DATA_DIR="$SCRATCH_DIR/data"
export HANSARD_LLM_ARTIFACTS_DIR="$SCRATCH_DIR/artifacts/llm"
export HANSARD_LLM_ENV="$SCRATCH_DIR/.env"
export VENV_DIR="$ENV_DIR"
EOF
echo "wrote $CFG"

echo "env ready: source $ENV_DIR/bin/activate"
python -c "import vllm; print('vllm', vllm.__version__)"
