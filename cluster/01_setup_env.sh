#!/usr/bin/env bash
# One-time x86_64 venv + config on a BMRC login node (walkthrough: cluster/README.md).
# Prerequisites (README §§1–2):
#   export HANSARD_SCRATCH=/well/<group>/users/$USER/hansard-scratch
#   mkdir -p "$HANSARD_SCRATCH"/{hf_cache,artifacts/llm,data}
#   git clone <repo-url> "$HANSARD_SCRATCH/hansard"
# Then from the clone:
#   HANSARD_SCRATCH=... bash cluster/01_setup_env.sh
# Creates hansard-env + ~/.config/hansard_llm.env for sbatch jobs.
# Does not mkdir the project root or clone. GH200: 01b_setup_env_gh200.sh.

set -euo pipefail

SCRATCH_DIR="${HANSARD_SCRATCH:?set HANSARD_SCRATCH to /well/<group>/users/$USER/hansard-scratch}"
REPO_DIR="$SCRATCH_DIR/hansard"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

if [ ! -d "$SCRATCH_DIR" ]; then
  echo "missing $SCRATCH_DIR — create the project root first (see cluster/README.md §1)" >&2
  exit 1
fi
for d in hf_cache artifacts/llm data; do
  if [ ! -d "$SCRATCH_DIR/$d" ]; then
    echo "missing $SCRATCH_DIR/$d — mkdir the §1 subdirs first (see cluster/README.md)" >&2
    exit 1
  fi
done
if [ ! -d "$REPO_DIR" ]; then
  echo "missing $REPO_DIR — clone the repo there first (see cluster/README.md §§1–2)" >&2
  exit 1
fi

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
uv pip install -e "$REPO_DIR"

CFG="$HOME/.config/hansard_llm.env"
mkdir -p "$(dirname "$CFG")"
cat > "$CFG" <<EOF
# Sourced by every cluster/*.sbatch job. module load = libffi etc. for the venv.
# Offline: compute nodes have no internet; 02_download_models.sh unsets these.
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

echo "env ready: source ~/.config/hansard_llm.env && source \"\$VENV_DIR/bin/activate\""
python -c "import vllm, hansard_llm; print('vllm', vllm.__version__)"
