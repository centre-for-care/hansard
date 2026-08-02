#!/usr/bin/env bash
# One-time environment setup on the cluster. Run on the login node AFTER
# reviewing 00_discovery.sh output and setting HANSARD_SCRATCH below (or in the
# environment) to a filesystem with >=500GB quota (model weights + caches must
# NOT go to a small $HOME).
#
#   HANSARD_SCRATCH=/scratch/$USER bash cluster/01_setup_env.sh
#
# Creates:
#   $HANSARD_SCRATCH/hansard-env/     uv virtualenv with vllm + sentence-transformers
#   $HANSARD_SCRATCH/hf_cache/        HuggingFace model cache
#   $HANSARD_SCRATCH/hansard/         clone of the code repo (if REPO_URL set)
#   $HANSARD_SCRATCH/artifacts/llm/   experiment artifacts (results store)
#   ~/.config/hansard_llm.env         env vars sourced by the sbatch templates

set -euo pipefail

SCRATCH_DIR="${HANSARD_SCRATCH:?set HANSARD_SCRATCH to your scratch path}"
REPO_URL="${REPO_URL:-}"          # optional: git URL; otherwise rsync the repo yourself
PYTHON_MODULE="${PYTHON_MODULE:-}" # e.g. "python/3.12" if the cluster needs a module

mkdir -p "$SCRATCH_DIR"/{hf_cache,artifacts/llm,data}

[ -n "$PYTHON_MODULE" ] && module load "$PYTHON_MODULE"

# uv: fast, no root, resolves once. Install to ~/.local/bin if absent.
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

ENV_DIR="$SCRATCH_DIR/hansard-env"
uv venv "$ENV_DIR" --python 3.12
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# vLLM wheels bundle their CUDA runtime; only the node driver matters.
uv pip install vllm sentence-transformers huggingface_hub[hf_transfer]

# The pipeline package (editable, so cluster-side pulls take effect).
if [ -n "$REPO_URL" ]; then
  [ -d "$SCRATCH_DIR/hansard" ] || git clone "$REPO_URL" "$SCRATCH_DIR/hansard"
fi
[ -d "$SCRATCH_DIR/hansard" ] && uv pip install -e "$SCRATCH_DIR/hansard"

# Central env file sourced by every sbatch template.
CFG="$HOME/.config/hansard_llm.env"
mkdir -p "$(dirname "$CFG")"
cat > "$CFG" <<EOF
export HANSARD_SCRATCH="$SCRATCH_DIR"
export HF_HOME="$SCRATCH_DIR/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export HANSARD_LLM_DATA_DIR="$SCRATCH_DIR/data"
export HANSARD_LLM_ARTIFACTS_DIR="$SCRATCH_DIR/artifacts/llm"
export HANSARD_LLM_ENV="$SCRATCH_DIR/.env"
export VENV_DIR="$ENV_DIR"
EOF
echo "wrote $CFG"

# Data note: copy the corpus + pilot artifacts up once, e.g. from your laptop:
#   rsync -av --progress hansard_eda/data/full_data_enriched.parquet  cluster:$SCRATCH_DIR/data/
#   rsync -av --progress hansard_eda/artifacts/llm/pilot_sample.parquet cluster:$SCRATCH_DIR/artifacts/llm/
#   rsync -av --progress hansard_eda/artifacts/llm/legacy               cluster:$SCRATCH_DIR/artifacts/llm/
# And create $SCRATCH_DIR/.env with LLM_BASE_URL/LLM_API_KEY (Nebius fallback);
# the vLLM sbatch templates override these per job.

echo "env ready: source $ENV_DIR/bin/activate"
python -c "import vllm; print('vllm', vllm.__version__)"
