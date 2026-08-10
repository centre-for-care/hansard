#!/usr/bin/env bash
# One-time venv + config on Oxford ARC (walkthrough: cluster/arc/README.md).
# Run on an interactive node (not the login node):
#   srun --clusters=htc -p interactive --pty /bin/bash
# Prerequisites (README §§1–2): mkdir $DATA/hansard_root/… and clone into hansard/.
# Then from the clone:
#   bash cluster/arc/01_setup_env.sh
# Writes ~/.config/hansard_llm.arc.env (not the BMRC hansard_llm.env).

set -euo pipefail

SCRATCH_DIR="${HANSARD_SCRATCH:-${DATA:?set DATA or HANSARD_SCRATCH}/hansard_root}"
REPO_DIR="$SCRATCH_DIR/hansard"
[ -d "$REPO_DIR" ] || { echo "missing $REPO_DIR — clone first (cluster/arc/README.md)" >&2; exit 1; }

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
export UV_CACHE_DIR="$SCRATCH_DIR/.uv-cache"   # keep caches off $HOME (15GiB)

ENV_DIR="$SCRATCH_DIR/hansard-env"
uv venv "$ENV_DIR" --python "$(command -v python3)"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

# sentence-transformers <5.6: 5.6 pulls torchcodec needing newer FFmpeg.
uv pip install vllm 'sentence-transformers~=5.1.0' huggingface_hub
uv pip install -e "$REPO_DIR"

CFG="$HOME/.config/hansard_llm.arc.env"
mkdir -p "$(dirname "$CFG")"
cat > "$CFG" <<EOF
# Sourced by cluster/arc/*.sbatch. Distinct from BMRC ~/.config/hansard_llm.env.
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

echo "env ready: source ~/.config/hansard_llm.arc.env && source \"\$VENV_DIR/bin/activate\""
python -c "import vllm, hansard_llm; print('vllm', vllm.__version__)"
