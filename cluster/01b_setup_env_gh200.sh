#!/usr/bin/env bash
# ARM (aarch64) venv for vLLM on GH200 nodes (details: cluster/README.md).
# Downloads happen on the login node; the GPU-node step is an offline unpack
# only — a venv must be created by an ARM python, which exists only there.
#
#   bash cluster/01b_setup_env_gh200.sh download    # login node
#   srun -A gpu_<group>.prj -p gpu_gh200_144gb --gres=gpu:1 -t 30 --mem=32G \
#        bash cluster/01b_setup_env_gh200.sh install

set -euo pipefail
PHASE="${1:?usage: $0 [download|install]}"

# shellcheck disable=SC1090
source "$HOME/.config/hansard_llm.env"
SCRATCH_DIR="${HANSARD_SCRATCH:?run 01_setup_env.sh first}"
WHEEL_DIR="$SCRATCH_DIR/wheels-aarch64"
export PIP_CACHE_DIR="$SCRATCH_DIR/.pip-cache"

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
module load "$PYTHON_MODULE" 2>/dev/null \
  || echo "module $PYTHON_MODULE not found — using system python3"

# vllm (brings torch) + hansard_llm runtime deps from pyproject.toml.
PKGS=(vllm huggingface_hub
      openai python-dotenv pandas pyarrow "duckdb==1.5.5" numpy
      scikit-learn matplotlib setuptools wheel)

case "$PHASE" in
  download)
    mkdir -p "$WHEEL_DIR"
    python3 -m pip download \
      --platform manylinux2014_aarch64 --only-binary=:all: \
      --python-version 3.11 --implementation cp \
      -d "$WHEEL_DIR" "${PKGS[@]}"
    echo "wheels ready: $(ls "$WHEEL_DIR" | wc -l) files, $(du -sh "$WHEEL_DIR" | cut -f1)"
    ;;

  install)
    [ "$(uname -m)" = "aarch64" ] || {
      echo "ERROR: $(uname -m) node — install must run on a GH200 node"; exit 1; }
    ENV_DIR="$SCRATCH_DIR/hansard-env-arm"
    python3 -m venv "$ENV_DIR"
    # shellcheck disable=SC1091
    source "$ENV_DIR/bin/activate"
    pip install --no-index --find-links "$WHEEL_DIR" "${PKGS[@]}"
    [ -d "$SCRATCH_DIR/hansard" ] && \
      pip install --no-index --no-deps --no-build-isolation -e "$SCRATCH_DIR/hansard"

    CFG="$HOME/.config/hansard_llm.env"
    grep -q VENV_DIR_ARM "$CFG" \
      && sed -i "s|^export VENV_DIR_ARM=.*|export VENV_DIR_ARM=\"$ENV_DIR\"|" "$CFG" \
      || echo "export VENV_DIR_ARM=\"$ENV_DIR\"" >> "$CFG"

    python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__, '| cuda', torch.cuda.is_available())"
    ;;

  *) echo "usage: $0 [download|install]"; exit 1 ;;
esac
