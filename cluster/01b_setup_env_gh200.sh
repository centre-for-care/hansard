#!/usr/bin/env bash
# ARM (aarch64) venv for vLLM on the GH200 nodes. ALL downloading happens on
# the login node into /well; the GPU node never touches the internet. The
# only thing that must run on a GH200 node is the final venv creation +
# wheel unpack (~2 min, offline), because a venv is wired to a concrete
# Python interpreter and only an ARM Python can be that interpreter — the
# login node's x86 binaries cannot execute on the Grace CPUs.
#
#   # phase 1 — LOGIN node (x86, internet): download aarch64 wheels to /well
#   bash cluster/01b_setup_env_gh200.sh download
#
#   # phase 2 — GH200 node (ARM, offline, one-time): unpack wheels into venv.
#   # Non-interactive srun: runs the install and releases the GPU immediately.
#   srun -A gpu_mills.prj -p gpu_gh200_144gb --gres=gpu:1 -t 30 --mem=32G \
#        bash cluster/01b_setup_env_gh200.sh install
#
# After phase 2 the venv lives in /well like everything else and every job
# just sources it. Requires 01_setup_env.sh to have run first (uses
# ~/.config/hansard_llm.env and $HANSARD_SCRATCH). The download phase fails
# fast on the login node if any dependency lacks an aarch64 wheel — nothing
# breaks inside a job.
#
# Appends VENV_DIR_ARM to ~/.config/hansard_llm.env; the sbatch templates
# pick x86 vs ARM venv automatically via `uname -m`.

set -euo pipefail
PHASE="${1:?usage: $0 [download|install]}"

# shellcheck disable=SC1090
source "$HOME/.config/hansard_llm.env"
SCRATCH_DIR="${HANSARD_SCRATCH:?run 01_setup_env.sh first}"
WHEEL_DIR="$SCRATCH_DIR/wheels-aarch64"

PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"
module load "$PYTHON_MODULE" 2>/dev/null \
  || echo "module $PYTHON_MODULE not found — using system python3"

# Everything the ARM venv needs: vllm (brings torch etc.) + the hansard_llm
# runtime deps (pyproject.toml), pinned loosely enough for aarch64 wheels.
PKGS=(vllm "huggingface_hub[hf_transfer]"
      openai python-dotenv pandas pyarrow "duckdb==1.5.5" numpy
      scikit-learn matplotlib setuptools wheel)

case "$PHASE" in
  download)
    [ "$(uname -m)" = "x86_64" ] || echo "note: expected to run on a login node"
    mkdir -p "$WHEEL_DIR"
    python3 -m pip download \
      --platform manylinux2014_aarch64 --only-binary=:all: \
      --python-version 3.11 --implementation cp \
      -d "$WHEEL_DIR" "${PKGS[@]}"
    echo; echo "wheels ready: $(ls "$WHEEL_DIR" | wc -l) files, $(du -sh "$WHEEL_DIR" | cut -f1)"
    echo "now: srun onto a GH200 node and run '$0 install'"
    ;;

  install)
    [ "$(uname -m)" = "aarch64" ] || {
      echo "ERROR: $(uname -m) node — the install phase must run on a GH200 node"; exit 1; }
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
    echo "recorded VENV_DIR_ARM in $CFG"

    python -c "import torch, vllm; print('torch', torch.__version__, '| vllm', vllm.__version__, '| cuda', torch.cuda.is_available())"
    ;;

  *) echo "usage: $0 [download|install]"; exit 1 ;;
esac
