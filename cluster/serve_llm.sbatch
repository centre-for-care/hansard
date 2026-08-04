#!/usr/bin/env bash
#SBATCH --job-name=vllm-serve
#SBATCH --account=gpu_mills.prj
#SBATCH --partition=gpu_a100_80gb  # gpu_gh200_144gb once it's up (down 2026-08)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Standalone server for interactive use; writes host:port for clients:
#
#   sbatch --export=ALL,MODEL=google/gemma-3-27b-it cluster/serve_llm.sbatch
#   export LLM_BASE_URL="http://$(cat logs/vllm-endpoint-<jobid>)/v1" LLM_API_KEY=local
#
# Sizing (80GB): <=32B bf16 -> gpu:1; Nemotron-49B -> VLLM_ARGS="--quantization fp8"
# (or gpu:2 + "--tensor-parallel-size 2" in bf16).

set -euo pipefail
source ~/.config/hansard_llm.env
if [ "$(uname -m)" = "aarch64" ]; then
  source "${VENV_DIR_ARM:?run 01b_setup_env_gh200.sh first}/bin/activate"
else
  source "$VENV_DIR/bin/activate"
fi
mkdir -p logs

MODEL="${MODEL:?export MODEL=<hf model id>}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
echo "$(hostname -s):$PORT" > "logs/vllm-endpoint-$SLURM_JOB_ID"

exec vllm serve "$MODEL" --port "$PORT" --host 0.0.0.0 ${VLLM_ARGS:-}
