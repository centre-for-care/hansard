#!/usr/bin/env bash
#SBATCH --job-name=vllm-serve
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_sku:H100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out

# Standalone vLLM on ARC HTC. Setup: cluster/arc/README.md.
#
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507 cluster/arc/serve_llm.sbatch
#   export LLM_BASE_URL="http://$(cat logs/vllm-endpoint-<jobid>)/v1" LLM_API_KEY=local

set -euo pipefail
source ~/.config/hansard_llm.arc.env
source "${VENV_DIR:?VENV_DIR unset — run cluster/arc/01_setup_env.sh}/bin/activate"
mkdir -p logs

MODEL="${MODEL:?export MODEL=<hf model id>}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
echo "$(hostname -s):$PORT" > "logs/vllm-endpoint-$SLURM_JOB_ID"

exec vllm serve "$MODEL" --port "$PORT" --host 0.0.0.0 ${VLLM_ARGS:-}
