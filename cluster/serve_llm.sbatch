#!/usr/bin/env bash
#SBATCH --job-name=vllm-serve
#SBATCH --partition=gpu            # EDIT: from 00_discovery output
#SBATCH --gres=gpu:1               # EDIT per model (see table below)
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# Standalone server job, for interactive experimentation from a login-node
# shell (or several client jobs sharing one server). Writes host:port to
# logs/vllm-endpoint-$SLURM_JOB_ID so clients can find it:
#
#   sbatch --export=ALL,MODEL=google/gemma-3-27b-it cluster/serve_llm.sbatch
#   # then on the login node:
#   export LLM_BASE_URL="http://$(cat logs/vllm-endpoint-<jobid>)/v1" LLM_API_KEY=local
#
# Sizing guide (80GB GPU): <=32B bf16 -> gpu:1. Nemotron-Super-49B ->
# gpu:1 + VLLM_ARGS="--quantization fp8" (or gpu:2 +
# "--tensor-parallel-size 2" in bf16). MoE 30B-A3B -> gpu:1.

set -euo pipefail
source ~/.config/hansard_llm.env
source "$VENV_DIR/bin/activate"
mkdir -p logs

MODEL="${MODEL:?export MODEL=<hf model id>}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
echo "$(hostname -s):$PORT" > "logs/vllm-endpoint-$SLURM_JOB_ID"

exec vllm serve "$MODEL" --port "$PORT" --host 0.0.0.0 ${VLLM_ARGS:-}
