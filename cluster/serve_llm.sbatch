#!/usr/bin/env bash
#SBATCH --job-name=vllm-serve
#SBATCH --account=gpu_mills.prj
#SBATCH --partition=gpu_a100_80gb  # gpu_gh200_144gb was DOWN at discovery (2026-08-03);
                                   # switch back once it's up — 144GB fits all panel LLMs bf16
#SBATCH --qos=gpu_bmrc_24hr        # priority boost; caps runtime at 24h
#SBATCH --gres=gpu:1               # per model (see table below)
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
# Arch-aware venv: GH200 nodes are ARM and need the venv from 01b.
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
