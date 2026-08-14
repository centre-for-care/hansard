#!/usr/bin/env bash
#SBATCH --job-name=hansard-grid
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_sku:H100
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=logs/%x-%j.out

# vLLM + hansard_llm.panel on ARC HTC (H100 on short ≤12h).
# Setup: cluster/arc/README.md. Submit from $HANSARD_SCRATCH/hansard.
#
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-4B-Instruct-2507,RUN_ARGS="--determinism" \
#          cluster/arc/run_grid.sbatch
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 cluster/arc/run_grid.sbatch
# Longer / L40S: sbatch -p medium -t 48:00:00 --constraint=gpu_sku:L40S ...

set -euo pipefail
source ~/.config/hansard_llm.arc.env
source "${VENV_DIR:?VENV_DIR unset — run cluster/arc/01_setup_env.sh}/bin/activate"
mkdir -p logs

MODEL="${MODEL:?export MODEL=<hf model id>}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
HOST=$(hostname -s)

echo "serving $MODEL on $HOST:$PORT"
vllm serve "$MODEL" --port "$PORT" --host 0.0.0.0 \
    ${VLLM_ARGS:-} > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!

for i in $(seq 1 90); do
  sleep 10
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then
    echo "server up after ~$((i * 10))s"; break
  fi
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died — see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  [ "$i" -eq 90 ] && { echo "server never came up"; exit 1; }
done

export LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
export LLM_API_KEY="local"
export LLM_BACKEND_NAME="vllm-arc"

python -m hansard_llm.panel --model "$MODEL" ${RUN_ARGS:-} \
    --workers "${WORKERS:-32}"

kill $VLLM_PID 2>/dev/null || true
echo "done"
