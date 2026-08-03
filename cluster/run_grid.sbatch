#!/usr/bin/env bash
#SBATCH --job-name=hansard-grid
#SBATCH --account=gpu_mills.prj
#SBATCH --partition=gpu_a100_80gb  # gpu_gh200_144gb was DOWN at discovery (2026-08-03);
                                   # switch back once it's up — 144GB fits all panel LLMs bf16
#SBATCH --qos=gpu_bmrc_24hr        # <=24h jobs get a large priority boost
#SBATCH --gres=gpu:1               # 80GB: Nemotron-49B needs VLLM_ARGS="--quantization fp8"
                                   # (or gpu:2 + "--tensor-parallel-size 2" for bf16)
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

# Same-job pattern: start a vLLM OpenAI-compatible server in the background,
# wait for it to come up, then point the existing pipeline client at it via
# LLM_BASE_URL — client.py is already OpenAI-compatible, so zero code changes.
#
#   sbatch --export=ALL,MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507,RUN_ARGS="--panel" \
#          cluster/run_grid.sbatch
#
# Env knobs:
#   MODEL      (required) HF id to serve
#   RUN_ARGS   args for `python -m hansard_llm.run` (the experiment to execute)
#   VLLM_ARGS  extra vllm serve args, e.g. "--quantization fp8" or
#              "--tensor-parallel-size 2" (Nemotron-49B: fp8 on one 80GB GPU)
#   PORT       default 8000+random

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
HOST=$(hostname -s)

echo "serving $MODEL on $HOST:$PORT"
vllm serve "$MODEL" --port "$PORT" --host 0.0.0.0 \
    ${VLLM_ARGS:-} > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!

# Health-poll until the server lists its model (up to ~15 min for big loads).
for i in $(seq 1 90); do
  sleep 10
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then
    echo "server up after ~$((i * 10))s"; break
  fi
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died — see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  [ "$i" -eq 90 ] && { echo "server never came up"; exit 1; }
done

# Point the pipeline at the local server; record the backend in provenance.
export LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
export LLM_API_KEY="local"
export LLM_BACKEND_NAME="vllm-${SLURM_CLUSTER_NAME:-cluster}"

python -m hansard_llm.run ${RUN_ARGS:-} --workers "${WORKERS:-32}"

kill $VLLM_PID 2>/dev/null || true
echo "done"
