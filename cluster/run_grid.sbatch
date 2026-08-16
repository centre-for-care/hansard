#!/usr/bin/env bash
#SBATCH --job-name=hansard-grid
#SBATCH --partition=gpu_a100_80gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

# vLLM in the background + hansard_llm.panel against LLM_BASE_URL.
# Setup: cluster/README.md. Always pass -A (no #SBATCH --account).
# Submit from $HANSARD_SCRATCH/hansard so logs/ lands next to the repo.
#
#   sbatch -A gpu_<group>.prj --qos gpu_bmrc_24hr \
#          --export=ALL,MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 cluster/run_grid.sbatch
#   sbatch -A gpu_<group>.prj --qos gpu_bmrc_24hr \
#          --export=ALL,MODEL=Qwen/Qwen3-14B,RUN_ARGS="--extended" cluster/run_grid.sbatch
#   sbatch -A gpu_<group>.prj --qos gpu_bmrc_24hr \
#          --export=ALL,MODEL=nvidia/Llama-3_3-Nemotron-Super-49B-v1_5,\
#          VLLM_ARGS="--quantization fp8 --trust-remote-code" cluster/run_grid.sbatch
#
# MODEL (required); RUN_ARGS (--extended, --determinism); VLLM_ARGS; PORT; WORKERS;
# MAX_MODEL_LEN (default 32768 — Qwen native 262k OOMs KV cache on 80GB).

set -euo pipefail
source ~/.config/hansard_llm.env
if [ "$(uname -m)" = "aarch64" ]; then
  source "${VENV_DIR_ARM:?run cluster/01b_setup_env_gh200.sh first}/bin/activate"
else
  source "${VENV_DIR:?VENV_DIR unset — run cluster/01_setup_env.sh}/bin/activate"
fi
mkdir -p logs

# A100 enables FlashInfer; vLLM then JIT-compiles the sampler and needs nvcc.
# BMRC nodes have the driver, not /usr/local/cuda. The 4B interactive run
# skipped this because RTX 6000 is CC 7.5. Native sampler is fine.
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${MODEL:?export MODEL=<hf model id>}"
PORT="${PORT:-$((8000 + RANDOM % 1000))}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
HOST=$(hostname -s)

echo "serving $MODEL on $HOST:$PORT (max-model-len $MAX_MODEL_LEN)"
vllm serve "$MODEL" --port "$PORT" --host 0.0.0.0 \
    --max-model-len "$MAX_MODEL_LEN" --trust-remote-code \
    ${VLLM_ARGS:-} > "logs/vllm-$SLURM_JOB_ID.log" 2>&1 &
VLLM_PID=$!

# Health-poll until the server lists its model (up to ~30 min for 32B/49B on GPFS).
for i in $(seq 1 180); do
  sleep 10
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q "$MODEL"; then
    echo "server up after ~$((i * 10))s"; break
  fi
  kill -0 $VLLM_PID 2>/dev/null || { echo "vLLM died — see logs/vllm-$SLURM_JOB_ID.log"; exit 1; }
  [ "$i" -eq 180 ] && { echo "server never came up"; exit 1; }
done

export LLM_BASE_URL="http://127.0.0.1:$PORT/v1"
export LLM_API_KEY="local"
export LLM_BACKEND_NAME="vllm-${SLURM_CLUSTER_NAME:-cluster}"

python -m hansard_llm.panel --model "$MODEL" ${RUN_ARGS:-} \
    --workers "${WORKERS:-32}"

kill $VLLM_PID 2>/dev/null || true
echo "done"
