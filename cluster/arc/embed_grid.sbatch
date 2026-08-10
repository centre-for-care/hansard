#!/usr/bin/env bash
#SBATCH --job-name=embed-grid
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu_sku:L40S
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-7
#SBATCH --output=logs/%x-%A_%a.out

# Embedder grid on ARC HTC (L40S default — many cards). short ≤12h.
# Setup: cluster/arc/README.md. Submit from $HANSARD_SCRATCH/hansard.
#
#   sbatch cluster/arc/embed_grid.sbatch
#   sbatch --array=0 cluster/arc/embed_grid.sbatch
# Longer: sbatch -p medium -t 48:00:00 cluster/arc/embed_grid.sbatch

set -euo pipefail
source ~/.config/hansard_llm.arc.env
source "${VENV_DIR:?VENV_DIR unset — run cluster/arc/01_setup_env.sh}/bin/activate"
mkdir -p logs

# Keep in sync with cluster/02_download_models.sh EMBEDDERS and embedder_grid.EMBEDDERS.
MODELS=(
  Qwen/Qwen3-Embedding-0.6B
  Qwen/Qwen3-Embedding-4B
  Qwen/Qwen3-Embedding-8B
  BAAI/bge-base-en-v1.5
  BAAI/bge-large-en-v1.5
  Alibaba-NLP/gte-large-en-v1.5
  intfloat/e5-large-v2
  nomic-ai/nomic-embed-text-v1.5
)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "embedder grid: $MODEL"
python -m hansard_llm.embedder_grid --model "$MODEL" --backend st
