#!/usr/bin/env bash
#SBATCH --job-name=embed-grid
#SBATCH --partition=gpu_a100_80gb  # embedders <=8B fit 40GB
#SBATCH --gres=gpu:1
#SBATCH --array=0-7                # one task per embedder (see MODELS below)
#SBATCH --output=logs/%x-%A_%a.out

# Embedder grid: one model per array task (hansard_llm.embedder_grid, ST backend).
# Setup: cluster/README.md. Always pass -A (no #SBATCH --account).
# Submit from $HANSARD_SCRATCH/hansard so logs/ lands next to the repo.
#
#   sbatch -A gpu_<group>.prj cluster/embed_grid.sbatch
#   sbatch -A gpu_<group>.prj --array=0 cluster/embed_grid.sbatch   # one model

set -euo pipefail
source ~/.config/hansard_llm.env
source "${VENV_DIR:?VENV_DIR unset — run cluster/01_setup_env.sh}/bin/activate"
mkdir -p logs

# Keep in sync with 02_download_models.sh EMBEDDERS and embedder_grid.EMBEDDERS.
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
